"""
Fine-tune LLaVA-NeXT (Mistral-7B) on ERIT with LoRA via HuggingFace PEFT.
Trains on ERIT_train, evaluates on ERIT_dev at end of each epoch.
"""

import argparse
import os
import random
import warnings
from datetime import datetime

import numpy as np
import pandas as pd
import torch
from PIL import Image
from tqdm import tqdm
from torch.utils.data import Dataset, DataLoader
from transformers import (
    LlavaNextProcessor,
    LlavaNextForConditionalGeneration,
    BitsAndBytesConfig,
)
from peft import LoraConfig, get_peft_model, TaskType

warnings.filterwarnings("ignore", message=".*pad_token_id.*")

EMOTIONS = ["happiness", "sadness", "disgust", "anger", "surprise", "fear", "neutral"]


def normalize_prediction(raw: str) -> str:
    raw_lower = raw.strip().lower().rstrip(".")
    if raw_lower in EMOTIONS:
        return raw_lower
    for emo in EMOTIONS:
        if emo in raw_lower:
            return emo
    alt_map = {
        "happy": "happiness", "sad": "sadness", "angry": "anger",
        "disgusted": "disgust", "surprised": "surprise", "fearful": "fear",
        "scared": "fear", "content": "happiness", "joy": "happiness",
    }
    for alt, emo in alt_map.items():
        if alt in raw_lower:
            return emo
    return raw_lower.split()[0] if raw_lower else "unknown"


class ERITDataset(Dataset):
    """ERIT dataset for fine-tuning: returns (conversation, image, label_text)."""

    def __init__(self, csv_path, img_dir, processor, mode="fusion"):
        self.df = pd.read_csv(csv_path)
        self.img_dir = img_dir
        self.processor = processor
        self.mode = mode

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        # Randomly sample from all valid labels (OriginalLabels) to handle ambiguity
        orig = str(row.get("OriginalLabels", row["caption"]))
        valid_labels = [l.strip().lower() for l in orig.split(",")]
        label = random.choice(valid_labels)
        instruction = str(row["instruction"]).strip().strip('"')

        # Load image
        img_path = os.path.join(self.img_dir, row["url"])
        try:
            image = Image.open(img_path).convert("RGB")
        except Exception:
            image = Image.new("RGB", (384, 384), (128, 128, 128))

        # Build prompt (fusion: image + text)
        if self.mode == "fusion":
            user_text = (
                f'The person in this image said: "{instruction}"\n'
                "Based on both the facial expression and what they said, "
                "what emotion are they expressing? "
                "Respond with exactly one word: happiness, sadness, disgust, "
                "anger, surprise, fear, or neutral."
            )
        elif self.mode == "image":
            user_text = (
                "What emotion is the person in this image expressing? "
                "Respond with exactly one word: happiness, sadness, disgust, "
                "anger, surprise, fear, or neutral."
            )
        else:
            user_text = (
                "Ignore the image. Focus only on the following text.\n"
                f'Someone said: "{instruction}"\n'
                "Based on this text alone, what emotion are they expressing? "
                "Respond with exactly one word: happiness, sadness, disgust, "
                "anger, surprise, fear, or neutral."
            )

        conversation = [{"role": "user", "content": [
            {"type": "image"},
            {"type": "text", "text": user_text},
        ]}]

        prompt = self.processor.apply_chat_template(
            conversation, add_generation_prompt=True
        )
        # Build proper assistant turn: space + label + EOS
        eos = self.processor.tokenizer.eos_token or "</s>"
        full_text = prompt + " " + label + eos

        return image, full_text, prompt, label


def collate_fn(batch, processor):
    images, full_texts, prompts, labels = zip(*batch)

    # Process full sequences (prompt + label) for training
    inputs = processor(
        text=list(full_texts),
        images=list(images),
        padding=True,
        truncation=True,
        return_tensors="pt",
    )

    # Process prompts only to find where labels start
    prompt_inputs = processor(
        text=list(prompts),
        images=list(images),
        padding=False,  # no padding for length measurement
        truncation=True,
        return_tensors=None,  # return lists
    )

    # Build labels: mask everything except the answer tokens
    input_ids = inputs["input_ids"]
    label_ids = input_ids.clone()

    for i in range(len(batch)):
        prompt_len = len(prompt_inputs["input_ids"][i])
        label_ids[i, :prompt_len] = -100

    # Mask padding tokens
    if "attention_mask" in inputs:
        label_ids[inputs["attention_mask"] == 0] = -100

    inputs["labels"] = label_ids
    return inputs


@torch.no_grad()
def evaluate(model, processor, eval_csv, eval_img_dir, device, n_print=3):
    """Zero-shot-style eval on dev set using greedy decoding."""
    df = pd.read_csv(eval_csv)
    correct = 0
    total = 0
    samples = []  # collect sample predictions to print

    model.eval()
    pbar = tqdm(df.iterrows(), total=len(df), desc="  Eval", leave=False)
    for _, row in pbar:
        # Accept any label from OriginalLabels as correct
        orig = str(row.get("OriginalLabels", row["caption"]))
        valid_labels = set(l.strip().lower() for l in orig.split(","))
        instruction = str(row["instruction"]).strip().strip('"')
        img_path = os.path.join(eval_img_dir, row["url"])

        try:
            image = Image.open(img_path).convert("RGB")
        except Exception:
            continue

        conversation = [{"role": "user", "content": [
            {"type": "image"},
            {"type": "text", "text": (
                f'The person in this image said: "{instruction}"\n'
                "Based on both the facial expression and what they said, "
                "what emotion are they expressing? "
                "Respond with exactly one word: happiness, sadness, disgust, "
                "anger, surprise, fear, or neutral."
            )},
        ]}]

        prompt = processor.apply_chat_template(
            conversation, add_generation_prompt=True
        )
        inputs = processor(
            text=prompt, images=image, return_tensors="pt"
        ).to(device)

        output_ids = model.generate(
            **inputs,
            max_new_tokens=10,
            do_sample=False,
            pad_token_id=processor.tokenizer.eos_token_id,
        )
        generated = output_ids[0][inputs["input_ids"].shape[-1]:]
        raw = processor.decode(generated, skip_special_tokens=True)
        pred = normalize_prediction(raw)

        hit = pred in valid_labels
        if hit:
            correct += 1
        total += 1

        if len(samples) < n_print:
            samples.append({
                "img": row["url"],
                "text": instruction[:60],
                "raw": raw.strip()[:40],
                "pred": pred,
                "gt": ", ".join(sorted(valid_labels)),
                "hit": hit,
            })

        pbar.set_postfix(acc=f"{correct/total:.3f}")
    pbar.close()

    # Print sample predictions
    print("  Sample predictions:")
    for s in samples:
        mark = "OK" if s["hit"] else "MISS"
        print(f"    [{mark}] raw='{s['raw']}' pred={s['pred']}  gt={{{s['gt']}}}")
        print(f"         img={s['img']}  text='{s['text']}...'")

    acc = correct / total if total > 0 else 0
    return acc, correct, total


def main():
    parser = argparse.ArgumentParser(description="Fine-tune LLaVA-NeXT on ERIT with LoRA")
    parser.add_argument("--model-id", type=str,
                        default="llava-hf/llava-v1.6-mistral-7b-hf")
    parser.add_argument("--erit-root", type=str,
                        default="/scratch/vemotionsys/rmfrieske/datasets/ERIT_new")
    parser.add_argument("--output-dir", type=str,
                        default="/home/rmfrieske/datasets/ERIT/checkpoints/llava_lora")
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--lora-rank", type=int, default=8)
    parser.add_argument("--lora-alpha", type=int, default=16)
    parser.add_argument("--lora-dropout", type=float, default=0.1)
    parser.add_argument("--lora-targets", type=str, default="q_proj,v_proj",
                        help="Comma-separated LoRA target modules")
    parser.add_argument("--lora-layers", type=int, default=4,
                        help="Apply LoRA to last N transformer layers only (0=all)")
    parser.add_argument("--grad-accum", type=int, default=4)
    parser.add_argument("--train-mode", type=str, default="fusion",
                        choices=["fusion", "image", "text"])
    parser.add_argument("--load-in-4bit", action="store_true",
                        help="Load base model in 4-bit quantization")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    # Paths
    train_csv = os.path.join(args.erit_root, "train/train.csv")
    train_img = os.path.join(args.erit_root, "train")
    dev_csv = os.path.join(args.erit_root, "dev/dev.csv")
    dev_img = os.path.join(args.erit_root, "dev")

    print(f"Loading processor: {args.model_id}")
    processor = LlavaNextProcessor.from_pretrained(args.model_id)
    # Ensure pad token is set
    if processor.tokenizer.pad_token is None:
        processor.tokenizer.pad_token = processor.tokenizer.eos_token

    # Model loading
    print(f"Loading model: {args.model_id}")
    model_kwargs = {
        "torch_dtype": torch.float16,
        "device_map": "auto",
    }
    if args.load_in_4bit:
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_use_double_quant=True,
        )
        model_kwargs["quantization_config"] = bnb_config

    model = LlavaNextForConditionalGeneration.from_pretrained(
        args.model_id, **model_kwargs
    )

    # Apply LoRA
    print(f"Applying LoRA: rank={args.lora_rank}, alpha={args.lora_alpha}")
    target_modules = [m.strip() for m in args.lora_targets.split(",")]
    n_text_layers = model.config.text_config.num_hidden_layers  # 32 for Mistral-7B
    lora_kwargs = dict(
        r=args.lora_rank,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        target_modules=target_modules,
        bias="none",
        task_type=TaskType.CAUSAL_LM,
    )
    if args.lora_layers > 0:
        layers = list(range(n_text_layers - args.lora_layers, n_text_layers))
        lora_kwargs["layers_to_transform"] = layers
        print(f"  target_modules={target_modules}, layers={layers}")
    else:
        print(f"  target_modules={target_modules}, layers=ALL {n_text_layers}")
    lora_config = LoraConfig(**lora_kwargs)
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    # Enable gradient checkpointing
    model.enable_input_require_grads()
    if hasattr(model, "gradient_checkpointing_enable"):
        model.gradient_checkpointing_enable()

    # Dataset
    print(f"Loading training data from {train_csv}")
    train_ds = ERITDataset(train_csv, train_img, processor, mode=args.train_mode)
    print(f"  {len(train_ds)} training samples")

    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=4,
        collate_fn=lambda b: collate_fn(b, processor),
        pin_memory=True,
    )

    # Optimizer + cosine LR scheduler
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.lr, weight_decay=0.01
    )
    total_steps = (len(train_loader) // args.grad_accum) * args.epochs
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=total_steps, eta_min=args.lr * 0.1
    )

    # Training loop
    device = next(model.parameters()).device
    best_acc = 0.0
    ts = datetime.now().strftime("%Y_%m_%d-%H%M%S")

    print(f"\nStarting training: {args.epochs} epochs, batch_size={args.batch_size}, "
          f"grad_accum={args.grad_accum}, effective_batch={args.batch_size * args.grad_accum}")

    for epoch in range(args.epochs):
        model.train()
        total_loss = 0.0
        num_steps = 0
        optimizer.zero_grad()

        pbar = tqdm(train_loader, desc=f"  Epoch {epoch+1}/{args.epochs}",
                     leave=False)
        for step, batch in enumerate(pbar):
            batch = {k: v.to(device) if hasattr(v, "to") else v
                     for k, v in batch.items()}

            outputs = model(**batch)
            loss = outputs.loss / args.grad_accum
            loss.backward()

            total_loss += outputs.loss.item()
            num_steps += 1

            if (step + 1) % args.grad_accum == 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()

            pbar.set_postfix(loss=f"{total_loss/num_steps:.4f}")
        pbar.close()

        # Final gradient step if needed
        if num_steps % args.grad_accum != 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()

        avg_loss = total_loss / max(num_steps, 1)
        print(f"Epoch {epoch+1}/{args.epochs}: avg_loss={avg_loss:.4f}")

        # Evaluate on dev set
        print("  Evaluating on dev set...")
        acc, correct, total = evaluate(model, processor, dev_csv, dev_img, device)
        print(f"  Dev accuracy: {acc:.4f} ({correct}/{total})")

        # Save every epoch
        epoch_path = os.path.join(args.output_dir, f"epoch{epoch+1}_{ts}")
        model.save_pretrained(epoch_path)
        processor.save_pretrained(epoch_path)
        print(f"  Epoch {epoch+1} saved to {epoch_path}")

        # Track best
        if acc > best_acc:
            best_acc = acc
            print(f"  New best dev accuracy: {best_acc:.4f}")

    print(f"\nTraining complete. Best dev accuracy: {best_acc:.4f}")


if __name__ == "__main__":
    main()
