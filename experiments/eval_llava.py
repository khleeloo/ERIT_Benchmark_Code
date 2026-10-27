"""
Evaluate LLaVA-NeXT (llava-v1.6-mistral-7b) on ERIT dataset.
Three conditions: image-only, text-only, fusion (image+text).
Uses processor.apply_chat_template for correct Mistral [INST] formatting.
"""

import argparse
import os
import time
import warnings
from datetime import datetime

import pandas as pd
import torch
from PIL import Image
from transformers import LlavaNextProcessor, LlavaNextForConditionalGeneration

# Suppress pad_token_id warning spam
warnings.filterwarnings("ignore", message=".*pad_token_id.*")

EMOTIONS = ["happiness", "sadness", "disgust", "anger", "surprise", "fear", "neutral"]


def normalize_prediction(raw: str) -> str:
    """Extract a valid emotion label from model output."""
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


def build_conv_image(instruction=None):
    """Build conversation for image-only or fusion."""
    if instruction:
        text = (
            f'The person in this image said: "{instruction}"\n'
            "Based on both the facial expression and what they said, "
            "what emotion are they expressing? "
            "Respond with exactly one word: happiness, sadness, disgust, "
            "anger, surprise, fear, or neutral."
        )
    else:
        text = (
            "What emotion is the person in this image expressing? "
            "Respond with exactly one word: happiness, sadness, disgust, "
            "anger, surprise, fear, or neutral."
        )
    return [{"role": "user", "content": [
        {"type": "image"},
        {"type": "text", "text": text},
    ]}]


def build_conv_text(instruction):
    """Build conversation for text-only (will use dummy image)."""
    return [{"role": "user", "content": [
        {"type": "image"},
        {"type": "text", "text": (
            "Ignore the image. Focus only on the following text.\n"
            f'Someone said: "{instruction}"\n'
            "Based on this text alone, what emotion are they expressing? "
            "Respond with exactly one word: happiness, sadness, disgust, "
            "anger, surprise, fear, or neutral."
        )},
    ]}]


def predict(model, processor, conversation, image, max_new_tokens=15):
    """Run a single prediction using apply_chat_template."""
    prompt = processor.apply_chat_template(conversation, add_generation_prompt=True)
    inputs = processor(text=prompt, images=image, return_tensors="pt").to(model.device)
    with torch.no_grad():
        output_ids = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            pad_token_id=processor.tokenizer.eos_token_id,
        )
    generated = output_ids[0][inputs["input_ids"].shape[-1]:]
    return processor.decode(generated, skip_special_tokens=True)


def main():
    parser = argparse.ArgumentParser(description="Evaluate LLaVA-NeXT on ERIT")
    parser.add_argument("--model-id", type=str,
                        default="llava-hf/llava-v1.6-mistral-7b-hf")
    parser.add_argument("--erit-root", type=str,
                        default="/scratch/vemotionsys/rmfrieske/ERIT")
    parser.add_argument("--split", type=str, default="test",
                        choices=["test", "dev", "train"])
    parser.add_argument("--output-dir", type=str,
                        default="/home/rmfrieske/datasets/ERIT/results")
    args = parser.parse_args()

    split_map = {"test": "ERIT_test/test.csv",
                 "dev": "ERIT_dev/dev.csv",
                 "train": "ERIT_train/train.csv"}
    csv_path = os.path.join(args.erit_root, split_map[args.split])
    img_dir = os.path.dirname(csv_path)

    print(f"Loading dataset from {csv_path}")
    df = pd.read_csv(csv_path)
    print(f"  {len(df)} samples")

    print(f"Loading model: {args.model_id}")
    processor = LlavaNextProcessor.from_pretrained(args.model_id)
    model = LlavaNextForConditionalGeneration.from_pretrained(
        args.model_id,
        torch_dtype=torch.float16,
        device_map="auto",
        low_cpu_mem_usage=True,
    )
    model.eval()
    print(f"  Model loaded, device: {model.device}")

    # Dummy image for text-only (LLaVA requires an image)
    dummy_image = Image.new("RGB", (64, 64), color=(128, 128, 128))

    # Sanity check
    first_row = df.iloc[0]
    test_img = Image.open(os.path.join(img_dir, first_row["url"])).convert("RGB")
    conv = build_conv_image()
    raw = predict(model, processor, conv, test_img)
    print(f"  Sanity: raw='{raw}' -> {normalize_prediction(raw)} (GT: {first_row['OriginalLabels']})")

    results = []
    correct = {"text": 0, "image": 0, "fusion": 0}
    correct_multi = {"text": 0, "image": 0, "fusion": 0}
    total = 0
    t0 = time.time()

    for idx, row in df.iterrows():
        img_filename = row["url"]
        instruction = str(row.get("instruction", ""))
        gt_label = str(row["caption"]).strip().lower()
        orig_labels = str(row.get("OriginalLabels", gt_label))
        gt_set = {normalize_prediction(l.strip()) for l in orig_labels.split(",")}

        img_path = os.path.join(img_dir, img_filename)
        if not os.path.exists(img_path):
            print(f"  [SKIP] {img_path}")
            continue

        image = Image.open(img_path).convert("RGB")
        total += 1

        # Text-only (with dummy image)
        conv_t = build_conv_text(instruction)
        raw_t = predict(model, processor, conv_t, dummy_image)
        pred_t = normalize_prediction(raw_t)

        # Image-only
        conv_i = build_conv_image()
        raw_i = predict(model, processor, conv_i, image)
        pred_i = normalize_prediction(raw_i)

        # Fusion
        conv_f = build_conv_image(instruction)
        raw_f = predict(model, processor, conv_f, image)
        pred_f = normalize_prediction(raw_f)

        if pred_t == gt_label: correct["text"] += 1
        if pred_i == gt_label: correct["image"] += 1
        if pred_f == gt_label: correct["fusion"] += 1
        if pred_t in gt_set: correct_multi["text"] += 1
        if pred_i in gt_set: correct_multi["image"] += 1
        if pred_f in gt_set: correct_multi["fusion"] += 1

        results.append({
            "ImageName": img_path,
            "Transcript": instruction,
            "Prediction_Fusion": pred_f,
            "Prediction_Text": pred_t,
            "Prediction_Image": pred_i,
            "GTLabel": gt_label,
            "OriginalLabels": orig_labels,
        })

        if (idx + 1) % 50 == 0:
            elapsed = time.time() - t0
            print(f"  [{idx+1}/{len(df)}] "
                  f"T:{correct['text']/total*100:.1f}% "
                  f"I:{correct['image']/total*100:.1f}% "
                  f"F:{correct['fusion']/total*100:.1f}% "
                  f"({elapsed:.0f}s)")

    os.makedirs(args.output_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y_%m_%d-%H%M%S")
    model_short = args.model_id.split("/")[-1]
    out_path = os.path.join(args.output_dir,
                            f"results_{model_short}_{args.split}_{timestamp}.csv")
    pd.DataFrame(results).to_csv(out_path, index=False)

    elapsed = time.time() - t0
    print(f"\n{'='*60}")
    print(f"Model: {args.model_id}")
    print(f"Split: {args.split} | Samples: {total}")
    print(f"TEXT   Accuracy: {correct['text']/total*100:.2f}% ({correct['text']}/{total})  multi: {correct_multi['text']/total*100:.2f}% ({correct_multi['text']}/{total})")
    print(f"IMAGE  Accuracy: {correct['image']/total*100:.2f}% ({correct['image']}/{total})  multi: {correct_multi['image']/total*100:.2f}% ({correct_multi['image']}/{total})")
    print(f"FUSION Accuracy: {correct['fusion']/total*100:.2f}% ({correct['fusion']}/{total})  multi: {correct_multi['fusion']/total*100:.2f}% ({correct_multi['fusion']}/{total})")
    print(f"Time: {elapsed:.0f}s ({elapsed/total:.1f}s/sample)")
    print(f"Results saved to: {out_path}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
