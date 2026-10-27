"""Evaluate fine-tuned LLaVA-NeXT LoRA checkpoint on ERIT test set."""

import argparse
import os
import warnings
from datetime import datetime

import pandas as pd
import torch
from PIL import Image
from tqdm import tqdm
from peft import PeftModel
from transformers import LlavaNextProcessor, LlavaNextForConditionalGeneration, BitsAndBytesConfig

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


def build_conv_fusion(instruction):
    return [{"role": "user", "content": [
        {"type": "image"},
        {"type": "text", "text": (
            f'The person in this image said: "{instruction}"\n'
            "Based on both the facial expression and what they said, "
            "what emotion are they expressing? "
            "Respond with exactly one word: happiness, sadness, disgust, "
            "anger, surprise, fear, or neutral."
        )},
    ]}]


def build_conv_image():
    return [{"role": "user", "content": [
        {"type": "image"},
        {"type": "text", "text": (
            "What emotion is the person in this image expressing? "
            "Respond with exactly one word: happiness, sadness, disgust, "
            "anger, surprise, fear, or neutral."
        )},
    ]}]


def build_conv_text(instruction):
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


def predict(model, processor, conversation, image, device):
    prompt = processor.apply_chat_template(conversation, add_generation_prompt=True)
    inputs = processor(text=prompt, images=image, return_tensors="pt").to(device)
    with torch.no_grad():
        output_ids = model.generate(
            **inputs, max_new_tokens=10, do_sample=False,
            pad_token_id=processor.tokenizer.eos_token_id,
        )
    generated = output_ids[0][inputs["input_ids"].shape[-1]:]
    raw = processor.decode(generated, skip_special_tokens=True)
    return normalize_prediction(raw), raw.strip()[:50]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True, help="Path to LoRA checkpoint dir")
    parser.add_argument("--model-id", default="llava-hf/llava-v1.6-mistral-7b-hf")
    parser.add_argument("--erit-root", default="/scratch/vemotionsys/rmfrieske/datasets/ERIT_new")
    parser.add_argument("--split", default="test", choices=["test", "dev", "dev+test"])
    parser.add_argument("--output-dir", default="/home/rmfrieske/datasets/ERIT/results")
    parser.add_argument("--load-in-4bit", action="store_true")
    args = parser.parse_args()

    splits = args.split.split("+")
    dfs = []
    img_dirs = {}
    for s in splits:
        csv_path = os.path.join(args.erit_root, s, f"{s}.csv")
        sdf = pd.read_csv(csv_path)
        sdf["_split"] = s
        sdf["_img_dir"] = os.path.join(args.erit_root, s)
        dfs.append(sdf)
        img_dirs[s] = os.path.join(args.erit_root, s)

    print(f"Loading processor: {args.model_id}")
    processor = LlavaNextProcessor.from_pretrained(args.model_id)

    print(f"Loading base model: {args.model_id}")
    quant_config = None
    if args.load_in_4bit:
        quant_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_quant_type="nf4",
        )
    model = LlavaNextForConditionalGeneration.from_pretrained(
        args.model_id,
        quantization_config=quant_config,
        torch_dtype=torch.float16,
        device_map="auto",
    )

    print(f"Loading LoRA adapter: {args.checkpoint}")
    model = PeftModel.from_pretrained(model, args.checkpoint)
    model.eval()

    device = next(model.parameters()).device
    df = pd.concat(dfs, ignore_index=True)
    print(f"Evaluating {len(df)} samples from {args.split}")

    # Dummy image for text-only
    dummy_image = Image.new("RGB", (64, 64), color=(128, 128, 128))

    results = []
    correct_strict = {"fusion": 0, "image": 0, "text": 0}
    correct_multi = {"fusion": 0, "image": 0, "text": 0}
    total = 0

    for _, row in tqdm(df.iterrows(), total=len(df), desc="Eval"):
        caption_label = row["caption"].strip().lower()
        orig = str(row.get("OriginalLabels", row["caption"]))
        valid_labels = set(l.strip().lower() for l in orig.split(","))
        instruction = str(row["instruction"]).strip().strip('"')
        img_dir = row["_img_dir"]
        img_path = os.path.join(img_dir, row["url"])

        try:
            image = Image.open(img_path).convert("RGB")
        except Exception:
            continue

        # Fusion
        pred_fus, raw_fus = predict(model, processor, build_conv_fusion(instruction), image, device)
        # Image only
        pred_img, _ = predict(model, processor, build_conv_image(), image, device)
        # Text only
        pred_txt, _ = predict(model, processor, build_conv_text(instruction), dummy_image, device)

        for mode, pred in [("fusion", pred_fus), ("image", pred_img), ("text", pred_txt)]:
            if pred == caption_label:
                correct_strict[mode] += 1
            if pred in valid_labels:
                correct_multi[mode] += 1
        total += 1

        results.append({
            "ImageName": row["url"],
            "Transcript": instruction,
            "Prediction_Fusion": pred_fus,
            "Prediction_Image": pred_img,
            "Prediction_Text": pred_txt,
            "GTLabel": caption_label,
            "OriginalLabels": orig,
            "RawOutput": raw_fus,
        })

    print(f"\n=== Results on {args.split} ({total} samples) ===")
    for mode in ["fusion", "image", "text"]:
        s_acc = correct_strict[mode] / total if total else 0
        m_acc = correct_multi[mode] / total if total else 0
        print(f"  {mode:7s} — strict: {s_acc:.4f} ({correct_strict[mode]}/{total})  "
              f"multi: {m_acc:.4f} ({correct_multi[mode]}/{total})")

    # Save CSV
    os.makedirs(args.output_dir, exist_ok=True)
    ts = datetime.now().strftime("%Y_%m_%d-%H%M%S")
    csv_out = os.path.join(args.output_dir, f"results_llava_lora_{args.split}_{ts}.csv")
    pd.DataFrame(results).to_csv(csv_out, index=False)
    print(f"  Results saved to {csv_out}")


if __name__ == "__main__":
    main()
