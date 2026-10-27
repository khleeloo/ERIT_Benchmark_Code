"""
Evaluate Qwen2-VL on ERIT dataset.
Three conditions: image-only, text-only, fusion (image+text).
Outputs results CSV compatible with existing pipeline format.
"""

import argparse
import os
import time
from datetime import datetime

import pandas as pd
import torch
from PIL import Image
from transformers import Qwen2VLForConditionalGeneration, AutoProcessor

EMOTIONS = ["happiness", "sadness", "disgust", "anger", "surprise", "fear", "neutral"]

# ── Prompt builders (Qwen2-VL uses chat template with message dicts) ────────

def make_messages_text(instruction):
    return [
        {"role": "system", "content": [{"type": "text", "text":
            "You are an emotion recognition model. Analyze the text and respond "
            "with only one label: happiness, sadness, disgust, anger, surprise, "
            "fear, or neutral. No additional text."}]},
        {"role": "user", "content": [{"type": "text", "text":
            f"Input: {instruction}"}]},
    ]

def make_messages_image():
    return [
        {"role": "system", "content": [{"type": "text", "text":
            "You are an image emotion recognition model. Analyze the image and "
            "respond with only one label: happiness, sadness, disgust, anger, "
            "surprise, fear, or neutral. No additional text."}]},
        {"role": "user", "content": [
            {"type": "image"},
            {"type": "text", "text": "What emotion is shown in this image?"},
        ]},
    ]

def make_messages_fusion(instruction):
    return [
        {"role": "system", "content": [{"type": "text", "text":
            "You are a multimodal emotion recognition model. Analyze both text "
            "and image and respond with only one label: happiness, sadness, "
            "disgust, anger, surprise, fear, or neutral. No additional text."}]},
        {"role": "user", "content": [
            {"type": "image"},
            {"type": "text", "text":
                f"Text: {instruction}\n"
                "Analyze the emotion in both the text and image."},
        ]},
    ]

# ── Helpers ─────────────────────────────────────────────────────────────────

def normalize_prediction(raw: str) -> str:
    raw_lower = raw.strip().lower()
    for emo in EMOTIONS:
        if emo in raw_lower:
            return emo
    return raw_lower.split()[0] if raw_lower else "unknown"


def predict(model, processor, messages, images=None, max_new_tokens=20):
    """Run a single Qwen2-VL prediction."""
    text_prompt = processor.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    if images is not None:
        inputs = processor(
            text=[text_prompt], images=images,
            padding=True, return_tensors="pt"
        ).to(model.device)
    else:
        inputs = processor(
            text=[text_prompt], images=None,
            padding=True, return_tensors="pt"
        ).to(model.device)

    with torch.no_grad():
        output_ids = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
        )
    generated = output_ids[0][inputs["input_ids"].shape[-1]:]
    return processor.decode(generated, skip_special_tokens=True)


# ── Main ────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Evaluate Qwen2-VL on ERIT")
    parser.add_argument("--model-id", type=str,
                        default="Qwen/Qwen2-VL-7B-Instruct",
                        help="HuggingFace model ID")
    parser.add_argument("--erit-root", type=str,
                        default="/scratch/vemotionsys/rmfrieske/ERIT")
    parser.add_argument("--split", type=str, default="test",
                        choices=["test", "dev", "train"])
    parser.add_argument("--output-dir", type=str,
                        default="/home/rmfrieske/datasets/ERIT/results")
    args = parser.parse_args()

    # ── Paths ───────────────────────────────────────────────────────────────
    split_map = {"test": "ERIT_test/test.csv",
                 "dev": "ERIT_dev/dev.csv",
                 "train": "ERIT_train/train.csv"}
    csv_path = os.path.join(args.erit_root, split_map[args.split])
    img_dir = os.path.dirname(csv_path)

    print(f"Loading dataset from {csv_path}")
    df = pd.read_csv(csv_path)
    print(f"  {len(df)} samples, columns: {list(df.columns)}")

    # ── Load model ──────────────────────────────────────────────────────────
    print(f"Loading model: {args.model_id}")
    model = Qwen2VLForConditionalGeneration.from_pretrained(
        args.model_id,
        torch_dtype=torch.float16,
        device_map="auto",
        low_cpu_mem_usage=True,
    )
    processor = AutoProcessor.from_pretrained(args.model_id)
    model.eval()
    print(f"  Model loaded on {model.device}")

    # ── Evaluate ────────────────────────────────────────────────────────────
    results = []
    correct = {"text": 0, "image": 0, "fusion": 0}
    total = 0
    t0 = time.time()

    for idx, row in df.iterrows():
        img_filename = row["url"]
        instruction = str(row.get("instruction", ""))
        gt_label = str(row["caption"]).strip().lower()

        img_path = os.path.join(img_dir, img_filename)
        if not os.path.exists(img_path):
            print(f"  [SKIP] Image not found: {img_path}")
            continue

        image = Image.open(img_path).convert("RGB")
        total += 1

        # ── Text-only ──
        msgs_t = make_messages_text(instruction)
        raw_t = predict(model, processor, msgs_t, images=None)
        pred_t = normalize_prediction(raw_t)

        # ── Image-only ──
        msgs_i = make_messages_image()
        raw_i = predict(model, processor, msgs_i, images=[image])
        pred_i = normalize_prediction(raw_i)

        # ── Fusion ──
        msgs_f = make_messages_fusion(instruction)
        raw_f = predict(model, processor, msgs_f, images=[image])
        pred_f = normalize_prediction(raw_f)

        if pred_t == gt_label:
            correct["text"] += 1
        if pred_i == gt_label:
            correct["image"] += 1
        if pred_f == gt_label:
            correct["fusion"] += 1

        results.append({
            "ImageName": img_path,
            "Transcript": instruction,
            "Prediction_Fusion": pred_f,
            "Prediction_Text": pred_t,
            "Prediction_Image": pred_i,
            "GTLabel": gt_label,
        })

        if (idx + 1) % 50 == 0:
            elapsed = time.time() - t0
            print(f"  [{idx+1}/{len(df)}] "
                  f"Acc — T:{correct['text']/total*100:.1f}% "
                  f"I:{correct['image']/total*100:.1f}% "
                  f"F:{correct['fusion']/total*100:.1f}% "
                  f"({elapsed:.0f}s)")

    # ── Save ────────────────────────────────────────────────────────────────
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
    print(f"TEXT  Accuracy: {correct['text']/total*100:.2f}% ({correct['text']}/{total})")
    print(f"IMAGE Accuracy: {correct['image']/total*100:.2f}% ({correct['image']}/{total})")
    print(f"FUSION Accuracy: {correct['fusion']/total*100:.2f}% ({correct['fusion']}/{total})")
    print(f"Time: {elapsed:.0f}s ({elapsed/total:.1f}s/sample)")
    print(f"Results saved to: {out_path}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
