"""
ERIT benchmark evaluation harness for Qwen2-VL.

Same structure as experiments/eval_benchmark.py but uses Qwen2-VL architecture
(Qwen2VLForConditionalGeneration + AutoProcessor).

Usage:
    python experiments/eval_benchmark_qwen2vl.py \
        --benchmark-dir /scratch/vemotionsys/rmfrieske/ERIT_benchmark \
        --model-id Qwen/Qwen2-VL-7B-Instruct \
        --output-dir /home/rmfrieske/datasets/ERIT/results/benchmark
"""

import argparse
import os
import time
import warnings
from datetime import datetime

import pandas as pd
import torch
from PIL import Image
from transformers import Qwen2VLForConditionalGeneration, AutoProcessor

warnings.filterwarnings("ignore", message=".*pad_token_id.*")

EMOTIONS = ["happiness", "sadness", "disgust", "anger", "surprise", "fear", "neutral"]

# ── Condition × modality relevance matrix ─────────────────────────────
CONDITION_MODALITY = {
    "baseline":      {"image": True,  "text": True,  "fusion": True},
    "no_keyword":    {"image": False, "text": True,  "fusion": True},
    "grayscale":     {"image": True,  "text": False, "fusion": True},
    "blurred_faces": {"image": True,  "text": False, "fusion": True},
    "text_dropout":  {"image": False, "text": True,  "fusion": True},
    "image_dropout": {"image": True,  "text": False, "fusion": True},
}


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


# ── Qwen2-VL prompt builders ─────────────────────────────────────────

def build_msgs_image():
    return [
        {"role": "system", "content": [{"type": "text", "text":
            "You are an emotion recognition model. Respond with exactly one word: "
            "happiness, sadness, disgust, anger, surprise, fear, or neutral."}]},
        {"role": "user", "content": [
            {"type": "image"},
            {"type": "text", "text": "What emotion is the person in this image expressing?"},
        ]},
    ]


def build_msgs_text(instruction):
    return [
        {"role": "system", "content": [{"type": "text", "text":
            "You are an emotion recognition model. Respond with exactly one word: "
            "happiness, sadness, disgust, anger, surprise, fear, or neutral."}]},
        {"role": "user", "content": [
            {"type": "text", "text":
                f'Someone said: "{instruction}"\n'
                "Based on this text alone, what emotion are they expressing?"},
        ]},
    ]


def build_msgs_fusion(instruction):
    return [
        {"role": "system", "content": [{"type": "text", "text":
            "You are a multimodal emotion recognition model. Respond with exactly one word: "
            "happiness, sadness, disgust, anger, surprise, fear, or neutral."}]},
        {"role": "user", "content": [
            {"type": "image"},
            {"type": "text", "text":
                f'The person in this image said: "{instruction}"\n'
                "Based on both the facial expression and what they said, "
                "what emotion are they expressing?"},
        ]},
    ]


def predict(model, processor, messages, images=None, max_new_tokens=15):
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


def evaluate_condition(model, processor, df, img_dir, condition):
    """Evaluate one condition across all relevant modalities."""
    relevance = CONDITION_MODALITY.get(condition, {})
    results = []
    modalities_to_run = [m for m in ["image", "text", "fusion"] if relevance.get(m, False)]

    if not modalities_to_run:
        return results

    print(f"  Modalities: {modalities_to_run}")
    t0 = time.time()
    n_done = 0

    for idx, row in df.iterrows():
        img_path = os.path.join(img_dir, row["url"])
        instruction = str(row.get("instruction", ""))
        gt_label = str(row["caption"]).strip().lower()
        orig_labels = str(row.get("OriginalLabels", gt_label))
        split = row.get("_split", "unknown")

        for modality in modalities_to_run:
            if modality == "image":
                if not os.path.exists(img_path):
                    continue
                image = Image.open(img_path).convert("RGB")
                msgs = build_msgs_image()
                raw = predict(model, processor, msgs, images=[image])
            elif modality == "text":
                msgs = build_msgs_text(instruction)
                raw = predict(model, processor, msgs, images=None)
            else:  # fusion
                if not os.path.exists(img_path):
                    continue
                image = Image.open(img_path).convert("RGB")
                msgs = build_msgs_fusion(instruction)
                raw = predict(model, processor, msgs, images=[image])

            pred = normalize_prediction(raw)
            results.append({
                "condition": condition,
                "modality": modality,
                "url": row["url"],
                "prediction": pred,
                "gt_label": gt_label,
                "original_labels": orig_labels,
                "split": split,
            })

        n_done += 1
        if n_done % 100 == 0:
            elapsed = time.time() - t0
            print(f"    [{n_done}/{len(df)}] {elapsed:.0f}s")

    elapsed = time.time() - t0
    print(f"  Done: {n_done} samples × {len(modalities_to_run)} modalities in {elapsed:.0f}s")
    return results


def main():
    parser = argparse.ArgumentParser(description="ERIT benchmark evaluation — Qwen2-VL")
    parser.add_argument("--benchmark-dir", required=True)
    parser.add_argument("--model-id", type=str, default="Qwen/Qwen2-VL-7B-Instruct")
    parser.add_argument("--output-dir", type=str,
                        default="/home/rmfrieske/datasets/ERIT/results/benchmark")
    parser.add_argument("--conditions", nargs="*", default=None)
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    # Discover conditions
    if args.conditions:
        conditions = args.conditions
    else:
        conditions = sorted([
            d for d in os.listdir(args.benchmark_dir)
            if os.path.isdir(os.path.join(args.benchmark_dir, d))
            and os.path.exists(os.path.join(args.benchmark_dir, d, "benchmark.csv"))
        ])

    print(f"Conditions to evaluate: {conditions}")

    # Load model
    print(f"Loading model: {args.model_id}")
    model = Qwen2VLForConditionalGeneration.from_pretrained(
        args.model_id,
        torch_dtype=torch.float16,
        device_map="auto",
        low_cpu_mem_usage=True,
    )
    processor = AutoProcessor.from_pretrained(args.model_id)
    model.eval()
    print(f"  Model loaded, device: {model.device}")

    all_results = []

    for cond in conditions:
        cond_dir = os.path.join(args.benchmark_dir, cond)
        csv_path = os.path.join(cond_dir, "benchmark.csv")

        print(f"\n{'='*50}")
        print(f"Condition: {cond}")
        print(f"{'='*50}")

        df = pd.read_csv(csv_path)
        print(f"  {len(df)} samples")

        results = evaluate_condition(model, processor, df, cond_dir, cond)
        all_results.extend(results)

    # Save consolidated results
    df_out = pd.DataFrame(all_results)
    model_short = args.model_id.split("/")[-1]
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_path = os.path.join(args.output_dir, f"benchmark_{model_short}_{timestamp}.csv")
    df_out.to_csv(out_path, index=False)
    print(f"\nSaved {len(df_out)} predictions to {out_path}")

    # Quick summary
    print(f"\n{'='*50}")
    print("Quick summary (strict accuracy):")
    print(f"{'='*50}")
    for cond in df_out["condition"].unique():
        for mod in df_out[df_out["condition"] == cond]["modality"].unique():
            mask = (df_out["condition"] == cond) & (df_out["modality"] == mod)
            subset = df_out[mask]
            acc = (subset["prediction"] == subset["gt_label"]).mean() * 100
            print(f"  {cond:16s} {mod:8s}  Acc={acc:.1f}%  (n={len(subset)})")


if __name__ == "__main__":
    main()
