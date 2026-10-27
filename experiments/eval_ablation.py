"""
Evaluate LLaVA-NeXT zero-shot on ERIT frame selection ablation variants.

Runs image-only and fusion (image+text) on DeepFace, random, and midpoint
frame variants. Text-only is identical across variants (same transcripts).

Usage:
    python experiments/eval_ablation.py \
        --erit-csv /scratch/vemotionsys/rmfrieske/ERIT/ERIT_test/test.csv \
        --random-csv /scratch/vemotionsys/rmfrieske/ERIT/ERIT_test_ablation/test_random.csv \
        --random-img-dir /scratch/vemotionsys/rmfrieske/ERIT/ERIT_test_ablation/random_frames \
        --midpoint-csv /scratch/vemotionsys/rmfrieske/ERIT/ERIT_test_ablation/test_midpoint.csv \
        --midpoint-img-dir /scratch/vemotionsys/rmfrieske/ERIT/ERIT_test_ablation/midpoint_frames \
        --output-dir /home/rmfrieske/datasets/ERIT/results/ablation
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

warnings.filterwarnings("ignore", message=".*pad_token_id.*")

EMOTIONS = ["happiness", "sadness", "disgust", "anger", "surprise", "fear", "neutral"]


def normalize_prediction(raw):
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
    return [{"role": "user", "content": [{"type": "image"}, {"type": "text", "text": text}]}]


def predict(model, processor, conversation, image, max_new_tokens=20):
    prompt = processor.apply_chat_template(conversation, add_generation_prompt=True)
    inputs = processor(images=image, text=prompt, return_tensors="pt").to(model.device)
    with torch.inference_mode():
        output_ids = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            pad_token_id=processor.tokenizer.eos_token_id,
        )
    generated = output_ids[0][inputs["input_ids"].shape[-1]:]
    return processor.decode(generated, skip_special_tokens=True)


def evaluate_variant(model, processor, df, img_dir, variant_name):
    """Evaluate image-only and fusion on a single variant."""
    correct = {"image": 0, "fusion": 0}
    correct_multi = {"image": 0, "fusion": 0}
    total = 0
    skipped = 0
    results = []
    t0 = time.time()

    for idx, row in df.iterrows():
        img_filename = row["url"]
        instruction = str(row.get("instruction", ""))
        gt_label = str(row["caption"]).strip().lower()
        orig_labels = str(row.get("OriginalLabels", gt_label))
        gt_set = {normalize_prediction(l.strip()) for l in orig_labels.split(",")}

        img_path = os.path.join(img_dir, img_filename)
        if not os.path.exists(img_path):
            skipped += 1
            continue

        image = Image.open(img_path).convert("RGB")
        total += 1

        # Image-only
        conv_i = build_conv_image()
        raw_i = predict(model, processor, conv_i, image)
        pred_i = normalize_prediction(raw_i)

        # Fusion
        conv_f = build_conv_image(instruction)
        raw_f = predict(model, processor, conv_f, image)
        pred_f = normalize_prediction(raw_f)

        if pred_i == gt_label: correct["image"] += 1
        if pred_f == gt_label: correct["fusion"] += 1
        if pred_i in gt_set: correct_multi["image"] += 1
        if pred_f in gt_set: correct_multi["fusion"] += 1

        results.append({
            "url": img_filename,
            "instruction": instruction,
            "pred_image": pred_i,
            "pred_fusion": pred_f,
            "gt_label": gt_label,
            "orig_labels": orig_labels,
            "variant": variant_name,
        })

        if (total) % 100 == 0:
            elapsed = time.time() - t0
            print(f"  [{variant_name}] {total} done | "
                  f"I:{correct['image']/total*100:.1f}% "
                  f"F:{correct['fusion']/total*100:.1f}% "
                  f"({elapsed:.0f}s)")

    elapsed = time.time() - t0
    print(f"\n  {variant_name}: {total} samples ({skipped} skipped), {elapsed:.0f}s")
    print(f"    Image  - Strict: {correct['image']/total*100:.2f}%  Multi-label: {correct_multi['image']/total*100:.2f}%")
    print(f"    Fusion - Strict: {correct['fusion']/total*100:.2f}%  Multi-label: {correct_multi['fusion']/total*100:.2f}%")

    return results, {
        "variant": variant_name,
        "n_samples": total,
        "image_strict": correct["image"] / total * 100 if total else 0,
        "image_multi": correct_multi["image"] / total * 100 if total else 0,
        "fusion_strict": correct["fusion"] / total * 100 if total else 0,
        "fusion_multi": correct_multi["fusion"] / total * 100 if total else 0,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-id", default="llava-hf/llava-v1.6-mistral-7b-hf")
    parser.add_argument("--erit-csv", required=True)
    parser.add_argument("--random-csv", required=True)
    parser.add_argument("--random-img-dir", required=True)
    parser.add_argument("--midpoint-csv", required=True)
    parser.add_argument("--midpoint-img-dir", required=True)
    parser.add_argument("--ensemble-csv", default=None)
    parser.add_argument("--ensemble-img-dir", default=None)
    parser.add_argument("--output-dir", default="/home/rmfrieske/datasets/ERIT/results/ablation")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    # Load model once
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

    # Variant 1: DeepFace (original ERIT)
    print("\n=== DeepFace (original ERIT) ===")
    df_deepface = pd.read_csv(args.erit_csv)
    erit_img_dir = os.path.dirname(args.erit_csv)
    res_df, stats_df = evaluate_variant(model, processor, df_deepface, erit_img_dir, "deepface")

    # Variant 2: Random
    print("\n=== Random ===")
    df_random = pd.read_csv(args.random_csv)
    res_rand, stats_rand = evaluate_variant(model, processor, df_random, args.random_img_dir, "random")

    # Variant 3: Midpoint
    print("\n=== Midpoint ===")
    df_midpoint = pd.read_csv(args.midpoint_csv)
    res_mid, stats_mid = evaluate_variant(model, processor, df_midpoint, args.midpoint_img_dir, "midpoint")

    all_stats = [stats_df, stats_rand, stats_mid]
    all_res = res_df + res_rand + res_mid

    # Variant 4: Ensemble (optional)
    if args.ensemble_csv and args.ensemble_img_dir:
        print("\n=== Ensemble ===")
        df_ensemble = pd.read_csv(args.ensemble_csv)
        res_ens, stats_ens = evaluate_variant(model, processor, df_ensemble, args.ensemble_img_dir, "ensemble")
        all_stats.append(stats_ens)
        all_res += res_ens

    # Summary table
    summary = pd.DataFrame(all_stats)
    print("\n" + "=" * 70)
    print("FRAME SELECTION ABLATION SUMMARY")
    print("=" * 70)
    print(summary.to_string(index=False))

    timestamp = datetime.now().strftime("%Y_%m_%d-%H%M%S")
    summary.to_csv(os.path.join(args.output_dir, f"ablation_summary_{timestamp}.csv"), index=False)

    # Save all predictions
    all_results = pd.DataFrame(all_res)
    all_results.to_csv(os.path.join(args.output_dir, f"ablation_all_predictions_{timestamp}.csv"), index=False)

    print(f"\nSaved to {args.output_dir}")


if __name__ == "__main__":
    main()
