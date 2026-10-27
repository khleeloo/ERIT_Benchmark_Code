"""
Build ERIT_new dataset from ensemble-selected frames + Whisper transcripts.

Assembles train/dev/test splits from:
  1. Ensemble frame selection outputs (images + emotion labels)
  2. Whisper Large V3 transcripts with confidence scores

Structure:
    ERIT_new/
        train/
            train.csv       (columns: url, instruction, OriginalLabels, caption,
                             GTEmotionScore, whisper_mean_conf, whisper_n_low_conf)
            *.jpg
        dev/  ...
        test/ ...

Usage:
    python erit/build_erit_new.py \
        --output-dir /scratch/vemotionsys/rmfrieske/datasets/ERIT_new \
        --transcript-dir /scratch/vemotionsys/rmfrieske/datasets/ERIT_new_transcripts
"""

import argparse
import json
import os
import shutil

import pandas as pd


# Columns to keep in the final CSVs
KEEP_COLS = [
    "url", "instruction", "OriginalLabels", "caption", "GTEmotionScore",
    "whisper_mean_conf", "whisper_n_low_conf",
]

TRANSCRIPT_DIR_DEFAULT = "/scratch/vemotionsys/rmfrieske/datasets/ERIT_new_transcripts"

SPLITS = {
    "train": {
        "ensemble_csv": "/scratch/vemotionsys/rmfrieske/ERIT/ERIT_train_ensemble/train_ensemble.csv",
        "ensemble_frames": "/scratch/vemotionsys/rmfrieske/ERIT/ERIT_train_ensemble/consensus_frames",
        "out_csv_name": "train.csv",
    },
    "dev": {
        "ensemble_csv": "/scratch/vemotionsys/rmfrieske/ERIT/ERIT_dev_ensemble/dev_ensemble.csv",
        "ensemble_frames": "/scratch/vemotionsys/rmfrieske/ERIT/ERIT_dev_ensemble/consensus_frames",
        "out_csv_name": "dev.csv",
    },
    "test": {
        "ensemble_csv": "/scratch/vemotionsys/rmfrieske/ERIT/ERIT_test_ensemble/test_ensemble.csv",
        "ensemble_frames": "/scratch/vemotionsys/rmfrieske/ERIT/ERIT_test_ensemble/consensus_frames",
        "out_csv_name": "test.csv",
    },
}


def video_name_from_ensemble_url(url):
    """'50_50_17_ensemble_100.jpg' -> '50_50_17'"""
    base = url.replace(".jpg", "")
    idx = base.rfind("_ensemble_")
    if idx == -1:
        # Fallback: try _frame_
        idx = base.rfind("_frame_")
    if idx == -1:
        return base
    return base[:idx]


def load_transcripts(transcript_dir, split_name):
    """Load Whisper transcript CSV and return dict: video_name -> row."""
    csv_path = os.path.join(transcript_dir, f"{split_name}_transcripts.csv")
    if not os.path.exists(csv_path):
        print(f"    WARNING: transcript file not found: {csv_path}")
        return {}

    df_t = pd.read_csv(csv_path)
    print(f"    Loaded {len(df_t)} transcripts from {csv_path}")
    return {row["video_name"]: row for _, row in df_t.iterrows()}


def build_split(split_name, config, output_dir, transcript_dir):
    split_dir = os.path.join(output_dir, split_name)
    os.makedirs(split_dir, exist_ok=True)

    csv_path = config["ensemble_csv"]
    frames_dir = config["ensemble_frames"]

    if not os.path.exists(csv_path):
        print(f"  WARNING: {csv_path} not found, skipping {split_name}")
        return 0

    # Load ensemble CSV
    df = pd.read_csv(csv_path)
    print(f"  {split_name}: {len(df)} samples from ensemble CSV")

    # Load Whisper transcripts
    transcripts = load_transcripts(transcript_dir, split_name)

    # Map video names for transcript lookup
    df["_video_name"] = df["url"].apply(video_name_from_ensemble_url)

    # Replace instructions with Whisper transcripts where available
    replaced = 0
    whisper_confs = []
    whisper_n_lows = []
    for idx, row in df.iterrows():
        vname = row["_video_name"]
        if vname in transcripts:
            t = transcripts[vname]
            transcript = str(t.get("transcript", "")).strip()
            if transcript:
                df.at[idx, "instruction"] = transcript
                replaced += 1
            whisper_confs.append(t.get("mean_confidence", None))
            whisper_n_lows.append(t.get("n_low_confidence", 0))
        else:
            whisper_confs.append(None)
            whisper_n_lows.append(None)

    df["whisper_mean_conf"] = whisper_confs
    df["whisper_n_low_conf"] = whisper_n_lows
    print(f"    Replaced {replaced}/{len(df)} instructions with Whisper transcripts")

    # Copy frames
    copied = 0
    missing = 0
    for fname in df["url"]:
        src = os.path.join(frames_dir, fname)
        dst = os.path.join(split_dir, fname)
        if os.path.exists(src):
            shutil.copy2(src, dst)
            copied += 1
        else:
            missing += 1
            print(f"    MISSING frame: {fname}")

    print(f"    Copied {copied} frames ({missing} missing)")

    # Save CSV
    cols = [c for c in KEEP_COLS if c in df.columns]
    df_out = df[cols]
    out_csv = os.path.join(split_dir, config["out_csv_name"])
    df_out.to_csv(out_csv, index=False)
    print(f"    Saved {out_csv}")

    return len(df_out)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", required=True,
                        help="Output directory for ERIT_new")
    parser.add_argument("--transcript-dir", default=TRANSCRIPT_DIR_DEFAULT,
                        help="Directory with Whisper transcript CSVs")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    print(f"Building ERIT_new at: {args.output_dir}")
    print(f"Transcripts from: {args.transcript_dir}\n")

    total = 0
    for split_name, config in SPLITS.items():
        n = build_split(split_name, config, args.output_dir, args.transcript_dir)
        total += n
        print()

    print(f"{'='*50}")
    print(f"ERIT_new complete: {total} total samples")
    print(f"Location: {args.output_dir}")
    for split_name in SPLITS:
        split_dir = os.path.join(args.output_dir, split_name)
        csv_name = SPLITS[split_name]["out_csv_name"]
        csv_path = os.path.join(split_dir, csv_name)
        if os.path.exists(csv_path):
            df = pd.read_csv(csv_path)
            n_imgs = len([f for f in os.listdir(split_dir) if f.endswith(".jpg")])
            has_conf = df["whisper_mean_conf"].notna().sum() if "whisper_mean_conf" in df.columns else 0
            print(f"  {split_name}: {len(df)} CSV rows, {n_imgs} images, "
                  f"{has_conf} with Whisper confidence")


if __name__ == "__main__":
    main()
