"""
Frame Selection Ablation for ERIT.

For each ERIT test sample, extract an alternative frame from the original
ElderReact video using two strategies (random, midpoint) and save new
CSVs that can be evaluated with the same pipeline as the original ERIT.

The DeepFace variant is the existing ERIT test set (no re-extraction needed).

Usage:
    python experiments/frame_selection_ablation.py \
        --erit-csv /scratch/vemotionsys/rmfrieske/ERIT/ERIT_test/test.csv \
        --video-dir /scratch/vemotionsys/rmfrieske/datasets/ElderReact/ElderReact_Data/ElderReact_test \
        --output-dir /scratch/vemotionsys/rmfrieske/ERIT/ERIT_test_ablation \
        --seed 42
"""

import argparse
import os
import random

import cv2
import numpy as np
import pandas as pd


def get_video_frame_count(video_path):
    """Return total number of frames in a video."""
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {video_path}")
    n_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()
    return n_frames


def extract_frame(video_path, frame_idx):
    """Extract a single frame from a video as a numpy array (BGR)."""
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {video_path}")
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
    ret, frame = cap.read()
    cap.release()
    if not ret:
        raise RuntimeError(f"Cannot read frame {frame_idx} from {video_path}")
    return frame


def video_name_from_erit_url(url):
    """
    ERIT url: '50_50_17_frame_0.jpg' -> video name: '50_50_17'
    Handles names with variable number of underscored segments.
    """
    # Remove .jpg, then split off _frame_N
    base = url.replace(".jpg", "")
    # Find last occurrence of '_frame_'
    idx = base.rfind("_frame_")
    if idx == -1:
        raise ValueError(f"Cannot parse video name from: {url}")
    return base[:idx]


def main():
    parser = argparse.ArgumentParser(description="Frame selection ablation for ERIT")
    parser.add_argument("--erit-csv", type=str, required=True,
                        help="Path to ERIT test CSV")
    parser.add_argument("--video-dir", type=str, required=True,
                        help="Path to ElderReact video directory (same split)")
    parser.add_argument("--output-dir", type=str, required=True,
                        help="Output directory for ablation variants")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)

    os.makedirs(args.output_dir, exist_ok=True)
    os.makedirs(os.path.join(args.output_dir, "random_frames"), exist_ok=True)
    os.makedirs(os.path.join(args.output_dir, "midpoint_frames"), exist_ok=True)

    df = pd.read_csv(args.erit_csv)
    print(f"Loaded {len(df)} ERIT samples")

    # Group by video name to avoid re-opening the same video
    df["video_name"] = df["url"].apply(video_name_from_erit_url)
    video_groups = df.groupby("video_name")

    random_rows = []
    midpoint_rows = []
    missing_videos = []

    for video_name, group in video_groups:
        video_path = os.path.join(args.video_dir, f"{video_name}.mp4")
        if not os.path.exists(video_path):
            print(f"  WARNING: Video not found: {video_path}")
            missing_videos.append(video_name)
            continue

        try:
            n_frames = get_video_frame_count(video_path)
        except RuntimeError as e:
            print(f"  ERROR: {e}")
            missing_videos.append(video_name)
            continue

        if n_frames == 0:
            print(f"  WARNING: 0 frames in {video_path}")
            missing_videos.append(video_name)
            continue

        # Midpoint frame index
        mid_idx = n_frames // 2

        # For each emotion label from this video, create alternative samples
        for row_idx, row in group.iterrows():
            # --- Random frame ---
            rand_idx = random.randint(0, n_frames - 1)
            rand_fname = f"{video_name}_random_{rand_idx}.jpg"
            rand_path = os.path.join(args.output_dir, "random_frames", rand_fname)

            try:
                frame = extract_frame(video_path, rand_idx)
                cv2.imwrite(rand_path, frame)
                random_rows.append({
                    "url": rand_fname,
                    "instruction": row["instruction"],
                    "OriginalLabels": row["OriginalLabels"],
                    "caption": row["caption"],
                    "GTEmotionScore": -1,  # N/A for random
                    "strategy": "random",
                    "frame_idx": rand_idx,
                    "n_frames": n_frames,
                    "original_url": row["url"],
                })
            except RuntimeError as e:
                print(f"  ERROR extracting random frame: {e}")

            # --- Midpoint frame ---
            mid_fname = f"{video_name}_midpoint_{mid_idx}.jpg"
            mid_path = os.path.join(args.output_dir, "midpoint_frames", mid_fname)

            try:
                # Only extract once per video (same midpoint for all emotions)
                if not os.path.exists(mid_path):
                    frame = extract_frame(video_path, mid_idx)
                    cv2.imwrite(mid_path, frame)
                midpoint_rows.append({
                    "url": mid_fname,
                    "instruction": row["instruction"],
                    "OriginalLabels": row["OriginalLabels"],
                    "caption": row["caption"],
                    "GTEmotionScore": -1,  # N/A for midpoint
                    "strategy": "midpoint",
                    "frame_idx": mid_idx,
                    "n_frames": n_frames,
                    "original_url": row["url"],
                })
            except RuntimeError as e:
                print(f"  ERROR extracting midpoint frame: {e}")

    # Save CSVs
    df_random = pd.DataFrame(random_rows)
    df_midpoint = pd.DataFrame(midpoint_rows)

    random_csv = os.path.join(args.output_dir, "test_random.csv")
    midpoint_csv = os.path.join(args.output_dir, "test_midpoint.csv")

    df_random.to_csv(random_csv, index=False)
    df_midpoint.to_csv(midpoint_csv, index=False)

    print(f"\nDone!")
    print(f"  Random:   {len(df_random)} samples -> {random_csv}")
    print(f"  Midpoint: {len(df_midpoint)} samples -> {midpoint_csv}")
    if missing_videos:
        print(f"  Missing videos: {len(missing_videos)}")
        for v in missing_videos:
            print(f"    {v}")


if __name__ == "__main__":
    main()
