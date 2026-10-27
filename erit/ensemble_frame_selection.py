"""
Ensemble Frame Selection for ERIT.

For each ElderReact test video, score EVERY frame using 3 heterogeneous
emotion classifiers, then select frames using consensus and disagreement.

Models:
  1. DeepFace (FER backend) - CNN-based emotion classifier
  2. CLIP zero-shot - text-prompt emotion probe  
  3. HuggingFace ViT (trpakov/vit-face-expression) - AffectNet-trained ViT

For each frame f and target emotion e, we compute:
  - Consensus:    S_consensus(f) = sum_i w_i * p_i(e|f)
  - Disagreement: S_disagreement(f) = variance across model probabilities

Output: Per-video JSON with all frame scores, plus selected frames.

Usage (interactive, 1 GPU):
    pip install deepface tf-keras  # one-time
    python erit/ensemble_frame_selection.py \
        --erit-csv /scratch/vemotionsys/rmfrieske/ERIT/ERIT_test/test.csv \
        --video-dir /scratch/vemotionsys/rmfrieske/datasets/ElderReact/ElderReact_Data/ElderReact_test \
        --output-dir /scratch/vemotionsys/rmfrieske/ERIT/ERIT_test_ensemble \
        --sample-rate 3
"""

import argparse
import json
import os
import warnings
from collections import defaultdict

import cv2
import clip
import numpy as np
import pandas as pd
import torch
from PIL import Image

warnings.filterwarnings("ignore")

# ── Emotion label mapping ──────────────────────────────────────────────
# All models must map to these 7 ERIT classes
ERIT_EMOTIONS = ["anger", "disgust", "fear", "happiness", "sadness", "surprise", "neutral"]

# DeepFace uses: angry, disgust, fear, happy, sad, surprise, neutral
DEEPFACE_TO_ERIT = {
    "angry": "anger", "disgust": "disgust", "fear": "fear",
    "happy": "happiness", "sad": "sadness", "surprise": "surprise",
    "neutral": "neutral",
}

# HF ViT (trpakov) uses: angry, disgust, fear, happy, neutral, sad, surprise
HF_VIT_TO_ERIT = {
    "angry": "anger", "disgust": "disgust", "fear": "fear",
    "happy": "happiness", "neutral": "neutral", "sad": "sadness",
    "surprise": "surprise",
}

# CLIP: we define prompts directly for ERIT emotions
CLIP_PROMPTS = [
    "a photo of a person expressing anger",
    "a photo of a person expressing disgust",
    "a photo of a person expressing fear",
    "a photo of a person expressing happiness",
    "a photo of a person expressing sadness",
    "a photo of a person expressing surprise",
    "a photo of a person with a neutral expression",
]


# ── Model Wrappers ─────────────────────────────────────────────────────

class DeepFaceClassifier:
    """DeepFace emotion analysis using FER backend."""

    def __init__(self):
        from deepface import DeepFace
        self.DeepFace = DeepFace
        self.name = "deepface"
        # Warm up
        dummy = np.zeros((64, 64, 3), dtype=np.uint8)
        try:
            self.DeepFace.analyze(dummy, actions=["emotion"],
                                 enforce_detection=False, silent=True)
        except Exception:
            pass

    def predict(self, frame_bgr):
        """Return dict {erit_emotion: probability} for a BGR frame."""
        try:
            results = self.DeepFace.analyze(
                frame_bgr, actions=["emotion"],
                enforce_detection=False, silent=True,
            )
            if isinstance(results, list):
                results = results[0]
            raw = results.get("emotion", {})
            probs = {}
            for df_key, erit_key in DEEPFACE_TO_ERIT.items():
                probs[erit_key] = raw.get(df_key, 0.0) / 100.0  # pct -> prob
            return probs
        except Exception:
            return {e: 1.0 / 7 for e in ERIT_EMOTIONS}


class CLIPClassifier:
    """CLIP zero-shot emotion classification."""

    def __init__(self, device="cuda"):
        self.device = device
        self.model, self.preprocess = clip.load("ViT-B/32", device=device)
        self.text_features = self._encode_prompts()
        self.name = "clip"

    def _encode_prompts(self):
        tokens = clip.tokenize(CLIP_PROMPTS).to(self.device)
        with torch.no_grad():
            text_features = self.model.encode_text(tokens)
            text_features = text_features / text_features.norm(dim=-1, keepdim=True)
        return text_features

    def predict(self, frame_bgr):
        """Return dict {erit_emotion: probability} for a BGR frame."""
        img_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        pil_img = Image.fromarray(img_rgb)
        img_input = self.preprocess(pil_img).unsqueeze(0).to(self.device)

        with torch.no_grad():
            img_features = self.model.encode_image(img_input)
            img_features = img_features / img_features.norm(dim=-1, keepdim=True)
            similarity = (100.0 * img_features @ self.text_features.T).softmax(dim=-1)
            probs_tensor = similarity[0].cpu().numpy()

        return {emo: float(probs_tensor[i]) for i, emo in enumerate(ERIT_EMOTIONS)}


class HFViTClassifier:
    """HuggingFace ViT trained on AffectNet (trpakov/vit-face-expression)."""

    def __init__(self, device="cuda"):
        from transformers import AutoImageProcessor, AutoModelForImageClassification
        model_name = "trpakov/vit-face-expression"
        self.processor = AutoImageProcessor.from_pretrained(model_name)
        self.model = AutoModelForImageClassification.from_pretrained(model_name)
        self.model.to(device).eval()
        self.device = device
        self.name = "hf_vit"

        # Get label mapping from model config
        self.id2label = self.model.config.id2label

    def predict(self, frame_bgr):
        """Return dict {erit_emotion: probability} for a BGR frame."""
        img_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        pil_img = Image.fromarray(img_rgb)

        inputs = self.processor(images=pil_img, return_tensors="pt").to(self.device)
        with torch.no_grad():
            outputs = self.model(**inputs)
            probs_tensor = torch.softmax(outputs.logits, dim=-1)[0].cpu().numpy()

        # Map HF labels to ERIT
        probs = {e: 0.0 for e in ERIT_EMOTIONS}
        for idx, p in enumerate(probs_tensor):
            hf_label = self.id2label[idx].lower()
            erit_label = HF_VIT_TO_ERIT.get(hf_label)
            if erit_label:
                probs[erit_label] = float(p)
        return probs


# ── Scoring functions ──────────────────────────────────────────────────

def compute_consensus(model_probs, emotion, weights=None):
    """
    Weighted average of p(emotion|frame) across models.
    model_probs: list of dicts {emotion: prob}
    """
    if weights is None:
        weights = [1.0 / len(model_probs)] * len(model_probs)
    return sum(w * mp.get(emotion, 0.0) for w, mp in zip(weights, model_probs))


def compute_disagreement(model_probs, emotion):
    """Variance of p(emotion|frame) across models."""
    vals = [mp.get(emotion, 0.0) for mp in model_probs]
    return float(np.var(vals))


def video_name_from_erit_url(url):
    """'50_50_17_frame_0.jpg' -> '50_50_17'"""
    base = url.replace(".jpg", "")
    idx = base.rfind("_frame_")
    if idx == -1:
        raise ValueError(f"Cannot parse video name from: {url}")
    return base[:idx]


# ── Main pipeline ─────────────────────────────────────────────────────

def process_video(video_path, target_emotions, models, sample_rate=3):
    """
    Process all frames of a video, return per-frame scores.

    Args:
        video_path: path to mp4
        target_emotions: list of ERIT emotion labels to score for
        models: list of classifier instances
        sample_rate: score every N-th frame (1=all, 3=every 3rd)

    Returns:
        list of dicts with frame_idx, per-model probs, consensus, disagreement
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open: {video_path}")

    n_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    results = []

    frame_idx = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        if frame_idx % sample_rate == 0:
            # Get predictions from all models
            all_probs = []
            for model in models:
                probs = model.predict(frame)
                all_probs.append(probs)

            # Compute scores for each target emotion
            emotion_scores = {}
            for emo in target_emotions:
                cons = compute_consensus(all_probs, emo)
                disag = compute_disagreement(all_probs, emo)
                emotion_scores[emo] = {
                    "consensus": cons,
                    "disagreement": disag,
                    "per_model": {m.name: mp.get(emo, 0.0)
                                  for m, mp in zip(models, all_probs)},
                }

            results.append({
                "frame_idx": frame_idx,
                "n_frames": n_frames,
                "emotion_scores": emotion_scores,
                "full_probs": {m.name: mp for m, mp in zip(models, all_probs)},
            })

        frame_idx += 1

    cap.release()
    return results


def select_frames(frame_scores, target_emotion, strategy="consensus"):
    """
    Select best frame for a target emotion.

    strategy:
      - 'consensus': highest consensus score
      - 'high_consensus_low_disagreement': highest consensus - lambda*disagreement
      - 'disagreement': highest disagreement (ambiguous frames)
    """
    if not frame_scores:
        return None

    if strategy == "consensus":
        best = max(frame_scores,
                   key=lambda f: f["emotion_scores"][target_emotion]["consensus"])
    elif strategy == "high_consensus_low_disagreement":
        # Consensus minus penalty for disagreement
        lam = 0.5
        best = max(frame_scores,
                   key=lambda f: (f["emotion_scores"][target_emotion]["consensus"]
                                  - lam * f["emotion_scores"][target_emotion]["disagreement"]))
    elif strategy == "disagreement":
        best = max(frame_scores,
                   key=lambda f: f["emotion_scores"][target_emotion]["disagreement"])
    else:
        raise ValueError(f"Unknown strategy: {strategy}")

    return best


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--erit-csv", required=True)
    parser.add_argument("--video-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--sample-rate", type=int, default=3,
                        help="Score every N-th frame (default: 3)")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--save-scores", action="store_true", default=True,
                        help="Save all per-frame scores as JSON")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    os.makedirs(os.path.join(args.output_dir, "consensus_frames"), exist_ok=True)
    os.makedirs(os.path.join(args.output_dir, "scores"), exist_ok=True)

    # Load ERIT CSV
    df = pd.read_csv(args.erit_csv)
    df["video_name"] = df["url"].apply(video_name_from_erit_url)
    print(f"Loaded {len(df)} ERIT samples from {df['video_name'].nunique()} videos")

    # Initialize models
    print("Loading models...")
    models = [
        DeepFaceClassifier(),
        CLIPClassifier(device=args.device),
        HFViTClassifier(device=args.device),
    ]
    print(f"  Loaded {len(models)} classifiers: {[m.name for m in models]}")

    # Process videos
    video_groups = df.groupby("video_name")
    all_selected = []
    n_videos = len(video_groups)

    for v_idx, (video_name, group) in enumerate(video_groups):
        video_path = os.path.join(args.video_dir, f"{video_name}.mp4")
        if not os.path.exists(video_path):
            print(f"  [{v_idx+1}/{n_videos}] MISSING: {video_name}")
            continue

        # Target emotions for this video
        target_emotions = set()
        for _, row in group.iterrows():
            target_emotions.add(str(row["caption"]).strip().lower())

        print(f"  [{v_idx+1}/{n_videos}] {video_name} -> {target_emotions}")

        try:
            frame_scores = process_video(
                video_path, list(target_emotions), models,
                sample_rate=args.sample_rate,
            )
        except RuntimeError as e:
            print(f"    ERROR: {e}")
            continue

        if not frame_scores:
            print(f"    WARNING: no frames scored")
            continue

        # Save raw scores
        if args.save_scores:
            score_path = os.path.join(args.output_dir, "scores", f"{video_name}.json")
            with open(score_path, "w") as f:
                json.dump(frame_scores, f, indent=2, default=str)

        # Select best frame per target emotion using consensus
        for _, row in group.iterrows():
            emo = str(row["caption"]).strip().lower()
            best = select_frames(frame_scores, emo, strategy="consensus")
            if best is None:
                continue

            best_idx = best["frame_idx"]

            # Extract and save the selected frame
            cap = cv2.VideoCapture(video_path)
            cap.set(cv2.CAP_PROP_POS_FRAMES, best_idx)
            ret, frame = cap.read()
            cap.release()

            if ret:
                fname = f"{video_name}_ensemble_{best_idx}.jpg"
                cv2.imwrite(
                    os.path.join(args.output_dir, "consensus_frames", fname),
                    frame,
                )

                all_selected.append({
                    "url": fname,
                    "instruction": row["instruction"],
                    "OriginalLabels": row["OriginalLabels"],
                    "caption": row["caption"],
                    "GTEmotionScore": best["emotion_scores"][emo]["consensus"],
                    "strategy": "ensemble_consensus",
                    "frame_idx": best_idx,
                    "n_frames": best["n_frames"],
                    "original_url": row["url"],
                    "disagreement": best["emotion_scores"][emo]["disagreement"],
                    "deepface_score": best["emotion_scores"][emo]["per_model"].get("deepface", -1),
                    "clip_score": best["emotion_scores"][emo]["per_model"].get("clip", -1),
                    "hf_vit_score": best["emotion_scores"][emo]["per_model"].get("hf_vit", -1),
                })

    # Save CSV — derive name from input CSV (e.g. train.csv -> train_ensemble.csv)
    df_selected = pd.DataFrame(all_selected)
    input_stem = os.path.splitext(os.path.basename(args.erit_csv))[0]
    csv_name = f"{input_stem}_ensemble.csv"
    csv_path = os.path.join(args.output_dir, csv_name)
    df_selected.to_csv(csv_path, index=False)

    print(f"\n{'='*60}")
    print(f"Done! {len(df_selected)} samples selected")
    print(f"  CSV:    {csv_path}")
    print(f"  Frames: {os.path.join(args.output_dir, 'consensus_frames')}")
    print(f"  Scores: {os.path.join(args.output_dir, 'scores')}")

    # Print summary statistics
    if len(df_selected) > 0:
        print(f"\n  Consensus score:    mean={df_selected['GTEmotionScore'].mean():.4f}, "
              f"std={df_selected['GTEmotionScore'].std():.4f}")
        print(f"  Disagreement:       mean={df_selected['disagreement'].mean():.4f}, "
              f"std={df_selected['disagreement'].std():.4f}")
        for m in ["deepface_score", "clip_score", "hf_vit_score"]:
            print(f"  {m:20s}: mean={df_selected[m].mean():.4f}")


if __name__ == "__main__":
    main()
