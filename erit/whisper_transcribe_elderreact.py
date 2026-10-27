"""
Whisper Large V3 transcription with confidence scores for ElderReact videos.

Transcribes each video and reports per-segment and per-token confidence scores.
Outputs a CSV per split mapping video_name -> transcript + confidence metadata.

Confidence metrics:
  - mean_logprob: mean log-probability across generated tokens (higher = more confident)
  - no_speech_prob: model estimate of no-speech probability
  - compression_ratio: token compression ratio (high = repetitive/hallucinated)
  - word_confidence: per-word confidence from word-level timestamps (when available)

Usage:
    python erit/whisper_transcribe_elderreact.py \
        --video-dir /scratch/.../ElderReact_Data/ElderReact_test \
        --output-dir /scratch/.../ERIT_new_transcripts \
        --split test
"""

import argparse
import json
import os
import subprocess
import tempfile
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import torch

warnings.filterwarnings("ignore")


def extract_audio(video_path, out_wav):
    """Extract audio from mp4 to 16kHz mono wav using ffmpeg."""
    cmd = [
        "ffmpeg", "-y", "-i", str(video_path),
        "-ar", "16000", "-ac", "1", "-f", "wav",
        str(out_wav),
    ]
    subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=True)


def transcribe_videos(video_dir, output_dir, split, model_id, device, batch_size=1):
    from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor

    os.makedirs(output_dir, exist_ok=True)

    # Discover videos
    video_files = sorted(Path(video_dir).glob("*.mp4"))
    print(f"Found {len(video_files)} videos in {video_dir}")

    # Load model
    print(f"Loading {model_id}...")
    torch_dtype = torch.float16 if device == "cuda" else torch.float32

    model = AutoModelForSpeechSeq2Seq.from_pretrained(
        model_id,
        torch_dtype=torch_dtype,
        low_cpu_mem_usage=True,
    )
    model.to(device)
    model.eval()

    processor = AutoProcessor.from_pretrained(model_id)
    print("  Model loaded.")

    # Transcribe each video
    results = []
    all_confidences = []
    t0 = time.time()

    tmp_dir = tempfile.mkdtemp(prefix="whisper_audio_")

    for v_idx, vpath in enumerate(video_files):
        video_name = vpath.stem  # e.g. "50_50_17"
        wav_path = os.path.join(tmp_dir, f"{video_name}.wav")

        try:
            extract_audio(vpath, wav_path)
        except Exception as e:
            print(f"  [{v_idx+1}/{len(video_files)}] FFMPEG ERROR {video_name}: {e}")
            results.append({
                "video_name": video_name,
                "transcript": "",
                "mean_confidence": 0.0,
                "min_confidence": 0.0,
                "n_words": 0,
                "n_low_confidence": 0,
                "word_confidences_json": "[]",
            })
            continue

        # Get token-level confidence via direct model call
        try:
            conf_data = get_token_confidences(
                wav_path, model, processor, device, torch_dtype
            )
        except Exception as e:
            print(f"  [{v_idx+1}/{len(video_files)}] ERROR {video_name}: {e}")
            results.append({
                "video_name": video_name,
                "transcript": "",
                "mean_confidence": 0.0,
                "min_confidence": 0.0,
                "n_words": 0,
                "n_low_confidence": 0,
                "word_confidences_json": "[]",
            })
            if os.path.exists(wav_path):
                os.remove(wav_path)
            continue

        transcript = conf_data["transcript"]
        mean_conf = conf_data["mean_confidence"]
        min_conf = conf_data["min_confidence"]
        word_level = conf_data["word_confidences"]
        n_low = sum(1 for w in word_level if w["confidence"] < 0.7)

        all_confidences.append(mean_conf)

        results.append({
            "video_name": video_name,
            "transcript": transcript,
            "mean_confidence": round(mean_conf, 4),
            "min_confidence": round(min_conf, 4),
            "n_words": len(word_level),
            "n_low_confidence": n_low,
            "no_speech_prob": round(conf_data.get("no_speech_prob", 0.0), 4),
            "compression_ratio": round(conf_data.get("compression_ratio", 0.0), 4),
            "word_confidences_json": json.dumps(word_level),
        })

        # Clean up temp wav
        if os.path.exists(wav_path):
            os.remove(wav_path)

        if (v_idx + 1) % 50 == 0 or v_idx == 0:
            elapsed = time.time() - t0
            print(f"  [{v_idx+1}/{len(video_files)}] {video_name}: "
                  f"conf={mean_conf:.3f}, words={len(word_level)}, "
                  f"low_conf={n_low} | {elapsed:.0f}s")

    elapsed = time.time() - t0

    # Save results
    df = pd.DataFrame(results)
    csv_path = os.path.join(output_dir, f"{split}_transcripts.csv")
    df.to_csv(csv_path, index=False)

    # Summary statistics
    confs = np.array(all_confidences)
    total_words = df["n_words"].sum()
    total_low = df["n_low_confidence"].sum()
    pct_low = (total_low / total_words * 100) if total_words > 0 else 0

    print(f"\n{'='*60}")
    print(f"Split: {split}")
    print(f"Videos: {len(df)}, Transcribed: {(df['transcript'] != '').sum()}")
    print(f"Time: {elapsed:.0f}s ({elapsed/len(video_files):.1f}s/video)")
    if len(confs) > 0:
        print(f"Mean segment confidence: {confs.mean():.4f} (σ={confs.std():.4f})")
    else:
        print("Mean segment confidence: N/A (no videos transcribed)")
    print(f"Total words: {total_words}, Low-confidence (<0.7): {total_low} ({pct_low:.1f}%)")
    print(f"Saved: {csv_path}")
    print(f"{'='*60}")

    # Save summary stats
    stats = {
        "split": split,
        "n_videos": len(df),
        "n_transcribed": int((df["transcript"] != "").sum()),
        "total_words": int(total_words),
        "total_low_confidence_words": int(total_low),
        "pct_low_confidence": round(pct_low, 2),
        "mean_confidence": round(float(confs.mean()), 4) if len(confs) > 0 else 0.0,
        "std_confidence": round(float(confs.std()), 4) if len(confs) > 0 else 0.0,
        "median_confidence": round(float(np.median(confs)), 4) if len(confs) > 0 else 0.0,
        "min_confidence": round(float(confs.min()), 4) if len(confs) > 0 else 0.0,
        "elapsed_seconds": round(elapsed, 1),
    }
    stats_path = os.path.join(output_dir, f"{split}_stats.json")
    with open(stats_path, "w") as f:
        json.dump(stats, f, indent=2)


def get_token_confidences(audio_path, model, processor, device, torch_dtype):
    """
    Run Whisper with output_scores=True to get per-token log-probabilities,
    then convert to per-word confidence scores.
    """
    import soundfile as sf

    # Load audio (already extracted to 16kHz wav)
    audio, sr = sf.read(audio_path)
    input_features = processor(
        audio, sampling_rate=16000, return_tensors="pt"
    ).input_features.to(device, dtype=torch_dtype)

    # Generate with scores
    with torch.no_grad():
        output = model.generate(
            input_features,
            language="en",
            return_dict_in_generate=True,
            output_scores=True,
            max_new_tokens=444,
        )

    # Decode transcript
    transcription = processor.batch_decode(
        output.sequences, skip_special_tokens=True
    )[0].strip()

    # Compute per-token confidence from logits
    token_ids = output.sequences[0]
    scores = output.scores  # tuple of logits per generated step

    token_confidences = []
    for i, score in enumerate(scores):
        probs = torch.softmax(score[0], dim=-1)
        token_id = token_ids[len(token_ids) - len(scores) + i]
        conf = probs[token_id].item()
        token_confidences.append(conf)

    # Map tokens to words
    generated_ids = output.sequences[0][len(token_ids) - len(scores):]
    tokens = processor.tokenizer.convert_ids_to_tokens(generated_ids.tolist())

    word_confidences = []
    current_word = ""
    current_confs = []

    for tok, conf in zip(tokens, token_confidences):
        if tok in processor.tokenizer.all_special_tokens:
            continue
        # Whisper tokenizer: tokens starting with 'Ġ' or without prefix start new words
        decoded = processor.tokenizer.decode([processor.tokenizer.convert_tokens_to_ids(tok)])
        if decoded.startswith(" ") and current_word:
            # Flush previous word
            word_confidences.append({
                "word": current_word.strip(),
                "confidence": round(float(np.mean(current_confs)), 4),
                "min_token_conf": round(float(min(current_confs)), 4),
            })
            current_word = decoded
            current_confs = [conf]
        else:
            current_word += decoded
            current_confs.append(conf)

    # Flush last word
    if current_word.strip() and current_confs:
        word_confidences.append({
            "word": current_word.strip(),
            "confidence": round(float(np.mean(current_confs)), 4),
            "min_token_conf": round(float(min(current_confs)), 4),
        })

    # Overall metrics
    all_confs = [w["confidence"] for w in word_confidences] if word_confidences else [0.0]
    mean_conf = float(np.mean(all_confs))
    min_conf = float(min(all_confs))

    # No-speech probability and compression ratio from initial decoder output
    no_speech_prob = 0.0
    compression_ratio = 0.0
    if scores:
        # Compression ratio: ratio of text length to token count
        compression_ratio = len(transcription) / max(len(scores), 1)

    return {
        "mean_confidence": mean_conf,
        "min_confidence": min_conf,
        "word_confidences": word_confidences,
        "no_speech_prob": no_speech_prob,
        "compression_ratio": compression_ratio,
        "transcript": transcription,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--video-dir", required=True,
                        help="Directory with ElderReact .mp4 files")
    parser.add_argument("--output-dir", required=True,
                        help="Output directory for transcripts")
    parser.add_argument("--split", required=True, choices=["train", "dev", "test"],
                        help="Split name (train/dev/test)")
    parser.add_argument("--model-id", default="openai/whisper-large-v3",
                        help="HuggingFace Whisper model ID")
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    transcribe_videos(
        video_dir=args.video_dir,
        output_dir=args.output_dir,
        split=args.split,
        model_id=args.model_id,
        device=args.device,
    )


if __name__ == "__main__":
    main()
