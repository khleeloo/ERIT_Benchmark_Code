"""
Generate confusion matrices and error analysis from ERIT result CSVs.
Produces:
  - Normalized confusion matrix figures (2 subfigures side by side)
  - Per-emotion accuracy breakdown
  - Top confusion pairs
  - Qualitative examples (fusion success / failure)

Usage:
  python experiments/confusion_analysis.py \
    --csv-a results/results_gpt4o_test.csv --label-a "GPT-4o" \
    --csv-b results/results_llava_test.csv --label-b "LLaVA-NeXT" \
    --output-dir results/figures
"""

import argparse
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix, classification_report

EMOTIONS = ["happiness", "sadness", "disgust", "anger", "surprise", "fear", "neutral"]


def load_results(csv_path, mode="fusion"):
    """Load a results CSV and return (y_true, y_pred) for the given mode."""
    df = pd.read_csv(csv_path)
    col_map = {
        "fusion": "Prediction_Fusion",
        "text": "Prediction_Text",
        "image": "Prediction_Image",
    }
    pred_col = col_map[mode]
    y_true = df["GTLabel"].astype(str).str.strip().str.lower().tolist()
    y_pred = df[pred_col].astype(str).str.strip().str.lower().tolist()
    return y_true, y_pred, df


def plot_confusion_pair(y_true_a, y_pred_a, label_a,
                        y_true_b, y_pred_b, label_b,
                        labels, out_path):
    """Plot two normalized confusion matrices side by side."""
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    for ax, y_true, y_pred, title in [
        (axes[0], y_true_a, y_pred_a, label_a),
        (axes[1], y_true_b, y_pred_b, label_b),
    ]:
        cm = confusion_matrix(y_true, y_pred, labels=labels)
        # Row-normalize
        row_sums = cm.sum(axis=1, keepdims=True)
        row_sums[row_sums == 0] = 1  # avoid div-by-zero
        cm_norm = cm.astype(float) / row_sums

        im = ax.imshow(cm_norm, interpolation="nearest", cmap="Blues",
                       vmin=0, vmax=1)
        ax.set_title(title, fontsize=13, fontweight="bold")
        ax.set_xticks(range(len(labels)))
        ax.set_yticks(range(len(labels)))
        short = [l[:3].capitalize() for l in labels]
        ax.set_xticklabels(short, rotation=45, ha="right", fontsize=9)
        ax.set_yticklabels(short, fontsize=9)
        ax.set_xlabel("Predicted")
        ax.set_ylabel("True")

        # Annotate cells
        for i in range(len(labels)):
            for j in range(len(labels)):
                val = cm_norm[i, j]
                color = "white" if val > 0.5 else "black"
                ax.text(j, i, f"{val:.2f}", ha="center", va="center",
                        fontsize=7, color=color)

    cbar = fig.colorbar(im, ax=axes.ravel().tolist(), shrink=0.75, pad=0.02,
                        aspect=30)
    cbar.set_label("Row-normalized proportion", fontsize=10)
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    print(f"  Saved confusion matrix figure: {out_path}")
    plt.close(fig)


def top_confusions(y_true, y_pred, labels, n=5):
    """Return the top-n off-diagonal (true→pred) confusion pairs."""
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    row_sums = cm.sum(axis=1, keepdims=True)
    row_sums[row_sums == 0] = 1
    cm_norm = cm.astype(float) / row_sums

    pairs = []
    for i, true_l in enumerate(labels):
        for j, pred_l in enumerate(labels):
            if i != j:
                pairs.append((true_l, pred_l, cm_norm[i, j], cm[i, j]))
    pairs.sort(key=lambda x: -x[2])
    return pairs[:n]


def compare_fusion_gain(df, labels):
    """Show per-emotion accuracy for image, text, fusion."""
    rows = []
    for emo in labels:
        mask = df["GTLabel"].str.lower().str.strip() == emo
        sub = df[mask]
        if len(sub) == 0:
            continue
        acc_i = (sub["Prediction_Image"].str.lower().str.strip() == emo).mean() * 100
        acc_t = (sub["Prediction_Text"].str.lower().str.strip() == emo).mean() * 100
        acc_f = (sub["Prediction_Fusion"].str.lower().str.strip() == emo).mean() * 100
        rows.append({
            "emotion": emo, "n": len(sub),
            "image_acc": acc_i, "text_acc": acc_t, "fusion_acc": acc_f,
            "fusion_gain_vs_image": acc_f - acc_i,
            "fusion_gain_vs_text": acc_f - acc_t,
        })
    return pd.DataFrame(rows)


def qualitative_examples(df, n_success=2, n_failure=2):
    """Find examples where fusion succeeds/fails vs unimodal."""
    gt = df["GTLabel"].str.lower().str.strip()
    pf = df["Prediction_Fusion"].str.lower().str.strip()
    pi = df["Prediction_Image"].str.lower().str.strip()
    pt = df["Prediction_Text"].str.lower().str.strip()

    # Fusion correct, both unimodal wrong
    success_mask = (pf == gt) & (pi != gt) & (pt != gt)
    # Fusion wrong, at least one unimodal correct
    fail_mask = (pf != gt) & ((pi == gt) | (pt == gt))

    successes = df[success_mask].head(n_success)
    failures = df[fail_mask].head(n_failure)
    return successes, failures


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv-a", required=True, help="Results CSV for model A")
    parser.add_argument("--label-a", default="Model A")
    parser.add_argument("--csv-b", required=True, help="Results CSV for model B")
    parser.add_argument("--label-b", default="Model B")
    parser.add_argument("--output-dir", default="/home/rmfrieske/datasets/ERIT/results/figures")
    parser.add_argument("--mode", default="fusion",
                        choices=["fusion", "text", "image"])
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    # ── Load ────────────────────────────────────────────────────────────────
    y_true_a, y_pred_a, df_a = load_results(args.csv_a, args.mode)
    y_true_b, y_pred_b, df_b = load_results(args.csv_b, args.mode)

    # ── Confusion matrices ──────────────────────────────────────────────────
    fig_path = os.path.join(args.output_dir,
                            f"confusion_{args.label_a}_{args.label_b}_{args.mode}.pdf")
    plot_confusion_pair(y_true_a, y_pred_a, f"{args.label_a} ({args.mode})",
                        y_true_b, y_pred_b, f"{args.label_b} ({args.mode})",
                        EMOTIONS, fig_path)

    # Also save PNG for quick viewing
    plot_confusion_pair(y_true_a, y_pred_a, f"{args.label_a} ({args.mode})",
                        y_true_b, y_pred_b, f"{args.label_b} ({args.mode})",
                        EMOTIONS, fig_path.replace(".pdf", ".png"))

    # ── Also generate image-only and text-only confusion matrices ───────
    for mode in ["image", "text"]:
        yt_a, yp_a, _ = load_results(args.csv_a, mode)
        yt_b, yp_b, _ = load_results(args.csv_b, mode)
        fp = os.path.join(args.output_dir,
                          f"confusion_{args.label_a}_{args.label_b}_{mode}.pdf")
        plot_confusion_pair(yt_a, yp_a, f"{args.label_a} ({mode})",
                            yt_b, yp_b, f"{args.label_b} ({mode})",
                            EMOTIONS, fp)

    # ── Top confusions ──────────────────────────────────────────────────────
    print(f"\n{'='*60}")
    print(f"Top confusions — {args.label_a} ({args.mode})")
    print(f"{'='*60}")
    for true_l, pred_l, rate, count in top_confusions(y_true_a, y_pred_a, EMOTIONS):
        print(f"  {true_l:>10} → {pred_l:<10}  {rate*100:5.1f}%  (n={count})")

    print(f"\nTop confusions — {args.label_b} ({args.mode})")
    print(f"{'='*60}")
    for true_l, pred_l, rate, count in top_confusions(y_true_b, y_pred_b, EMOTIONS):
        print(f"  {true_l:>10} → {pred_l:<10}  {rate*100:5.1f}%  (n={count})")

    # ── Per-emotion fusion gain ─────────────────────────────────────────────
    print(f"\n{'='*60}")
    print(f"Per-emotion fusion gain — {args.label_a}")
    print(f"{'='*60}")
    gain_a = compare_fusion_gain(df_a, EMOTIONS)
    print(gain_a.to_string(index=False, float_format="%.1f"))

    print(f"\nPer-emotion fusion gain — {args.label_b}")
    gain_b = compare_fusion_gain(df_b, EMOTIONS)
    print(gain_b.to_string(index=False, float_format="%.1f"))

    # Save gains to CSV
    gain_a.to_csv(os.path.join(args.output_dir, f"fusion_gain_{args.label_a}.csv"), index=False)
    gain_b.to_csv(os.path.join(args.output_dir, f"fusion_gain_{args.label_b}.csv"), index=False)

    # ── Classification reports ──────────────────────────────────────────────
    print(f"\n{'='*60}")
    print(f"Classification Report — {args.label_a} ({args.mode})")
    print(f"{'='*60}")
    print(classification_report(y_true_a, y_pred_a, labels=EMOTIONS,
                                target_names=EMOTIONS, zero_division=0))

    print(f"Classification Report — {args.label_b} ({args.mode})")
    print(classification_report(y_true_b, y_pred_b, labels=EMOTIONS,
                                target_names=EMOTIONS, zero_division=0))

    # ── Qualitative examples ────────────────────────────────────────────────
    for label, dframe in [(args.label_a, df_a), (args.label_b, df_b)]:
        successes, failures = qualitative_examples(dframe)
        print(f"\n{'='*60}")
        print(f"Qualitative examples — {label}")
        print(f"{'='*60}")
        if len(successes) > 0:
            print("\n  FUSION SUCCEEDS (both unimodal wrong):")
            for _, r in successes.iterrows():
                print(f"    GT: {r['GTLabel']}")
                print(f"    Text: \"{r['Transcript'][:80]}...\"")
                print(f"    Pred — Image: {r['Prediction_Image']}, "
                      f"Text: {r['Prediction_Text']}, "
                      f"Fusion: {r['Prediction_Fusion']}")
                print(f"    Image: {os.path.basename(r['ImageName'])}")
                print()
        if len(failures) > 0:
            print("  FUSION FAILS (unimodal correct):")
            for _, r in failures.iterrows():
                print(f"    GT: {r['GTLabel']}")
                print(f"    Text: \"{r['Transcript'][:80]}...\"")
                print(f"    Pred — Image: {r['Prediction_Image']}, "
                      f"Text: {r['Prediction_Text']}, "
                      f"Fusion: {r['Prediction_Fusion']}")
                print(f"    Image: {os.path.basename(r['ImageName'])}")
                print()

    print("Done.")


if __name__ == "__main__":
    main()
