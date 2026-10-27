"""
Analyze ERIT benchmark results: compute metrics, stratified slices, LaTeX tables.

Reads the consolidated CSV from experiments/eval_benchmark.py and produces:
  1. Main results table (condition × modality)
  2. Stratified analysis (label agreement, ASR confidence, modality conflict)
  3. Delta heatmap (accuracy drop from baseline per perturbation)
  4. LaTeX output for all tables

Usage:
    python experiments/analyze_benchmark.py \
        --results-csv results/benchmark/benchmark_llava-v1.6-mistral-7b-hf_*.csv \
        --erit-root /scratch/vemotionsys/rmfrieske/ERIT \
        --transcript-dir /scratch/vemotionsys/rmfrieske/datasets/ERIT_new_transcripts \
        --output-dir results/benchmark/analysis
"""

import argparse
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


EMOTIONS = ["happiness", "sadness", "disgust", "anger", "surprise", "fear", "neutral"]


# ── Metrics ───────────────────────────────────────────────────────────

def accuracy(gt, pred):
    return (gt == pred).mean() * 100


def multi_label_accuracy(orig_labels, pred):
    """Prediction correct if it matches ANY label in the annotation set."""
    correct = 0
    for ol, p in zip(orig_labels, pred):
        gt_set = {l.strip().lower() for l in str(ol).split(",")}
        if p.strip().lower() in gt_set:
            correct += 1
    return correct / len(pred) * 100


def balanced_accuracy(gt, pred):
    classes = gt.unique()
    recalls = []
    for c in classes:
        mask = gt == c
        if mask.sum() == 0:
            continue
        recalls.append((pred[mask] == c).mean())
    return float(np.mean(recalls)) * 100


def macro_f1(gt, pred):
    classes = gt.unique()
    f1s = []
    for c in classes:
        tp = ((pred == c) & (gt == c)).sum()
        fp = ((pred == c) & (gt != c)).sum()
        fn = ((pred != c) & (gt == c)).sum()
        p = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        r = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1s.append(2 * p * r / (p + r) if (p + r) > 0 else 0.0)
    return float(np.mean(f1s)) * 100


def compute_metrics(df_subset):
    gt = df_subset["gt_label"].str.lower()
    pred = df_subset["prediction"].str.lower()
    ol = df_subset["original_labels"]
    return {
        "Acc": accuracy(gt, pred),
        "Multi-Acc": multi_label_accuracy(ol, pred),
        "BAcc": balanced_accuracy(gt, pred),
        "F1": macro_f1(gt, pred),
        "N": len(df_subset),
    }


# ── Stratified slicing ───────────────────────────────────────────────

def add_agreement_column(df):
    """Add 'agreement' column: 'high' if single label, 'low' if multi-label."""
    def count_labels(ol):
        return len([l.strip() for l in str(ol).split(",") if l.strip()])
    df["n_labels"] = df["original_labels"].apply(count_labels)
    df["agreement"] = df["n_labels"].apply(lambda n: "high" if n == 1 else "low")
    return df


def add_asr_confidence_column(df, transcript_dir):
    """Add 'asr_conf' column from Whisper transcript CSVs."""
    # Load all transcript CSVs
    conf_map = {}
    for split in ["train", "dev", "test"]:
        csv_path = os.path.join(transcript_dir, f"{split}_transcripts.csv")
        if not os.path.exists(csv_path):
            continue
        tdf = pd.read_csv(csv_path)
        for _, row in tdf.iterrows():
            vname = str(row.get("video_name", ""))
            conf_map[vname] = row.get("mean_confidence", None)

    # Map url -> video_name -> confidence
    def url_to_vname(url):
        base = str(url).replace(".jpg", "")
        idx = base.rfind("_frame_")
        return base[:idx] if idx != -1 else base

    df["_vname"] = df["url"].apply(url_to_vname)
    df["asr_conf"] = df["_vname"].map(conf_map)
    df["asr_conf_bin"] = df["asr_conf"].apply(
        lambda c: "high" if pd.notna(c) and c >= 0.85 else
                  ("low" if pd.notna(c) else "unknown")
    )
    return df


# Pairs of emotions with opposing valence / contradictory affect
OPPOSITE_PAIRS = {
    frozenset({"happiness", "sadness"}),
    frozenset({"happiness", "anger"}),
    frozenset({"happiness", "disgust"}),
    frozenset({"happiness", "fear"}),
    frozenset({"surprise", "neutral"}),
    frozenset({"anger", "neutral"}),
    frozenset({"fear", "neutral"}),
    frozenset({"sadness", "neutral"}),
}


def add_modality_conflict_column(df):
    """
    Add conflict columns by comparing image-only vs text-only predictions
    for the same sample under baseline condition.

    Columns added:
      - conflict:      'agree' | 'conflict'
      - conflict_type: 'agree' | 'opposite' | 'mild'
      - img_pred:      image-only prediction for this sample
      - txt_pred:      text-only prediction for this sample
    """
    baseline = df[df["condition"] == "baseline"]
    # Drop duplicate URLs (same image across splits) — keep first occurrence
    img_preds = (baseline[baseline["modality"] == "image"]
                 .drop_duplicates(subset="url")
                 .set_index("url")["prediction"])
    txt_preds = (baseline[baseline["modality"] == "text"]
                 .drop_duplicates(subset="url")
                 .set_index("url")["prediction"])

    conflict_info = {}  # url -> (img_pred, txt_pred, conflict_type)
    for url in img_preds.index:
        if url in txt_preds.index:
            ip = str(img_preds[url]).strip().lower()
            tp = str(txt_preds[url]).strip().lower()
            if ip == tp:
                conflict_info[url] = (ip, tp, "agree")
            elif frozenset({ip, tp}) in OPPOSITE_PAIRS:
                conflict_info[url] = (ip, tp, "opposite")
            else:
                conflict_info[url] = (ip, tp, "mild")

    df["conflict"] = df["url"].apply(
        lambda u: "agree" if conflict_info.get(u, ("", "", "agree"))[2] == "agree" else "conflict"
    )
    df["conflict_type"] = df["url"].apply(
        lambda u: conflict_info.get(u, ("", "", "agree"))[2]
    )
    df["img_pred"] = df["url"].apply(
        lambda u: conflict_info.get(u, ("", "", ""))[0]
    )
    df["txt_pred"] = df["url"].apply(
        lambda u: conflict_info.get(u, ("", "", ""))[1]
    )
    return df


# ── LaTeX output ──────────────────────────────────────────────────────

def metrics_to_latex_row(label, metrics, bold_col=None):
    """Format one row of a metrics table."""
    cols = ["Acc", "Multi-Acc", "BAcc", "F1"]
    vals = []
    for c in cols:
        v = f"{metrics[c]:.1f}"
        if bold_col == c:
            v = f"\\textbf{{{v}}}"
        vals.append(v)
    n = f"{metrics['N']:,}"
    return f"{label} & {' & '.join(vals)} & {n} \\\\"


def main():
    parser = argparse.ArgumentParser(description="Analyze ERIT benchmark results")
    parser.add_argument("--results-csv", required=True,
                        help="Path to benchmark results CSV")
    parser.add_argument("--erit-root", default="/scratch/vemotionsys/rmfrieske/ERIT",
                        help="Root of ERIT splits")
    parser.add_argument("--transcript-dir",
                        default="/scratch/vemotionsys/rmfrieske/datasets/ERIT_new_transcripts",
                        help="Directory with Whisper transcripts")
    parser.add_argument("--output-dir", default="results/benchmark/analysis",
                        help="Output directory for analysis")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    print(f"Loading results from {args.results_csv}")
    df = pd.read_csv(args.results_csv)
    print(f"  {len(df)} prediction rows")
    print(f"  Conditions: {sorted(df['condition'].unique())}")
    print(f"  Modalities: {sorted(df['modality'].unique())}")

    # ── Table 1: Main results (condition × modality) ──────────────────
    print(f"\n{'='*60}")
    print("TABLE 1: Main benchmark results")
    print(f"{'='*60}")

    rows_main = []
    for cond in ["baseline", "no_keyword", "grayscale", "blurred_faces",
                 "text_dropout", "image_dropout"]:
        for mod in ["image", "text", "fusion"]:
            mask = (df["condition"] == cond) & (df["modality"] == mod)
            subset = df[mask]
            if len(subset) == 0:
                continue
            m = compute_metrics(subset)
            m["condition"] = cond
            m["modality"] = mod
            rows_main.append(m)
            print(f"  {cond:16s} {mod:8s}  "
                  f"Acc={m['Acc']:.1f}  MAcc={m['Multi-Acc']:.1f}  "
                  f"BAcc={m['BAcc']:.1f}  F1={m['F1']:.1f}  N={m['N']}")

    df_main = pd.DataFrame(rows_main)
    df_main.to_csv(os.path.join(args.output_dir, "main_results.csv"), index=False)

    # LaTeX Table 1
    print("\n% --- LaTeX Table 1: Main results ---")
    print(r"\begin{table}[t]")
    print(r"\centering")
    print(r"\caption{Benchmark results across perturbation conditions and modalities (Acc\%).}")
    print(r"\label{tab:benchmark}")
    print(r"\resizebox{\columnwidth}{!}{%")
    print(r"\begin{tabular}{ll rrrr r}")
    print(r"\toprule")
    print(r"\textbf{Condition} & \textbf{Modality} & \textbf{Acc} & \textbf{Multi-Acc} & \textbf{BAcc} & \textbf{F1} & \textbf{N} \\")
    print(r"\midrule")
    for _, r in df_main.iterrows():
        label = f"{r['condition']:16s} & {r['modality']:8s}"
        print(metrics_to_latex_row(label, r))
    print(r"\bottomrule")
    print(r"\end{tabular}}")
    print(r"\end{table}")

    # ── Enrichment for stratified analysis ────────────────────────────
    df = add_agreement_column(df)
    df = add_asr_confidence_column(df, args.transcript_dir)
    df = add_modality_conflict_column(df)

    # ── Table 2: Stratified by label agreement ────────────────────────
    print(f"\n{'='*60}")
    print("TABLE 2: Stratified by label agreement (baseline only)")
    print(f"{'='*60}")

    baseline = df[df["condition"] == "baseline"]
    rows_agree = []
    for agree in ["high", "low"]:
        for mod in ["image", "text", "fusion"]:
            mask = (baseline["agreement"] == agree) & (baseline["modality"] == mod)
            subset = baseline[mask]
            if len(subset) == 0:
                continue
            m = compute_metrics(subset)
            m["agreement"] = agree
            m["modality"] = mod
            rows_agree.append(m)
            print(f"  {agree:5s} agreement  {mod:8s}  "
                  f"Acc={m['Acc']:.1f}  MAcc={m['Multi-Acc']:.1f}  N={m['N']}")

    df_agree = pd.DataFrame(rows_agree)
    df_agree.to_csv(os.path.join(args.output_dir, "stratified_agreement.csv"), index=False)

    # ── Table 3: Stratified by ASR confidence ─────────────────────────
    print(f"\n{'='*60}")
    print("TABLE 3: Stratified by ASR confidence (baseline only)")
    print(f"{'='*60}")

    rows_asr = []
    for conf_bin in ["high", "low"]:
        for mod in ["image", "text", "fusion"]:
            mask = (baseline["asr_conf_bin"] == conf_bin) & (baseline["modality"] == mod)
            subset = baseline[mask]
            if len(subset) == 0:
                continue
            m = compute_metrics(subset)
            m["asr_conf"] = conf_bin
            m["modality"] = mod
            rows_asr.append(m)
            print(f"  {conf_bin:5s} ASR conf  {mod:8s}  "
                  f"Acc={m['Acc']:.1f}  MAcc={m['Multi-Acc']:.1f}  N={m['N']}")

    df_asr = pd.DataFrame(rows_asr)
    df_asr.to_csv(os.path.join(args.output_dir, "stratified_asr_conf.csv"), index=False)

    # ── Table 4: Modality conflict analysis ───────────────────────────
    print(f"\n{'='*60}")
    print("TABLE 4: Modality conflict analysis (baseline, all modalities)")
    print(f"{'='*60}")

    rows_conflict = []
    for conflict_val in ["agree", "conflict"]:
        for mod in ["image", "text", "fusion"]:
            mask = (baseline["conflict"] == conflict_val) & (baseline["modality"] == mod)
            subset = baseline[mask]
            if len(subset) == 0:
                continue
            m = compute_metrics(subset)
            m["conflict"] = conflict_val
            m["modality"] = mod
            rows_conflict.append(m)
            print(f"  {conflict_val:10s}  {mod:8s}  "
                  f"Acc={m['Acc']:.1f}  MAcc={m['Multi-Acc']:.1f}  N={m['N']}")

    df_conflict = pd.DataFrame(rows_conflict)
    df_conflict.to_csv(os.path.join(args.output_dir, "stratified_conflict.csv"), index=False)

    # ── Table 4b: Opposite vs mild conflict breakdown ─────────────────
    print(f"\n{'='*60}")
    print("TABLE 4b: Opposite vs mild conflict (baseline, all modalities)")
    print(f"{'='*60}")

    rows_conflict_type = []
    for ctype in ["agree", "mild", "opposite"]:
        for mod in ["image", "text", "fusion"]:
            mask = (baseline["conflict_type"] == ctype) & (baseline["modality"] == mod)
            subset = baseline[mask]
            if len(subset) == 0:
                continue
            m = compute_metrics(subset)
            m["conflict_type"] = ctype
            m["modality"] = mod
            rows_conflict_type.append(m)
            print(f"  {ctype:10s}  {mod:8s}  "
                  f"Acc={m['Acc']:.1f}  MAcc={m['Multi-Acc']:.1f}  N={m['N']}")

    df_conflict_type = pd.DataFrame(rows_conflict_type)
    df_conflict_type.to_csv(os.path.join(args.output_dir, "stratified_conflict_type.csv"), index=False)

    # ── Conflict pair frequency table ─────────────────────────────────
    print(f"\n{'='*60}")
    print("Conflict pair frequencies (image-pred vs text-pred)")
    print(f"{'='*60}")

    conflict_samples = baseline[
        (baseline["modality"] == "fusion") & (baseline["conflict"] == "conflict")
    ].copy()
    if len(conflict_samples) > 0:
        pair_counts = (
            conflict_samples
            .groupby(["img_pred", "txt_pred"])
            .size()
            .reset_index(name="count")
            .sort_values("count", ascending=False)
        )
        pair_counts["opposite"] = pair_counts.apply(
            lambda r: "YES" if frozenset({r["img_pred"], r["txt_pred"]}) in OPPOSITE_PAIRS else "",
            axis=1
        )
        print(pair_counts.head(20).to_string(index=False))
        pair_counts.to_csv(os.path.join(args.output_dir, "conflict_pairs.csv"), index=False)

    # ── Per-sample conflict detail CSV ────────────────────────────────
    detail_cols = ["url", "gt_label", "original_labels", "img_pred", "txt_pred",
                   "conflict", "conflict_type"]
    fusion_baseline = baseline[baseline["modality"] == "fusion"]
    if len(fusion_baseline) > 0:
        detail_df = fusion_baseline[detail_cols].drop_duplicates(subset=["url"])
        detail_df.to_csv(os.path.join(args.output_dir, "per_sample_conflict.csv"), index=False)
        n_agree = (detail_df["conflict_type"] == "agree").sum()
        n_mild = (detail_df["conflict_type"] == "mild").sum()
        n_opposite = (detail_df["conflict_type"] == "opposite").sum()
        print(f"\n  Summary: {n_agree} agree, {n_mild} mild conflict, {n_opposite} opposite conflict")

    # ── Delta heatmap ─────────────────────────────────────────────────
    print(f"\n{'='*60}")
    print("Generating delta heatmap...")
    print(f"{'='*60}")

    # Compute baseline accuracies
    baseline_acc = {}
    for mod in ["image", "text", "fusion"]:
        mask = (df["condition"] == "baseline") & (df["modality"] == mod)
        subset = df[mask]
        if len(subset) > 0:
            baseline_acc[mod] = accuracy(subset["gt_label"].str.lower(),
                                         subset["prediction"].str.lower())

    perturb_conds = ["no_keyword", "grayscale", "blurred_faces",
                     "text_dropout", "image_dropout"]
    modalities = ["image", "text", "fusion"]

    delta_matrix = np.full((len(perturb_conds), len(modalities)), np.nan)
    for i, cond in enumerate(perturb_conds):
        for j, mod in enumerate(modalities):
            mask = (df["condition"] == cond) & (df["modality"] == mod)
            subset = df[mask]
            if len(subset) > 0 and mod in baseline_acc:
                acc_perturbed = accuracy(subset["gt_label"].str.lower(),
                                         subset["prediction"].str.lower())
                delta_matrix[i, j] = acc_perturbed - baseline_acc[mod]

    fig, ax = plt.subplots(figsize=(6, 4))
    im = ax.imshow(delta_matrix, cmap="RdYlGn", aspect="auto",
                   vmin=-20, vmax=5)

    ax.set_xticks(range(len(modalities)))
    ax.set_xticklabels(["Image-only", "Text-only", "Fusion"], fontsize=10)
    ax.set_yticks(range(len(perturb_conds)))
    ax.set_yticklabels([c.replace("_", " ").title() for c in perturb_conds], fontsize=10)

    # Annotate cells
    for i in range(len(perturb_conds)):
        for j in range(len(modalities)):
            val = delta_matrix[i, j]
            if np.isnan(val):
                ax.text(j, i, "—", ha="center", va="center", fontsize=10, color="gray")
            else:
                sign = "+" if val >= 0 else ""
                color = "black" if abs(val) < 10 else "white"
                ax.text(j, i, f"{sign}{val:.1f}", ha="center", va="center",
                        fontsize=10, fontweight="bold", color=color)

    plt.colorbar(im, ax=ax, label="Δ Accuracy (%)", shrink=0.8)
    ax.set_title("Accuracy change from baseline", fontsize=12, fontweight="bold")
    plt.tight_layout()

    heatmap_path = os.path.join(args.output_dir, "delta_heatmap.pdf")
    plt.savefig(heatmap_path, dpi=150, bbox_inches="tight")
    plt.savefig(heatmap_path.replace(".pdf", ".png"), dpi=150, bbox_inches="tight")
    print(f"  Saved: {heatmap_path}")
    plt.close()

    # ── Combined LaTeX for stratified tables ──────────────────────────
    print("\n% --- LaTeX Table: Stratified Analysis ---")
    print(r"\begin{table}[t]")
    print(r"\centering")
    print(r"\caption{Stratified analysis on baseline condition. \textit{High/Low agreement}: single vs.\ multi-label annotations. \textit{ASR conf}: Whisper confidence $\geq$/$<$ 0.85. \textit{Conflict}: image-only and text-only predictions disagree.}")
    print(r"\label{tab:stratified}")
    print(r"\resizebox{\columnwidth}{!}{%")
    print(r"\begin{tabular}{ll rrrr r}")
    print(r"\toprule")
    print(r"\textbf{Slice} & \textbf{Modality} & \textbf{Acc} & \textbf{Multi-Acc} & \textbf{BAcc} & \textbf{F1} & \textbf{N} \\")
    print(r"\midrule")
    print(r"\multicolumn{7}{l}{\textit{Label agreement}} \\")
    for _, r in df_agree.iterrows():
        print(metrics_to_latex_row(f"  {r['agreement'].title()} agreement & {r['modality']}", r))
    print(r"\midrule")
    print(r"\multicolumn{7}{l}{\textit{ASR confidence}} \\")
    for _, r in df_asr.iterrows():
        print(metrics_to_latex_row(f"  {r['asr_conf'].title()} conf & {r['modality']}", r))
    print(r"\midrule")
    print(r"\multicolumn{7}{l}{\textit{Modality conflict}} \\")
    for _, r in df_conflict.iterrows():
        print(metrics_to_latex_row(f"  {r['conflict'].title()} & {r['modality']}", r))
    print(r"\midrule")
    print(r"\multicolumn{7}{l}{\textit{Conflict type (fusion only)}} \\")
    for _, r in df_conflict_type[df_conflict_type["modality"] == "fusion"].iterrows():
        print(metrics_to_latex_row(f"  {r['conflict_type'].title()} & fusion", r))
    print(r"\bottomrule")
    print(r"\end{tabular}}")
    print(r"\end{table}")

    print(f"\nAll outputs saved to {args.output_dir}/")


if __name__ == "__main__":
    main()
