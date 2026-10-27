"""
Statistical significance testing for ERIT results.
- McNemar's test: fusion vs image-only, fusion vs text-only (per model)
- Bootstrap 95% confidence intervals on accuracy
- Outputs a summary table ready for LaTeX

Usage:
  python experiments/statistical_tests.py \
    --csvs results/results_gpt4o_test.csv results/results_llava_test.csv \
    --labels "GPT-4o" "LLaVA-NeXT" \
    --n-bootstrap 1000
"""

import argparse
import os

import numpy as np
import pandas as pd
from scipy import stats

EMOTIONS = ["happiness", "sadness", "disgust", "anger", "surprise", "fear", "neutral"]


def load_correctness(csv_path):
    """Load results and return per-sample correctness arrays for each mode."""
    df = pd.read_csv(csv_path)
    gt = df["GTLabel"].astype(str).str.strip().str.lower()
    result = {}
    for mode, col in [("text", "Prediction_Text"),
                      ("image", "Prediction_Image"),
                      ("fusion", "Prediction_Fusion")]:
        pred = df[col].astype(str).str.strip().str.lower()
        result[mode] = (pred == gt).values.astype(int)
    return result, gt.values


def mcnemar_test(correct_a, correct_b):
    """
    McNemar's test comparing two paired binary correctness arrays.
    Returns (chi2, p_value, n_discordant).
    """
    # Contingency: a correct & b wrong, a wrong & b correct
    b_val = ((correct_a == 1) & (correct_b == 0)).sum()  # fusion right, unimodal wrong
    c_val = ((correct_a == 0) & (correct_b == 1)).sum()  # fusion wrong, unimodal right

    n_discordant = b_val + c_val
    if n_discordant == 0:
        return 0.0, 1.0, 0

    # McNemar with continuity correction
    chi2 = (abs(b_val - c_val) - 1) ** 2 / (b_val + c_val)
    p_value = 1 - stats.chi2.cdf(chi2, df=1)
    return chi2, p_value, n_discordant


def bootstrap_ci(correct, n_bootstrap=1000, ci=95, seed=42):
    """Bootstrap confidence interval for accuracy."""
    rng = np.random.RandomState(seed)
    n = len(correct)
    accs = []
    for _ in range(n_bootstrap):
        idx = rng.randint(0, n, size=n)
        accs.append(correct[idx].mean())
    lo = np.percentile(accs, (100 - ci) / 2)
    hi = np.percentile(accs, 100 - (100 - ci) / 2)
    return lo * 100, hi * 100


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--csvs", nargs="+", required=True,
                        help="Result CSV files")
    parser.add_argument("--labels", nargs="+", required=True,
                        help="Model labels (same order as csvs)")
    parser.add_argument("--n-bootstrap", type=int, default=1000)
    parser.add_argument("--output-dir", default="/home/rmfrieske/datasets/ERIT/results")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    # ── Per-model results ───────────────────────────────────────────────────
    rows_table = []
    rows_mcnemar = []

    for csv_path, label in zip(args.csvs, args.labels):
        print(f"\n{'='*70}")
        print(f"Model: {label}  ({csv_path})")
        print(f"{'='*70}")

        correct, gt = load_correctness(csv_path)
        n = len(gt)

        for mode in ["image", "text", "fusion"]:
            acc = correct[mode].mean() * 100
            lo, hi = bootstrap_ci(correct[mode], args.n_bootstrap)
            margin = (hi - lo) / 2
            rows_table.append({
                "Model": label, "Mode": mode,
                "Accuracy": acc,
                "CI_low": lo, "CI_high": hi,
                "Acc_str": f"{acc:.2f} ± {margin:.1f}",
                "N": n,
            })
            print(f"  {mode:>6}: {acc:.2f}%  95% CI [{lo:.2f}, {hi:.2f}]")

        # McNemar: fusion vs image
        chi2_i, p_i, nd_i = mcnemar_test(correct["fusion"], correct["image"])
        # McNemar: fusion vs text
        chi2_t, p_t, nd_t = mcnemar_test(correct["fusion"], correct["text"])

        sig_i = "***" if p_i < 0.001 else "**" if p_i < 0.01 else "*" if p_i < 0.05 else "ns"
        sig_t = "***" if p_t < 0.001 else "**" if p_t < 0.01 else "*" if p_t < 0.05 else "ns"

        rows_mcnemar.append({
            "Model": label,
            "Comparison": "Fusion vs Image",
            "chi2": chi2_i, "p_value": p_i,
            "n_discordant": nd_i, "sig": sig_i,
        })
        rows_mcnemar.append({
            "Model": label,
            "Comparison": "Fusion vs Text",
            "chi2": chi2_t, "p_value": p_t,
            "n_discordant": nd_t, "sig": sig_t,
        })

        print(f"\n  McNemar's test:")
        print(f"    Fusion vs Image: χ²={chi2_i:.2f}, p={p_i:.4f} {sig_i} (discordant={nd_i})")
        print(f"    Fusion vs Text:  χ²={chi2_t:.2f}, p={p_t:.4f} {sig_t} (discordant={nd_t})")

    # ── Save tables ─────────────────────────────────────────────────────────
    df_table = pd.DataFrame(rows_table)
    df_mcnemar = pd.DataFrame(rows_mcnemar)

    df_table.to_csv(os.path.join(args.output_dir, "accuracy_with_ci.csv"), index=False)
    df_mcnemar.to_csv(os.path.join(args.output_dir, "mcnemar_tests.csv"), index=False)

    # ── LaTeX table ─────────────────────────────────────────────────────────
    print(f"\n{'='*70}")
    print("LaTeX table rows (paste into Table 2a):")
    print(f"{'='*70}")

    # Pivot for LaTeX output
    for label in args.labels:
        sub = df_table[df_table["Model"] == label]
        mcn = df_mcnemar[df_mcnemar["Model"] == label]
        sig_i = mcn[mcn["Comparison"] == "Fusion vs Image"]["sig"].values[0]
        sig_t = mcn[mcn["Comparison"] == "Fusion vs Text"]["sig"].values[0]

        img_row = sub[sub["Mode"] == "image"].iloc[0]
        txt_row = sub[sub["Mode"] == "text"].iloc[0]
        fus_row = sub[sub["Mode"] == "fusion"].iloc[0]

        sig_mark_i = "$^{\\dagger}$" if sig_i != "ns" else ""
        sig_mark_t = "$^{\\dagger}$" if sig_t != "ns" else ""

        print(f"% {label}")
        print(f"Image  & {img_row['Acc_str']} \\\\")
        print(f"Text   & {txt_row['Acc_str']} \\\\")
        print(f"Fusion & \\textbf{{{fus_row['Acc_str']}}}{sig_mark_i}{sig_mark_t} \\\\")
        print()

    print(f"\nResults saved to {args.output_dir}/accuracy_with_ci.csv")
    print(f"McNemar tests saved to {args.output_dir}/mcnemar_tests.csv")


if __name__ == "__main__":
    main()
