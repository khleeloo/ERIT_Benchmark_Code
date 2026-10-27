"""Extended evaluation: Acc, Balanced Acc, Macro F1 across runs.
No sklearn required — metrics computed directly with pandas/numpy.
"""

import pandas as pd
import numpy as np


def accuracy_score(gt, pred):
    return (gt == pred).mean()


def balanced_accuracy_score(gt, pred):
    classes = gt.unique()
    recalls = []
    for c in classes:
        mask = gt == c
        if mask.sum() == 0:
            continue
        recalls.append((pred[mask] == c).mean())
    return float(np.mean(recalls))


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
    return float(np.mean(f1s))


def classification_report(gt, pred):
    classes = sorted(gt.unique())
    lines = [f"{'':>12}  {'precision':>9}  {'recall':>7}  {'f1-score':>8}  {'support':>7}"]
    for c in classes:
        tp = ((pred == c) & (gt == c)).sum()
        fp = ((pred == c) & (gt != c)).sum()
        fn = ((pred != c) & (gt == c)).sum()
        sup = (gt == c).sum()
        p = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        r = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f = 2 * p * r / (p + r) if (p + r) > 0 else 0.0
        lines.append(f"{c:>12}  {p:>9.2f}  {r:>7.2f}  {f:>8.2f}  {sup:>7}")
    lines.append("")
    acc = accuracy_score(gt, pred)
    lines.append(f"{'accuracy':>12}  {'':>9}  {'':>7}  {acc:>8.2f}  {len(gt):>7}")
    lines.append(f"{'macro avg':>12}  {'':>9}  {'':>7}  {macro_f1(gt, pred):>8.2f}  {len(gt):>7}")
    return "\n".join(lines)

LAST = "results/results_llava_lora_test_2026_04_14-152553.csv"
PREV = "results/results_llava_lora_test_2026_04_14-150037.csv"
ZS_LLAVA = "results/results_llava-v1.6-mistral-7b-hf_test_2026_04_14-115050.csv"
ZS_QWEN  = "results/results_Qwen2-VL-7B-Instruct_test_2026_04_14-120413.csv"


def evaluate(csv_path, label):
    df = pd.read_csv(csv_path)
    gt = df["GTLabel"].str.lower()
    rows = []
    for mode in ["Fusion", "Image", "Text"]:
        col = f"Prediction_{mode}"
        if col not in df.columns:
            continue
        pred = df[col].str.lower()
        n    = len(df)
        acc  = accuracy_score(gt, pred) * 100
        bacc = balanced_accuracy_score(gt, pred) * 100
        f1   = macro_f1(gt, pred) * 100
        ci   = 1.96 * np.sqrt(acc / 100 * (1 - acc / 100) / n) * 100
        rows.append({
            "Run": label, "Mode": mode, "N": n,
            "Acc%": round(acc, 2), "±CI": round(ci, 2),
            "BalAcc%": round(bacc, 2), "MacroF1%": round(f1, 2),
        })
    return pd.DataFrame(rows)


df_zs_llava = evaluate(ZS_LLAVA, "LLaVA zero-shot")
df_zs_qwen  = evaluate(ZS_QWEN,  "Qwen2-VL zero-shot")
df_prev     = evaluate(PREV,     "LoRA-prev (eff_bs=16)")
df_last     = evaluate(LAST,     "LoRA-last (eff_bs=32)")

combined = pd.concat([df_zs_llava, df_zs_qwen, df_prev, df_last], ignore_index=True)

print("\n" + "=" * 72)
print("  ERIT TEST SET — Accuracy / Balanced Accuracy / Macro F1")
print("=" * 72)
print(combined.to_string(index=False))
print("=" * 72)

# Per-class report for last LoRA run, Fusion mode
print("\n─── Per-class report: LoRA-last, Fusion ───")
df = pd.read_csv(LAST)
print(classification_report(
    df["GTLabel"].str.lower(),
    df["Prediction_Fusion"].str.lower(),
))
