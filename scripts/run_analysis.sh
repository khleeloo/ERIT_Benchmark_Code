#!/bin/bash
# ============================================================================
# Run confusion matrices + statistical significance analysis
# Run AFTER model evaluations complete (run_eval_models.sh)
#
# Usage (no GPU needed, can run on login node):
#   bash /home/rmfrieske/datasets/ERIT/scripts/run_analysis.sh
#
# You MUST set the CSV filenames below to match your actual result files.
# ============================================================================

set -e

ERIT_DIR="/home/rmfrieske/datasets/ERIT"
RESULTS_DIR="${ERIT_DIR}/results"
FIGURES_DIR="${RESULTS_DIR}/figures"

mkdir -p "${FIGURES_DIR}"

eval "$(conda shell.bash hook)"
conda activate llama

# ════════════════════════════════════════════════════════════════════════════
# CONFIGURE: Set these to your actual result CSV filenames after running
# the model evaluations. Use `ls results/results_*.csv` to find them.
#
# Example filenames (update timestamps):
#   results_llava-v1.6-mistral-7b-hf_test_2026_04_14-123456.csv
#   results_Qwen2-VL-7B-Instruct_test_2026_04_14-130000.csv
# ════════════════════════════════════════════════════════════════════════════

# Auto-detect latest result files
LLAVA_TEST=$(ls -t "${RESULTS_DIR}"/results_llava*test*.csv 2>/dev/null | head -1)
QWEN_TEST=$(ls -t "${RESULTS_DIR}"/results_Qwen*test*.csv 2>/dev/null | head -1)

# Fallback: manually specify if auto-detect fails
# LLAVA_TEST="${RESULTS_DIR}/results_llava-v1.6-mistral-7b-hf_test_TIMESTAMP.csv"
# QWEN_TEST="${RESULTS_DIR}/results_Qwen2-VL-7B-Instruct_test_TIMESTAMP.csv"

echo "============================================"
echo "Using result files:"
echo "  LLaVA:  ${LLAVA_TEST}"
echo "  Qwen2:  ${QWEN_TEST}"
echo "============================================"

if [[ -z "$LLAVA_TEST" || -z "$QWEN_TEST" ]]; then
    echo "ERROR: Could not find result CSVs. Available files:"
    ls "${RESULTS_DIR}"/results_*.csv 2>/dev/null || echo "  (none)"
    echo ""
    echo "Run run_eval_models.sh first, or set paths manually in this script."
    exit 1
fi

# ── Step 1: Confusion matrices + error analysis ────────────────────────────
echo ""
echo "=========================================="
echo "  Generating confusion matrices"
echo "=========================================="
python "${ERIT_DIR}/experiments/confusion_analysis.py" \
    --csv-a "${LLAVA_TEST}" --label-a "LLaVA-NeXT" \
    --csv-b "${QWEN_TEST}" --label-b "Qwen2-VL" \
    --output-dir "${FIGURES_DIR}" \
    --mode fusion 2>&1 | tee "${RESULTS_DIR}/confusion_analysis.log"

# ── Step 2: Statistical significance tests ─────────────────────────────────
echo ""
echo "=========================================="
echo "  Running statistical tests"
echo "=========================================="
python "${ERIT_DIR}/experiments/statistical_tests.py" \
    --csvs "${LLAVA_TEST}" "${QWEN_TEST}" \
    --labels "LLaVA-NeXT" "Qwen2-VL" \
    --n-bootstrap 1000 \
    --output-dir "${RESULTS_DIR}" 2>&1 | tee "${RESULTS_DIR}/statistical_tests.log"

echo ""
echo "=========================================="
echo "  Analysis complete!"
echo "=========================================="
echo "Figures:  ${FIGURES_DIR}/"
ls "${FIGURES_DIR}"/*.pdf 2>/dev/null
echo ""
echo "Tables:   ${RESULTS_DIR}/"
ls "${RESULTS_DIR}"/*.csv 2>/dev/null
echo ""
echo "Logs:     ${RESULTS_DIR}/"
ls "${RESULTS_DIR}"/*.log 2>/dev/null
