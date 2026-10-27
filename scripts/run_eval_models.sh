#!/bin/bash
# ============================================================================
# Run ERIT evaluation experiments (LLaVA-NeXT + Qwen2-VL)
# Run inside an interactive GPU session:
#   srun --partition=preempt --account=vemotionsys --gres=gpu:1 \
#        --cpus-per-task=4 --mem=48G --time=02:00:00 --pty bash
#   Then: bash /home/rmfrieske/datasets/ERIT/scripts/run_eval_models.sh
# ============================================================================

set -e

ERIT_DIR="/home/rmfrieske/datasets/ERIT"
RESULTS_DIR="${ERIT_DIR}/results"
ERIT_ROOT="/scratch/vemotionsys/rmfrieske/ERIT"

mkdir -p "${RESULTS_DIR}"

# Activate conda env (adjust name if needed)
eval "$(conda shell.bash hook)"
conda activate llama  # or whichever env has transformers >= 4.40

echo "============================================"
echo "GPU check:"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
echo "============================================"

# ── Step 1: Verify dependencies ─────────────────────────────────────────────
echo "Verifying dependencies..."
python -c "from transformers import LlavaNextProcessor; print('  LLaVA: OK')"
python -c "from transformers import Qwen2VLForConditionalGeneration; print('  Qwen2-VL: OK')"
python -c "import sklearn, matplotlib, scipy; print('  Analysis libs: OK')"
echo "Dependencies OK."

# ── Step 2: Run LLaVA-NeXT on test + dev ────────────────────────────────────
echo ""
echo "=========================================="
echo "  Running LLaVA-NeXT on ERIT test"
echo "  (First run will download ~15GB model)"
echo "=========================================="
python "${ERIT_DIR}/experiments/eval_llava.py" \
    --model-id "llava-hf/llava-v1.6-mistral-7b-hf" \
    --erit-root "${ERIT_ROOT}" \
    --split test \
    --output-dir "${RESULTS_DIR}"

echo ""
echo "=========================================="
echo "  Running LLaVA-NeXT on ERIT dev"
echo "=========================================="
python "${ERIT_DIR}/experiments/eval_llava.py" \
    --model-id "llava-hf/llava-v1.6-mistral-7b-hf" \
    --erit-root "${ERIT_ROOT}" \
    --split dev \
    --output-dir "${RESULTS_DIR}"

# ── Step 3: Run Qwen2-VL on test + dev ─────────────────────────────────────
echo ""
echo "=========================================="
echo "  Running Qwen2-VL on ERIT test"
echo "=========================================="
python "${ERIT_DIR}/experiments/eval_qwen2vl.py" \
    --model-id "Qwen/Qwen2-VL-7B-Instruct" \
    --erit-root "${ERIT_ROOT}" \
    --split test \
    --output-dir "${RESULTS_DIR}"

echo ""
echo "=========================================="
echo "  Running Qwen2-VL on ERIT dev"
echo "=========================================="
python "${ERIT_DIR}/experiments/eval_qwen2vl.py" \
    --model-id "Qwen/Qwen2-VL-7B-Instruct" \
    --erit-root "${ERIT_ROOT}" \
    --split dev \
    --output-dir "${RESULTS_DIR}"

echo ""
echo "=========================================="
echo "  All model evaluations complete!"
echo "  Results in: ${RESULTS_DIR}"
echo "=========================================="
ls -la "${RESULTS_DIR}"/results_*.csv
