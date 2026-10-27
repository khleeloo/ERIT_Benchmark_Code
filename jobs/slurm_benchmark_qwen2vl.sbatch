#!/bin/bash
#SBATCH --job-name=qwen2vl_bench
#SBATCH --output=/home/rmfrieske/datasets/ERIT/slurm_logs/qwen2vl_bench_%j.out
#SBATCH --error=/home/rmfrieske/datasets/ERIT/slurm_logs/qwen2vl_bench_%j.err
#SBATCH --partition=normal
#SBATCH --account=vemotionsys
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --time=24:00:00

set -e

echo "=== Job $SLURM_JOB_ID started at $(date) ==="
echo "Node: $(hostname), GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader)"

eval "$(conda shell.bash hook)"
conda activate llama

cd /home/rmfrieske/datasets/ERIT

BENCHMARK_DIR="/scratch/vemotionsys/rmfrieske/ERIT_benchmark"
ERIT_ROOT="/scratch/vemotionsys/rmfrieske/datasets/ERIT_new"
TRANSCRIPT_DIR="/scratch/vemotionsys/rmfrieske/datasets/ERIT_new_transcripts"
RESULTS_DIR="/home/rmfrieske/datasets/ERIT/results/benchmark"

# ── Step 1: Evaluate Qwen2-VL on all conditions ──────────────────────
echo ""
echo "=========================================="
echo "Step 1: Evaluating Qwen2-VL"
echo "=========================================="
python experiments/eval_benchmark_qwen2vl.py \
    --benchmark-dir "${BENCHMARK_DIR}" \
    --model-id Qwen/Qwen2-VL-7B-Instruct \
    --output-dir "${RESULTS_DIR}"

# ── Step 2: Analyze results ───────────────────────────────────────────
echo ""
echo "=========================================="
echo "Step 2: Analyzing results"
echo "=========================================="
LATEST_CSV=$(ls -t "${RESULTS_DIR}"/benchmark_Qwen2-VL-7B-Instruct_*.csv 2>/dev/null | head -1)

if [ -n "${LATEST_CSV}" ]; then
    python experiments/analyze_benchmark.py \
        --results-csv "${LATEST_CSV}" \
        --erit-root "${ERIT_ROOT}" \
        --transcript-dir "${TRANSCRIPT_DIR}" \
        --output-dir "${RESULTS_DIR}/analysis_qwen2vl"
    echo "Analysis complete."
else
    echo "ERROR: No results CSV found in ${RESULTS_DIR}"
fi

echo "=== Job finished at $(date) ==="
