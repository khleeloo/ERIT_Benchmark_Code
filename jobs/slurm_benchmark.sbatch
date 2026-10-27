#!/bin/bash
#SBATCH --job-name=erit_benchmark
#SBATCH --output=/home/rmfrieske/datasets/ERIT/slurm_logs/benchmark_%j.out
#SBATCH --error=/home/rmfrieske/datasets/ERIT/slurm_logs/benchmark_%j.err
#SBATCH --partition=normal
#SBATCH --account=vemotionsys
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=24:00:00

eval "$(conda shell.bash hook)"
conda activate llama

cd /home/rmfrieske/datasets/ERIT

ERIT_ROOT="/scratch/vemotionsys/rmfrieske/datasets/ERIT_new"
BENCHMARK_DIR="/scratch/vemotionsys/rmfrieske/ERIT_benchmark"
TRANSCRIPT_DIR="/scratch/vemotionsys/rmfrieske/datasets/ERIT_new_transcripts"
RESULTS_DIR="/home/rmfrieske/datasets/ERIT/results/benchmark"

# ── Step 1: Generate perturbation conditions ──────────────────────────
echo "=========================================="
echo "Step 1: Generating perturbation conditions"
echo "=========================================="
python erit/generate_perturbations.py \
    --erit-root "${ERIT_ROOT}" \
    --output-dir "${BENCHMARK_DIR}"

# ── Step 2: Evaluate LLaVA-NeXT on all conditions ────────────────────
echo ""
echo "=========================================="
echo "Step 2: Evaluating LLaVA-NeXT"
echo "=========================================="
python experiments/eval_benchmark.py \
    --benchmark-dir "${BENCHMARK_DIR}" \
    --model-id llava-hf/llava-v1.6-mistral-7b-hf \
    --output-dir "${RESULTS_DIR}"

# ── Step 3: Analyze results ───────────────────────────────────────────
echo ""
echo "=========================================="
echo "Step 3: Analyzing results"
echo "=========================================="
# Find the most recent results CSV
LATEST_CSV=$(ls -t "${RESULTS_DIR}"/benchmark_llava-v1.6-mistral-7b-hf_*.csv 2>/dev/null | head -1)

if [ -n "${LATEST_CSV}" ]; then
    python experiments/analyze_benchmark.py \
        --results-csv "${LATEST_CSV}" \
        --erit-root "${ERIT_ROOT}" \
        --transcript-dir "${TRANSCRIPT_DIR}" \
        --output-dir "${RESULTS_DIR}/analysis"
    echo "Analysis complete."
else
    echo "ERROR: No results CSV found in ${RESULTS_DIR}"
fi
