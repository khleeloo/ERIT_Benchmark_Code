#!/bin/bash
#SBATCH --job-name=llava_lora_new
#SBATCH --partition=preempt
#SBATCH --account=vemotionsys
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=08:00:00
#SBATCH --output=/home/rmfrieske/datasets/ERIT/slurm_logs/llava_lora_new_%j.out
#SBATCH --error=/home/rmfrieske/datasets/ERIT/slurm_logs/llava_lora_new_%j.err

set -e

echo "=== Job $SLURM_JOB_ID started at $(date) ==="
echo "Node: $(hostname), GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader)"

eval "$(conda shell.bash hook)"
conda activate llama

cd /home/rmfrieske/datasets/ERIT

ERIT_ROOT="/scratch/vemotionsys/rmfrieske/datasets/ERIT_new"
CKPT_DIR="/home/rmfrieske/datasets/ERIT/checkpoints/llava_lora_new"
RESULTS_DIR="/home/rmfrieske/datasets/ERIT/results"

# ── Step 1: Fine-tune LoRA on ERIT_new train ─────────────────────────
echo "=========================================="
echo "Step 1: Fine-tuning LLaVA-NeXT LoRA on ERIT_new"
echo "=========================================="
python experiments/finetune_llava_lora.py \
    --model-id llava-hf/llava-v1.6-mistral-7b-hf \
    --erit-root "${ERIT_ROOT}" \
    --output-dir "${CKPT_DIR}" \
    --epochs 5 \
    --batch-size 4 \
    --grad-accum 4 \
    --lr 1e-4 \
    --lora-rank 8 \
    --lora-alpha 16 \
    --lora-dropout 0.1 \
    --lora-targets q_proj,v_proj \
    --lora-layers 4 \
    --train-mode fusion \
    --load-in-4bit

# ── Step 2: Find best checkpoint (highest dev accuracy from training log) ─
# Use the most recent epoch5 checkpoint as final, or best if saved
BEST_CKPT=$(ls -dt "${CKPT_DIR}"/epoch5_* 2>/dev/null | head -1)
if [ -z "${BEST_CKPT}" ]; then
    BEST_CKPT=$(ls -dt "${CKPT_DIR}"/epoch*_* 2>/dev/null | head -1)
fi

if [ -z "${BEST_CKPT}" ]; then
    echo "ERROR: No checkpoint found in ${CKPT_DIR}"
    exit 1
fi

echo ""
echo "=========================================="
echo "Step 2: Evaluating LoRA on dev+test (checkpoint: ${BEST_CKPT})"
echo "=========================================="
python experiments/eval_llava_lora.py \
    --checkpoint "${BEST_CKPT}" \
    --model-id llava-hf/llava-v1.6-mistral-7b-hf \
    --erit-root "${ERIT_ROOT}" \
    --split test \
    --output-dir "${RESULTS_DIR}" \
    --load-in-4bit

echo "=== Job finished at $(date) ==="
