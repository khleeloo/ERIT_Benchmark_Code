#!/bin/bash
#SBATCH --job-name=llava_lora_erit
#SBATCH --partition=preempt
#SBATCH --account=vemotionsys
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=06:00:00
#SBATCH --output=/home/rmfrieske/datasets/ERIT/slurm_logs/llava_lora_%j.out
#SBATCH --error=/home/rmfrieske/datasets/ERIT/slurm_logs/llava_lora_%j.err

set -e

echo "=== Job $SLURM_JOB_ID started at $(date) ==="
echo "Node: $(hostname), GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader)"

# Activate conda
eval "$(conda shell.bash hook)"
conda activate llama

cd /home/rmfrieske/datasets/ERIT

python experiments/finetune_llava_lora.py \
    --model-id llava-hf/llava-v1.6-mistral-7b-hf \
    --erit-root /scratch/vemotionsys/rmfrieske/ERIT \
    --output-dir /home/rmfrieske/datasets/ERIT/checkpoints/llava_lora \
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

echo "=== Job finished at $(date) ==="
