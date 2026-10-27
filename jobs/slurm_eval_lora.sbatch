#!/bin/bash
#SBATCH --job-name=eval_lora
#SBATCH --partition=preempt
#SBATCH --account=vemotionsys
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --time=00:30:00
#SBATCH --output=/home/rmfrieske/datasets/ERIT/slurm_logs/eval_lora_%j.out
#SBATCH --error=/home/rmfrieske/datasets/ERIT/slurm_logs/eval_lora_%j.err

eval "$(conda shell.bash hook)"
conda activate llama
cd /home/rmfrieske/datasets/ERIT

python experiments/eval_llava_lora.py \
  --checkpoint checkpoints/llava_lora/epoch5_2026_04_14-152650 \
  --split test \
  --load-in-4bit

echo "=== Done ==="
