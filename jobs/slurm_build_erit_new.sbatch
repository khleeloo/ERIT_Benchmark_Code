#!/bin/bash
#SBATCH --job-name=erit_build_new
#SBATCH --output=/home/rmfrieske/datasets/ERIT/slurm_logs/build_erit_new_%j.out
#SBATCH --error=/home/rmfrieske/datasets/ERIT/slurm_logs/build_erit_new_%j.err
#SBATCH --partition=preempt
#SBATCH --account=vemotionsys
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=00:30:00
#SBATCH --dependency=afterok:374721:374722

eval "$(conda shell.bash hook)"
conda activate llama

cd /home/rmfrieske/datasets/ERIT

echo "=========================================="
echo "Building ERIT_new dataset"
echo "=========================================="
python erit/build_erit_new.py \
    --output-dir /scratch/vemotionsys/rmfrieske/datasets/ERIT_new \
    --transcript-dir /scratch/vemotionsys/rmfrieske/datasets/ERIT_new_transcripts
