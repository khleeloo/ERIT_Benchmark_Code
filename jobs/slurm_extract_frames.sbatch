#!/bin/bash
#SBATCH --job-name=erit_extract_frames
#SBATCH --output=/home/rmfrieske/datasets/ERIT/slurm_logs/extract_frames_%j.out
#SBATCH --error=/home/rmfrieske/datasets/ERIT/slurm_logs/extract_frames_%j.err
#SBATCH --partition=cpu
#SBATCH --account=vemotionsys
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=01:00:00

eval "$(conda shell.bash hook)"
conda activate llama

cd /home/rmfrieske/datasets/ERIT

python experiments/frame_selection_ablation.py \
    --erit-csv /scratch/vemotionsys/rmfrieske/ERIT/ERIT_test/test.csv \
    --video-dir /scratch/vemotionsys/rmfrieske/datasets/ElderReact/ElderReact_Data/ElderReact_test \
    --output-dir /scratch/vemotionsys/rmfrieske/ERIT/ERIT_test_ablation \
    --seed 42
