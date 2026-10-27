#!/bin/bash
#SBATCH --job-name=erit_ensemble_train_dev
#SBATCH --output=/home/rmfrieske/datasets/ERIT/slurm_logs/ensemble_train_dev_%j.out
#SBATCH --error=/home/rmfrieske/datasets/ERIT/slurm_logs/ensemble_train_dev_%j.err
#SBATCH --partition=preempt
#SBATCH --account=vemotionsys
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=12:00:00

eval "$(conda shell.bash hook)"
conda activate llama

cd /home/rmfrieske/datasets/ERIT

echo "=========================================="
echo "TRAIN split ensemble extraction"
echo "=========================================="
python erit/ensemble_frame_selection.py \
    --erit-csv /scratch/vemotionsys/rmfrieske/ERIT/ERIT_train/train.csv \
    --video-dir /scratch/vemotionsys/rmfrieske/datasets/ElderReact/ElderReact_Data/ElderReact_train \
    --output-dir /scratch/vemotionsys/rmfrieske/ERIT/ERIT_train_ensemble \
    --sample-rate 3

echo ""
echo "=========================================="
echo "DEV split ensemble extraction"
echo "=========================================="
python erit/ensemble_frame_selection.py \
    --erit-csv /scratch/vemotionsys/rmfrieske/ERIT/ERIT_dev/dev.csv \
    --video-dir /scratch/vemotionsys/rmfrieske/datasets/ElderReact/ElderReact_Data/ElderReact_dev \
    --output-dir /scratch/vemotionsys/rmfrieske/ERIT/ERIT_dev_ensemble \
    --sample-rate 3

echo ""
echo "=========================================="
echo "Ensemble extraction complete."
echo "Run erit/build_erit_new.py AFTER Whisper transcription (job 374704) finishes."
echo "=========================================="
