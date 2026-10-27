#!/bin/bash
#SBATCH --job-name=erit_ens_dev
#SBATCH --output=/home/rmfrieske/datasets/ERIT/slurm_logs/ensemble_dev_%j.out
#SBATCH --error=/home/rmfrieske/datasets/ERIT/slurm_logs/ensemble_dev_%j.err
#SBATCH --partition=normal
#SBATCH --account=vemotionsys
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=06:00:00

eval "$(conda shell.bash hook)"
conda activate llama

cd /home/rmfrieske/datasets/ERIT

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
echo "Building ERIT_new dataset"
echo "=========================================="
python erit/build_erit_new.py \
    --output-dir /scratch/vemotionsys/rmfrieske/datasets/ERIT_new \
    --transcript-dir /scratch/vemotionsys/rmfrieske/datasets/ERIT_new_transcripts
