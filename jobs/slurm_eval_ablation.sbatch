#!/bin/bash
#SBATCH --job-name=erit_ablation_eval
#SBATCH --output=/home/rmfrieske/datasets/ERIT/slurm_logs/ablation_eval_%j.out
#SBATCH --error=/home/rmfrieske/datasets/ERIT/slurm_logs/ablation_eval_%j.err
#SBATCH --partition=preempt
#SBATCH --account=vemotionsys
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=04:00:00

eval "$(conda shell.bash hook)"
conda activate llama

cd /home/rmfrieske/datasets/ERIT

python experiments/eval_ablation.py \
    --erit-csv /scratch/vemotionsys/rmfrieske/ERIT/ERIT_test/test.csv \
    --random-csv /scratch/vemotionsys/rmfrieske/ERIT/ERIT_test_ablation/test_random.csv \
    --random-img-dir /scratch/vemotionsys/rmfrieske/ERIT/ERIT_test_ablation/random_frames \
    --midpoint-csv /scratch/vemotionsys/rmfrieske/ERIT/ERIT_test_ablation/test_midpoint.csv \
    --midpoint-img-dir /scratch/vemotionsys/rmfrieske/ERIT/ERIT_test_ablation/midpoint_frames \
    --ensemble-csv /scratch/vemotionsys/rmfrieske/ERIT/ERIT_test_ensemble/test_ensemble.csv \
    --ensemble-img-dir /scratch/vemotionsys/rmfrieske/ERIT/ERIT_test_ensemble/consensus_frames \
    --output-dir /home/rmfrieske/datasets/ERIT/results/ablation
