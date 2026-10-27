#!/bin/bash
#SBATCH --job-name=erit_whisper_transcribe
#SBATCH --output=/home/rmfrieske/datasets/ERIT/slurm_logs/whisper_transcribe_%A_%a.out
#SBATCH --error=/home/rmfrieske/datasets/ERIT/slurm_logs/whisper_transcribe_%A_%a.err
#SBATCH --partition=normal
#SBATCH --account=vemotionsys
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=06:00:00
#SBATCH --array=0-1

eval "$(conda shell.bash hook)"
conda activate llama

cd /home/rmfrieske/datasets/ERIT

# Map array index to split
SPLITS=(train dev test)
SPLIT=${SPLITS[$SLURM_ARRAY_TASK_ID]}

VIDEO_DIR="/scratch/vemotionsys/rmfrieske/datasets/ElderReact/ElderReact_Data/ElderReact_${SPLIT}"
OUTPUT_DIR="/scratch/vemotionsys/rmfrieske/datasets/ERIT_new_transcripts"

echo "=========================================="
echo "Whisper Large V3 transcription: ${SPLIT}"
echo "Videos: ${VIDEO_DIR}"
echo "Output: ${OUTPUT_DIR}"
echo "=========================================="

python erit/whisper_transcribe_elderreact.py \
    --video-dir "${VIDEO_DIR}" \
    --output-dir "${OUTPUT_DIR}" \
    --split "${SPLIT}" \
    --model-id openai/whisper-large-v3
