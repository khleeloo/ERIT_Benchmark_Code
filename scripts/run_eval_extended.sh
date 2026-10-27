#!/usr/bin/env bash
set -e

ERIT_DIR="/home/rmfrieske/datasets/ERIT"
cd "${ERIT_DIR}"
python3 experiments/eval_metrics_extended.py
