#!/usr/bin/env bash
set -euo pipefail
cd /home/ref/RGB_SFM_V3
export OMP_NUM_THREADS=4
export OPENBLAS_NUM_THREADS=4
log_dir=overnight_logs/non_aligned_translation_1015
trap 'printf "failed\n" > "$log_dir/status.txt"' ERR
run_device=$(.venv/bin/python -c 'import torch; print("cuda:0" if torch.cuda.is_available() else "cpu")')
printf 'baseline_evaluation (%s)\n' "$run_device" > "$log_dir/status.txt"
.venv/bin/python -u scripts/evaluate_non_aligned_translation.py --model baseline --device "$run_device" > "$log_dir/baseline.log" 2>&1
printf 'aligned_model_evaluation (%s)\n' "$run_device" > "$log_dir/status.txt"
.venv/bin/python -u scripts/evaluate_non_aligned_translation.py --model aligned_8_16_32 --device "$run_device" > "$log_dir/aligned.log" 2>&1
printf 'training_1_7 (%s)\n' "$run_device" > "$log_dir/status.txt"
.venv/bin/python -u scripts/train_caltech_mergingvit.py --translation-shifts 1 2 3 4 5 6 7 --content-size 160 --output-dir runs/train/mergingvit_caltech101_non_aligned_1015 --device "$run_device" > "$log_dir/train.log" 2>&1
printf 'augmented_evaluation (%s)\n' "$run_device" > "$log_dir/status.txt"
.venv/bin/python -u scripts/evaluate_non_aligned_translation.py --model augment_1_7 --device "$run_device" > "$log_dir/augmented.log" 2>&1
printf 'complete\n' > "$log_dir/status.txt"
