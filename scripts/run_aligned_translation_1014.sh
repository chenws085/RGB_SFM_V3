#!/usr/bin/env bash
set -euo pipefail
cd /home/ref/RGB_SFM_V3
export CUDA_VISIBLE_DEVICES=1
export OMP_NUM_THREADS=4
export OPENBLAS_NUM_THREADS=4
log_dir=overnight_logs/aligned_translation_1014
trap 'printf "failed\n" > "$log_dir/status.txt"' ERR
printf 'training\n' > "$log_dir/status.txt"
.venv/bin/python -u scripts/train_caltech_mergingvit.py --aligned-translation --content-size 160 --output-dir runs/train/mergingvit_caltech101_aligned_translation_1014 --device cuda:0 > "$log_dir/train.log" 2>&1
printf 'evaluating\n' > "$log_dir/status.txt"
.venv/bin/python -u scripts/evaluate_aligned_augmentation.py --augmented runs/train/mergingvit_caltech101_aligned_translation_1014/MergingViT_best.pth --output-dir plots/kmeans/caltech_aligned_translation_1014 --device cuda:0 > "$log_dir/evaluation.log" 2>&1
printf 'complete\n' > "$log_dir/status.txt"
