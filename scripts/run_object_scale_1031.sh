#!/usr/bin/env bash
set -euo pipefail
cd /home/ref/RGB_SFM_V3
stamp=$(date -u +%Y%m%dT%H%M%SZ)
base="runs/train/object_scale_1031_${stamp}"
mkdir -p "$base"
printf '%s\n' "$base" > overnight_logs/object_scale_training_1031/latest_run.txt
.venv/bin/python -u scripts/train_object_scale.py --output "$base/B" --epochs 20 --per-class 16 --batch-size 16 --device cuda:1 > "$base/B.log" 2>&1
.venv/bin/python -u scripts/train_object_scale.py --output "$base/D" --content-merging --epochs 20 --per-class 16 --batch-size 16 --device cuda:1 > "$base/D.log" 2>&1
