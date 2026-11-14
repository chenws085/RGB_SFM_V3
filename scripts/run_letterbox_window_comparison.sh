#!/usr/bin/env bash
set -euo pipefail
cd /home/ref/RGB_SFM_V3
run=runs/train/window_letterbox160_20261109
out=plots/reports/attention_translation_letterbox160_20261109
.venv/bin/python scripts/train_caltech_mergingvit.py --output-dir "$run" --attention-mode window --window-size 7 --device cuda:1 --epochs 100 --batch-size 32 --workers 8 --lr 0.0003 --patience 20 --seed 42 --content-size 160
.venv/bin/python scripts/evaluate_attention_translation.py "$out" --letterbox-run "$run"
.venv/bin/python scripts/report_attention_translation.py "$out"
