#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
export OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4
log_dir=overnight_logs/allclass_translation_1020
mkdir -p "$log_dir"
exec 9>"$log_dir/run.lock"
flock -n 9 || { echo 'All-class translation evaluation already running' >&2; exit 1; }
trap 'printf "failed\n" > "$log_dir/status.txt"' ERR
for model in baseline aligned_8_16_32 augment_1_7; do
    printf 'evaluating %s\n' "$model" > "$log_dir/status.txt"
    .venv/bin/python -u scripts/evaluate_allclass_translation.py --model "$model" --per-class 5 --sample-seed 42 --device "${ALLCLASS_DEVICE:-cuda:1}" > "$log_dir/$model.log" 2>&1
done
printf 'evaluation_complete\n' > "$log_dir/status.txt"
