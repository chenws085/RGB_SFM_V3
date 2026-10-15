#!/usr/bin/env bash
# Queue scale work after the entire translation pipeline releases its lock.
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
export OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4
if [[ -d .venv/wsl-cuda ]]; then
    export LD_LIBRARY_PATH="$PWD/.venv/wsl-cuda:/usr/lib/wsl/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
fi
log_dir=overnight_logs/caltech_scale_1015
dependency=overnight_logs/non_aligned_translation_restart_1015
s0=runs/train/mergingvit_caltech101_scale_S0_1015
s1=runs/train/mergingvit_caltech101_scale_S1_1015
out=plots/kmeans/caltech_scale_1015
mkdir -p "$log_dir"
exec 9>"$log_dir/run.lock"
flock -n 9 || { echo 'Scale queue already running' >&2; exit 1; }
trap 'code=$?; printf "failed (exit %s; inspect launcher/train/evaluation logs)\n" "$code" > "$log_dir/status.txt"' ERR
printf 'waiting_for_translation_training_and_evaluation\n' > "$log_dir/status.txt"
exec 8>"$dependency/run.lock"
flock -x 8
if [[ "$(cat "$dependency/status.txt")" != complete ]]; then
    printf 'blocked: translation pipeline did not finish successfully\n' > "$log_dir/status.txt"
    exit 1
fi
exec 8>&-
run_device=$(.venv/bin/python -c 'import torch; print("cuda:0" if torch.cuda.is_available() else "cpu")')
printf 'training_S0 (%s)\n' "$run_device" > "$log_dir/status.txt"
.venv/bin/python -u scripts/train_caltech_mergingvit.py \
    --content-size 160 --output-dir "$s0" --device "$run_device" > "$log_dir/train_S0.log" 2>&1
printf 'training_S1 (%s)\n' "$run_device" > "$log_dir/status.txt"
.venv/bin/python -u scripts/train_caltech_mergingvit.py \
    --content-size 160 --scale-content-sizes 96 128 160 192 \
    --output-dir "$s1" --device "$run_device" > "$log_dir/train_S1.log" 2>&1
printf 'evaluating_scale (%s)\n' "$run_device" > "$log_dir/status.txt"
.venv/bin/python -u scripts/evaluate_caltech_scale.py \
    --s0 "$s0/MergingViT_best.pth" --s1 "$s1/MergingViT_best.pth" \
    --translation-checkpoint runs/train/mergingvit_caltech101_non_aligned_restart_1015/MergingViT_best.pth \
    --output-dir "$out" --device "$run_device" > "$log_dir/evaluation.log" 2>&1
printf 'complete\n' > "$log_dir/status.txt"
