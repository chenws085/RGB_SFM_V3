#!/usr/bin/env bash
# Fresh training on this machine, followed by the original spherical protocol.
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
export OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4
if [[ -d .venv/wsl-cuda ]]; then
    export LD_LIBRARY_PATH="$PWD/.venv/wsl-cuda:/usr/lib/wsl/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
fi
log_dir=overnight_logs/non_aligned_translation_restart_1015
train_dir=runs/train/mergingvit_caltech101_non_aligned_restart_1015
eval_dir=plots/kmeans/caltech_non_aligned_restart_1015
mkdir -p "$log_dir" "$eval_dir"
exec 9>"$log_dir/run.lock"
flock -n 9 || { echo 'Experiment already running' >&2; exit 1; }
trap 'printf "failed (exit %s)\n" "$?" > "$log_dir/status.txt"' ERR
run_device=$(.venv/bin/python -c 'import torch; print("cuda:0" if torch.cuda.is_available() else "cpu")')
printf 'training_from_scratch (%s)\n' "$run_device" > "$log_dir/status.txt"
.venv/bin/python -u scripts/train_caltech_mergingvit.py \
    --translation-shifts 1 2 3 4 5 6 7 --content-size 160 \
    --output-dir "$train_dir" --device "$run_device" > "$log_dir/train.log" 2>&1

# Archived controls were already evaluated with this split and FAISS protocol.
# Preserve their provenance; these checkpoints are not retrained on this machine.
for model in baseline aligned_8_16_32; do
    mkdir -p "$eval_dir/$model"
    for artifact in summary.csv paired_metrics.csv run_manifest.json cache_identity.json translation_stability.png; do
        cp "plots/kmeans/caltech_non_aligned_1015/$model/$artifact" "$eval_dir/$model/$artifact"
    done
done
printf 'evaluating_1_7 (%s)\n' "$run_device" > "$log_dir/status.txt"
.venv/bin/python -u scripts/evaluate_non_aligned_translation.py \
    --model augment_1_7 --checkpoint "$train_dir/MergingViT_best.pth" \
    --output-dir "$eval_dir" --device "$run_device" > "$log_dir/evaluation.log" 2>&1
printf 'complete\n' > "$log_dir/status.txt"
