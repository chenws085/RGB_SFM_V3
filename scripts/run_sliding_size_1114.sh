#!/usr/bin/env bash
set -euo pipefail
cd /home/ref/RGB_SFM_V3
.venv/bin/python scripts/sliding_size_sweep.py runs/train/sliding_size_20261114
python3 scripts/publish_sliding_size.py runs/train/sliding_size_20261114
