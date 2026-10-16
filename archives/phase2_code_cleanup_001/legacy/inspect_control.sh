#!/usr/bin/env bash
set -euo pipefail
run_dir="$1"
mkdir -p "$run_dir"
cp inspect_control.py inspect_control.sh inspect_control.yaml "$run_dir/"
cd "$run_dir"
model=$(find /robotrl/run_004/logs -name model_199.pt -print -quit)
test -n "$model"
timeout 900 python inspect_control.py --model "$model" --task Isaac-Velocity-Flat-UnitreeGo2 --num_envs 4 physics=newton_mjwarp 2>&1 | tee control_diagnostic.log
