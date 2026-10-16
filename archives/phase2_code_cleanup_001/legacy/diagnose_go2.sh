#!/usr/bin/env bash
set -euo pipefail
run_dir="$1"
mkdir -p "$run_dir"
cp evaluate_go2.py diagnose_go2.sh diagnose_go2.yaml "$run_dir/"
cd "$run_dir"
for checkpoint in 0 199; do
    model=$(find /robotrl/run_004/logs -name "model_${checkpoint}.pt" -print -quit)
    test -n "$model"
    timeout 900 python evaluate_go2.py --model "$model" --output "forward_${checkpoint}.json" --forward_speed 0.5 --task Isaac-Velocity-Flat-UnitreeGo2 --num_envs 16 physics=newton_mjwarp 2>&1 | tee "forward_${checkpoint}.log"
done
echo ROBOTRL_DIAGNOSTIC_PASSED
