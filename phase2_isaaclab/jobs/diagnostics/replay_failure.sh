#!/usr/bin/env bash
set -euo pipefail
out=/robotrl/experiments/phase2_go2_baseline/run_003/failure_replay_001
test ! -e "$out/evaluation.json"
mkdir -p "$out"
cp evaluate_go2.py replay_failure.sh train_baseline.yaml "$out/"
cd "$out"
model=$(find /robotrl/experiments/phase2_go2_baseline/run_003/logs -name model_299.pt -print -quit)
test -n "$model"
timeout 900 python evaluate_go2.py --model "$model" --output evaluation.json --trace_env 11 --forward_speed 0.5 --eval_seed 10000 --task Isaac-Velocity-Flat-UnitreeGo2 --num_envs 16 physics=newton_mjwarp env.scene.robot.spawn.usd_path=/robotrl/assets/Go2/go2.usd 2>&1 | tee evaluation.log
echo ROBOTRL_FAILURE_REPLAY_PASSED
