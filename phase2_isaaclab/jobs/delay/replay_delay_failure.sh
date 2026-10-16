#!/usr/bin/env bash
set -euo pipefail
out=/robotrl/experiments/phase2_go2_delay_training/run_001/failure_replay_001
test ! -e "$out/evaluation.json"
mkdir -p "$out"
cp evaluate_go2.py action_delay.py replay_delay_failure.sh train_baseline.yaml "$out/"
cd "$out"
model=$(find /robotrl/experiments/phase2_go2_delay_training/run_001/logs -name model_299.pt -print -quit)
test -n "$model"
timeout 900 python evaluate_go2.py --model "$model" --output evaluation.json --trace_env 15 --scenario nominal --action_delay_steps 1 --forward_speed 0.5 --eval_seed 20000 --task Isaac-Velocity-Flat-UnitreeGo2 --num_envs 16 physics=newton_mjwarp env.scene.robot.spawn.usd_path=/robotrl/assets/Go2/go2.usd 2>&1 | tee evaluation.log
echo ROBOTRL_DELAY_FAILURE_REPLAY_PASSED | tee completed.txt
