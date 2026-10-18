#!/usr/bin/env bash
set -euo pipefail
out=/robotrl/experiments/phase2_go2_behavior/diagnostic_002
test ! -e "$out"
mkdir -p "$out"
cp evaluate_go2.py behavior_protocol.py friction_audit.py replay_stop_failure.sh train_baseline.yaml context_manifest.json "$out/"
cd "$out"
model=$(find /robotrl/experiments/phase2_go2_baseline/run_003/logs -name model_299.pt -print -quit)
test -n "$model"
timeout 900 python evaluate_go2.py --model "$model" --output seed_1_stop_restart.json --behavior stop_restart --audit_friction --trace_all --eval_seed 40000 --task Isaac-Velocity-Flat-UnitreeGo2 --num_envs 32 physics=newton_mjwarp env.scene.robot.spawn.usd_path=/robotrl/assets/Go2/go2.usd 2>&1 | tee seed_1_stop_restart.log
echo GO2_STOP_FAILURE_REPLAY_COMPLETED | tee completed.txt
