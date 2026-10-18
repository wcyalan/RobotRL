#!/usr/bin/env bash
set -euo pipefail
out=/robotrl/experiments/phase2_go2_behavior/run_001
test ! -e "$out"
mkdir -p "$out"
cp evaluate_go2.py behavior_protocol.py friction_audit.py evaluate_behaviors.sh train_baseline.yaml "$out/"
cd "$out"
for pair in '42 run_001' '0 run_002' '1 run_003'; do
    read -r seed run <<< "$pair"
    model=$(find "/robotrl/experiments/phase2_go2_baseline/$run/logs" -name model_299.pt -print -quit)
    test -n "$model"
    for behavior in turn_left turn_right stop_restart; do
        timeout 900 python evaluate_go2.py --model "$model" --output "seed_${seed}_${behavior}.json" --behavior "$behavior" --audit_friction --eval_seed 40000 --task Isaac-Velocity-Flat-UnitreeGo2 --num_envs 32 physics=newton_mjwarp env.scene.robot.spawn.usd_path=/robotrl/assets/Go2/go2.usd 2>&1 | tee "seed_${seed}_${behavior}.log"
    done
done
echo GO2_BEHAVIOR_EVALUATION_PASSED | tee completed.txt
