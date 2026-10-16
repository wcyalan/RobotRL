#!/usr/bin/env bash
set -euo pipefail
out=/robotrl/phase2_go2_robustness/run_002
mkdir -p "$out"
cp evaluate_go2.py friction_audit.py audit_friction.sh train_baseline.yaml "$out/"
cd "$out"
model=$(find /robotrl/run_010/logs -name model_299.pt -print -quit)
for scenario in nominal low_friction low_friction_both; do
    test ! -e "$scenario.json"
    timeout 900 python evaluate_go2.py --model "$model" --output "$scenario.json" --scenario "$scenario" --audit_friction --eval_seed 20000 --forward_speed 0.5 --task Isaac-Velocity-Flat-UnitreeGo2 --num_envs 16 physics=newton_mjwarp env.scene.robot.spawn.usd_path=/robotrl/assets/Go2/go2.usd 2>&1 | tee "$scenario.log"
done
echo ROBOTRL_FRICTION_AUDIT_PASSED | tee completed.txt
