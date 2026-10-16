#!/usr/bin/env bash
set -euo pipefail
out=/robotrl/experiments/phase2_go2_robustness/run_003
mkdir -p "$out"
cp evaluate_go2.py friction_audit.py evaluate_low_friction.sh train_baseline.yaml "$out/"
cd "$out"
for pair in '0 /robotrl/experiments/phase2_go2_baseline/run_002' '1 /robotrl/experiments/phase2_go2_baseline/run_003' '42 /robotrl/experiments/phase2_go2_baseline/run_001'; do
    read -r seed source <<< "$pair"
    model=$(find "$source/logs" -name model_299.pt -print -quit)
    test -n "$model"
    test ! -e "seed_${seed}.json"
    timeout 900 python evaluate_go2.py --model "$model" --output "seed_${seed}.json" --scenario low_friction_both --audit_friction --eval_seed 20000 --forward_speed 0.5 --task Isaac-Velocity-Flat-UnitreeGo2 --num_envs 16 physics=newton_mjwarp env.scene.robot.spawn.usd_path=/robotrl/assets/Go2/go2.usd 2>&1 | tee "seed_${seed}.log"
done
echo ROBOTRL_LOW_FRICTION_PASSED | tee completed.txt
