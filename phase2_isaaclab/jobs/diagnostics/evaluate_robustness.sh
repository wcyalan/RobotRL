#!/usr/bin/env bash
set -euo pipefail
out=/robotrl/experiments/phase2_go2_robustness/run_001
test ! -e "$out/completed.txt"
mkdir -p "$out"
cp evaluate_go2.py evaluate_robustness.sh train_baseline.yaml "$out/"
cd "$out"
for pair in '42 /robotrl/experiments/phase2_go2_baseline/run_001' '0 /robotrl/experiments/phase2_go2_baseline/run_002' '1 /robotrl/experiments/phase2_go2_baseline/run_003'; do
    read -r seed source <<< "$pair"
    model=$(find "$source/logs" -name model_299.pt -print -quit)
    test -n "$model"
    for scenario in nominal low_friction heavy_base; do
        result="seed_${seed}_${scenario}"
        test ! -e "$result.json"
        timeout 900 python evaluate_go2.py --model "$model" --output "$result.json" --scenario "$scenario" --eval_seed 20000 --forward_speed 0.5 --task Isaac-Velocity-Flat-UnitreeGo2 --num_envs 16 physics=newton_mjwarp env.scene.robot.spawn.usd_path=/robotrl/assets/Go2/go2.usd 2>&1 | tee "$result.log"
    done
done
echo ROBOTRL_ROBUSTNESS_PASSED | tee completed.txt
