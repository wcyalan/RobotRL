#!/usr/bin/env bash
set -euo pipefail
source_dir="$PWD"
out=/robotrl/experiments/phase2_go2_mass_friction/diagnostic_001
test ! -e "$out"
mkdir -p "$out"
cp evaluate_go2.py friction_audit.py replay_dr_outlier.sh train_baseline.yaml "$out/"
for run in run_002 run_004; do
    mkdir "$out/$run"
    (
        cd "$out/$run"
        model=$(find "/robotrl/experiments/phase2_go2_mass_friction/$run/logs" -name model_299.pt -print -quit)
        test -n "$model"
        timeout 900 python "$out/evaluate_go2.py" --model "$model" --output heavy_high.json --dr_condition heavy_high --audit_friction --trace_env 25 --eval_seed 30000 --forward_speed 0.5 --task Isaac-Velocity-Flat-UnitreeGo2 --num_envs 32 physics=newton_mjwarp env.scene.robot.spawn.usd_path=/robotrl/assets/Go2/go2.usd 2>&1 | tee evaluation.log
    )
done
echo DR_OUTLIER_REPLAY_PASSED | tee "$out/completed.txt"
