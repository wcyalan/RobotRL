#!/usr/bin/env bash
set -euo pipefail
source_dir="$PWD"
root=/robotrl/experiments/phase2_go2_mass_friction
test ! -e "$root/run_001"
test ! -e "$root/run_002"
test ! -e "$root/run_003"
test ! -e "$root/run_004"
# Validate the newly added fixed impulse using an existing smoke checkpoint.
mkdir "$root/evaluation_check_001"
(
    cd "$root/evaluation_check_001"
    cp "$source_dir/"*.py "$source_dir/train_mass_friction.sh" "$source_dir/train_baseline.yaml" .
    model=$(find "$root/smoke_001/fixed/logs" -name model_2.pt -print -quit)
    test -n "$model"
    timeout 900 python evaluate_go2.py --model "$model" --output evaluation.json --dr_condition nominal_push --audit_friction --eval_seed 30000 --forward_speed 0.5 --task Isaac-Velocity-Flat-UnitreeGo2 --num_envs 32 physics=newton_mjwarp env.scene.robot.spawn.usd_path=/robotrl/assets/Go2/go2.usd 2>&1 | tee evaluation.log
    python -c 'import json; d=json.load(open("evaluation.json")); assert d["fixed_push"] and d["fixed_push"]["env_ids"]; assert len(d["episodes"])==32; assert all(abs(v-1)<1e-5 for a in d["friction_audit"] for v in a["active_contacts"]["ground_sliding_friction"]); print("FIXED_PUSH_CHECK_PASSED")'
)
number=0
for group in fixed randomized; do
    for seed in 0 1; do
        number=$((number+1))
        name=$(printf 'run_%03d' "$number")
        out="$root/$name"
        mkdir "$out"
        (
            cd "$out"
            cp "$source_dir/"*.py "$source_dir/train_mass_friction.sh" "$source_dir/train_baseline.yaml" .
            printf '{"group":"%s","seed":%s,"num_envs":4096,"iterations":300,"steps":29491200}\n' "$group" "$seed" > protocol.json
            timeout 7200 python train_mass_friction.py --dr_group "$group" --rl_library rsl_rl --task Isaac-Velocity-Flat-UnitreeGo2 --seed "$seed" physics=newton_mjwarp env.scene.robot.spawn.usd_path=/robotrl/assets/Go2/go2.usd 2>&1 | tee training.log
            test -f mass_friction_audit.json
            model=$(find logs -name model_299.pt -print -quit)
            test -n "$model"
            for condition in nominal light_low heavy_high heavy_low nominal_push; do
                timeout 900 python evaluate_go2.py --model "$model" --output "$condition.json" --dr_condition "$condition" --audit_friction --eval_seed 30000 --forward_speed 0.5 --task Isaac-Velocity-Flat-UnitreeGo2 --num_envs 32 physics=newton_mjwarp env.scene.robot.spawn.usd_path=/robotrl/assets/Go2/go2.usd 2>&1 | tee "$condition.log"
            done
            echo MASS_FRICTION_RUN_PASSED | tee completed.txt
        )
    done
done
echo MASS_FRICTION_FOUR_RUNS_PASSED | tee "$root/completed.txt"
