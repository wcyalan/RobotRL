#!/usr/bin/env bash
set -euo pipefail
source_dir="$PWD"
profile=${1:-turn}
family=phase2_go2_turn_coverage
eval_seeds='41000 42000'
if [ "$profile" = stop_sequence ]; then
    family=phase2_go2_stop_sequence
    eval_seeds='43000 44000'
elif [ "$profile" != turn ]; then exit 2; fi
root=/robotrl/experiments/$family/validation_001
test ! -e "$root"
mkdir -p "$root"
cp "$source_dir/"*.py "$source_dir/"*.sh "$source_dir/train_baseline.yaml" "$source_dir/context_manifest.json" "$root/"
for eval_seed in $eval_seeds; do
    for pair in '0 run_001 run_002' '1 run_002 run_003'; do
        read -r seed run baseline <<< "$pair"
        out="$root/eval_${eval_seed}/seed_${seed}"
        mkdir -p "$out"
        for group in coverage baseline; do
            if [ "$group" = coverage ]; then
                model=$(find "/robotrl/experiments/$family/$run/logs" -name model_299.pt -print -quit)
            else
                if [ "$profile" = stop_sequence ]; then
                    model=$(find "/robotrl/experiments/phase2_go2_turn_coverage/$run/logs" -name model_299.pt -print -quit)
                else
                    model=$(find "/robotrl/experiments/phase2_go2_baseline/$baseline/logs" -name model_299.pt -print -quit)
                fi
            fi
            test -n "$model"
            for behavior in turn_left turn_right stop_restart; do
                trace=()
                if [ "$group" = coverage ] && [ "$behavior" = stop_restart ]; then trace=(--trace_all); fi
                (
                    cd "$out"
                    timeout 900 python "$root/evaluate_go2.py" --model "$model" --output "${group}_${behavior}.json" --behavior "$behavior" --audit_friction "${trace[@]}" --eval_seed "$eval_seed" --num_envs 32 --task Isaac-Velocity-Flat-UnitreeGo2 physics=newton_mjwarp env.scene.robot.spawn.usd_path=/robotrl/assets/Go2/go2.usd 2>&1 | tee "${group}_${behavior}.log"
                )
            done
            if [ "$profile" = stop_sequence ]; then
                (
                    cd "$out"
                    timeout 900 python "$root/evaluate_go2.py" --model "$model" --output "${group}_forward.json" --dr_condition nominal --forward_speed .5 --audit_friction --eval_seed "$eval_seed" --num_envs 32 --task Isaac-Velocity-Flat-UnitreeGo2 physics=newton_mjwarp env.scene.robot.spawn.usd_path=/robotrl/assets/Go2/go2.usd 2>&1 | tee "${group}_forward.log"
                )
            fi
        done
    done
done
echo TURN_COVERAGE_VALIDATION_COMPLETED | tee "$root/completed.txt"
