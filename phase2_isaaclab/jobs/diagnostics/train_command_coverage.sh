#!/usr/bin/env bash
set -euo pipefail
mode=${1:?Specify smoke or full}
variant=${2:-mixed}
source_dir="$PWD"
root=/robotrl/experiments/phase2_go2_command_coverage
if [ "$variant" = turn_only ]; then
    root=/robotrl/experiments/phase2_go2_turn_coverage
elif [ "$variant" = turn_stop ]; then
    root=/robotrl/experiments/phase2_go2_stop_stability
elif [ "$variant" = stop_sequence ]; then
    root=/robotrl/experiments/phase2_go2_stop_sequence
elif [ "$variant" != mixed ]; then
    exit 2
fi
if [ "$mode" = smoke ]; then
    out="$root/smoke_001"
    test ! -e "$out"
    mkdir -p "$out"
    cp "$source_dir/"*.py "$source_dir/"*.sh "$source_dir/train_baseline.yaml" "$source_dir/context_manifest.json" "$out/"
    cd "$out"
    timeout 1200 python train_command_coverage.py --coverage_variant "$variant" --rl_library rsl_rl --task Isaac-Velocity-Flat-UnitreeGo2 --seed 0 --num_envs 256 --max_iterations 24 physics=newton_mjwarp env.scene.robot.spawn.usd_path=/robotrl/assets/Go2/go2.usd 2>&1 | tee training.log
    test -f command_audit.json
    model=$(find logs -name model_23.pt -print -quit)
    test -n "$model"
    timeout 900 python evaluate_go2.py --model "$model" --output smoke_evaluation.json --behavior stop_restart --eval_seed 40000 --num_envs 32 --task Isaac-Velocity-Flat-UnitreeGo2 physics=newton_mjwarp env.scene.robot.spawn.usd_path=/robotrl/assets/Go2/go2.usd 2>&1 | tee evaluation.log
    echo COMMAND_COVERAGE_SMOKE_COMPLETED | tee completed.txt
elif [ "$mode" = full ]; then
    test -f "$root/smoke_001/completed.txt"
    for pair in '0 run_001 run_002' '1 run_002 run_003'; do
        read -r seed run baseline <<< "$pair"
        out="$root/$run"
        test ! -e "$out"
        mkdir -p "$out"
        (
            cd "$out"
            cp "$source_dir/"*.py "$source_dir/"*.sh "$source_dir/train_baseline.yaml" "$source_dir/context_manifest.json" .
            timeout 7200 python train_command_coverage.py --coverage_variant "$variant" --rl_library rsl_rl --task Isaac-Velocity-Flat-UnitreeGo2 --seed "$seed" physics=newton_mjwarp env.scene.robot.spawn.usd_path=/robotrl/assets/Go2/go2.usd 2>&1 | tee training.log
            test -f command_audit.json
            for group in coverage baseline; do
                if [ "$group" = coverage ]; then
                    model=$(find "$out/logs" -name model_299.pt -print -quit)
                else
                    if [ "$variant" = turn_stop ] || [ "$variant" = stop_sequence ]; then
                        model=$(find "/robotrl/experiments/phase2_go2_turn_coverage/$run/logs" -name model_299.pt -print -quit)
                    else
                        model=$(find "/robotrl/experiments/phase2_go2_baseline/$baseline/logs" -name model_299.pt -print -quit)
                    fi
                fi
                test -n "$model"
                for behavior in turn_left turn_right stop_restart; do
                    timeout 900 python evaluate_go2.py --model "$model" --output "${group}_${behavior}.json" --behavior "$behavior" --audit_friction --eval_seed 40000 --num_envs 32 --task Isaac-Velocity-Flat-UnitreeGo2 physics=newton_mjwarp env.scene.robot.spawn.usd_path=/robotrl/assets/Go2/go2.usd 2>&1 | tee "${group}_${behavior}.log"
                done
                timeout 900 python evaluate_go2.py --model "$model" --output "${group}_forward.json" --dr_condition nominal --forward_speed .5 --audit_friction --eval_seed 40000 --num_envs 32 --task Isaac-Velocity-Flat-UnitreeGo2 physics=newton_mjwarp env.scene.robot.spawn.usd_path=/robotrl/assets/Go2/go2.usd 2>&1 | tee "${group}_forward.log"
            done
            echo COMMAND_COVERAGE_RUN_COMPLETED | tee completed.txt
        )
    done
else
    echo "Unknown mode: $mode" >&2
    exit 2
fi
