#!/usr/bin/env bash
set -euo pipefail
source_dir="$PWD"
root=/robotrl/experiments/phase2_go2_stop_sequence_extended
parent=/robotrl/experiments/phase2_go2_stop_sequence
mkdir -p "$root"
for phase in smoke run_001 run_002; do
    seed=0
    run=run_001
    iterations=300
    envs=4096
    if [ "$phase" = smoke ]; then iterations=24; envs=256; fi
    if [ "$phase" = run_002 ]; then seed=1; run=run_002; fi
    out="$root/$phase"
    test ! -e "$out"
    model=$(find "$parent/$run/logs" -name model_299.pt -print -quit)
    test -n "$model"
    mkdir -p "$out"
    (
        cd "$out"
        cp "$source_dir/"*.py "$source_dir/"*.sh "$source_dir/train_baseline.yaml" "$source_dir/context_manifest.json" .
        timeout 7200 python resume_command_coverage.py --checkpoint "$model" --coverage_variant stop_sequence --rl_library rsl_rl --task Isaac-Velocity-Flat-UnitreeGo2 --seed "$seed" --num_envs "$envs" --max_iterations "$iterations" physics=newton_mjwarp env.scene.robot.spawn.usd_path=/robotrl/assets/Go2/go2.usd 2>&1 | tee training.log
        test -f command_audit.json
        test -f model_extended.pt
        if [ "$phase" = smoke ]; then
            timeout 900 python evaluate_go2.py --model model_extended.pt --output smoke_evaluation.json --behavior stop_restart --eval_seed 40000 --num_envs 32 --task Isaac-Velocity-Flat-UnitreeGo2 physics=newton_mjwarp env.scene.robot.spawn.usd_path=/robotrl/assets/Go2/go2.usd 2>&1 | tee evaluation.log
        else
            for eval_seed in 43000 44000; do
                mkdir -p "eval_$eval_seed"
                for group in coverage baseline; do
                    checkpoint="$model"
                    if [ "$group" = coverage ]; then checkpoint="$out/model_extended.pt"; fi
                    for behavior in turn_left turn_right stop_restart forward; do
                        args=(--behavior "$behavior")
                        if [ "$behavior" = forward ]; then args=(--dr_condition nominal --forward_speed .5); fi
                        if [ "$behavior" = stop_restart ] && [ "$group" = coverage ]; then args+=(--trace_all); fi
                        timeout 900 python evaluate_go2.py --model "$checkpoint" --output "eval_$eval_seed/${group}_${behavior}.json" "${args[@]}" --audit_friction --eval_seed "$eval_seed" --num_envs 32 --task Isaac-Velocity-Flat-UnitreeGo2 physics=newton_mjwarp env.scene.robot.spawn.usd_path=/robotrl/assets/Go2/go2.usd 2>&1 | tee "eval_$eval_seed/${group}_${behavior}.log"
                    done
                done
            done
        fi
        echo RESUME_COMPLETED > completed.txt
    )
done
echo EXTENDED_EXPERIMENT_COMPLETED > "$root/completed.txt"
