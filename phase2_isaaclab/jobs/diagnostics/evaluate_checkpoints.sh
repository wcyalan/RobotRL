#!/usr/bin/env bash
set -euo pipefail
source_dir="$PWD"
family=/robotrl/experiments/phase2_go2_stop_sequence_extended
mode=${1:-screen}
if [ "$mode" = validate ]; then
    root="$family/validation_001"
    test ! -e "$root"
    mkdir -p "$root"
    cp "$source_dir/"*.py "$source_dir/"*.sh "$source_dir/train_baseline.yaml" "$source_dir/context_manifest.json" "$source_dir/frozen_models.json" "$root/"
    cd "$root"
    python - <<'PY'
import hashlib, json
from pathlib import Path
for entry in json.loads(Path('frozen_models.json').read_text())['models']:
    assert hashlib.sha256(Path(entry['path']).read_bytes()).hexdigest() == entry['sha256']
print('FROZEN_MODELS_VERIFIED', flush=True)
PY
    for eval_seed in 45000 46000; do
        for run in run_001 run_002; do
            out="$root/eval_$eval_seed/$run"
            mkdir -p "$out"
            for group in coverage baseline; do
                if [ "$group" = coverage ]; then
                    model=$(find "$family/$run/logs" -name model_450.pt -print -quit)
                else
                    model=$(find "/robotrl/experiments/phase2_go2_stop_sequence/$run/logs" -name model_299.pt -print -quit)
                fi
                test -n "$model"
                for behavior in turn_left turn_right stop_restart forward; do
                    args=(--behavior "$behavior")
                    if [ "$behavior" = forward ]; then args=(--dr_condition nominal --forward_speed .5); fi
                    if [ "$group" = coverage ] && [ "$behavior" = stop_restart ]; then args+=(--trace_all); fi
                    (
                        cd "$out"
                        timeout 900 python "$root/evaluate_go2.py" --model "$model" --output "${group}_${behavior}.json" "${args[@]}" --audit_friction --eval_seed "$eval_seed" --num_envs 32 --task Isaac-Velocity-Flat-UnitreeGo2 physics=newton_mjwarp env.scene.robot.spawn.usd_path=/robotrl/assets/Go2/go2.usd 2>&1 | tee "${group}_${behavior}.log"
                    )
                done
            done
        done
    done
    echo FROZEN_CHECKPOINT_VALIDATION_COMPLETED > "$root/completed.txt"
    exit 0
elif [ "$mode" != screen ]; then
    exit 2
fi
root="$family/checkpoints_001"
test ! -e "$root"
mkdir -p "$root"
cp "$source_dir/"*.py "$source_dir/"*.sh "$source_dir/train_baseline.yaml" "$source_dir/context_manifest.json" "$root/"
# Screening only. Independent evaluation follows model selection.
for run in run_001 run_002; do
    for iteration in 350 400 450 500 550; do
        model=$(find "$family/$run/logs" -name "model_${iteration}.pt" -print -quit)
        test -n "$model"
        out="$root/$run/model_$iteration"
        mkdir -p "$out"
        for behavior in turn_left turn_right stop_restart; do
            (
                cd "$out"
                timeout 900 python "$root/evaluate_go2.py" --model "$model" --output "${behavior}.json" --behavior "$behavior" --audit_friction --eval_seed 43000 --num_envs 32 --task Isaac-Velocity-Flat-UnitreeGo2 physics=newton_mjwarp env.scene.robot.spawn.usd_path=/robotrl/assets/Go2/go2.usd 2>&1 | tee "${behavior}.log"
            )
        done
    done
done
echo CHECKPOINT_SCREEN_COMPLETED > "$root/completed.txt"
