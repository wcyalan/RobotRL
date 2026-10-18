#!/usr/bin/env bash
set -euo pipefail
source_dir="$PWD"
root=/robotrl/experiments/phase2_go2_deployment/run_001
test ! -e "$root/started.txt"
echo EXPORT_VALIDATION_STARTED > "$root/started.txt"
mkdir -p "$root/source"
cp "$source_dir/"*.py "$source_dir/"*.sh "$source_dir/train_baseline.yaml" "$source_dir/context_manifest.json" "$root/source/"
python - <<'PY'
import hashlib,json
from pathlib import Path
root=Path('/robotrl/experiments/phase2_go2_deployment/run_001')
for run in ['run_001','run_002']:
    m=json.loads((root/run/'manifest.json').read_text())
    model=next((Path('/robotrl/experiments/phase2_go2_stop_sequence_extended')/run/'logs').glob('**/model_450.pt'))
    assert hashlib.sha256(model.read_bytes()).hexdigest()==m['checkpoint_sha256']
    assert hashlib.sha256((root/run/'policy.pt').read_bytes()).hexdigest()==m['policy_sha256']
print('EXPORT_HASHES_VERIFIED',flush=True)
PY
for run in run_001 run_002; do
    model=$(find "/robotrl/experiments/phase2_go2_stop_sequence_extended/$run/logs" -name model_450.pt -print -quit)
    (
        cd "$root/$run"
        timeout 900 python "$root/source/evaluate_go2.py" --model "$model" --jit_policy "$root/$run/policy.pt" --verify_jit --output equivalence.json --behavior stop_restart --eval_seed 45000 --num_envs 32 --task Isaac-Velocity-Flat-UnitreeGo2 physics=newton_mjwarp env.scene.robot.spawn.usd_path=/robotrl/assets/Go2/go2.usd 2>&1 | tee equivalence.log
    )
done
# Both official comparisons must pass before independent exported-policy evaluation.
for run in run_001 run_002; do
    model=$(find "/robotrl/experiments/phase2_go2_stop_sequence_extended/$run/logs" -name model_450.pt -print -quit)
    for eval_seed in 45000 46000; do
        out="$root/$run/eval_$eval_seed"
        mkdir -p "$out"
        for behavior in turn_left turn_right stop_restart forward; do
            args=(--behavior "$behavior")
            if [ "$behavior" = forward ]; then args=(--dr_condition nominal --forward_speed .5); fi
            (
                cd "$out"
                timeout 900 python "$root/source/evaluate_go2.py" --model "$model" --jit_policy "$root/$run/policy.pt" --output "$behavior.json" "${args[@]}" --audit_friction --eval_seed "$eval_seed" --num_envs 32 --task Isaac-Velocity-Flat-UnitreeGo2 physics=newton_mjwarp env.scene.robot.spawn.usd_path=/robotrl/assets/Go2/go2.usd 2>&1 | tee "$behavior.log"
            )
        done
    done
done
echo EXPORT_VALIDATION_COMPLETED > "$root/completed.txt"
