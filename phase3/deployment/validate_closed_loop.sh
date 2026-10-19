#!/usr/bin/env bash
set -euo pipefail
context="$PWD"
root=/robotrl/experiments/phase3_g1_deployment
test -f "$root/completed.txt"
test ! -e "$root/closed_loop_started.txt"
mkdir -p "$root/closed_loop_source"
cp "$context/"*.py "$context/"*.sh "$context/"*.json "$context/environment.yaml" "$root/closed_loop_source/"
python - "$root" <<'PY'
import hashlib,json,sys
from pathlib import Path
root=Path(sys.argv[1])
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
for name in ('run_001','run_002'):
    m=json.loads((root/name/'export/manifest.json').read_text())
    v=json.loads((root/name/'equivalence/summary.json').read_text())
    assert v['passed'] and v['policy_sha256']==sha(root/name/'export/policy.pt')==m['policy_sha256']
assert sha(root/'closed_loop_source/protocol.json')=='894cad975e0ca4bf06fe1c605e7fe0dc21b46caddfd2a5d786ddafa2fdcf17dc'
(root/'closed_loop_source_manifest.json').write_text(json.dumps({p.name:sha(p) for p in (root/'closed_loop_source').iterdir()},indent=2)+'\n')
PY
echo started > "$root/closed_loop_started.txt"
for run in run_001 run_002; do
    for seed in 20000 21000; do
        for condition in stand forward_low forward_medium turn_left turn_right stop_restart; do
            out="$root/$run/closed_loop/eval_$seed/$condition"
            mkdir -p "$(dirname "$out")"
            (
                cd "$root/$run"
                python "$root/closed_loop_source/evaluate_export.py" --export "$root/$run/export" --protocol "$root/closed_loop_source/protocol.json" --condition "$condition" --output "$out" --eval_seed "$seed" --task Isaac-Velocity-Flat-G1 physics=newton_mjwarp env.scene.robot.spawn.usd_path=/robotrl/assets/G1/g1_minimal.usd > "${out}.log" 2>&1
            )
            echo "CLOSED_LOOP_GROUP_COMPLETE $run $seed $condition"
        done
    done
done
echo complete > "$root/closed_loop_completed.txt"
