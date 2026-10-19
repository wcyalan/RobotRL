#!/usr/bin/env bash
set -euo pipefail
context="$PWD"
root=/robotrl/experiments/phase3_g1_deployment
test ! -e "$root/started.txt"
mkdir -p "$root/source"
cp "$context/"*.py "$context/"*.sh "$context/environment.yaml" "$root/source/"
python - "$root/source" <<'PY'
import hashlib, json, sys
from pathlib import Path
root = Path(sys.argv[1])
(root.parent/'source_manifest.json').write_text(json.dumps({p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in root.iterdir()}, indent=2)+'\n')
PY
echo started > "$root/started.txt"
for run in run_001 run_002; do
    model=$(cat "/robotrl/experiments/phase3_g1_baseline/$run/final_checkpoint.txt")
    mkdir -p "$root/$run"
    python "$root/source/export_policy.py" --checkpoint "$model" --output "$root/$run/export" > "$root/$run/export.log" 2>&1
    (
        cd "$root/$run"
        python "$root/source/verify_export.py" --model "$model" --export "$root/$run/export" --output "$root/$run/equivalence" --task Isaac-Velocity-Flat-G1 physics=newton_mjwarp env.scene.robot.spawn.usd_path=/robotrl/assets/G1/g1_minimal.usd 2>&1 | tee equivalence.log
    )
done
echo complete > "$root/completed.txt"
