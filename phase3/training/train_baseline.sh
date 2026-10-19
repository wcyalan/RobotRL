#!/usr/bin/env bash
set -euo pipefail
out="$1"
mode="$2"
source_dir="$PWD"
asset=/robotrl/assets/G1/g1_minimal.usd
if [ "$mode" = evaluator ]; then
    mkdir -p "$out"
    model=$(cat /robotrl/experiments/phase3_g1_smoke/run_001/final_checkpoint.txt)
    python evaluate_behaviors.py --model "$model" --protocol protocol.json --condition stop_restart --output "$out" --eval_seed 19000 --smoke --task Isaac-Velocity-Flat-G1 physics=newton_mjwarp env.scene.robot.spawn.usd_path="$asset" 2>&1 | tee "$out/evaluation.log"
    exit 0
fi
seed="$mode"
case "$seed" in 0|1) ;; *) exit 2 ;; esac
test ! -e "$out/training.log"
python - <<'PY'
import hashlib,json
from pathlib import Path
root=Path('/robotrl/experiments/phase3_g1_smoke/run_001')
assert json.loads((root/'training_check.json').read_text())['passed']
assert json.loads((root/'reload/summary.json').read_text())['passed']
e=json.loads((root/'behavior_smoke/summary.json').read_text())
assert e['completed'] and e['evaluator_smoke'] and len(e['command_switches'])==4
assert e['protocol_sha256']==hashlib.sha256(Path('protocol.json').read_bytes()).hexdigest()
assert hashlib.sha256(Path('/robotrl/assets/G1/g1_minimal.usd').read_bytes()).hexdigest()=='9dfe7a710aa791e49abf2d9ea74ad3163e291f02f21f59bda9bfcc40f3fab428'
PY
mkdir -p "$out/source"
cp inspect_g1.py evaluate_behaviors.py verify_training.py train_baseline.sh environment.yaml protocol.json "$out/source/"
cp /workspace/isaaclab/source/isaaclab_rl/isaaclab_rl/entrypoints/backends/train_rsl_rl.py "$out/source/official_train_rsl_rl.py"
cd "$out"
python "$source_dir/inspect_g1.py" --output "$out/configuration" --task Isaac-Velocity-Flat-G1 physics=newton_mjwarp 2>&1 | tee configuration.log
python /workspace/isaaclab/scripts/reinforcement_learning/train.py --rl_library rsl_rl --task Isaac-Velocity-Flat-G1 --seed "$seed" physics=newton_mjwarp env.scene.robot.spawn.usd_path="$asset" 2>&1 | tee training.log
python "$source_dir/verify_training.py" "$out" --iterations 1500 --num_envs 4096 2>&1 | tee training_check.log
model=$(cat final_checkpoint.txt)
for eval_seed in 20000 21000; do
    for condition in stand forward_low forward_medium turn_left turn_right stop_restart; do
        result="$out/evaluation/eval_$eval_seed/$condition"
        mkdir -p "$result"
        python "$source_dir/evaluate_behaviors.py" --model "$model" --protocol "$out/source/protocol.json" --condition "$condition" --output "$result" --eval_seed "$eval_seed" --task Isaac-Velocity-Flat-G1 physics=newton_mjwarp env.scene.robot.spawn.usd_path="$asset" 2>&1 | tee "$result/evaluation.log"
    done
done
