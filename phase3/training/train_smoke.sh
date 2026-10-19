#!/usr/bin/env bash
set -euo pipefail
out="$1"
source_dir="$PWD"
test ! -e "$out/training.log"
mkdir -p "$out/source"
cp inspect_g1.py evaluate_g1.py verify_training.py train_smoke.sh environment.yaml "$out/source/"
cp /workspace/isaaclab/source/isaaclab_rl/isaaclab_rl/entrypoints/backends/train_rsl_rl.py "$out/source/official_train_rsl_rl.py"
python - <<'PY'
import hashlib,json
from pathlib import Path
p=Path('/robotrl/experiments/phase3_g1_environment/run_001/smoke')
s=json.loads((p/'smoke.json').read_text())
assert s['passed'] and s['automatic_reset_observed'] and s['steps']==1100 and s['num_envs']==4
assert hashlib.sha256(Path('/robotrl/assets/G1/g1_minimal.usd').read_bytes()).hexdigest()=='9dfe7a710aa791e49abf2d9ea74ad3163e291f02f21f59bda9bfcc40f3fab428'
PY
cd "$out"
python "$source_dir/inspect_g1.py" --output "$out/configuration" --task Isaac-Velocity-Flat-G1 physics=newton_mjwarp 2>&1 | tee configuration.log
python /workspace/isaaclab/scripts/reinforcement_learning/train.py --rl_library rsl_rl --task Isaac-Velocity-Flat-G1 --num_envs 128 --max_iterations 20 --seed 0 physics=newton_mjwarp env.scene.robot.spawn.usd_path=/robotrl/assets/G1/g1_minimal.usd 2>&1 | tee training.log
python "$source_dir/verify_training.py" "$out" 2>&1 | tee training_check.log
model=$(cat final_checkpoint.txt)
python "$source_dir/evaluate_g1.py" --model "$model" --output "$out/reload" --eval_seed 10000 --task Isaac-Velocity-Flat-G1 --num_envs 4 physics=newton_mjwarp env.scene.robot.spawn.usd_path=/robotrl/assets/G1/g1_minimal.usd 2>&1 | tee reload.log
