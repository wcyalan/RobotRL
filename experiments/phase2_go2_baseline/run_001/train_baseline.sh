#!/usr/bin/env bash
set -euo pipefail
run_dir="$1"
mkdir -p "$run_dir"
cp baseline_config.py evaluate_go2.py train_baseline.sh train_baseline.yaml "$run_dir/"
cd "$run_dir"
cp /robotrl/assets/Go2/SHA256SUMS asset_sha256.txt
cp -r /workspace/isaaclab/source/isaaclab_tasks/isaaclab_tasks/core/velocity/config/go2 official_go2_source
python baseline_config.py 2>&1 | tee official_defaults.log
timeout 7200 python /workspace/isaaclab/scripts/reinforcement_learning/train.py --rl_library rsl_rl --task Isaac-Velocity-Flat-UnitreeGo2 physics=newton_mjwarp env.scene.robot.spawn.usd_path=/robotrl/assets/Go2/go2.usd 2>&1 | tee training.log
last=$(python -c 'import json; print(json.load(open("official_defaults.json"))["max_iterations"]-1)')
for checkpoint in 0 "$last"; do
    model=$(find logs -name "model_${checkpoint}.pt" -print -quit)
    test -n "$model"
    timeout 900 python evaluate_go2.py --model "$model" --output "random_${checkpoint}.json" --task Isaac-Velocity-Flat-UnitreeGo2 --num_envs 16 physics=newton_mjwarp env.scene.robot.spawn.usd_path=/robotrl/assets/Go2/go2.usd 2>&1 | tee "random_${checkpoint}.log"
    timeout 900 python evaluate_go2.py --model "$model" --output "forward_${checkpoint}.json" --forward_speed 0.5 --task Isaac-Velocity-Flat-UnitreeGo2 --num_envs 16 physics=newton_mjwarp env.scene.robot.spawn.usd_path=/robotrl/assets/Go2/go2.usd 2>&1 | tee "forward_${checkpoint}.log"
done
echo ROBOTRL_OFFICIAL_BASELINE_PASSED
