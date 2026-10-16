#!/usr/bin/env bash
set -euo pipefail
source_dir="$PWD"
out=/robotrl/experiments/phase2_go2_delay_training/run_001
test ! -e "$out/training.log"
mkdir -p "$out/smoke"
cp train_delay_randomized.py verify_delay_training.py random_action_delay.py action_delay.py evaluate_go2.py train_delay_randomized.sh train_baseline.yaml "$out/"
cd "$out/smoke"
timeout 900 python "$out/train_delay_randomized.py" --rl_library rsl_rl --task Isaac-Velocity-Flat-UnitreeGo2 --seed 42 --num_envs 64 --max_iterations 3 physics=newton_mjwarp env.scene.robot.spawn.usd_path=/robotrl/assets/Go2/go2.usd 2>&1 | tee training.log
test -f delay_training_audit.json
python "$out/verify_delay_training.py"
cd "$out"
timeout 7200 python train_delay_randomized.py --rl_library rsl_rl --task Isaac-Velocity-Flat-UnitreeGo2 --seed 42 physics=newton_mjwarp env.scene.robot.spawn.usd_path=/robotrl/assets/Go2/go2.usd 2>&1 | tee training.log
python verify_delay_training.py
model=$(find logs -name model_299.pt -print -quit)
test -n "$model"
for delay in 0 1 2; do
    timeout 900 python evaluate_go2.py --model "$model" --output "delay_${delay}.json" --action_delay_steps "$delay" --scenario nominal --eval_seed 20000 --forward_speed 0.5 --task Isaac-Velocity-Flat-UnitreeGo2 --num_envs 16 physics=newton_mjwarp env.scene.robot.spawn.usd_path=/robotrl/assets/Go2/go2.usd 2>&1 | tee "delay_${delay}.log"
done
echo ROBOTRL_DELAY_TRAINING_PASSED | tee completed.txt
