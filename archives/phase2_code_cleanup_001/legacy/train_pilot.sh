#!/usr/bin/env bash
set -euo pipefail
run_dir="$1"
mkdir -p "$run_dir"
cp evaluate_go2.py train_pilot.sh train_pilot.yaml "$run_dir/"
cd "$run_dir"
# Check evaluation integration on the prior smoke checkpoint before spending the pilot budget.
smoke_model=$(find /robotrl/run_002/logs -name model_9.pt -print -quit)
test -n "$smoke_model"
timeout 900 python evaluate_go2.py --model "$smoke_model" --output evaluation_smoke.json --task Isaac-Velocity-Flat-UnitreeGo2 --num_envs 16 physics=newton_mjwarp 2>&1 | tee evaluation_smoke.log
timeout 1800 python /workspace/isaaclab/scripts/reinforcement_learning/train.py --rl_library rsl_rl --task Isaac-Velocity-Flat-UnitreeGo2 --num_envs 256 --max_iterations 200 --seed 0 physics=newton_mjwarp 2>&1 | tee training.log
for checkpoint in 0 199; do
    model=$(find logs -name "model_${checkpoint}.pt" -print -quit)
    test -n "$model"
    timeout 900 python evaluate_go2.py --model "$model" --output "evaluation_${checkpoint}.json" --task Isaac-Velocity-Flat-UnitreeGo2 --num_envs 16 physics=newton_mjwarp 2>&1 | tee "evaluation_${checkpoint}.log"
done
echo ROBOTRL_PILOT_PASSED
