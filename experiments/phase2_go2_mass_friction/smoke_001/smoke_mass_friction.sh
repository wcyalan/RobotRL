#!/usr/bin/env bash
set -euo pipefail
out=/robotrl/phase2_go2_mass_friction/smoke_001
test ! -e "$out"
mkdir -p "$out"
cp train_mass_friction.py friction_audit.py smoke_mass_friction.sh train_baseline.yaml "$out/"
for group in fixed randomized; do
    mkdir "$out/$group"
    (
        cd "$out/$group"
        timeout 900 python "$out/train_mass_friction.py" --dr_group "$group" --rl_library rsl_rl --task Isaac-Velocity-Flat-UnitreeGo2 --seed 0 --num_envs 64 --max_iterations 3 physics=newton_mjwarp env.scene.robot.spawn.usd_path=/robotrl/assets/Go2/go2.usd 2>&1 | tee training.log
        test -f mass_friction_audit.json
    )
done
echo MASS_FRICTION_SMOKE_PASSED | tee "$out/completed.txt"
