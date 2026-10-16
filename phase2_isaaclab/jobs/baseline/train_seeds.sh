#!/usr/bin/env bash
set -euo pipefail
# One allocation, two sequential runs; identical evaluation protocol to seed 42.
bash train_baseline.sh /robotrl/experiments/phase2_go2_baseline/run_002 0
bash train_baseline.sh /robotrl/experiments/phase2_go2_baseline/run_003 1
echo ROBOTRL_TWO_SEEDS_PASSED
