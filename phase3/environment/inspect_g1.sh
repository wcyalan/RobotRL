#!/usr/bin/env bash
set -euo pipefail
out="$1"
shift
mkdir -p "$out"
cp inspect_g1.py inspect_g1.sh environment.yaml "$out/"
python inspect_g1.py --output "$out" --task Isaac-Velocity-Flat-G1 physics=newton_mjwarp "$@" 2>&1 | tee "$out/inspection.log"
