#!/usr/bin/env bash
set -euo pipefail
export OMP_NUM_THREADS=2 MKL_NUM_THREADS=2
cd /workspace/isaaclab
echo 'ROBOTRL_KITLESS_PROBE_START'
cat VERSION
python -c 'import os, importlib.metadata as m; print("driver_capabilities:", os.getenv("NVIDIA_DRIVER_CAPABILITIES")); print({n: m.version(n) for n in ["torch", "warp-lang", "newton"]})'
python -c 'import torch, importlib.metadata as m; assert torch.cuda.is_available(); print(torch.cuda.get_device_name()); names = {d.metadata["Name"].lower() for d in m.distributions()}; assert "isaacsim" not in names, "Full Isaac Sim distribution unexpectedly installed"'
timeout 900 python scripts/environments/zero_agent.py --task Isaac-Velocity-Flat-UnitreeGo2 --num_envs 4 --max_steps 200 physics=newton_mjwarp
echo 'ROBOTRL_KITLESS_PROBE_PASSED'
