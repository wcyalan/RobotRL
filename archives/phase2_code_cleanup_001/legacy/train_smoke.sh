#!/usr/bin/env bash
set -euo pipefail
run_dir="$1"
mkdir -p "$run_dir"
cp validate_go2.py train_smoke.sh train_smoke.yaml "$run_dir/"
cd "$run_dir"
timeout 900 python validate_go2.py --task Isaac-Velocity-Flat-UnitreeGo2 --num_envs 4 physics=newton_mjwarp 2>&1 | tee validation.log
timeout 1200 python /workspace/isaaclab/scripts/reinforcement_learning/train.py --rl_library rsl_rl --task Isaac-Velocity-Flat-UnitreeGo2 --num_envs 64 --max_iterations 10 --seed 0 physics=newton_mjwarp 2>&1 | tee training.log
python - <<'PY'
import json
from pathlib import Path
import torch
models = sorted(Path('logs').rglob('model_*.pt'))
assert models, 'No checkpoints saved'
summary = []
for path in models:
    # Only read checkpoints produced by this task.
    checkpoint = torch.load(path, map_location='cpu', weights_only=False)
    tensors = []
    def visit(x):
        if isinstance(x, torch.Tensor):
            assert torch.isfinite(x).all().item(), f'Nonfinite checkpoint: {path}'
            tensors.append(x)
        elif isinstance(x, dict):
            for y in x.values(): visit(y)
        elif isinstance(x, (list, tuple)):
            for y in x: visit(y)
    visit(checkpoint)
    assert tensors
    summary.append({'path': str(path), 'bytes': path.stat().st_size, 'finite_tensors': len(tensors)})
Path('training_summary.json').write_text(json.dumps({'passed': True, 'num_envs': 64, 'iterations_requested': 10, 'checkpoints': summary}, indent=2))
print('ROBOTRL_TRAIN_SMOKE_PASSED', flush=True)
PY
