"""Check official PPO checkpoints and finite TensorBoard losses after a run."""
import argparse
import hashlib
import json
import math
import sys
from pathlib import Path
import torch
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
parser = argparse.ArgumentParser()
parser.add_argument('root')
parser.add_argument('--iterations', type=int, default=20)
parser.add_argument('--num_envs', type=int, default=128)
opts = parser.parse_args()
root = Path(opts.root)
models = sorted(root.glob('logs/rsl_rl/g1_flat/*/model_*.pt'), key=lambda p: int(p.stem.split('_')[-1]))
assert models and models[-1].name == f'model_{opts.iterations - 1}.pt', 'missing final checkpoint'
assert len(models) >= 2, 'need two saved checkpoints to verify parameter updates'
first, last = models[0], models[-1]
def flatten(data, prefix=''):
    result = {}
    for k,v in data.items():
        name = prefix + str(k)
        if isinstance(v, torch.Tensor):
            assert torch.isfinite(v).all(), name
            result[name] = v
        elif isinstance(v, dict):
            result.update(flatten(v,name+'/'))
    return result
old = flatten(torch.load(first, map_location='cpu', weights_only=False))
new = flatten(torch.load(last, map_location='cpu', weights_only=False))
changed = [k for k in new if k in old and new[k].shape == old[k].shape and not torch.equal(new[k], old[k])]
actor_changed = [k for k in changed if 'actor' in k.lower() and 'optimizer' not in k.lower()]
critic_changed = [k for k in changed if 'critic' in k.lower() and 'optimizer' not in k.lower()]
assert actor_changed and critic_changed, f'No actor/critic update evidence. Changed keys: {changed[:20]}'
events = EventAccumulator(str(last.parent), size_guidance={'scalars': 0}).Reload()
scalars = {k: events.Scalars(k) for k in events.Tags()['scalars']}
losses = {k: v for k,v in scalars.items() if 'loss' in k.lower()}
assert losses, 'missing loss metrics'
assert all(math.isfinite(e.value) for values in scalars.values() for e in values)
assert all(len(values) >= opts.iterations for values in losses.values()), 'missing logged training iterations'
result = dict(passed=True, first_checkpoint=str(first), final_checkpoint=str(last),
              final_checkpoint_sha256=hashlib.sha256(last.read_bytes()).hexdigest(),
              actor_tensors_changed=len(actor_changed), critic_tensors_changed=len(critic_changed),
              loss_metrics={k:{'count':len(v),'first':v[0].value,'last':v[-1].value} for k,v in losses.items()},
              all_logged_scalars_finite=True, training_envs=opts.num_envs, iterations=opts.iterations, rollout_steps=24, environment_steps=opts.num_envs*opts.iterations*24)
(root/'training_check.json').write_text(json.dumps(result,indent=2)+'\n')
(root/'final_checkpoint.txt').write_text(str(last.resolve())+'\n')
print('TRAINING_CHECK_PASSED',json.dumps(result),flush=True)
