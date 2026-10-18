"""Resume official PPO with explicit optimizer-state verification and lineage."""
import hashlib
import inspect
import json
from pathlib import Path
import runpy
import sys

import torch
import yaml
from rsl_rl.runners import OnPolicyRunner

checkpoint = Path(sys.argv[sys.argv.index('--checkpoint') + 1]).resolve()
smoke_run = '--num_envs' in sys.argv and sys.argv[sys.argv.index('--num_envs') + 1] == '256'
saved = torch.load(checkpoint, map_location='cpu', weights_only=False)
assert saved['iter'] == 299 and saved['optimizer_state_dict']['state']
report = {'parent': str(checkpoint), 'parent_sha256': hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
          'parent_iteration': saved['iter'], 'note': 'Resume optimizer and policy; simulator and RNG restart, not bitwise uninterrupted training.'}

def equal(a, b):
    if torch.is_tensor(a):
        return torch.equal(a.cpu(), b.cpu())
    if isinstance(a, dict):
        return a.keys() == b.keys() and all(equal(a[k], b[k]) for k in a)
    if isinstance(a, (list, tuple)):
        return len(a) == len(b) and all(equal(x, y) for x, y in zip(a, b))
    return a == b

original_load, original_learn = OnPolicyRunner.load, OnPolicyRunner.learn
Path('runtime_runner.py').write_text(inspect.getsource(OnPolicyRunner), encoding='utf-8')

def load(self, *args, **kwargs):
    result = original_load(self, *args, **kwargs)
    assert self.current_learning_iteration == saved['iter']
    assert equal(self.alg.optimizer.state_dict(), saved['optimizer_state_dict']), 'Optimizer not fully restored'
    report['optimizer_restored'] = True
    Path('resume_audit.json').write_text(json.dumps(report, indent=2), encoding='utf-8')
    return result

def learn(self, num_learning_iterations, *args, **kwargs):
    assert report.get('optimizer_restored')
    for name in ['agent', 'env']:
        old = yaml.load((checkpoint.parent / 'params' / (name + '.yaml')).read_text(), Loader=yaml.BaseLoader)
        current = yaml.load(next(Path('logs').glob('**/params/' + name + '.yaml')).read_text(), Loader=yaml.BaseLoader)
        if name == 'agent':
            for key in ['max_iterations', 'load_checkpoint', 'resume', 'load_run']:
                old.pop(key, None)
                current.pop(key, None)
        else:
            # Only the disposable smoke run uses fewer environments.
            if smoke_run:
                old['scene']['num_envs'] = current['scene']['num_envs']
                old['scene']['terrain']['num_envs'] = current['scene']['terrain']['num_envs']
            for key in ['log_dir']:
                old.pop(key, None)
                current.pop(key, None)
        assert old == current, name + ' config changed unexpectedly'
    report['configs_match_parent'] = True
    report['additional_iterations'] = num_learning_iterations
    result = original_learn(self, num_learning_iterations, *args, **kwargs)
    report['final_iteration'] = self.current_learning_iteration
    assert self.current_learning_iteration in (299 + num_learning_iterations - 1, 299 + num_learning_iterations)
    self.save(str(Path('model_extended.pt').resolve()))
    report['final_sha256'] = hashlib.sha256(Path('model_extended.pt').read_bytes()).hexdigest()
    Path('resume_audit.json').write_text(json.dumps(report, indent=2), encoding='utf-8')
    return result

OnPolicyRunner.load, OnPolicyRunner.learn = load, learn
runpy.run_path(str(Path(__file__).with_name('train_command_coverage.py')), run_name='__main__')
