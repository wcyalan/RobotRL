"""Export the audited G1 flat MLP to TorchScript, without a training runtime."""
import argparse
import hashlib
import json
from pathlib import Path
import shutil

import torch
from torch import nn


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    opts = parser.parse_args()
    params = opts.checkpoint.parent / 'params'
    agent = (params / 'agent.yaml').read_text(encoding='utf-8')
    # Deliberately reject architectures/preprocessing outside the audited baseline.
    assert agent.count('activation: elu') == 2
    assert agent.count('obs_normalization: false') == 2
    assert 'clip_actions: null' in agent
    assert 'obs_groups:\n  actor:\n  - policy\n' in agent
    state = torch.load(opts.checkpoint, map_location='cpu', weights_only=False)['actor_state_dict']
    expected = {'distribution.std_param'} | {f'mlp.{i}.{k}' for i in (0, 2, 4, 6) for k in ('weight', 'bias')}
    assert set(state) == expected
    assert all(torch.isfinite(value).all() for value in state.values())
    actor = nn.Sequential(nn.Linear(123, 256), nn.ELU(), nn.Linear(256, 128), nn.ELU(),
                          nn.Linear(128, 128), nn.ELU(), nn.Linear(128, 37)).eval()
    actor.load_state_dict({k.removeprefix('mlp.'): v for k, v in state.items() if k.startswith('mlp.')}, strict=True)
    opts.output.mkdir(parents=True, exist_ok=False)
    for name in ('agent.yaml', 'env.yaml'):
        shutil.copy2(params / name, opts.output / name)
    path = opts.output / 'policy.pt'
    torch.jit.script(actor).save(str(path))
    loaded = torch.jit.load(str(path)).eval()
    torch.manual_seed(20261019)
    maximum = 0.0
    with torch.inference_mode():
        for batch in (1, 32, 256):
            for scale in (0.1, 1.0, 5.0):
                obs = torch.randn(batch, 123) * scale
                expected_action, actual = actor(obs), loaded(obs)
                torch.testing.assert_close(actual, expected_action, atol=1e-5, rtol=1e-5)
                maximum = max(maximum, (actual - expected_action).abs().max().item())
    report = dict(checkpoint=str(opts.checkpoint.resolve()), checkpoint_sha256=sha(opts.checkpoint),
                  policy_sha256=sha(path), torch_version=torch.__version__,
                  config_sha256={name: sha(opts.output/name) for name in ('agent.yaml', 'env.yaml')},
                  exporter_sha256=sha(Path(__file__)), input_shape=['batch', 123], output_shape=['batch', 37],
                  dtype='float32', hidden_dims=[256, 128, 128], activation='ELU',
                  observation_normalization=False, history=False, recurrent=False,
                  deterministic_mean_action=True, clip_actions=None,
                  synthetic_max_abs_error=maximum, tolerance=dict(atol=1e-5, rtol=1e-5),
                  action_semantics='Unscaled action; environment applies default_joint_pos + 0.5 * action.',
                  status='Synthetic check passed; official runtime comparison on real observations still required.')
    (opts.output/'manifest.json').write_text(json.dumps(report, indent=2)+'\n', encoding='utf-8')
    print(json.dumps(report))


if __name__ == '__main__':
    main()
