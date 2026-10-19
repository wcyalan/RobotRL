"""Compare official inference and a reloaded export on actual G1 observations.

This is an interface/numerical check, not a new behavior acceptance protocol.
The official policy drives the environment; exports receive identical inputs.
"""
import argparse
import hashlib
import importlib.metadata
import json
from pathlib import Path

import warp as wp
wp.config.enable_backward = False
import numpy as np
import torch
import gymnasium as gym
from tensordict import TensorDict
from rsl_rl.runners import OnPolicyRunner
from isaaclab.app import launch_simulation
from isaaclab_rl.entrypoints.simple_agents import _parse_args
from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper, handle_deprecated_rsl_rl_cfg
from isaaclab_tasks.utils import resolve_task_config


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


p = argparse.ArgumentParser(add_help=False)
p.add_argument('--model', required=True)
p.add_argument('--export', required=True)
p.add_argument('--output', required=True)
opts, remaining = p.parse_known_args()
args = _parse_args(remaining, 'zero')
assert args.task == 'Isaac-Velocity-Flat-G1'
directory = Path(opts.export)
manifest = json.loads((directory/'manifest.json').read_text())
assert sha(opts.model) == manifest['checkpoint_sha256']
assert sha(directory/'policy.pt') == manifest['policy_sha256']
for name, digest in manifest['config_sha256'].items():
    assert sha(directory/name) == digest
    assert sha(Path(opts.model).parent/'params'/name) == digest
cfg, agent = resolve_task_config(args.task, 'rsl_rl_cfg_entry_point')
agent = handle_deprecated_rsl_rl_cfg(agent, importlib.metadata.version('rsl-rl-lib'))
assert sha('/robotrl/assets/G1/g1_minimal.usd') == '9dfe7a710aa791e49abf2d9ea74ad3163e291f02f21f59bda9bfcc40f3fab428'
assert agent.actor.hidden_dims == [256, 128, 128]
assert agent.actor.activation == 'elu' and not agent.actor.obs_normalization
assert agent.clip_actions is None
cfg.scene.num_envs = 32
cfg.seed = agent.seed = 22000
cfg.commands.base_velocity.heading_command = False
cfg.commands.base_velocity.rel_heading_envs = cfg.commands.base_velocity.rel_standing_envs = 0.0
cfg.commands.base_velocity.resampling_time_range = (1000.0, 1000.0)
args.device = cfg.sim.device
out = Path(opts.output)
out.mkdir(parents=True, exist_ok=False)
with launch_simulation(cfg, args):
    raw = gym.make(args.task, cfg=cfg)
    env = RslRlVecEnvWrapper(raw, clip_actions=agent.clip_actions)
    try:
        base = raw.unwrapped
        runner = OnPolicyRunner(env, agent.to_dict(), log_dir=None, device=agent.device)
        runner.load(opts.model)
        official = runner.get_inference_policy(device=base.device)
        exported = torch.jit.load(str(directory/'policy.pt'), map_location=base.device).eval()
        raw.reset(seed=22000)
        obs = env.get_observations()
        maximum, resets, samples = 0.0, 0, 0
        saved_obs, saved_actions = [], []
        commands = [[0.,0.,0.], [.3,0.,0.], [.6,0.,0.], [0.,0.,.5], [0.,0.,-.5], [0.,0.,0.]]
        with torch.inference_mode():
            for step in range(1500):
                # Set every step so resets cannot leave an unintended sampled command.
                base.command_manager.get_command('base_velocity')[:] = torch.tensor(commands[step//250], device=base.device)
                obs = TensorDict(base.observation_manager.compute(), batch_size=[base.num_envs])
                x = obs['policy']
                assert x.shape == (32, 123) and torch.isfinite(x).all()
                expected, actual = official(obs), exported(x)
                assert actual.shape == (32, 37) and torch.isfinite(expected).all() and torch.isfinite(actual).all()
                torch.testing.assert_close(actual, expected, atol=1e-5, rtol=1e-5)
                maximum = max(maximum, (actual-expected).abs().max().item())
                samples += x.shape[0]
                if step % 10 == 0:
                    saved_obs.append(x.cpu().numpy().copy())
                    saved_actions.append(expected.cpu().numpy().copy())
                obs, reward, done, _ = env.step(expected)
                assert torch.isfinite(reward).all()
                resets += int(done.sum())
        assert resets > 0, 'No automatic reset exercised'
        raw.reset(seed=22001)
        obs = env.get_observations()
        with torch.inference_mode():
            torch.testing.assert_close(exported(obs['policy']), official(obs), atol=1e-5, rtol=1e-5)
        np.savez_compressed(out/'observations_actions.npz', observations=np.concatenate(saved_obs), actions=np.concatenate(saved_actions))
        report = dict(passed=True, purpose='Official/export action equivalence; not behavior acceptance',
                      checkpoint_sha256=sha(opts.model), policy_sha256=sha(directory/'policy.pt'),
                      verifier_sha256=sha(__file__), eval_seed=22000, num_envs=32, steps=1500,
                      compared_observations=samples, max_abs_error=maximum, atol=1e-5, rtol=1e-5,
                      reset_transitions=resets, explicit_reset_checked=True, commands=commands,
                      step_dt=base.step_dt, joint_names=base.scene['robot'].joint_names,
                      saved_observations=4800, torch_version=torch.__version__,
                      samples_sha256=sha(out/'observations_actions.npz'),
                      rsl_rl_version=importlib.metadata.version('rsl-rl-lib'))
        (out/'summary.json').write_text(json.dumps(report, indent=2)+'\n')
        print('EXPORT_EQUIVALENCE_PASSED', json.dumps(report), flush=True)
    finally:
        env.close()
