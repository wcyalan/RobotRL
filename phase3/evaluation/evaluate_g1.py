"""Reload an official G1 checkpoint in a fresh process and record first episodes."""
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
from rsl_rl.runners import OnPolicyRunner
from isaaclab.app import launch_simulation
from isaaclab_rl.entrypoints.simple_agents import _parse_args
from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper, handle_deprecated_rsl_rl_cfg
from isaaclab_tasks.utils import resolve_task_config

p = argparse.ArgumentParser(add_help=False)
p.add_argument('--model', required=True)
p.add_argument('--output', required=True)
p.add_argument('--eval_seed', type=int, default=10000)
opts, remaining = p.parse_known_args()
args = _parse_args(remaining, 'zero')
cfg, agent = resolve_task_config(args.task, 'rsl_rl_cfg_entry_point')
agent = handle_deprecated_rsl_rl_cfg(agent, importlib.metadata.version('rsl-rl-lib'))
cfg.scene.num_envs = args.num_envs or 4
cfg.seed = agent.seed = opts.eval_seed
args.device = cfg.sim.device
out = Path(opts.output)
out.mkdir(parents=True, exist_ok=True)
def tensor(v):
    return v if isinstance(v, torch.Tensor) else v.torch
with launch_simulation(cfg, args):
    raw = gym.make(args.task, cfg=cfg)
    env = RslRlVecEnvWrapper(raw, clip_actions=agent.clip_actions)
    try:
        runner = OnPolicyRunner(env, agent.to_dict(), log_dir=None, device=agent.device)
        runner.load(opts.model)
        policy = runner.get_inference_policy(device=raw.unwrapped.device)
        raw.reset(seed=opts.eval_seed)
        obs = env.get_observations()
        base = raw.unwrapped
        robot = base.scene['robot']
        active = torch.ones(base.num_envs, dtype=torch.bool, device=base.device)
        returns = torch.zeros(base.num_envs, device=base.device)
        lengths = torch.zeros(base.num_envs, dtype=torch.long, device=base.device)
        early = torch.zeros_like(active)
        timeout = torch.zeros_like(active)
        trace = {k: [] for k in ['root_pose', 'joint_pos', 'command', 'active', 'done_after_step']}
        max_action, max_contact = 0.0, 0.0
        for step in range(base.max_episode_length + 2):
            assert all(torch.isfinite(v).all() for v in obs.values())
            with torch.inference_mode():
                action = policy(obs)
            assert torch.isfinite(action).all()
            max_action = max(max_action, action.abs().max().item())
            trace['root_pose'].append(tensor(robot.data.root_pose_w).cpu().numpy().copy())
            trace['joint_pos'].append(tensor(robot.data.joint_pos).cpu().numpy().copy())
            trace['command'].append(base.command_manager.get_command('base_velocity').cpu().numpy().copy())
            trace['active'].append(active.cpu().numpy().copy())
            obs, reward, done, _ = env.step(action)
            assert torch.isfinite(reward).all()
            force = tensor(base.scene['contact_forces'].data.net_forces_w)
            assert torch.isfinite(force).all()
            max_contact = max(max_contact, force.norm(dim=-1).max().item())
            returns += reward * active
            lengths += active.long()
            done = done.bool()
            newly_done = active & done
            timeout |= newly_done & base.termination_manager.time_outs
            early |= newly_done & base.termination_manager.terminated
            trace['done_after_step'].append(done.cpu().numpy().copy())
            active &= ~done
            if not active.any():
                break
        assert not active.any(), 'not all first episodes ended'
        reset_obs, _ = raw.reset(seed=opts.eval_seed + 1)
        assert all(torch.isfinite(v).all() for v in reset_obs.values())
        np.savez_compressed(out/'trajectory.npz', **{k: np.asarray(v) for k,v in trace.items()})
        result = dict(passed=True, purpose='checkpoint reload and numerical smoke test; no behavior success claim',
                      model=str(Path(opts.model).resolve()), model_sha256=hashlib.sha256(Path(opts.model).read_bytes()).hexdigest(),
                      eval_seed=opts.eval_seed, num_envs=base.num_envs, task=args.task,
                      returns=returns.cpu().tolist(), lengths=lengths.cpu().tolist(),
                      early_termination=early.cpu().tolist(), time_out=timeout.cpu().tolist(),
                      mean_return=returns.mean().item(), max_abs_action=max_action, max_contact_force_n=max_contact,
                      joint_names=robot.joint_names, step_dt=base.step_dt,
                      trace_semantics='pre-action states; done_after_step marks the following transition; use active mask to exclude reset episodes')
        (out/'summary.json').write_text(json.dumps(result, indent=2)+'\n')
        print('G1_RELOAD_PASSED', json.dumps(result), flush=True)
    finally:
        env.close()
