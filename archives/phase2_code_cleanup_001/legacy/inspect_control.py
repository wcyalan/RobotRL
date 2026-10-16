"""Finite diagnostic: standing, joint response, learned policy, forced base contact."""
import argparse
import json
import importlib.metadata
from pathlib import Path
import warp as wp
wp.config.enable_backward = False
import torch
import gymnasium as gym
from rsl_rl.runners import OnPolicyRunner
from isaaclab.app import launch_simulation
from isaaclab_rl.entrypoints.simple_agents import _parse_args
from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper, handle_deprecated_rsl_rl_cfg
from isaaclab_tasks.utils import resolve_task_config

parser = argparse.ArgumentParser(add_help=False)
parser.add_argument('--model', required=True)
opts, rest = parser.parse_known_args()
args = _parse_args(rest, 'zero')
cfg, agent = resolve_task_config(args.task, 'rsl_rl_cfg_entry_point')
agent = handle_deprecated_rsl_rl_cfg(agent, importlib.metadata.version('rsl-rl-lib'))
cfg.scene.num_envs = 4
cfg.scene.robot.spawn.usd_path = '/robotrl/assets/Go2/go2.usd'
cfg.sim.physics.solver_cfg.nconmax = 256
cfg.sim.physics.solver_cfg.njmax = 512
cfg.seed = 10000
args.device = cfg.sim.device
# Keep contacts visible instead of auto-resetting immediately in the deliberate-contact phase.
# These overrides are diagnostic-only, never used for policy training.
cfg.terminations.base_contact = None
cfg.episode_length_s = 30.0
cmd = cfg.commands.base_velocity
cmd.heading_command = False
cmd.rel_heading_envs = cmd.rel_standing_envs = 0.0
cmd.resampling_time_range = (1000., 1000.)
cmd.ranges.lin_vel_x = (0.5, 0.5)
cmd.ranges.lin_vel_y = cmd.ranges.ang_vel_z = (0., 0.)

def tensor(x):
    return x.torch if hasattr(x, 'torch') else x

with launch_simulation(cfg, args):
    raw = gym.make(args.task, cfg=cfg)
    env = RslRlVecEnvWrapper(raw, clip_actions=agent.clip_actions)
    try:
        base = raw.unwrapped
        robot = base.scene['robot']
        sensor = base.scene['contact_forces']
        runner = OnPolicyRunner(env, agent.to_dict(), log_dir=None, device=agent.device)
        runner.load(opts.model)
        policy = runner.get_inference_policy(device=base.device)
        names = robot.joint_names
        body_names = sensor.body_names
        report = {'joint_names': names, 'contact_body_names': body_names,
                  'diagnostic_overrides': 'base-contact termination disabled; horizon 30 s; fixed 0.5 m/s command; nconmax=256, njmax=512 to accommodate deliberate base penetration; training randomization retained',
                  'phases': {}}
        calf = [i for i, name in enumerate(names) if 'calf' in name]
        assert len(calf) == 4
        with torch.inference_mode():
            for phase in ['zero_action', 'calf_positive_offset', 'policy', 'forced_base_contact']:
                raw.reset(seed=10000)
                if phase == 'forced_base_contact':
                    pose = tensor(robot.data.root_pose_w).clone()
                    pose[:, 2] = 0.02
                    robot.write_root_pose_to_sim(pose)
                rows = []
                force_peak = torch.zeros(len(body_names), device=base.device)
                for step in range(150):
                    action = torch.zeros(raw.action_space.shape, device=base.device)
                    if phase == 'calf_positive_offset':
                        action[:, calf] = 0.4  # target increment = 0.1 rad
                    elif phase == 'policy':
                        action = policy(env.get_observations())
                    obs, reward, terminated, truncated, _ = raw.step(action)
                    assert not (terminated | truncated).any(), 'Unexpected diagnostic reset'
                    q = tensor(robot.data.joint_pos)
                    target = base.action_manager.get_term('joint_pos').processed_actions
                    expected = tensor(robot.data.default_joint_pos) + 0.25 * action
                    assert torch.allclose(target, expected, atol=1e-5), 'Action mapping differs from configured default + 0.25 action'
                    forces = torch.linalg.vector_norm(tensor(sensor.data.net_forces_w), dim=-1)
                    assert torch.isfinite(q).all() and torch.isfinite(forces).all()
                    force_peak = torch.maximum(force_peak, forces.max(dim=0).values)
                    if step >= 100:
                        rows.append(torch.cat([
                            tensor(robot.data.root_pos_w)[:, 2].mean().reshape(1),
                            q.mean(dim=0), target.mean(dim=0), forces.mean(dim=0)]).cpu())
                average = torch.stack(rows).mean(dim=0)
                j = len(names)
                report['phases'][phase] = {
                    'mean_root_height_m': average[0].item(),
                    'mean_joint_position_rad': average[1:1+j].tolist(),
                    'mean_joint_target_rad': average[1+j:1+2*j].tolist(),
                    'mean_contact_force_norm_N': dict(zip(body_names, average[1+2*j:].tolist())),
                    'peak_contact_force_norm_N': dict(zip(body_names, force_peak.cpu().tolist())),
                    'action_mapping_passed': True}
                Path('control_diagnostic.json').write_text(json.dumps(report, indent=2))
                print('CONTROL_PHASE_COMPLETE', phase, report['phases'][phase]['mean_root_height_m'], flush=True)
        print('ROBOTRL_CONTROL_DIAGNOSTIC_PASSED', flush=True)
    finally:
        env.close()
