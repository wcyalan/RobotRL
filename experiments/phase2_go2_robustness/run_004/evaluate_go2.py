"""Independent first-episode evaluation, without renderer or parameter updates."""
import argparse
import hashlib
import json
import importlib.metadata
from pathlib import Path
from action_delay import ActionDelay
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
parser.add_argument('--output', required=True)
parser.add_argument('--eval_seed', type=int, default=10000)
parser.add_argument('--trace_env', type=int, default=0)
parser.add_argument('--scenario', choices=['original', 'nominal', 'low_friction', 'low_friction_both', 'heavy_base'], default='original')
parser.add_argument('--audit_friction', action='store_true')
parser.add_argument('--action_delay_steps', type=int, default=0)
parser.add_argument('--forward_speed', type=float, default=None,
                    help='Controlled diagnostic: fixed body-frame forward speed, zero lateral/yaw command.')
opts, remainder = parser.parse_known_args()
args = _parse_args(remainder, 'zero')
cfg, agent = resolve_task_config(args.task, 'rsl_rl_cfg_entry_point')
agent = handle_deprecated_rsl_rl_cfg(agent, importlib.metadata.version('rsl-rl-lib'))
cfg.scene.num_envs = args.num_envs or 16
cfg.seed = opts.eval_seed
scenario_settings = {'name': opts.scenario}
if opts.scenario != 'original':
    mass_scale = 1.5 if opts.scenario == 'heavy_base' else 1.0
    static_mu, dynamic_mu = (0.3, 0.3) if opts.scenario in ['low_friction', 'low_friction_both'] else (0.8, 0.6)
    if opts.scenario == 'low_friction_both':
        cfg.scene.terrain.physics_material.static_friction = 0.3
        cfg.scene.terrain.physics_material.dynamic_friction = 0.3
        scenario_settings['ground_friction_config'] = 0.3
    cfg.events.add_base_mass.params['mass_distribution_params'] = (mass_scale, mass_scale)
    cfg.events.physics_material.params['static_friction_range'] = (static_mu, static_mu)
    cfg.events.physics_material.params['dynamic_friction_range'] = (dynamic_mu, dynamic_mu)
    scenario_settings.update(base_mass_scale=mass_scale, robot_static_friction=static_mu,
                             robot_dynamic_friction=dynamic_mu,
                             note='Fixed base mass, other baseline randomizations/pushes retained; simulated stress conditions, not calibrated hardware. Ground/material combining remains unchanged.')
if opts.forward_speed is not None:
    command_cfg = cfg.commands.base_velocity
    command_cfg.heading_command = False
    command_cfg.rel_heading_envs = 0.0
    command_cfg.rel_standing_envs = 0.0
    command_cfg.resampling_time_range = (1000.0, 1000.0)
    command_cfg.ranges.lin_vel_x = (opts.forward_speed, opts.forward_speed)
    command_cfg.ranges.lin_vel_y = (0.0, 0.0)
    command_cfg.ranges.ang_vel_z = (0.0, 0.0)
agent.seed = opts.eval_seed
args.device = cfg.sim.device
torch.manual_seed(opts.eval_seed)
with launch_simulation(cfg, args):
    raw = gym.make(args.task, cfg=cfg)
    env = RslRlVecEnvWrapper(raw, clip_actions=agent.clip_actions)
    try:
        runner = OnPolicyRunner(env, agent.to_dict(), log_dir=None, device=agent.device)
        runner.load(opts.model)
        policy = runner.get_inference_policy(device=raw.unwrapped.device)
        base = raw.unwrapped
        n, device = base.num_envs, base.device
        assert 0 <= opts.trace_env < n
        trace_env = opts.trace_env
        # Runner construction/loading must not alter evaluation RNG initialization.
        raw.reset(seed=opts.eval_seed)
        obs = env.get_observations()
        active = torch.ones(n, dtype=torch.bool, device=device)
        returns = torch.zeros(n, device=device)
        lengths = torch.zeros(n, dtype=torch.long, device=device)
        xy_error = torch.zeros(n, device=device)
        yaw_error = torch.zeros(n, device=device)
        diagnostics = {k: torch.zeros(n, device=device) for k in
                       ['forward_speed_m_s', 'xy_speed_m_s', 'root_height_m', 'upright_cosine']}
        diagnostic_steps = torch.zeros(n, device=device)
        ended = torch.zeros(n, dtype=torch.bool, device=device)
        timed_out = torch.zeros_like(ended)
        trajectory = []
        friction_snapshots = []
        action_delay = ActionDelay(opts.action_delay_steps)
        action_delay_trace = []
        with torch.inference_mode():
            for step_index in range(base.max_episode_length + 2):
                robot = base.scene['robot']
                command = base.command_manager.get_command('base_velocity')
                linear = robot.data.root_lin_vel_b.torch
                angular = robot.data.root_ang_vel_b.torch
                xy = torch.linalg.vector_norm(linear[:, :2] - command[:, :2], dim=1)
                yaw = (angular[:, 2] - command[:, 2]).abs()
                assert torch.isfinite(xy).all() and torch.isfinite(yaw).all()
                if opts.forward_speed is not None:
                    expected = torch.tensor([opts.forward_speed, 0.0, 0.0], device=device)
                    assert torch.allclose(command[active], expected.expand_as(command[active]), atol=1e-6)
                q = robot.data.root_quat_w.torch
                measured = {'forward_speed_m_s': linear[:, 0],
                            'xy_speed_m_s': torch.linalg.vector_norm(linear[:, :2], dim=1),
                            'root_height_m': robot.data.root_pos_w.torch[:, 2],
                            'upright_cosine': 1 - 2 * (q[:, 0]**2 + q[:, 1]**2)}
                # Exclude the first second of spawn settling from posture/speed diagnostics.
                settled = active & (lengths * base.step_dt >= 1.0)
                diagnostic_steps += settled
                for key, value in measured.items():
                    assert torch.isfinite(value).all()
                    diagnostics[key] += value * settled
                xy_error += xy * active
                yaw_error += yaw * active
                if active[trace_env]:
                    trajectory.append({'position': robot.data.root_pos_w.torch[trace_env].cpu().tolist(),
                                       'quaternion_xyzw': robot.data.root_quat_w.torch[trace_env].cpu().tolist(),
                                       'body_linear_velocity': linear[trace_env].cpu().tolist(),
                                       'body_angular_velocity': angular[trace_env].cpu().tolist(),
                                       'joint_positions': robot.data.joint_pos.torch[trace_env].cpu().tolist(),
                                       'command': command[trace_env].cpu().tolist()})
                actions = policy(obs)
                assert torch.isfinite(actions).all()
                if agent.clip_actions is not None:
                    actions = actions.clamp(-agent.clip_actions, agent.clip_actions)
                applied_actions = action_delay.apply(actions)
                if step_index < 8:
                    action_delay_trace.append({'step':step_index,
                        'commanded':actions.cpu().tolist(), 'applied':applied_actions.cpu().tolist()})
                observation, reward, terminated, truncated, _ = raw.step(applied_actions)
                action_delay.reset(terminated | truncated)
                if opts.audit_friction and step_index in [50, 500]:
                    from friction_audit import snapshot
                    friction_snapshots.append({'step':step_index+1, **snapshot(base)})
                assert torch.isfinite(reward).all()
                assert all(torch.isfinite(v).all() for v in observation.values())
                returns += reward * active
                lengths += active.long()
                ended |= terminated & active
                timed_out |= truncated & active
                active &= ~(terminated | truncated)
                obs = env.get_observations()
                if not active.any():
                    break
        assert not active.any(), 'Incomplete episodes: do not report censored episodes as full episodes'
        rows = [{'env_index': i, 'return': returns[i].item(),
                 'length_steps': lengths[i].item(), 'length_seconds': lengths[i].item()*base.step_dt,
                 'terminated': ended[i].item(), 'time_out': timed_out[i].item(),
                 'mean_xy_velocity_error_m_s': (xy_error[i]/lengths[i]).item(),
                 'mean_yaw_velocity_error_rad_s': (yaw_error[i]/lengths[i]).item()} for i in range(n)]
        report = {'checkpoint': opts.model, 'sha256': hashlib.sha256(Path(opts.model).read_bytes()).hexdigest(),
                  'scenario': scenario_settings,
                  'friction_audit': friction_snapshots,
                  'action_delay': {'steps':opts.action_delay_steps, 'milliseconds':opts.action_delay_steps*base.step_dt*1000,
                      'initial_queue':'zero normalized action (nominal joint targets)',
                      'reset':'clear delayed commands for automatically reset lanes',
                      'observation_note':'Previous-action observations and action-rate rewards follow applied actions; sensor latency is not added.',
                      'first_eight_steps_all_envs':action_delay_trace},
                  'forward_speed_command_m_s': opts.forward_speed,
                  'step_dt': base.step_dt,
                  'eval_seed': opts.eval_seed, 'episodes': rows,
                  'return_mean': returns.mean().item(), 'return_std': returns.std(unbiased=False).item(),
                  'mean_length_seconds': (lengths.float().mean()*base.step_dt).item(),
                  'termination_rate': ended.float().mean().item(),
                  'time_out_rate': timed_out.float().mean().item(),
                  'mean_xy_velocity_error_m_s': (xy_error/lengths).mean().item(),
                  'mean_yaw_velocity_error_rad_s': (yaw_error/lengths).mean().item(),
                  'protocol': 'Deterministic policy; training task randomization/noise retained except an explicit fixed-command override when forward_speed_command_m_s is set; first complete episode per lane. Velocity errors sampled before stepping, averaged per episode then across episodes.',
                  'termination_note': 'Task non-timeout termination is base_contact (proxy for falling, not visual posture inspection).',
                  f'trajectory_env_{trace_env}': trajectory,
                  'trace_env': trace_env,
                  'trajectory_note': 'Pre-step poses only; final post-step terminal pose is excluded because of automatic reset.'}
        report['settled_diagnostics'] = {
            'exclude_first_seconds': 1.0,
            'episodes_with_samples': int((diagnostic_steps > 0).sum().item()),
            'per_episode': [{k: (v[i]/diagnostic_steps[i]).item() if diagnostic_steps[i] > 0 else None
                             for k, v in diagnostics.items()} for i in range(n)]}
        for key, value in diagnostics.items():
            valid = diagnostic_steps > 0
            report['settled_diagnostics'][key] = ((value[valid]/diagnostic_steps[valid]).mean().item()
                                                  if valid.any() else None)
        Path(opts.output).write_text(json.dumps(report, indent=2))
        print('ROBOTRL_EVALUATION_PASSED', {k:v for k,v in report.items() if k not in ['episodes','action_delay','friction_audit'] and not k.startswith('trajectory_env_')}, flush=True)
    finally:
        env.close()
