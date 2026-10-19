"""Run the frozen behavior protocol using only a loaded TorchScript actor.

No PPO runner or training checkpoint is loaded. The existing environment adapter
and score calculations are retained for comparison with the original baseline.
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
from isaaclab.app import launch_simulation
from isaaclab_rl.entrypoints.simple_agents import _parse_args
from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper, handle_deprecated_rsl_rl_cfg
from isaaclab_tasks.utils import resolve_task_config

p = argparse.ArgumentParser(add_help=False)
p.add_argument('--export', required=True, help='Directory containing policy.pt and manifest.json')
p.add_argument('--protocol', required=True)
p.add_argument('--condition', required=True)
p.add_argument('--output', required=True)
p.add_argument('--eval_seed', type=int, required=True)
p.add_argument('--smoke', action='store_true', help='Test evaluator plumbing only; never count as official behavior acceptance')
opts, remaining = p.parse_known_args()
export = Path(opts.export)
manifest = json.loads((export/'manifest.json').read_text())
sha = lambda path: hashlib.sha256(Path(path).read_bytes()).hexdigest()
assert sha(export/'policy.pt') == manifest['policy_sha256']
assert manifest['input_shape'] == ['batch', 123] and manifest['output_shape'] == ['batch', 37]
assert manifest['observation_normalization'] is False and manifest['clip_actions'] is None
for name, digest in manifest['config_sha256'].items():
    assert sha(export/name) == digest
assert sha('/robotrl/assets/G1/g1_minimal.usd') == '9dfe7a710aa791e49abf2d9ea74ad3163e291f02f21f59bda9bfcc40f3fab428'
assert sha(opts.protocol) == '894cad975e0ca4bf06fe1c605e7fe0dc21b46caddfd2a5d786ddafa2fdcf17dc'
protocol = json.loads(Path(opts.protocol).read_text())
case = protocol['conditions'][opts.condition]
args = _parse_args(remaining, 'zero')
assert args.task == protocol['task']
cfg, agent = resolve_task_config(args.task, 'rsl_rl_cfg_entry_point')
agent = handle_deprecated_rsl_rl_cfg(agent, importlib.metadata.version('rsl-rl-lib'))
cfg.scene.num_envs = 4 if opts.smoke else protocol['num_envs']
cfg.seed = agent.seed = opts.eval_seed
cfg.episode_length_s = case['duration_s']
cfg.events.add_base_mass.params.update(mass_distribution_params=(1.,1.), operation='scale', distribution='uniform')
cfg.events.push_robot = None
for key in cfg.events.reset_base.params['velocity_range']:
    cfg.events.reset_base.params['velocity_range'][key] = (0.,0.)
cmd = cfg.commands.base_velocity
cmd.heading_command = False
cmd.rel_heading_envs = cmd.rel_standing_envs = 0.
cmd.resampling_time_range = (1000.,1000.)
cmd.ranges.lin_vel_x = cmd.ranges.lin_vel_y = cmd.ranges.ang_vel_z = (0.,0.)
args.device = cfg.sim.device
out = Path(opts.output)
out.mkdir(parents=True, exist_ok=False)
def tensor(v):
    return v if isinstance(v, torch.Tensor) else v.torch
with launch_simulation(cfg, args):
    raw = gym.make(args.task, cfg=cfg)
    env = RslRlVecEnvWrapper(raw, clip_actions=agent.clip_actions)
    try:
        base = raw.unwrapped
        n, device, dt = base.num_envs, base.device, base.step_dt
        policy = torch.jit.load(str(export/'policy.pt'), map_location=device).eval()
        assert agent.clip_actions is None
        reference = json.loads((export.parent/'equivalence/summary.json').read_text())
        assert reference['passed'] and reference['policy_sha256'] == manifest['policy_sha256']
        assert reference['joint_names'] == base.scene['robot'].joint_names
        assert dt == reference['step_dt'] == 0.02
        raw.reset(seed=opts.eval_seed)
        obs = env.get_observations()
        robot = base.scene['robot']
        active = torch.ones(n, dtype=torch.bool, device=device)
        lengths = torch.zeros(n, dtype=torch.long, device=device)
        early = torch.zeros_like(active)
        timed_out = torch.zeros_like(active)
        returns = torch.zeros(n,device=device)
        windows = case['measurement_windows']
        sums = torch.zeros((len(windows),n,3),device=device)
        counts = torch.zeros((len(windows),n),device=device)
        stable_count = torch.zeros(n,dtype=torch.long,device=device)
        settle = torch.full((n,),float('inf'),device=device)
        threshold = protocol['thresholds']
        trace = {k:[] for k in ['root_pose','joint_pos','command','active','done_after_step']}
        previous = None
        command_switches = []
        with torch.inference_mode():
            for step in range(base.max_episode_length):
                t = step*dt
                target = next(values for start,end,values in case['segments'] if start <= t < end)
                target_tensor = torch.tensor(target,device=device,dtype=torch.float32)
                command = base.command_manager.get_command('base_velocity')
                command[:] = target_tensor
                if target != previous:
                    obs = TensorDict(base.observation_manager.compute(),batch_size=[n])
                    previous = target
                    command_switches.append({'time_s':t,'command':target})
                assert torch.allclose(obs['policy'][active,9:12],target_tensor.expand(int(active.sum()),3),atol=1e-6)
                assert all(torch.isfinite(v).all() for v in obs.values())
                lin = tensor(robot.data.root_lin_vel_b)
                ang = tensor(robot.data.root_ang_vel_b)
                xy = (lin[:,:2]-target_tensor[:2]).norm(dim=-1)
                yaw = (ang[:,2]-target_tensor[2]).abs()
                q = tensor(robot.data.root_quat_w)
                upright = 1-2*(q[:,0]**2+q[:,1]**2)
                tilt = torch.rad2deg(torch.acos(upright.clamp(-1,1)))
                metrics = torch.stack((xy,yaw,tilt),dim=-1)
                assert torch.isfinite(metrics).all()
                for i,(start,end,_) in enumerate(windows):
                    if start <= t < end:
                        sums[i] += metrics*active[:,None]
                        counts[i] += active
                if opts.condition == 'stop_restart' and 12 <= t < 22:
                    stable = active & (xy <= threshold['stopped_xy_speed_m_s']) & (yaw <= threshold['stopped_yaw_speed_rad_s'])
                    stable_count = torch.where(stable,stable_count+1,0)
                    reached = (stable_count*dt >= threshold['stop_stable_window_s']) & torch.isinf(settle)
                    settle[reached] = t+dt-case['stop_time_s']
                for key,value in [('root_pose',tensor(robot.data.root_pose_w)),('joint_pos',tensor(robot.data.joint_pos)),('command',command),('active',active)]:
                    trace[key].append(value[:2].cpu().numpy().copy())
                action = policy(obs['policy'])
                assert torch.isfinite(action).all()
                obs,reward,done,_ = env.step(action)
                assert torch.isfinite(reward).all()
                returns += reward*active
                lengths += active.long()
                done = done.bool()
                early |= active & base.termination_manager.terminated
                timed_out |= active & base.termination_manager.time_outs
                trace['done_after_step'].append(done[:2].cpu().numpy().copy())
                active &= ~done
                # Run the complete horizon even if all first episodes fail, so all
                # scheduled command switches are exercised. Metrics exclude resets.
        assert not active.any(), 'some first episodes did not terminate by horizon'
        passed = (~early) & timed_out & (lengths == base.max_episode_length)
        window_results=[]
        for i,(start,end,kind) in enumerate(windows):
            means=sums[i]/counts[i].clamp(min=1)[:,None]
            expected=round((end-start)/dt)
            xy_limit=threshold['stopped_xy_speed_m_s' if kind=='stopped' else 'moving_xy_error_m_s']
            yaw_limit=threshold['stopped_yaw_speed_rad_s' if kind=='stopped' else 'moving_yaw_error_rad_s']
            passed &= (counts[i]==expected) & (means[:,0]<=xy_limit) & (means[:,1]<=yaw_limit) & (means[:,2]<=threshold['mean_tilt_deg'])
            window_results.append({'start_s':start,'end_s':end,'kind':kind,'expected_samples':expected,'counts':counts[i].cpu().tolist(),
                                   'mean_xy_error_m_s':means[:,0].cpu().tolist(),'mean_yaw_error_rad_s':means[:,1].cpu().tolist(),'mean_tilt_deg':means[:,2].cpu().tolist(),
                                   'note':'Means with zero/partial sample counts cannot pass and are not complete-episode performance.'})
        if opts.condition == 'stop_restart':
            passed &= settle <= threshold['stop_stable_by_s']
        np.savez_compressed(out/'trajectory.npz',**{k:np.asarray(v) for k,v in trace.items()})
        result=dict(completed=True, evaluator_smoke=opts.smoke, condition=opts.condition,eval_seed=opts.eval_seed,num_envs=n,
                    model_sha256=manifest['checkpoint_sha256'],policy_sha256=manifest['policy_sha256'],
                    ppo_runner_constructed=False,policy_format='TorchScript',evaluator_sha256=sha(__file__),
                    protocol_sha256=hashlib.sha256(Path(opts.protocol).read_bytes()).hexdigest(),
                    pass_count=int(passed.sum()),group_passed=None if opts.smoke else bool(passed.sum()>=protocol['minimum_pass_count']),
                    lane_passed=passed.cpu().tolist(),early_termination=early.cpu().tolist(),time_out=timed_out.cpu().tolist(),lengths=lengths.cpu().tolist(),
                    returns=returns.cpu().tolist(),windows=window_results,command_switches=command_switches,
                    stop_stable_time_s=[v if np.isfinite(v) else None for v in settle.cpu().tolist()],step_dt=dt,
                    trace_lanes=[0,1],trace_semantics='pre-action state, active excludes reset episodes; next transition termination is done_after_step')
        (out/'summary.json').write_text(json.dumps(result,indent=2)+'\n')
        print('BEHAVIOR_EVALUATION_COMPLETE',opts.condition,opts.eval_seed,result['pass_count'],flush=True)
    finally:
        env.close()
