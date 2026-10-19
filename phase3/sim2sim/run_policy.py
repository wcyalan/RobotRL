"""Run an exported G1 policy with native MuJoCo CPU dynamics, without Isaac Lab.

This first pilot uses one episode per condition and a fixed initial pose. It is
not the original 32-episode acceptance test. Never concatenate reset episodes.
"""
import argparse
import json
import os
import time
from pathlib import Path

import mujoco
import numpy as np
import torch

CODE_ROOT = Path(__file__).resolve().parents[2]
ROOT = Path(os.environ.get('ROBOTRL_ROOT', CODE_ROOT)).resolve()


class G1Env:
    def __init__(self, model_dir):
        self.config = json.loads((model_dir/'interface.json').read_text())
        self.model = mujoco.MjModel.from_xml_path(str(model_dir/'g1.xml'))
        self.data = mujoco.MjData(self.model)
        self.names = self.config['joint_names']
        ids = [self.model.joint(n).id for n in self.names]
        assert len(set(ids)) == 37
        self.qadr = self.model.jnt_qposadr[ids]
        self.vadr = self.model.jnt_dofadr[ids]
        self.actuator_ids = np.array([self.model.actuator(n+'_pd').id for n in self.names])
        assert np.array_equal(self.model.actuator_trnid[self.actuator_ids,0], ids)
        self.default = np.array(self.config['default_joint_pos'])
        self.root_id = self.model.body(self.config['root_body']).id
        self.torso_geom = self.model.geom('torso_link_collision').id
        self.previous_action = np.zeros(37, dtype=np.float32)
        self.noise_amplitudes = np.array(self.config['noise_amplitudes'])
        self.dt = self.config['policy_dt']
        self.substeps = self.config['physics_steps_per_action']
        assert abs(self.model.opt.timestep*self.substeps-self.dt) < 1e-12
        assert self.model.nmocap == 0
        self.reset(20000)

    def reset(self, seed):
        mujoco.mj_resetData(self.model,self.data)
        self.data.qpos[:3] = [0,0,self.config['initial_height']]
        self.data.qpos[3:7] = [1,0,0,0]  # MuJoCo wxyz, not Isaac Lab xyzw.
        self.data.qpos[self.qadr] = self.default
        self.previous_action.fill(0)
        self.data.ctrl[self.actuator_ids] = self.default
        self.rng = np.random.default_rng(seed)
        mujoco.mj_forward(self.model,self.data)

    def state(self):
        mujoco.mj_forward(self.model,self.data)
        velocity = np.empty(6)
        # Source BaseArticulationData.root_lin_vel_b aliases root_com_lin_vel_b.
        # BODY gives COM velocity; use world orientation then rotate into LINK
        # axes, not the slightly rotated principal-inertia axes of BODY local.
        mujoco.mj_objectVelocity(self.model,self.data,mujoco.mjtObj.mjOBJ_BODY,
                                self.root_id,velocity,0)
        rotation = self.data.xmat[self.root_id].reshape(3,3)
        gravity = rotation.T @ np.array([0.,0.,-1.])
        return rotation.T @ velocity[3:], rotation.T @ velocity[:3], gravity

    def observation(self, command, noise=True):
        lin,ang,gravity = self.state()
        obs = np.concatenate((lin,ang,gravity,command,
                              self.data.qpos[self.qadr]-self.default,
                              self.data.qvel[self.vadr],self.previous_action))
        if noise:
            obs += self.rng.uniform(-1.,1.,123)*self.noise_amplitudes
        assert obs.shape == (123,) and np.isfinite(obs).all()
        return obs.astype(np.float32)

    def target(self, action):
        action = np.asarray(action,dtype=np.float32)
        assert action.shape == (37,) and np.isfinite(action).all()
        return self.default + self.config['action_scale']*action

    def step(self, action):
        self.data.ctrl[self.actuator_ids] = self.target(action)
        self.previous_action[:] = action  # Store raw action, not position target.
        max_torso_force = 0.
        for _ in range(self.substeps):
            mujoco.mj_step(self.model,self.data)
            assert np.isfinite(self.data.qpos).all() and np.isfinite(self.data.qvel).all()
            assert not any(w.number for w in self.data.warning), 'MuJoCo numerical warning'
            total_force = np.zeros(3)
            for index, contact in enumerate(self.data.contact):
                if self.torso_geom in (contact.geom1,contact.geom2):
                    wrench = np.empty(6)
                    mujoco.mj_contactForce(self.model,self.data,index,wrench)
                    sign = 1 if contact.geom2 == self.torso_geom else -1
                    total_force += sign*contact.frame.reshape(3,3).T @ wrench[:3]
            max_torso_force = max(max_torso_force,float(np.linalg.norm(total_force)))
        # Native solver contact sample semantics differ from Isaac Lab sensors.
        return max_torso_force > 1., max_torso_force


def evaluate(env, policy, case, threshold, condition, eval_seed, output, noise, video=False):
    env.reset(eval_seed)
    frames = {key: [] for key in ('qpos','qvel','command','action','metrics')}
    sums = np.zeros((len(case['measurement_windows']),3))
    counts = np.zeros(len(sums),dtype=int)
    stable, settle, early, max_contact = 0, None, False, 0.
    writer = renderer = None
    if video:
        import imageio.v2 as imageio
        writer = imageio.get_writer(output/'forward_low.mp4',fps=25,codec='libx264',quality=7)
        renderer = mujoco.Renderer(env.model,height=540,width=960)
        camera = mujoco.MjvCamera()
        camera.distance,camera.azimuth,camera.elevation = 2.5,135,-15
        option = mujoco.MjvOption()
        option.geomgroup[3] = 0
    try:
        with torch.inference_mode():
            for step in range(round(case['duration_s']/env.dt)):
                t = step*env.dt
                cmd = next(v for start,end,v in case['segments'] if start <= t < end)
                obs = env.observation(cmd,noise=noise)
                lin,ang,gravity = env.state()
                metrics = np.array([np.linalg.norm(lin[:2]-cmd[:2]), abs(ang[2]-cmd[2]),
                                    np.rad2deg(np.arccos(np.clip(-gravity[2],-1,1)))])
                for i,(start,end,_) in enumerate(case['measurement_windows']):
                    if start <= t < end:
                        sums[i] += metrics
                        counts[i] += 1
                if condition == 'stop_restart' and 12 <= t < 22:
                    stable = stable+1 if metrics[0] <= threshold['stopped_xy_speed_m_s'] and metrics[1] <= threshold['stopped_yaw_speed_rad_s'] else 0
                    if stable*env.dt >= threshold['stop_stable_window_s'] and settle is None:
                        settle = t+env.dt-case['stop_time_s']
                action = policy(torch.from_numpy(obs[None])).numpy()[0]
                for key,value in [('qpos',env.data.qpos),('qvel',env.data.qvel),('command',cmd),('action',action),('metrics',metrics)]:
                    frames[key].append(np.array(value).copy())
                if renderer is not None and step % 2 == 0:
                    camera.lookat[:] = env.data.xpos[env.root_id]
                    renderer.update_scene(env.data,camera=camera,scene_option=option)
                    writer.append_data(renderer.render())
                early,force = env.step(action)
                max_contact = max(max_contact,force)
                if early:
                    break
    finally:
        if writer is not None:
            writer.close()
            renderer.close()
    windows, passed = [], not early
    for i,(start,end,kind) in enumerate(case['measurement_windows']):
        means = sums[i]/counts[i] if counts[i] else None
        expected = round((end-start)/env.dt)
        limits = [threshold['stopped_xy_speed_m_s' if kind == 'stopped' else 'moving_xy_error_m_s'],
                  threshold['stopped_yaw_speed_rad_s' if kind == 'stopped' else 'moving_yaw_error_rad_s'],threshold['mean_tilt_deg']]
        ok = bool(counts[i] == expected and np.all(means <= limits))
        passed &= ok
        windows.append(dict(start_s=start,end_s=end,kind=kind,samples=int(counts[i]),expected_samples=expected,
                            means=None if means is None else means.tolist(),passed=ok))
    if condition == 'stop_restart':
        passed &= settle is not None and settle <= threshold['stop_stable_by_s']
    np.savez_compressed(output/(condition+'.npz'),**{k:np.array(v) for k,v in frames.items()})
    return dict(condition=condition,steps=len(frames['action']),duration_s=len(frames['action'])*env.dt,
                early_termination=early,max_torso_contact_force_n=max_contact,windows=windows,
                stop_stable_time_s=settle,pilot_episode_passed=bool(passed),
                final_root_position=env.data.qpos[:3].tolist(),
                max_abs_action=float(np.max(np.abs(frames['action']))))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model-dir',type=Path,default=ROOT/'assets/G1/native_mujoco')
    parser.add_argument('--output',type=Path,default=ROOT/'experiments/phase3_g1_sim2sim/run_001/pilot')
    parser.add_argument('--seeds',type=int,nargs='+',choices=[0,1],default=[0,1])
    parser.add_argument('--eval-seed',type=int,default=20000)
    parser.add_argument('--conditions',nargs='+',default=['stand','forward_low','forward_medium','turn_left','turn_right','stop_restart'])
    parser.add_argument('--no-noise',action='store_true')
    parser.add_argument('--video',action='store_true')
    args = parser.parse_args()
    torch.set_num_threads(1)
    protocol = json.loads((CODE_ROOT/'phase3/evaluation/protocol.json').read_text())
    assert all(c in protocol['conditions'] for c in args.conditions)
    args.output.mkdir(parents=True,exist_ok=False)
    env = G1Env(args.model_dir)
    results = []
    start = time.monotonic()
    for seed in args.seeds:
        export = ROOT/f'experiments/phase3_g1_deployment/run_{seed+1:03d}/export/policy.pt'
        policy = torch.jit.load(str(export),map_location='cpu').eval()
        folder = args.output/f'seed_{seed}'
        folder.mkdir()
        for condition in args.conditions:
            result = evaluate(env,policy,protocol['conditions'][condition],protocol['thresholds'],condition,
                              args.eval_seed,folder,not args.no_noise,args.video and condition=='forward_low')
            result['training_seed'] = seed
            results.append(result)
            (folder/(condition+'.json')).write_text(json.dumps(result,indent=2)+'\n')
            print(json.dumps(result),flush=True)
    report = dict(stage='native MuJoCo single-episode pilot; not group acceptance',
                  mujoco_version=mujoco.__version__,torch_version=torch.__version__,
                  wall_seconds=time.monotonic()-start,initial_pose='default joints, base z=.74, yaw=0, zero velocities',
                  evaluation_seed=args.eval_seed,observation_noise=not args.no_noise,
                  metric_order=['xy_error_m_s','yaw_error_rad_s','tilt_deg'],
                  interface=env.config,results=results)
    (args.output/'summary.json').write_text(json.dumps(report,indent=2)+'\n')


if __name__ == '__main__':
    main()
