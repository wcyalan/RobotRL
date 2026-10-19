"""Regression checks for G1 USD geometry, native control and policy observations.

Run: python -m unittest discover -s phase3/sim2sim -p test_interface.py -v
Requires the local original USD, generated MJCF and exported policies.
"""
import unittest
import json
import tempfile
from pathlib import Path
from unittest.mock import patch

import mujoco
import numpy as np
import torch
from pxr import Gf, Usd, UsdPhysics

from build_model import CODE_ROOT, ROOT, transform
from run_policy import G1Env, evaluate


class InterfaceTests(unittest.TestCase):
    def setUp(self):
        self.env = G1Env(ROOT/'assets/G1/native_mujoco')

    def test_usd_kinematics_and_inertia(self):
        env = self.env
        stage = Usd.Stage.Open(str(ROOT/'assets/G1/g1_minimal.usd'))
        bodies = {str(p.GetPath()):p for p in stage.Traverse() if p.HasAPI(UsdPhysics.RigidBodyAPI)}
        joints = []
        for p in stage.Traverse():
            if not p.IsA(UsdPhysics.Joint):
                continue
            j = UsdPhysics.Joint(p)
            parent,child = str(j.GetBody0Rel().GetTargets()[0]),str(j.GetBody1Rel().GetTargets()[0])
            f0 = transform(j.GetLocalPos0Attr().Get(),j.GetLocalRot0Attr().Get())
            f1 = transform(j.GetLocalPos1Attr().Get(),j.GetLocalRot1Attr().Get())
            axis = None if p.IsA(UsdPhysics.FixedJoint) else np.eye(3)['XYZ'.index(UsdPhysics.RevoluteJoint(p).GetAxisAttr().Get())]
            joints.append((parent,child,p.GetName(),f0,np.linalg.inv(f1),axis))
        rng = np.random.default_rng(42)
        for _ in range(12):
            q = env.default+rng.uniform(-.1,.1,37)
            quat = rng.normal(size=4);quat /= np.linalg.norm(quat)
            pos = rng.uniform(-1,1,3);pos[2] += 2
            env.data.qpos[:3],env.data.qpos[3:7] = pos,quat
            env.data.qpos[env.qadr] = q
            mujoco.mj_forward(env.model,env.data)
            matrices = {'/g1/pelvis':transform(Gf.Vec3d(*pos),Gf.Quatd(float(quat[0]),Gf.Vec3d(*quat[1:])))}
            pending = list(joints)
            while pending:
                ready = [j for j in pending if j[0] in matrices]
                self.assertTrue(ready)
                for parent,child,name,f0,f1inv,axis in ready:
                    rotation = np.eye(4)
                    if axis is not None:
                        angle = q[env.names.index(name)]
                        quaternion = np.r_[np.cos(angle/2),axis*np.sin(angle/2)]
                        r = np.empty(9);mujoco.mju_quat2Mat(r,quaternion)
                        rotation[:3,:3] = r.reshape(3,3)
                    matrices[child] = matrices[parent] @ f0 @ rotation @ f1inv
                pending = [j for j in pending if j[1] not in matrices]
            for path,expected in matrices.items():
                bid = env.model.body(bodies[path].GetName()).id
                np.testing.assert_allclose(env.data.xpos[bid],expected[:3,3],atol=3e-6)
                np.testing.assert_allclose(env.data.xmat[bid].reshape(3,3),expected[:3,:3],atol=3e-6)
        for path,p in bodies.items():
            bid = env.model.body(p.GetName()).id
            mass = UsdPhysics.MassAPI(p)
            self.assertAlmostEqual(env.model.body_mass[bid],mass.GetMassAttr().Get(),places=8)
            np.testing.assert_allclose(np.sort(env.model.body_inertia[bid]),
                                       np.sort(mass.GetDiagonalInertiaAttr().Get()),atol=1e-9)
        self.assertEqual(env.model.nmocap,0)
        self.assertEqual(sum(env.model.geom_contype == 2),3)

    def test_observation_frame_and_order(self):
        env = self.env
        angle = np.pi/2
        env.data.qpos[3:7] = [np.cos(angle/2),0,0,np.sin(angle/2)]
        env.data.qvel[:6] = [.4,.2,.1,.3,.2,.5]
        offsets = np.arange(37)*.001
        env.data.qpos[env.qadr] = env.default+offsets
        env.data.qvel[env.vadr] = np.arange(37)*.02
        env.previous_action[:] = np.arange(37)*.03
        command = [.6,-.1,.5]
        obs = env.observation(command,noise=False)
        # Analytic COM correction for a root link rotating about its origin:
        # v_com = v_link + omega cross [0,0,-.07605] (all in link axes).
        np.testing.assert_allclose(obs[:3],[.18479,-.377185,.1],atol=1e-6)
        np.testing.assert_allclose(obs[3:6],[.3,.2,.5],atol=1e-6)
        np.testing.assert_allclose(obs[6:9],[0,0,-1],atol=1e-6)
        np.testing.assert_allclose(obs[9:12],command,atol=1e-6)
        np.testing.assert_allclose(obs[12:49],offsets,atol=1e-6)
        np.testing.assert_allclose(obs[49:86],np.arange(37)*.02,atol=1e-6)
        np.testing.assert_allclose(obs[86:],env.previous_action,atol=1e-6)
        env.reset(20000)
        self.assertFalse(env.previous_action.any())

    def test_pd_targets_and_saturation(self):
        env = self.env
        action = np.linspace(-30,30,37).astype(np.float32)
        target = env.target(action)
        np.testing.assert_allclose(target,env.default+.5*action,atol=1e-6)
        env.data.qvel[env.vadr] = np.linspace(-2,2,37)
        env.data.ctrl[env.actuator_ids] = target
        mujoco.mj_forward(env.model,env.data)
        expected=[]
        for i,name in enumerate(env.names):
            p=env.config['joint_parameters'][name]
            force=p['stiffness']*(target[i]-env.data.qpos[env.qadr[i]])-p['damping']*env.data.qvel[env.vadr[i]]
            expected.append(np.clip(force,-p['joint_effort_limit'],p['joint_effort_limit']))
        np.testing.assert_allclose(env.data.qfrc_actuator[env.vadr],expected,atol=1e-6)

    def test_recorded_observations_keep_exported_actions(self):
        torch.set_num_threads(1)
        for seed in (0,1):
            run = ROOT/f'experiments/phase3_g1_deployment/run_{seed+1:03d}'
            policy = torch.jit.load(str(run/'export/policy.pt'),map_location='cpu').eval()
            with np.load(run/'equivalence/observations_actions.npz') as data,torch.inference_mode():
                action = policy(torch.from_numpy(data['observations'])).numpy()
                np.testing.assert_allclose(action,data['actions'],atol=1e-5,rtol=1e-5)

    def test_early_fall_is_serializable_failure(self):
        protocol=json.loads((CODE_ROOT/'phase3/evaluation/protocol.json').read_text())
        scratch=(ROOT/'.task_context').resolve()
        scratch.mkdir(exist_ok=True)
        with tempfile.TemporaryDirectory(dir=scratch) as folder,patch.object(self.env,'step',return_value=(True,10.)):
            self.assertTrue(Path(folder).resolve().is_relative_to(scratch))
            report=evaluate(self.env,lambda obs:torch.zeros((1,37)),protocol['conditions']['stand'],
                            protocol['thresholds'],'stand',20000,Path(folder),True)
        self.assertEqual(report['steps'],1)
        self.assertTrue(report['early_termination'])
        self.assertFalse(report['pilot_episode_passed'])
        self.assertIsNone(report['windows'][0]['means'])
        json.dumps(report,allow_nan=False)


if __name__ == '__main__':
    unittest.main()
