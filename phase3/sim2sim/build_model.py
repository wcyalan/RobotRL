"""Convert the trained 37-DOF G1 USD to a native MuJoCo dynamics model.

This is deliberately specific to g1_minimal.usd, not a general USD converter.
The three box-shaped collision meshes are converted to exact analytic boxes.
Contact solving is native MuJoCo; matching geometry does not imply identical
Newton contact dynamics. No Isaac Lab or Newton imports are needed.
"""
import argparse
import json
import os
import re
import xml.etree.ElementTree as ET
from pathlib import Path

import mujoco
import numpy as np
from pxr import Gf, Usd, UsdGeom, UsdPhysics

CODE_ROOT = Path(__file__).resolve().parents[2]
ROOT = Path(os.environ.get('ROBOTRL_ROOT', CODE_ROOT)).resolve()


def numbers(values):
    return " ".join(f"{float(x):.12g}" for x in values)


def transform(position, rotation):
    matrix = Gf.Matrix4d(1)
    matrix.SetRotate(Gf.Quatd(rotation))
    matrix.SetTranslateOnly(Gf.Vec3d(position))
    return np.array(matrix).T


def pose(matrix):
    quat = np.empty(4)
    mujoco.mju_mat2Quat(quat, np.ascontiguousarray(matrix[:3, :3]).ravel())
    return dict(pos=numbers(matrix[:3, 3]), quat=numbers(quat))


def parameter(value, name):
    if not isinstance(value, dict):
        return float(value)
    matches = [v for expr, v in value.items() if re.fullmatch(expr, name)]
    assert len(matches) == 1, (name, value)
    return float(matches[0])


def build(asset, output, source):
    output.mkdir(parents=True, exist_ok=True)
    stage = Usd.Stage.Open(str(asset))
    assert UsdGeom.GetStageMetersPerUnit(stage) == 1 and UsdGeom.GetStageUpAxis(stage) == 'Z'
    cache = UsdGeom.XformCache()
    env = json.loads((source / 'official_env.json').read_text())
    smoke = json.loads((source / 'smoke.json').read_text())
    names, q0 = smoke['joint_names'], smoke['default_joint_pos']
    assert len(names) == len(set(names)) == len(q0) == 37
    robot = env['scene']['robot']
    joint_parameters = {}
    for name in names:
        matches = [a for a in robot['actuators'].values()
                   if any(re.fullmatch(expr, name) for expr in a['joint_names_expr'])]
        assert len(matches) == 1
        a = matches[0]
        joint_parameters[name] = {k: parameter(a[k], name)
                                  for k in ('stiffness', 'damping', 'armature', 'joint_effort_limit')}
    bodies = {str(p.GetPath()): p for p in stage.Traverse() if p.HasAPI(UsdPhysics.RigidBodyAPI)}
    joints, children = [], set()
    rest_error = 0.
    for prim in stage.Traverse():
        if not prim.IsA(UsdPhysics.Joint):
            continue
        j = UsdPhysics.Joint(prim)
        parent, child = (str(j.GetBody0Rel().GetTargets()[0]), str(j.GetBody1Rel().GetTargets()[0]))
        f0 = transform(j.GetLocalPos0Attr().Get(), j.GetLocalRot0Attr().Get())
        f1 = transform(j.GetLocalPos1Attr().Get(), j.GetLocalRot1Attr().Get())
        rest = f0 @ np.linalg.inv(f1)
        expected = np.array(cache.GetLocalToWorldTransform(bodies[child])).T
        actual = np.array(cache.GetLocalToWorldTransform(bodies[parent])).T @ rest
        rest_error = max(rest_error, float(np.max(abs(expected - actual))))
        assert prim.IsA(UsdPhysics.RevoluteJoint) or prim.IsA(UsdPhysics.FixedJoint)
        joints.append((parent, child, prim, rest, f1))
        children.add(child)
    roots = set(bodies) - children
    assert len(roots) == 1 and len(bodies) == 44 and rest_error < 2e-5
    root = roots.pop()
    substeps = env['sim']['physics']['num_substeps']
    dt = env['sim']['dt'] / substeps
    xml = ET.Element('mujoco', model='G1_native_MuJoCo')
    ET.SubElement(xml, 'compiler', angle='radian', autolimits='true', inertiafromgeom='false', fusestatic='false')
    ET.SubElement(xml, 'option', timestep=str(dt), gravity=numbers(env['sim']['gravity']),
                  integrator='implicitfast', solver='Newton', iterations='100', ls_iterations='50',
                  tolerance='1e-6', cone='pyramidal')
    visual = ET.SubElement(xml, 'visual')
    ET.SubElement(visual, 'global', offwidth='960', offheight='540')
    assets, world = ET.SubElement(xml, 'asset'), ET.SubElement(xml, 'worldbody')
    ET.SubElement(world, 'light', pos='0 0 5', directional='true', dir='0 0 -1')
    # Explicit target-domain contact settings, frozen before the first pilot.
    # Native MuJoCo has one sliding coefficient, not separate static/dynamic mu.
    ET.SubElement(world, 'geom', name='ground', type='plane', size='0 0 .1',
                  friction='.8 .005 .0001', solref='.02 1', solimp='.9 .95 .001',
                  condim='3', contype='1', conaffinity='2', rgba='.28 .32 .36 1')
    nodes = {root: ET.SubElement(world, 'body', name=bodies[root].GetName(),
                                pos=numbers(robot['init_state']['pos']))}
    ET.SubElement(nodes[root], 'freejoint', name='floating_base')
    while joints:
        ready = [j for j in joints if j[0] in nodes]
        assert ready, 'Disconnected/cyclic USD graph'
        for parent, child, prim, rest, f1 in ready:
            node = ET.SubElement(nodes[parent], 'body', name=bodies[child].GetName(), **pose(rest))
            nodes[child] = node
            if prim.IsA(UsdPhysics.RevoluteJoint):
                j = UsdPhysics.RevoluteJoint(prim)
                name = prim.GetName()
                params = joint_parameters[name]
                axis = f1[:3, :3] @ np.eye(3)['XYZ'.index(j.GetAxisAttr().Get())]
                limits = np.deg2rad([j.GetLowerLimitAttr().Get(), j.GetUpperLimitAttr().Get()])
                ET.SubElement(node, 'joint', name=name, type='hinge', pos=numbers(f1[:3, 3]),
                              axis=numbers(axis), range=numbers(limits), damping='0',
                              armature=str(params['armature']), actuatorfrclimited='true',
                              actuatorfrcrange=numbers([-params['joint_effort_limit'],params['joint_effort_limit']]))
        joints = [j for j in joints if j[1] not in nodes]
    mass_sum = 0.
    for path, prim in bodies.items():
        mass = UsdPhysics.MassAPI(prim)
        value = float(mass.GetMassAttr().Get())
        inertia = np.array(mass.GetDiagonalInertiaAttr().Get(), dtype=float)
        assert value > 0 and np.all(inertia > 0)
        mass_sum += value
        axes = mass.GetPrincipalAxesAttr().Get()
        com = np.array(mass.GetCenterOfMassAttr().Get(), dtype=float)
        if axes.GetLength() == 0 or not np.isfinite(com).all():
            # USD sentinels on two negligible fixed auxiliary links. Their
            # isotropic inertia has no preferred principal-axis orientation.
            assert prim.GetName() in ('pelvis_contour_link', 'imu_link')
            assert value < 1.1e-6 and np.allclose(inertia, inertia[0])
            axes, com = Gf.Quatf(1), np.zeros(3)
        ET.SubElement(nodes[path], 'inertial', pos=numbers(com),
                      quat=numbers([axes.GetReal(), *axes.GetImaginary()]), mass=str(value),
                      diaginertia=numbers(inertia))
    meshes = output / 'meshes'
    meshes.mkdir(exist_ok=True)
    collision_count = visual_count = 0
    for prim in Usd.PrimRange.Stage(stage, Usd.TraverseInstanceProxies()):
        if not prim.IsA(UsdGeom.Mesh):
            continue
        collision = prim.HasAPI(UsdPhysics.CollisionAPI)
        if not collision and UsdGeom.Imageable(prim).ComputeVisibility() == 'invisible':
            continue
        ancestor = prim.GetParent()
        while str(ancestor.GetPath()) not in bodies:
            ancestor = ancestor.GetParent()
            assert ancestor
        local = np.linalg.inv(np.array(cache.GetLocalToWorldTransform(ancestor)).T) @ np.array(cache.GetLocalToWorldTransform(prim)).T
        node = nodes[str(ancestor.GetPath())]
        points = np.array(UsdGeom.Mesh(prim).GetPointsAttr().Get(), dtype=float)
        if collision:
            # Assert the expected unit cube before replacing its convex hull.
            assert points.shape == (8, 3) and np.allclose(abs(points), .5)
            scale = np.linalg.norm(local[:3, :3], axis=0)
            local[:3, :3] /= scale
            assert np.allclose(local[:3, :3].T @ local[:3, :3], np.eye(3), atol=1e-6)
            ET.SubElement(node, 'geom', name=ancestor.GetName()+'_collision', type='box',
                          size=numbers(scale/2), **pose(local), contype='2', conaffinity='1',
                          friction='.8 .005 .0001', condim='3', solref='.02 1', solimp='.9 .95 .001',
                          group='3', rgba='.8 .3 .2 .3')
            collision_count += 1
        else:
            mesh = UsdGeom.Mesh(prim)
            assert np.all(np.array(mesh.GetFaceVertexCountsAttr().Get()) == 3)
            points = points @ local[:3, :3].T + local[:3, 3]
            faces = np.array(mesh.GetFaceVertexIndicesAttr().Get()).reshape(-1, 3)
            if mesh.GetOrientationAttr().Get() == 'leftHanded':
                faces = faces[:, ::-1]
            name = f'visual_{visual_count}'
            with (meshes / f'{name}.obj').open('w', encoding='ascii') as stream:
                for point in points:
                    stream.write('v ' + numbers(point) + '\n')
                for face in faces:
                    stream.write('f ' + ' '.join(str(int(i)+1) for i in face) + '\n')
            ET.SubElement(assets, 'mesh', name=name, file=f'meshes/{name}.obj')
            ET.SubElement(node, 'geom', type='mesh', mesh=name, contype='0', conaffinity='0',
                          group='1', rgba='.65 .68 .72 1')
            visual_count += 1
    actuator = ET.SubElement(xml, 'actuator')
    for name in names:
        p = joint_parameters[name]
        # The pinned Newton backend uses POSITION_VELOCITY: separate P and D
        # actuators, with a joint-level limit on their sum.
        ET.SubElement(actuator, 'position', name=name+'_pd', joint=name,
                      kp=str(p['stiffness']), ctrllimited='false', forcelimited='false')
        ET.SubElement(actuator, 'velocity', name=name+'_velocity', joint=name,
                      kv=str(p['damping']), ctrllimited='false', forcelimited='false')
    ET.indent(xml)
    path = output / 'g1.xml'
    ET.ElementTree(xml).write(path, encoding='utf-8', xml_declaration=True)
    model = mujoco.MjModel.from_xml_path(str(path))
    assert (model.nq, model.nv, model.nu, model.nbody) == (44, 43, 74, 45)
    assert collision_count == 3 and visual_count == 43
    assert abs(mujoco.mj_getTotalmass(model)-mass_sum) < 1e-6
    interface = dict(joint_names=names, default_joint_pos=q0, root_body=bodies[root].GetName(),
                     initial_height=robot['init_state']['pos'][2], action_scale=.5,
                     physics_dt=dt, policy_dt=env['sim']['dt']*env['decimation'],
                     physics_steps_per_action=substeps*env['decimation'], joint_parameters=joint_parameters,
                     noise_amplitudes=[.1]*3+[.2]*3+[.05]*3+[0.]*3+[.01]*37+[1.5]*37+[0.]*37,
                     observation_terms=['root_com_lin_vel_in_link_frame:3','root_ang_vel_in_link_frame:3','projected_gravity:3',
                                        'command:3','joint_pos_relative:37','joint_vel:37','previous_action:37'],
                     native_contact=dict(sliding_friction=.8, torsional_friction=.005,
                                         rolling_friction=.0001, condim=3, solref=[.02,1], solimp=[.9,.95,.001]),
                     mass_kg=mass_sum, body_count=44, collision_count=collision_count,
                     visual_count=visual_count, zero_pose_usd_matrix_error=rest_error,
                     mujoco_version=mujoco.__version__,
                     limitations=['Native box contacts replace Newton convex-mesh contacts.',
                                  'Native contact parameters are a declared target-domain choice, not a calibrated match.',
                                  'PD gains and effort limits follow saved config; runtime backend parity still requires a reference export.',
                                  'No self collision, pushes, mass randomization or hardware state estimator.'])
    (output / 'interface.json').write_text(json.dumps(interface, indent=2)+'\n', encoding='utf-8')
    print(json.dumps(dict(model=str(path), mass_kg=mass_sum, nq=model.nq,nv=model.nv,nu=model.nu)))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--asset', type=Path, default=ROOT/'assets/G1/g1_minimal.usd')
    parser.add_argument('--output', type=Path, default=ROOT/'assets/G1/native_mujoco')
    parser.add_argument('--source', type=Path, default=ROOT/'experiments/phase3_g1_environment/run_001/smoke')
    args = parser.parse_args()
    build(args.asset, args.output, args.source)
