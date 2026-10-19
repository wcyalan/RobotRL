"""Replay recorded G1 poses using the exact training USD; never step physics.

Requires usd-core, mujoco, numpy, imageio[ffmpeg], Pillow.
Joint frames: https://openusd.org/release/api/class_usd_physics_joint.html
Run from the repository root: python phase3/visualization/replay_g1.py --seed 0
"""
import argparse
import hashlib
import importlib.metadata
import json
import math
from pathlib import Path
import xml.etree.ElementTree as ET

import imageio.v2 as imageio
import mujoco
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from pxr import Gf, Usd, UsdGeom, UsdPhysics

ROOT = Path(__file__).resolve().parents[2]
ASSET_HASH = "9dfe7a710aa791e49abf2d9ea74ad3163e291f02f21f59bda9bfcc40f3fab428"
CONDITIONS = ("stand", "forward_low", "forward_medium", "turn_left", "turn_right", "stop_restart")


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def transform(position, rotation):
    matrix = Gf.Matrix4d(1)
    matrix.SetRotate(Gf.Quatd(rotation))
    matrix.SetTranslateOnly(Gf.Vec3d(position))
    return np.array(matrix).T  # USD row-vector convention -> column vectors


def numbers(values):
    return " ".join(f"{float(x):.9g}" for x in values)


def build_visual(asset, names):
    assert digest(asset) == ASSET_HASH, "Asset must match the trained model"
    stage = Usd.Stage.Open(str(asset))
    cache = UsdGeom.XformCache()
    bodies = {str(p.GetPath()): p for p in stage.Traverse() if p.HasAPI(UsdPhysics.RigidBodyAPI)}
    assert len(bodies) == 44
    joints, children = [], set()
    rest_error = 0.0
    for prim in stage.Traverse():
        if not prim.IsA(UsdPhysics.Joint):
            continue
        joint = UsdPhysics.Joint(prim)
        parent, child = (str(joint.GetBody0Rel().GetTargets()[0]), str(joint.GetBody1Rel().GetTargets()[0]))
        frame0 = transform(joint.GetLocalPos0Attr().Get(), joint.GetLocalRot0Attr().Get())
        frame1 = transform(joint.GetLocalPos1Attr().Get(), joint.GetLocalRot1Attr().Get())
        axis, index = None, None
        if prim.IsA(UsdPhysics.RevoluteJoint):
            axis = "XYZ".index(UsdPhysics.RevoluteJoint(prim).GetAxisAttr().Get())
            index = names.index(prim.GetName())
        else:
            assert prim.IsA(UsdPhysics.FixedJoint)
        inverse = np.linalg.inv(frame1)
        predicted = np.array(cache.GetLocalToWorldTransform(bodies[parent])).T @ frame0 @ inverse
        rest_error = max(rest_error, float(np.max(np.abs(predicted - np.array(cache.GetLocalToWorldTransform(bodies[child])).T))))
        joints.append((parent, child, frame0, inverse, axis, index))
        children.add(child)
    assert rest_error < 2e-5, rest_error
    assert sorted(j[-1] for j in joints if j[-1] is not None) == list(range(37))
    roots = set(bodies) - children
    assert len(roots) == 1
    root = roots.pop()
    ordered, visited = [], {root}
    while joints:
        ready = [j for j in joints if j[0] in visited]
        assert ready, "Disconnected or cyclic joint graph"
        for joint in ready:
            ordered.append(joint)
            visited.add(joint[1])
        joints = [j for j in joints if j[1] not in visited]

    xml = ET.Element("mujoco", model="G1_USD_pose_replay")
    ET.SubElement(xml, "compiler", angle="radian")
    visual = ET.SubElement(xml, "visual")
    ET.SubElement(visual, "global", offwidth="960", offheight="540")
    ET.SubElement(visual, "headlight", ambient=".45 .45 .45", diffuse=".7 .7 .7")
    assets, world = ET.SubElement(xml, "asset"), ET.SubElement(xml, "worldbody")
    ET.SubElement(assets, "texture", name="grid", type="2d", builtin="checker", rgb1=".22 .27 .31", rgb2=".32 .38 .42", width="512", height="512")
    ET.SubElement(assets, "material", name="floor", texture="grid", texrepeat="2 2", texuniform="true")
    ET.SubElement(world, "geom", type="plane", size="0 0 .1", material="floor", contype="0", conaffinity="0")
    ET.SubElement(world, "light", pos="0 0 5", dir="0 0 -1", directional="true")
    nodes = {path: ET.SubElement(world, "body", name=prim.GetName(), mocap="true") for path, prim in bodies.items()}
    mesh_dir = asset.parent / "replay_meshes"
    mesh_dir.mkdir(exist_ok=True)
    mesh_count = 0
    for prim in Usd.PrimRange.Stage(stage, Usd.TraverseInstanceProxies()):
        if not prim.IsA(UsdGeom.Mesh) or UsdGeom.Imageable(prim).ComputeVisibility() == "invisible":
            continue
        ancestor = prim.GetParent()
        while str(ancestor.GetPath()) not in bodies:
            ancestor = ancestor.GetParent()
            assert ancestor
        body_path = str(ancestor.GetPath())
        mesh = UsdGeom.Mesh(prim)
        points = np.array(mesh.GetPointsAttr().Get(), dtype=float)
        local = np.linalg.inv(np.array(cache.GetLocalToWorldTransform(ancestor)).T) @ np.array(cache.GetLocalToWorldTransform(prim)).T
        points = points @ local[:3, :3].T + local[:3, 3]
        counts = np.array(mesh.GetFaceVertexCountsAttr().Get())
        assert np.all(counts == 3), "Only triangle meshes are supported"
        faces = np.array(mesh.GetFaceVertexIndicesAttr().Get()).reshape(-1, 3)
        if mesh.GetOrientationAttr().Get() == "leftHanded":
            faces = faces[:, ::-1]
        mesh_name = f"mesh_{mesh_count}"
        file = mesh_dir / f"{mesh_name}.obj"
        # Derived cache is deterministic, tied to the verified USD hash.
        if not file.exists():
            with file.open("w", encoding="ascii") as stream:
                for point in points:
                    stream.write("v " + numbers(point) + "\n")
                for face in faces:
                    stream.write("f " + " ".join(str(int(i)+1) for i in face) + "\n")
        color = [.7, .7, .7]
        targets = prim.GetRelationship("material:binding").GetTargets()
        if targets:
            material = stage.GetPrimAtPath(targets[0])
            for shader in material.GetChildren():
                value = shader.GetAttribute("inputs:diffuse_color_constant").Get()
                if value is not None:
                    color = list(value)
        ET.SubElement(assets, "mesh", name=mesh_name, file=file.as_posix())
        ET.SubElement(nodes[body_path], "geom", type="mesh", mesh=mesh_name, rgba=numbers(color+[1]), contype="0", conaffinity="0")
        mesh_count += 1
    assert mesh_count == 43
    model = mujoco.MjModel.from_xml_string(ET.tostring(xml, encoding="unicode"))
    ids = {path: int(model.body(prim.GetName()).mocapid[0]) for path, prim in bodies.items()}
    return model, root, ordered, ids, {"body_count": len(bodies), "visual_mesh_count": mesh_count, "zero_pose_max_matrix_error": rest_error}


def apply_pose(data, root, joints, ids, pose, angles, origin):
    x, y, z, w = pose[3:]
    assert abs(np.linalg.norm(pose[3:])-1) < 1e-4
    transforms = {root: transform(Gf.Vec3d(*(pose[:3]-origin)), Gf.Quatd(float(w), Gf.Vec3d(float(x), float(y), float(z))))}
    for parent, child, frame0, inverse, axis, index in joints:
        rotation = np.eye(4)
        if index is not None:
            vector = np.eye(3)[axis]
            skew = np.array([[0, -vector[2], vector[1]], [vector[2], 0, -vector[0]], [-vector[1], vector[0], 0]])
            rotation[:3, :3] = np.eye(3) + math.sin(angles[index])*skew + (1-math.cos(angles[index]))*(skew@skew)
        transforms[child] = transforms[parent] @ frame0 @ rotation @ inverse
    for path, value in transforms.items():
        data.mocap_pos[ids[path]] = value[:3, 3]
        quat = np.empty(4)
        mujoco.mju_mat2Quat(quat, np.ascontiguousarray(value[:3, :3]).ravel())
        data.mocap_quat[ids[path]] = quat


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, choices=(0, 1), required=True)
    parser.add_argument("--eval-seed", type=int, default=20000)
    parser.add_argument("--lane", type=int, choices=(0, 1), default=0)
    parser.add_argument("--preview-only", action="store_true")
    args = parser.parse_args()
    run = ROOT / f"experiments/phase3_g1_baseline/run_{args.seed+1:03d}"
    output = run / "videos"
    output.mkdir(exist_ok=True)
    smoke = json.loads((ROOT / "experiments/phase3_g1_environment/run_001/smoke/smoke.json").read_text())
    asset = ROOT / "assets/G1/g1_minimal.usd"
    model, root, joints, ids, checks = build_visual(asset, smoke["joint_names"])
    data = mujoco.MjData(model)
    font = ImageFont.truetype("C:/Windows/Fonts/arial.ttf", 20)
    small = ImageFont.truetype("C:/Windows/Fonts/arial.ttf", 16)
    camera = mujoco.MjvCamera()
    camera.distance, camera.elevation = 2.6, -15
    stem = f"g1_seed{args.seed}_eval{args.eval_seed}_lane{args.lane}"
    video = output / f"{stem}.mp4"
    provenance, total = [], 0
    contact_sheet = Image.new("RGB", (2400, 1620))
    writer = None if args.preview_only else imageio.get_writer(video, fps=25, codec="libx264", quality=7, macro_block_size=1, pixelformat="yuv420p")
    try:
        with mujoco.Renderer(model, height=540, width=960) as renderer:
            for condition_index, condition in enumerate(CONDITIONS):
                directory = run / f"evaluation/eval_{args.eval_seed}/{condition}"
                source = directory / "trajectory.npz"
                record = json.loads((directory / "summary.json").read_text())
                trace = np.load(source)
                n = len(trace["root_pose"])
                assert record["step_dt"] == .02 and np.all(trace["active"][:, args.lane])
                assert trace["joint_pos"].shape == (n, 2, 37)
                origin = trace["root_pose"][0, args.lane, :3].astype(float).copy()
                origin[2] = 0
                x, y, z, w = trace["root_pose"][0, args.lane, 3:]
                yaw = math.atan2(2*(w*z+x*y), 1-2*(y*y+z*z))
                camera.azimuth = math.degrees(yaw)+130
                previews = {0, n//4//2*2, n//2//2*2, (3*n//4)//2*2, n-2}
                indices = sorted(previews) if args.preview_only else range(0, n, 2)
                for i in indices:
                    pose = trace["root_pose"][i, args.lane].astype(float)
                    apply_pose(data, root, joints, ids, pose, trace["joint_pos"][i, args.lane], origin)
                    mujoco.mj_forward(model, data)
                    camera.lookat[:] = pose[:3]-origin
                    camera.lookat[2] = .65
                    renderer.update_scene(data, camera=camera)
                    frame = Image.fromarray(renderer.render())
                    draw = ImageDraw.Draw(frame)
                    draw.rectangle((0, 0, 960, 79), fill=(17, 24, 34))
                    result = "PASS" if record["lane_passed"][args.lane] else "FAIL: continuous stop stability"
                    draw.text((18, 10), f"G1 | Training seed {args.seed} | {condition} | {result}", font=font, fill="white")
                    vx, vy, wz = trace["command"][i, args.lane]
                    draw.text((18, 43), f"Time {i*.02:05.2f} s | Command vx {vx:.2f}, vy {vy:.2f} m/s, yaw {wz:.2f} rad/s", font=small, fill=(170, 220, 240))
                    draw.rectangle((0, 510, 960, 540), fill=(17, 24, 34))
                    draw.text((18, 516), f"Recorded poses / original USD | Real time | Eval {args.eval_seed}, lane {args.lane} | No physics stepping", font=small, fill="white")
                    if writer:
                        writer.append_data(np.asarray(frame))
                        total += 1
                    if i in previews:
                        column = sorted(previews).index(i)
                        contact_sheet.paste(frame.resize((480, 270)), (column*480, condition_index*270))
                provenance.append({"condition": condition, "source_sha256": digest(source), "summary_sha256": digest(directory / "summary.json"), "model_sha256": record["model_sha256"], "protocol_sha256": record["protocol_sha256"], "start_video_s": sum(p["duration_s"] for p in provenance), "duration_s": n*.02, "lane_passed": record["lane_passed"][args.lane]})
                print(f"Rendered seed={args.seed} {condition}", flush=True)
    finally:
        if writer:
            writer.close()
    contact_sheet.save(output / f"{stem}_preview.jpg", quality=92)
    if not args.preview_only:
        reader = imageio.get_reader(video)
        meta, count = reader.get_meta_data(), reader.count_frames()
        reader.close()
        assert count == total == 3250 and abs(meta["duration"]-130) < .05
        report = {"video": video.name, "training_seed": args.seed, "eval_seed": args.eval_seed, "lane": args.lane, "selection": "first saved lane, first frozen evaluation seed; no selection by visual quality", "asset_sha256": digest(asset), "joint_order": smoke["joint_names"], "method": "Original USD meshes and joint frames; root xyzw and 37 joint positions; MuJoCo mocap + mj_forward, never mj_step. Not a new simulation or original Isaac rendering.", "fps": 25, "frames": count, "duration_s": meta["duration"], "resolution": meta["size"], "renderer_checks": checks, "segments": provenance, "versions": {p: importlib.metadata.version(p) for p in ("usd-core", "mujoco", "imageio", "imageio-ffmpeg", "Pillow")}}
        (output / f"{stem}_metadata.json").write_text(json.dumps(report, indent=2)+"\n", encoding="utf-8")


if __name__ == "__main__":
    main()
