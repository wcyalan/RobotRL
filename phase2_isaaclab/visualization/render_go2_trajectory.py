"""Render recorded Newton poses with a MuJoCo Go2 visual model; no physics stepping."""
from pathlib import Path
import json
import hashlib
import math
import argparse
import xml.etree.ElementTree as ET
import numpy as np
import mujoco
import imageio.v2 as imageio
from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parents[2]
parser = argparse.ArgumentParser()
parser.add_argument('--run', default='run_001')
parser.add_argument('--input', default='forward_299.json')
parser.add_argument('--env', type=int, default=0)
parser.add_argument('--slowdown', type=int, default=1)
parser.add_argument('--title', default='Go2 | Official baseline | Forward command 0.50 m/s')
args = parser.parse_args()
run = ROOT / 'experiments/phase2_go2_baseline' / args.run
source = run / args.input
record = json.loads(source.read_text(encoding='utf-8'))
frames = record[f'trajectory_env_{args.env}']
fps = round(1 / record['step_dt'])
assert len(frames) > 0 and fps == 50 and args.slowdown >= 1
joint_names = [f'{leg}_{joint}_joint' for leg in ['FL', 'FR', 'RL', 'RR'] for joint in ['hip', 'thigh', 'calf']]
asset = ROOT / 'assets/unitree_go2'
tree = ET.parse(asset / 'go2.xml')
xml = tree.getroot()
xml.find('compiler').set('meshdir', (asset / 'assets').as_posix())
visual = ET.SubElement(xml, 'visual')
ET.SubElement(visual, 'global', offwidth='960', offheight='540')
ET.SubElement(visual, 'headlight', diffuse='.7 .7 .7', ambient='.4 .4 .4')
assets = xml.find('asset')
ET.SubElement(assets, 'texture', name='ground_tex', type='2d', builtin='checker', rgb1='.23 .28 .34', rgb2='.3 .36 .42', width='512', height='512')
ET.SubElement(assets, 'material', name='ground_mat', texture='ground_tex', texrepeat='2 2', texuniform='true', reflectance='.05')
world = xml.find('worldbody')
ET.SubElement(world, 'geom', name='floor', type='plane', size='0 0 .1', material='ground_mat')
ET.SubElement(world, 'light', pos='0 0 5', dir='0 0 -1', directional='true')
model = mujoco.MjModel.from_xml_string(ET.tostring(xml, encoding='unicode'))
data = mujoco.MjData(model)
addresses = [model.joint(name).qposadr[0] for name in joint_names]
assert len(set(addresses)) == 12
out = source.parent / 'videos'
out.mkdir(exist_ok=True)
stem = 'go2_forward_20s' if args.input == 'forward_299.json' and args.env == 0 and args.slowdown == 1 else f'go2_env{args.env}_{args.slowdown}x_slow'
output = out / f'{stem}.mp4'
camera = mujoco.MjvCamera()
camera.distance = 1.65
camera.elevation = -22
q = frames[0]['quaternion_xyzw']
yaw = math.atan2(2*(q[3]*q[2]+q[0]*q[1]), 1-2*(q[1]**2+q[2]**2))
camera.azimuth = math.degrees(yaw) + 130
origin = np.array(frames[0]['position'])
origin[2] = 0
font = ImageFont.truetype('C:/Windows/Fonts/arial.ttf', 20)
small = ImageFont.truetype('C:/Windows/Fonts/arial.ttf', 16)
options = mujoco.MjvOption()
options.geomgroup[3] = 0
with mujoco.Renderer(model, height=540, width=960) as renderer, imageio.get_writer(output, fps=fps, codec='libx264', quality=8, macro_block_size=1, pixelformat='yuv420p') as writer:
    for i, frame in enumerate(frames):
        data.qpos[:3] = np.array(frame['position']) - origin
        x,y,z,w = frame['quaternion_xyzw']
        data.qpos[3:7] = [w,x,y,z]
        assert abs(np.linalg.norm(data.qpos[3:7])-1) < 1e-4
        data.qpos[addresses] = frame['joint_positions']
        mujoco.mj_forward(model, data)
        camera.lookat[:] = data.qpos[:3]
        camera.lookat[2] = .22
        renderer.update_scene(data, camera=camera, scene_option=options)
        image = Image.fromarray(renderer.render())
        draw = ImageDraw.Draw(image)
        draw.rectangle((0,0,960,77), fill=(17,24,34))
        draw.text((22,12), args.title, font=font, fill='white')
        draw.text((22,44), f'Time {i/fps:05.2f} s    Forward {frame["body_linear_velocity"][0]:.2f} m/s    Height {frame["position"][2]:.3f} m', font=font, fill=(160,215,238))
        draw.rectangle((0,511,960,540), fill=(17,24,34))
        speed_label = 'Real time' if args.slowdown == 1 else f'{args.slowdown}x slow motion'
        draw.text((22,517), f'Recorded Newton trajectory | {speed_label} | Env {args.env} / eval seed {record["eval_seed"]}', font=small, fill=(205,213,224))
        for _ in range(args.slowdown):
            writer.append_data(np.asarray(image))
        if i in {0,len(frames)//4,len(frames)//2,3*len(frames)//4,len(frames)-1}:
            image.save(out / f'preview_{i:04d}.png')
        if i % 200 == 0:
            print(f'Rendered {i}/{len(frames)}', flush=True)
reader = imageio.get_reader(output)
meta = reader.get_meta_data()
count = reader.count_frames()
reader.close()
assert count == len(frames)*args.slowdown and abs(meta['duration']-len(frames)/fps*args.slowdown) < .05
report = {'video': output.name, 'frames': count, 'fps': meta['fps'], 'duration_seconds': meta['duration'], 'resolution': meta['size'], 'source_sha256': hashlib.sha256(source.read_bytes()).hexdigest(), 'joint_order': joint_names, 'quaternion_conversion': 'recorded xyzw to MuJoCo wxyz', 'method': 'Recorded poses only; mj_forward for visual kinematics, no mj_step. Menagerie Go2 visual model, not original Isaac renderer or a new simulation.', 'visual_model_source': 'https://github.com/google-deepmind/mujoco_menagerie/tree/main/unitree_go2'}
(out / f'{stem}_metadata.json').write_text(json.dumps(report, indent=2), encoding='utf-8')
print(json.dumps(report), flush=True)
