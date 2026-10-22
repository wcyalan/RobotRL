"""Render saved G1 states: six independent episodes, two policies side by side.

No training, policy inference or mj_step. Requires mujoco, numpy, Pillow,
imageio and imageio-ffmpeg. Inputs are the run_003 frozen model/meshes and
preselected eval_seed=20000, lane=0 traces (one per policy and condition).
"""
import argparse
import hashlib
import json
from pathlib import Path
import xml.etree.ElementTree as ET

CONDITIONS = ('stand', 'forward_low', 'forward_medium', 'turn_left', 'turn_right', 'stop_restart')
LABELS = ('Stand', 'Forward 0.3 m/s', 'Forward 0.6 m/s', 'Turn left', 'Turn right', 'Stop and restart')


def frame_indices(steps, dt, fps):
    stride = round(1 / (dt * fps))
    if stride < 1 or abs(stride * dt * fps - 1) > 1e-9 or steps % stride:
        raise ValueError('FPS must evenly sample saved policy steps')
    return range(0, steps, stride)


def command_at(case, time):
    for start, end, command in case['segments']:
        if start <= time < end:
            return command
    raise ValueError('Time outside command schedule')


def load_json(path):
    return json.loads(path.read_text(encoding='utf-8'))


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def render(inputs, output, fps=25, preview_only=False):
    import importlib.metadata as metadata
    import numpy as np
    import mujoco
    import imageio.v2 as imageio
    from PIL import Image, ImageDraw, ImageFont

    if output.exists():
        raise FileExistsError(f'Refusing to overwrite {output}')
    manifest = load_json(inputs / 'manifest.json')
    for relative, expected in manifest['files'].items():
        path = (inputs / relative).resolve()
        path.relative_to(inputs.resolve())
        if sha256(path) != expected:
            raise ValueError(f'Input hash mismatch: {relative}')
    plan = load_json(inputs / 'plan.json')
    if plan['eval_seed'] != 20000 or plan['lane'] != 0 or plan['training_seeds'] != [0, 1]:
        raise ValueError('Expected preselected seeds 0/1, eval 20000, lane 0')
    # Add a visual-only checkerboard so translation stays visible with a
    # following camera. Keep the frozen file and all physical parameters intact.
    xml = ET.parse(inputs / 'model/g1.xml')
    asset = xml.getroot().find('asset')
    for mesh in asset.findall('mesh'):
        mesh.set('file', str((inputs/'model'/mesh.get('file')).resolve()))
    ET.SubElement(asset, 'texture', name='demo_grid', type='2d', builtin='checker',
                  rgb1='.22 .27 .31', rgb2='.32 .38 .42', width='512', height='512')
    ET.SubElement(asset, 'material', name='demo_floor', texture='demo_grid',
                  texrepeat='2 2', texuniform='true')
    ground = xml.getroot().find('.//geom[@name="ground"]')
    ground.set('material', 'demo_floor')
    ground.set('rgba', '1 1 1 1')
    model = mujoco.MjModel.from_xml_string(ET.tostring(xml.getroot(), encoding='unicode'))
    interface = load_json(inputs / 'model/interface.json')
    dt = interface['policy_dt']
    data = [mujoco.MjData(model), mujoco.MjData(model)]
    root_id = model.body(interface['root_body']).id
    # A visual replay must never simulate a new outcome.
    renderers = [mujoco.Renderer(model, height=480, width=640) for _ in range(2)]
    cameras = [mujoco.MjvCamera() for _ in range(2)]
    option = mujoco.MjvOption()
    option.geomgroup[3] = 0
    for camera in cameras:
        camera.distance, camera.azimuth, camera.elevation = 2.8, 135, -18
    font = ImageFont.load_default(size=20)
    small = ImageFont.load_default(size=16)
    title_font = ImageFont.load_default(size=28)
    output.mkdir(parents=True)
    writer = None
    if not preview_only:
        writer = imageio.get_writer(output / 'g1_multicommand_replay.mp4', fps=fps,
                                   codec='libx264', quality=7, macro_block_size=16)
    samples, chapters, frame_count = [], [], 0
    try:
        for index, (condition, label) in enumerate(zip(CONDITIONS, LABELS)):
            case = plan['protocol']['conditions'][condition]
            episodes = [plan['episodes'][f'{seed}/{condition}'] for seed in (0, 1)]
            traces = []
            for seed in (0, 1):
                with np.load(inputs / f'traces/seed_{seed}_{condition}.npz', allow_pickle=False) as source:
                    traces.append({k: source[k] for k in ('qpos', 'qvel', 'metrics', 'command')})
            steps = round(case['duration_s'] / dt)
            for trace in traces:
                if trace['qpos'].shape != (steps, model.nq) or trace['qvel'].shape != (steps, model.nv):
                    raise ValueError('Incomplete selected trajectory')
                if trace['metrics'].shape != (steps, 3):
                    raise ValueError('Invalid metric shape')
                if not all(np.isfinite(v).all() for v in trace.values()):
                    raise ValueError('Nonfinite trajectory')
            indices = frame_indices(steps, dt, fps)
            sample_step = round((18 if condition == 'stop_restart' else 10) / dt)
            for step in ([sample_step] if preview_only else indices):
                t = step * dt
                command = command_at(case, t)
                canvas = Image.new('RGB', (1280, 720), '#111827')
                draw = ImageDraw.Draw(canvas)
                draw.text((24, 14), f'G1 | {index+1}/6  {label}', font=title_font, fill='white')
                draw.text((24, 54), 'Saved MuJoCo evaluation replay | eval 20000, lane 0 | no new physics', font=small, fill='#b8c3d9')
                draw.text((24, 81), f'Episode time {t:05.2f}s / {case["duration_s"]}s   Command: vx={command[0]:+.1f} m/s, yaw={command[2]:+.1f} rad/s', font=font, fill='#7dd3fc')
                for seed in (0, 1):
                    trace = traces[seed]
                    np.testing.assert_allclose(trace['command'][step], command, atol=1e-7, rtol=0)
                    data[seed].qpos[:] = trace['qpos'][step]
                    data[seed].qvel[:] = trace['qvel'][step]
                    mujoco.mj_forward(model, data[seed])
                    cameras[seed].lookat[:] = data[seed].xpos[root_id]
                    renderers[seed].update_scene(data[seed], camera=cameras[seed], scene_option=option)
                    canvas.paste(Image.fromarray(renderers[seed].render()), (640*seed, 120))
                    x = seed*640+20
                    draw.rectangle((x-8, 128, x+240, 160), fill='#111827')
                    draw.text((x, 132), f'Training seed {seed}', font=font, fill='white')
                    metric = trace['metrics'][step]
                    draw.text((x, 608), f'XY error {metric[0]:.3f} m/s | yaw error {metric[1]:.3f} rad/s', font=small, fill='white')
                    count = plan['groups'][f'{seed}/{condition}']['pass_count']
                    draw.text((x, 636), f'Group: {count}/32 passed | shown episode: '+('PASS' if episodes[seed]['episode_passed'] else 'FAIL'),
                              font=font, fill='#fbbf24' if count < 29 else '#86efac')
                draw.text((24, 677), 'Independent episodes: reset between chapters. Stop/restart is one continuous 30s episode.', font=small, fill='#b8c3d9')
                if step == sample_step:
                    samples.append(canvas.copy())
                if writer is not None:
                    writer.append_data(np.asarray(canvas))
                    frame_count += 1
            chapters.append(dict(condition=condition, start_s=sum(c['duration_s'] for c in chapters),
                                 duration_s=case['duration_s'], expected_frames=len(indices)))
            print(f'RENDERED {condition}', flush=True)
        # Six half-size chapter frames in two columns, three rows.
        sheet = Image.new('RGB', (1280, 1080), '#111827')
        for index, sample in enumerate(samples):
            sheet.paste(sample.resize((640, 360)), ((index % 2)*640, (index // 2)*360))
        sheet.save(output / 'preview.png')
    finally:
        if writer is not None:
            writer.close()
        for renderer in renderers:
            renderer.close()
    result = dict(completed=True, kind='kinematic replay of saved native-MuJoCo evaluation states',
                  new_physics_steps=0, new_training=False, fps=fps, width=1280, height=720,
                  frame_count=frame_count, duration_s=frame_count/fps, preview_only=preview_only,
                  selection=dict(training_seeds=[0, 1], eval_seed=20000, lane=0, rule='fixed before rendering; not best episode selection'),
                  chapters=chapters, renderer_sha256=sha256(Path(__file__)),
                  visual_only_changes=['checkerboard ground material', 'following camera', 'text overlays'],
                  input_manifest_sha256=sha256(inputs/'manifest.json'),
                  versions={name: metadata.version(name) for name in ('mujoco', 'numpy', 'Pillow', 'imageio', 'imageio-ffmpeg')},
                  limits=['Six independent episodes, not a single continuous six-command rollout.',
                          'Stop/restart shows an actual failed acceptance case; thresholds unchanged.',
                          'Pose replay does not revalidate contacts, dynamics or hardware safety.'])
    if not preview_only and frame_count != 3250:
        raise ValueError(f'Unexpected video frame count: {frame_count}')
    (output/'metadata.json').write_text(json.dumps(result, indent=2)+'\n', encoding='utf-8')
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--inputs', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--preview-only', action='store_true')
    args = parser.parse_args()
    print(json.dumps(render(args.inputs, args.output, preview_only=args.preview_only)))
