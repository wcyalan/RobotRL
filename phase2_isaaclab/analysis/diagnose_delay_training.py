"""Summarize recorded training windows and failure trajectories without refitting."""
import json
import math
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
out = ROOT/'experiments/phase2_go2_delay_training/run_001'
chunks = re.split(r'Learning iteration ', (out/'training.log').read_text(encoding='utf-8'))[1:]
assert len(chunks) == 300
keys = ['Mean reward', 'Mean episode length', 'Metrics/base_velocity/error_vel_xy', 'Episode_Termination/base_contact']
windows = []
for start in [0, 50, 100, 150, 200, 250, 280]:
    group = chunks[start:start+20]
    windows.append({'iterations': [start, start+19], **{
        key: sum(float(re.search(re.escape(key)+r':\s*([-\d.]+)', c)[1]) for c in group)/len(group)
        for key in keys}})
failures = []
for path, env in [(out/'delay_2.json', 0), (out/'failure_replay_001/evaluation.json', 15)]:
    data = json.loads(path.read_text(encoding='utf-8'))
    original = json.loads((out/('delay_1.json' if env == 15 else 'delay_2.json')).read_text(encoding='utf-8'))
    assert data['sha256'] == original['sha256']
    for key in ['eval_seed', 'scenario', 'forward_speed_command_m_s', 'step_dt']:
        assert data[key] == original[key]
    assert data['action_delay']['steps'] == original['action_delay']['steps']
    frames = data[f'trajectory_env_{env}']
    points = []
    for i in sorted(set([0, len(frames)//2, max(0,len(frames)-101), max(0,len(frames)-51), len(frames)-1])):
        row = frames[i]
        x, y, z, w = row['quaternion_xyzw']
        upright = max(-1, min(1, 1-2*(x*x+y*y)))
        points.append({'time_s': i*data['step_dt'], 'height_m': row['position'][2],
                       'tilt_degrees': math.degrees(math.acos(upright)),
                       'body_velocity': row['body_linear_velocity']})
    episode = next(e for e in data['episodes'] if e['env_index'] == env)
    failures.append({'source': str(path.relative_to(out)), 'episode': episode, 'points': points})
(out/'diagnosis.json').write_text(json.dumps({'training_windows':windows,'failures':failures},indent=2), encoding='utf-8')
lines = ['# 延迟随机化训练诊断', '', '## 学习进度', '',
         '下表为训练日志连续 20 轮的均值；训练使用随机指令和随机延迟，不能与固定前进评估数值直接横比。', '',
         '| 轮数（从 0 起） | 回报 | episode 步数 | XY 速度误差 | base_contact 日志值 |',
         '|---|---:|---:|---:|---:|']
for row in windows:
    lines.append(f"| {row['iterations'][0]}–{row['iterations'][1]} | {row[keys[0]]:.2f} | {row[keys[1]]:.1f} | {row[keys[2]]:.3f} | {row[keys[3]]:.3f} |")
lines += ['', '末段回报仍上升、速度误差仍下降，说明延长训练值得作为下一项受控实验；这些趋势不能证明延长后必然优于基线。', '', '## 失败轨迹', '']
for failure in failures:
    ep = failure['episode']
    lines += [f"### {failure['source']}，环境 {ep['env_index']}", '',
              f"提前终止：{ep['terminated']}；episode 长度 {ep['length_seconds']:.2f} s。", '',
              '| 时刻 s | 机身高度 m | 相对竖直方向倾角 ° |', '|---|---:|---:|']
    lines += [f"| {r['time_s']:.2f} | {r['height_m']:.3f} | {r['tilt_degrees']:.1f} |" for r in failure['points']]
    lines += ['']
lines += ['20 ms 原始评估中环境 15 在 14.14 s 提前终止；这次相同模型、评估 seed 和配置下完成 20 s，未复现失败。原始结果不被覆盖，也不能因此宣称问题解决。复现轨迹不是原始失败轨迹；非确定性是可能原因，尚未定位差异来源。', '',
          '40 ms 原始轨迹的机身倾角在 1.52 s 达到 20.7°，2.52 s 达到 177.1°、高度约 0.061 m，支持翻倒的判断。', '',
          '轨迹只包含 step 前姿态，最后一个触发终止的 step 后姿态因自动 reset 未记录。', '',
          '当前记录不足以把失稳唯一归因于延迟、推力、初始化或关节动作；不据此调整奖励。下一项建议保持模型结构和奖励不变，把延迟随机化训练延长，并保留 300 轮结果作为对照。单独增加训练预算后的提升不能解释为同预算算法优势。']
(out/'diagnosis.md').write_text('\n'.join(lines)+'\n', encoding='utf-8')
print(json.dumps({'windows': windows[-2:], 'failures': failures},indent=2))
