"""Compare the seed-42 delay intervention with its unchanged baseline."""
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
out = ROOT / 'experiments/phase2_go2_delay_training/run_001'
baseline = ROOT / 'experiments/phase2_go2_robustness/run_004'
assert (out / 'completed.txt').read_text().strip() == 'ROBOTRL_DELAY_TRAINING_PASSED'
model = next(out.glob('logs/**/model_299.pt'))
sha = hashlib.sha256(model.read_bytes()).hexdigest()
rows = []
for delay in range(3):
    pair = []
    for label, path in [('baseline', baseline / f'seed_42_delay_{delay}.json'),
                        ('randomized', out / f'delay_{delay}.json')]:
        data = json.loads(path.read_text(encoding='utf-8'))
        assert data['eval_seed'] == 20000 and len(data['episodes']) == 16
        assert data['action_delay']['steps'] == delay
        if label == 'randomized':
            assert data['sha256'] == sha
        trace = data['action_delay']['first_eight_steps_all_envs']
        for i, step in enumerate(trace):
            expected = trace[i-delay]['commanded'] if i >= delay else [[0]*len(a) for a in step['applied']]
            assert step['applied'] == expected
        pair.append({'policy': label, 'speed': data['settled_diagnostics']['forward_speed_m_s'],
                     'xy_error': data['mean_xy_velocity_error_m_s'],
                     'height': data['settled_diagnostics']['root_height_m'],
                     'terminated': sum(e['terminated'] for e in data['episodes'])})
    rows.append({'delay_ms': delay*20, 'results': pair,
                 'error_reduction_percent': 100*(1-pair[1]['xy_error']/pair[0]['xy_error'])})
(out / 'summary.json').write_text(json.dumps({'model_sha256': sha, 'results': rows}, indent=2), encoding='utf-8')
lines = ['# Go2 延迟随机化训练：seed=42', '',
         '训练时每个环境在每局开始随机选择 0、20 或 40 ms 动作延迟，整局保持不变；重置时清空队列。4096 个环境、300 轮，共 29,491,200 步。', '',
         '评估沿用 seed=20000、0.5 m/s 前进指令和标称场景；每个延迟各 16 局。', '',
         '| 延迟 | 原策略速度 | 新策略速度 | 原 XY 误差 | 新 XY 误差 | 误差降低 | 原/新提前终止 |',
         '|---|---:|---:|---:|---:|---:|---:|']
for row in rows:
    a, b = row['results']
    lines.append(f"| {row['delay_ms']} ms | {a['speed']:.3f} | {b['speed']:.3f} | {a['xy_error']:.3f} | {b['xy_error']:.3f} | {row['error_reduction_percent']:.1f}% | {a['terminated']}/16、{b['terminated']}/16 |")
improved = all(r['error_reduction_percent'] > 0 for r in rows)
lines += ['', '速度、误差单位均为 m/s。误差降低为负表示退化。速度排除首秒，误差包含启动阶段。', '',
          '本轮三个条件的误差均降低。' if improved else '本轮未取得三个条件下一致改善，不能替换原始基线。具体退化或改善应逐行比较。', '',
          '短训练 72 步和正式训练 7200 步的延迟审计通过，模型哈希与评估动作队列通过核验。agent.yaml 与原始 seed=42 基线完全相同，env.yaml 仅日志路径不同；延迟由外层包装器实现。', '',
          '这是单个训练 seed 的初步干预实验，评估 seed 已用于前期诊断，不能视为独立最终验收。尚未验证其他训练 seed、未见评估 seed、传感器延迟或真机表现。']
(out / 'README.md').write_text('\n'.join(lines)+'\n', encoding='utf-8')
protocol = json.loads((out / 'protocol.json').read_text(encoding='utf-8'))
protocol['status'] = 'completed; smoke and training audits passed; three fixed-delay evaluations verified'
(out / 'protocol.json').write_text(json.dumps(protocol, indent=2), encoding='utf-8')
print(json.dumps(rows, indent=2))
