"""Verify saved configs/checkpoints and summarize matched baseline evaluations."""
from pathlib import Path
import hashlib
import json
import re
import statistics

ROOT = Path(__file__).resolve().parents[1]
base = ROOT / 'experiments/phase2_go2_baseline'
rows = []
reference = {}
for name, seed in [('run_001',42),('run_002',0),('run_003',1)]:
    run = base / name
    for cfg in ['agent','env']:
        text = next(run.glob(f'logs/**/params/{cfg}.yaml')).read_text(encoding='utf-8')
        assert re.search(rf'^seed: {seed}$',text,re.M)
        normalized = re.sub(r'^(seed|log_dir):.*\n?', '', text, flags=re.M)
        if cfg not in reference: reference[cfg] = normalized
        assert normalized == reference[cfg], (name,cfg)
    row = {'run':name,'seed':seed,'training_seconds':float(re.search(r'Training time: ([\d.]+)',(run/'training.log').read_text(encoding='utf-8')).group(1))}
    model = next(run.glob('logs/**/model_299.pt'))
    sha = hashlib.sha256(model.read_bytes()).hexdigest()
    for kind in ['random','forward']:
        d = json.loads((run/f'{kind}_299.json').read_text(encoding='utf-8'))
        assert d['sha256'] == sha and d['eval_seed'] == 10000 and len(d['episodes']) == 16
        row[kind] = {k:d[k] for k in ['return_mean','return_std','mean_xy_velocity_error_m_s','mean_yaw_velocity_error_rad_s','termination_rate','mean_length_seconds']}
        row[kind].update({k:d['settled_diagnostics'][k] for k in ['forward_speed_m_s','root_height_m','upright_cosine']})
        row[kind]['terminated_episodes'] = [e for e in d['episodes'] if e['terminated']]
    rows.append(row)
report = {'configuration_check':'Agent and environment YAMLs identical except seed and log directory.',
          'steps_per_run':29491200,'evaluation_seed':10000,'episodes_per_policy_per_condition':16,
          'limitations':['Three training seeds only; no real-robot validation.',
                         'Random-command evaluation shares initialization seed, not guaranteed identical full trajectories.',
                         'Video contains env 0 only; failed env 11 trajectory was not saved.',
                         'Velocity and posture diagnostics exclude first second; episode return and velocity errors do not.'],
          'random_return_across_seed_mean':statistics.mean(r['random']['return_mean'] for r in rows),
          'random_return_across_seed_population_std':statistics.pstdev(r['random']['return_mean'] for r in rows),
          'results':rows}
(base/'comparison_001.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
lines=['# Go2 官方基线：三个训练 seed 对比','',
       '配置核验：actor/critic 与环境 YAML 除训练 seed、日志目录外完全一致。每组 4096 环境、300 轮、29,491,200 步；最终 model_299 权重 SHA-256 与评估记录一致。','',
       '独立评估：固定评估 seed=10000，每种条件各 16 局；随机指令和固定 0.5 m/s 前进。','',
       '| 运行 | seed | 训练秒数 | 随机指令回报 | 随机指令 XY 误差 m/s | 固定前进均速 m/s | 前进机身高度 m | 前进提前终止 |',
       '|---|---:|---:|---:|---:|---:|---:|---:|']
for r in rows:
    lines.append(f"| [{r['run']}]({r['run']}/) | {r['seed']} | {r['training_seconds']:.2f} | {r['random']['return_mean']:.2f} | {r['random']['mean_xy_velocity_error_m_s']:.3f} | {r['forward']['forward_speed_m_s']:.3f} | {r['forward']['root_height_m']:.3f} | {len(r['forward']['terminated_episodes'])}/16 |")
lines += ['', '三组随机指令评估均无 base 接触提前终止；seed=1 的固定前进评估在 env 11 于 2.02 秒发生 base 接触终止，其姿态指标也提示翻转。其余两组固定前进均完成。不能据此宣布完全稳定，也不能通过挑选成功 seed 消除失败记录。', '',
          '下一步：对 seed=1 做相同条件的复测，保存失败环境的逐帧状态，并扩展独立评估 seed；确认失败模式后再设计鲁棒性实验。', '',
          '现有视频只展示 env 0，不能作为 env 11 失败过程的可视化证据。视频为记录轨迹的外观渲染，不是新的闭环仿真。', '',
          '随机指令使用相同初始评估 seed，但状态与随机数消费会随策略变化；固定前进指令比较更直接。三个训练 seed 只是初步重复性证据。']
(base/'README.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
batch=base/'seed_batch_001.json'
d=json.loads(batch.read_text(encoding='utf-8')); d['status']='completed; both training runs and all evaluations passed; one final-policy episode terminated early'
d['comparison']='comparison_001.json'
batch.write_text(json.dumps(d,indent=2),encoding='utf-8')
print(json.dumps({'configs_verified':True,'checkpoint_hashes_verified':True,'results':rows},indent=2))
