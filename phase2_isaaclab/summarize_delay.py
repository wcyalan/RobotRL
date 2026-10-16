"""Verify command latency and saved policy hashes, then summarize delay sensitivity."""
from pathlib import Path
import json
import hashlib
import statistics

ROOT = Path(__file__).resolve().parents[1]
out = ROOT/'experiments/phase2_go2_robustness/run_004'
rows=[]
for seed,run in [(0,'run_002'),(1,'run_003'),(42,'run_001')]:
    model=next((ROOT/'experiments/phase2_go2_baseline'/run).glob('logs/**/model_299.pt'))
    sha=hashlib.sha256(model.read_bytes()).hexdigest()
    for delay in [0,1,2]:
        d=json.loads((out/f'seed_{seed}_delay_{delay}.json').read_text(encoding='utf-8'))
        assert d['sha256']==sha and d['eval_seed']==20000 and len(d['episodes'])==16
        assert d['action_delay']['steps']==delay and d['action_delay']['milliseconds']==20*delay
        trace=d['action_delay']['first_eight_steps_all_envs']
        for i,t in enumerate(trace):
            expected=trace[i-delay]['commanded'] if i>=delay else [[0]*len(row) for row in t['applied']]
            assert t['applied']==expected
        rows.append({'seed':seed,'delay_ms':delay*20,'speed':d['settled_diagnostics']['forward_speed_m_s'],
                     'xy_error':d['mean_xy_velocity_error_m_s'],'height':d['settled_diagnostics']['root_height_m'],
                     'terminated':sum(e['terminated'] for e in d['episodes']),
                     'timed_out':sum(e['time_out'] for e in d['episodes']),
                     'mean_length_seconds':d['mean_length_seconds']})
aggregate=[]
for delay in [0,20,40]:
    group=[r for r in rows if r['delay_ms']==delay]
    aggregate.append({'delay_ms':delay,**{k:statistics.mean(r[k] for r in group) for k in ['speed','xy_error','height']},
                      'terminated':sum(r['terminated'] for r in group),'episodes':48})
summary={'status':'completed','checkpoint_hashes_verified':True,'first_eight_steps_delay_verified':True,
         'results':rows,'aggregate':aggregate,
         'limitations':'One evaluation seed; only added command delay tested, not sensor latency. Applied previous-action observation; zero queue at reset. No direct foot-slip or real-world validation.'}
(out/'summary.json').write_text(json.dumps(summary,indent=2),encoding='utf-8')
p=out/'protocol.json'; d=json.loads(p.read_text()); d['status']='completed; 9 evaluations and command-delay trace verification passed'; p.write_text(json.dumps(d,indent=2))
lines=['# Go2 动作延迟敏感性测试','','三个训练 seed（0、1、42），评估 seed=20000，固定前进指令 0.5 m/s，标称 base 质量。模型权重不变。','',
'| 额外动作延迟 | 平均前进速度 m/s | 平均 XY 速度误差 m/s | 平均机身高度 m | 提前终止 |',
'|---|---:|---:|---:|---:|']
for r in aggregate: lines.append(f"| {r['delay_ms']} ms | {r['speed']:.3f} | {r['xy_error']:.3f} | {r['height']:.3f} | {r['terminated']}/48 |")
lines+=['','144 局全部完成 20 秒；三个 seed 均随额外延迟增加而出现跟踪误差增大、机身降低。没有提前终止不等于性能没有退化。','',
'模型 SHA-256 已核验；各记录前 8 步所有环境的实际动作均与延迟队列预期一致。重置时队列填零，零动作对应默认关节目标；观测中的上一动作来自实际执行动作。','',
'误差统计包含启动阶段，姿态/平均前进速度排除首秒。只测试额外动作延迟，未添加观测延迟；只有一个评估 seed，未直接测量打滑，不作为真机时延承受能力保证。', '',
'该结果可作为后续延迟随机化训练的对照；当前尚未修改或重训策略。']
(out/'README.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
print(json.dumps(aggregate,indent=2))
