"""Validate active contact evidence and compare corrected low-friction evaluations."""
from pathlib import Path
import json
import hashlib

root = Path(__file__).resolve().parents[2]
base = root/'experiments/phase2_go2_robustness'
out = base/'run_003'
rows=[]
for seed,run in [(0,'run_002'),(1,'run_003'),(42,'run_001')]:
    d=json.loads((out/f'seed_{seed}.json').read_text(encoding='utf-8'))
    nominal=json.loads((base/'run_001'/f'seed_{seed}_nominal.json').read_text(encoding='utf-8'))
    model=next((root/'experiments/phase2_go2_baseline'/run).glob('logs/**/model_299.pt'))
    assert hashlib.sha256(model.read_bytes()).hexdigest()==d['sha256']==nominal['sha256']
    assert len(d['episodes'])==16 and d['eval_seed']==20000
    contacts=0
    for audit in d['friction_audit']:
        a=audit['active_contacts']
        assert a['ground_count']>0
        assert all(abs(mu-.3)<1e-5 for mu in a['ground_sliding_friction'])
        contacts+=a['ground_count']
    row={'seed':seed,'nominal_speed':nominal['settled_diagnostics']['forward_speed_m_s'],
         'low_friction_speed':d['settled_diagnostics']['forward_speed_m_s'],
         'nominal_xy_error':nominal['mean_xy_velocity_error_m_s'],
         'low_friction_xy_error':d['mean_xy_velocity_error_m_s'],
         'height':d['settled_diagnostics']['root_height_m'],
         'terminated':sum(e['terminated'] for e in d['episodes']),
         'timed_out':sum(e['time_out'] for e in d['episodes']),
         'audited_ground_contacts':contacts}
    rows.append(row)
summary={'status':'completed','results':rows,'contact_friction_verified':0.3,
         'audit_steps':[51,501],'note':'All constraint-eligible ground contacts at the two sampled steps have mu=0.3; not a continuous-time contact audit. Only one evaluation seed; no slip metric or real-world validation.'}
(out/'summary.json').write_text(json.dumps(summary,indent=2),encoding='utf-8')
p=out/'protocol.json'; protocol=json.loads(p.read_text()); protocol['status']='completed; valid ground contacts verified at both snapshots'; p.write_text(json.dumps(protocol,indent=2))
lines=['# 修正后的低摩擦评估','','三个固定最终模型；0.5 m/s 前进，评估 seed=20000，每组 16 局；机器人和地面摩擦均为 0.3，base 质量为标称值。其余随机化与推扰保持原协议。', '',
'| 训练 seed | 对照速度 | 低摩擦速度 | 对照 XY 误差 | 低摩擦 XY 误差 | 提前终止 |',
'|---|---:|---:|---:|---:|---:|']
for r in rows:
    lines.append(f"| {r['seed']} | {r['nominal_speed']:.3f} | {r['low_friction_speed']:.3f} | {r['nominal_xy_error']:.3f} | {r['low_friction_xy_error']:.3f} | {r['terminated']}/16 |")
lines+=['','速度和误差单位均为 m/s。原始 seed_*.json 包含每局结果。模型 SHA-256 与原基线核验一致。', '',
'核验方式：按 nacon 截取有效槽位，筛选正接触维数、distance < includemargin 的接触，按 plane 类型识别地面，保存 world、geom/body 名称。第 51、501 步全部被筛选出的有效地面接触均为 0.3。', '',
'限定：这是两个时刻的核验，不是逐帧接触审计；仅一个评估 seed。未直接统计脚部滑移，不能把完成率直接等同于完全无打滑或真机可部署。旧 run_001 的 robot-only low_friction 仍标记为无效低接触摩擦实验。']
(out/'README.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
print(json.dumps(summary,indent=2))
