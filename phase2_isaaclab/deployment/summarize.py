"""Audit exported actor equivalence and standalone simulation acceptance."""
import hashlib
import json
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'phase2_isaaclab/analysis'))
from behavior_acceptance import assess_behavior
OUT=ROOT/'experiments/phase2_go2_deployment/run_001'
rows,comparisons,missing=[],[],[]
for seed,run in enumerate(['run_001','run_002']):
    folder=OUT/run
    manifest=json.loads((folder/'manifest.json').read_text())
    assert hashlib.sha256((folder/'policy.pt').read_bytes()).hexdigest()==manifest['policy_sha256']
    for name,sha in manifest['config_sha256'].items():
        assert hashlib.sha256((folder/name).read_bytes()).hexdigest()==sha
    for name in ['equivalence']+[f'eval_{ev}/{b}' for ev in [45000,46000] for b in ['turn_left','turn_right','stop_restart','forward']]:
        path=folder/(name+'.json')
        if not path.exists():
            missing.append(path.relative_to(OUT).as_posix());continue
        d=json.loads(path.read_text()); e=d['export_validation']
        assert d['sha256']==manifest['checkpoint_sha256'] and e['jit_sha256']==manifest['policy_sha256']
        if name=='equivalence':
            assert e['official_comparison'] and e['ppo_runner_constructed'] and e['compared_observations']>=32000
            comparisons.append({'training_seed':seed,**e})
            continue
        ev=int(name.split('/')[0][5:]); behavior=name.split('/')[1]
        assert not e['ppo_runner_constructed'] and not e['official_comparison'] and d['eval_seed']==ev
        reference=json.loads((ROOT/'experiments/phase2_go2_stop_sequence_extended/validation_001'/f'eval_{ev}'/run/f'coverage_{behavior}.json').read_text())
        assert d['scenario']==reference['scenario'] and len(d['episodes'])==32
        result=assess_behavior(d,behavior) if behavior!='forward' else {'passed':None,'passed_episodes':None}
        if behavior=='forward': assert d['forward_speed_command_m_s']==.5
        rows.append({'training_seed':seed,'eval_seed':ev,'behavior':behavior,
            'source_sha256':hashlib.sha256(path.read_bytes()).hexdigest(),
            'xy_error':d['mean_xy_velocity_error_m_s'],'original_xy_error':reference['mean_xy_velocity_error_m_s'],
            'termination_rate':d['termination_rate'],**result})
report={'complete':not missing,'missing':missing,'equivalence':comparisons,'conditions':len(rows),
        'all_behavior_conditions_passed':not missing and all(r['passed'] for r in rows if r['behavior']!='forward'), 'results':rows}
(OUT/'acceptance.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
print('Complete',report['complete'],'conditions',len(rows),'/16; official comparisons',len(comparisons),'/2')
for r in rows: print(r['training_seed'],r['eval_seed'],r['behavior'],r['passed_episodes'],r['xy_error'],r['termination_rate'])
