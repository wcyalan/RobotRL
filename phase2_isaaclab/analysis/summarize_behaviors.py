"""Apply predeclared Go2 behavior acceptance thresholds to all recorded lanes."""
import hashlib
import json
from pathlib import Path
from behavior_acceptance import assess_behavior

ROOT = Path(__file__).resolve().parents[2]
out = ROOT/'experiments/phase2_go2_behavior/run_001'
results = []
for seed, model_run in [(42,'run_001'),(0,'run_002'),(1,'run_003')]:
    model=next((ROOT/'experiments/phase2_go2_baseline'/model_run).glob('logs/**/model_299.pt'))
    sha=hashlib.sha256(model.read_bytes()).hexdigest()
    for profile in ['turn_left','turn_right','stop_restart']:
        path=out/f'seed_{seed}_{profile}.json'
        if not path.exists():continue
        data=json.loads(path.read_text(encoding='utf-8'))
        assert data['sha256']==sha and data['eval_seed']==40000 and len(data['episodes'])==32
        result = assess_behavior(data, profile)
        results.append({'seed':seed, 'behavior':profile, **result,
                        'source_sha256':hashlib.sha256(path.read_bytes()).hexdigest()})
report={'completed_conditions':len(results),'required_conditions':9,
        'all_passed':len(results)==9 and all(r['passed'] for r in results),'results':results,
        'video_review':'not implied by numerical acceptance; inspect representative and failing trajectories separately'}
out.mkdir(parents=True,exist_ok=True)
(out/'acceptance.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
print(json.dumps({**report,'results':[{k:v for k,v in r.items() if k!='rows'} for r in results]},indent=2))
