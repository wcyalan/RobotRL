"""Post-hoc episode and recorded-trajectory diagnostics; never overwrite evaluations."""
import json
import hashlib
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
root = ROOT/'experiments/phase2_go2_mass_friction'
output = root/'diagnostic_001'
output.mkdir(exist_ok=True)
results = {'scope':'Post-hoc diagnostic. Recovery metrics use only stored env 0, not all 32 episodes.',
           'recovery_rule':'After 5 s, first 0.5 s consecutive XY error below 0.1 m/s, searched until 10 s.',
           'heavy_high':[], 'push_env0':[], 'source_sha256':{}}
for run in ['run_001','run_002','run_003','run_004']:
    path = root/run/'heavy_high.json'
    d = json.loads(path.read_text())
    errors = np.array([e['mean_xy_velocity_error_m_s'] for e in d['episodes']])
    results['source_sha256'][str(path.relative_to(root))] = hashlib.sha256(path.read_bytes()).hexdigest()
    results['heavy_high'].append({'run':run,'mean_error':float(errors.mean()),
        'median_error':float(np.median(errors)), 'min_error':float(errors.min()),'max_error':float(errors.max()),
        'episodes_above_0_2': [e for e in d['episodes'] if e['mean_xy_velocity_error_m_s']>.2],
        'note':'0.2 m/s threshold chosen after inspecting data, for diagnosis only.'})
    for condition in ['nominal','nominal_push']:
        path=root/run/(condition+'.json');d=json.loads(path.read_text())
        results['source_sha256'][str(path.relative_to(root))]=hashlib.sha256(path.read_bytes()).hexdigest()
        frames=d['trajectory_env_0'];dt=d['step_dt'];assert dt==.02 and len(frames)==1000
        v=np.array([f['body_linear_velocity'] for f in frames]);q=np.array([f['quaternion_xyzw'] for f in frames])
        error=np.linalg.norm(v[:,:2]-[.5,0],axis=1)
        tilt=np.degrees(np.arccos(np.clip(1-2*(q[:,0]**2+q[:,1]**2),-1,1)))
        recovery=next((round(i*dt-5,3) for i in range(251,476) if np.all(error[i:i+25]<.1)),None)
        results['push_env0'].append({'run':run,'condition':condition,'env':0,
            'pre_3_to_5s_mean_error':float(error[150:250].mean()),
            'post_5_to_7s_peak_error':float(error[250:350].max()),
            'post_5_to_5_5s_mean_error':float(error[250:275].mean()),
            'post_5_5_to_7s_mean_error':float(error[275:350].mean()),
            'post_5_to_7s_peak_tilt_degrees':float(tilt[250:350].max()),
            'time_to_0_5s_below_threshold_s':recovery,
            'note':'The nominal row is an unpushed comparison; its threshold time is not a recovery time.'})
for run in ['run_002','run_004']:
    path=output/run/'heavy_high.json'
    if not path.exists():continue
    d=json.loads(path.read_text());original=json.loads((root/run/'heavy_high.json').read_text())
    assert d['sha256']==original['sha256'] and d['scenario']==original['scenario'] and d['eval_seed']==30000
    frames=d['trajectory_env_25'];v=np.array([f['body_linear_velocity'] for f in frames]);q=np.array([f['quaternion_xyzw'] for f in frames])
    tilt=np.degrees(np.arccos(np.clip(1-2*(q[:,0]**2+q[:,1]**2),-1,1)))
    replay={'run':run,'episode':d['episodes'][25], 'mean_v_after_1s':v[50:].mean(0).tolist(),
            'mean_height_after_1s':float(np.mean([f['position'][2] for f in frames[50:]])),
            'max_tilt_after_1s':float(tilt[50:].max())}
    results.setdefault('replays',[]).append(replay)
(output/'analysis.json').write_text(json.dumps(results,indent=2),encoding='utf-8')
print(json.dumps(results,indent=2))
