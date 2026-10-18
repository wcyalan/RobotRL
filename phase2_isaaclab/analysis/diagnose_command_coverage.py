"""Describe command sampling and left/right tracking without changing formal results."""
import hashlib
import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / 'experiments/phase2_go2_command_coverage'
report = {'sampling': [], 'tracking': [], 'limitations': [
    'Sampling counts are resampling events, not time-weighted exposure.',
    'Training audit did not retain separate left/right counts or per-mode tracking errors.',
    'Trajectory images cover env 0; other lanes have velocity and upright samples only.',
    'Analysis is post hoc; formal acceptance thresholds and results remain unchanged.']}
for seed, run in [(0, 'run_001'), (1, 'run_002')]:
    audit_path = OUT / run / 'command_audit.json'
    audit = json.loads(audit_path.read_text(encoding='utf-8'))
    counts = np.array(audit['sample_counts'])
    report['sampling'].append({'seed': seed, 'mode_order': audit['mode_order'],
                               'counts': counts.tolist(), 'fractions': (counts/counts.sum()).tolist(),
                               'source_sha256': hashlib.sha256(audit_path.read_bytes()).hexdigest()})
    for group in ['baseline', 'coverage']:
        for behavior in ['turn_left', 'turn_right']:
            path = OUT / run / f'{group}_{behavior}.json'
            data = json.loads(path.read_text(encoding='utf-8'))
            samples = data['behavior_samples']
            t = np.array([s['time_s'] for s in samples])
            active = np.array([s['active'] for s in samples])
            yaw = np.array([s['yaw_rate'] for s in samples])
            command = np.array([s['command'] for s in samples])
            target = .5 if behavior == 'turn_left' else -.5
            assert np.allclose(command, [0, 0, target])
            rows = []
            for env, episode in enumerate(data['episodes']):
                mask = (t >= 1) & active[:, env]
                values = yaw[mask, env]
                rows.append({'env': env, 'samples': len(values), 'terminated': episode['terminated'],
                             'mean_yaw': float(values.mean()) if len(values) else None,
                             'mean_abs_yaw_error': float(np.abs(values-target).mean()) if len(values) else None})
            surviving = [r for r in rows if not r['terminated'] and r['samples'] == 950]
            windows = []
            for start, end in [(1, 5), (5, 10), (10, 15), (15, 20)]:
                mask = (t >= start) & (t < end) & active[:, 0]
                windows.append({'start': start, 'end': end, 'mean_yaw': float(yaw[mask, 0].mean())})
            report['tracking'].append({'seed': seed, 'group': group, 'behavior': behavior,
                'target': target, 'terminated_envs': [r['env'] for r in rows if r['terminated']],
                'surviving_episodes': len(surviving),
                'survivor_mean_yaw': float(np.mean([r['mean_yaw'] for r in surviving])),
                'survivor_mean_abs_yaw_error': float(np.mean([r['mean_abs_yaw_error'] for r in surviving])),
                'env0_windows': windows, 'rows': rows,
                'source_sha256': hashlib.sha256(path.read_bytes()).hexdigest()})
(OUT / 'diagnosis.json').write_text(json.dumps(report, indent=2), encoding='utf-8')
print(json.dumps({**report, 'tracking': [{k:v for k,v in r.items() if k != 'rows'} for r in report['tracking']]}, indent=2))
