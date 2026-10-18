"""Post-hoc behavior diagnostics; does not change acceptance thresholds."""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('--directory', type=Path, default=ROOT / 'experiments/phase2_go2_behavior/run_001')
parser.add_argument('--pattern', default='seed_*.json', help='Input glob, including recursive patterns for validation runs')
args = parser.parse_args()
OUT = args.directory.resolve()
results = []
for path in sorted(OUT.glob(args.pattern)):
    data = json.loads(path.read_text(encoding='utf-8'))
    samples = data['behavior_samples']
    t = np.array([s['time_s'] for s in samples])
    active = np.array([s['active'] for s in samples])
    velocity = np.array([s['linear_b'] for s in samples])
    speed = np.linalg.norm(velocity[:, :, :2], axis=-1)
    yaw = np.array([s['yaw_rate'] for s in samples])
    command = np.array([s['command'] for s in samples])
    tilt = np.degrees(np.arccos(np.clip(np.array([s['upright_cosine'] for s in samples]), -1, 1)))
    rows = []
    for env in range(len(data['episodes'])):
        row = {'env': env}
        mask = (t >= 1) & active[:, env]
        if 'stop_restart' not in path.stem:
            row['mean_yaw_rate_after_1s'] = float(yaw[mask, env].mean())
        else:
            stops = []
            for start in [5, 15]:
                response = None
                window = round(.5 / data['step_dt'])
                for i in np.flatnonzero((t >= start) & (t < start + 5)):
                    j = i + window
                    if j <= len(t) and t[j - 1] < start + 5 and np.all(active[i:j, env]) and np.all(speed[i:j, env] <= .1) and np.all(np.abs(yaw[i:j, env]) <= .15):
                        response = round(float(t[i] - start), 3)
                        break
                mask = (t >= start + 1) & (t < start + 5) & active[:, env]
                stops.append({'switch_time_s': start, 'stable_window_start_delay_s': response,
                              'settled_mean_speed': float(speed[mask, env].mean()) if mask.any() else None,
                              'settled_mean_abs_yaw': float(np.abs(yaw[mask, env]).mean()) if mask.any() else None})
                # Separate speed and yaw to explain failures without changing acceptance.
                for label, values, limit in [('speed', speed[:, env], .1), ('yaw', np.abs(yaw[:, env]), .15)]:
                    delay = None
                    for i in np.flatnonzero((t >= start) & (t < start + 5)):
                        j = i + window
                        if j <= len(t) and t[j - 1] < start + 5 and np.all(active[i:j, env]) and np.all(values[i:j] <= limit):
                            delay = round(float(t[i] - start), 3)
                            break
                    stops[-1][label + '_only_window_start_delay_s'] = delay
            row['stops'] = stops
            windows = []
            for start, end, threshold in [(1, 5, .15), (6, 10, .1), (11, 15, .15), (16, 20, .1)]:
                mask = (t >= start) & (t < end) & active[:, env]
                count = int(mask.sum())
                xy = float(np.linalg.norm(velocity[mask, env, :2] - command[mask, :2], axis=-1).mean()) if count else None
                yaw_error = float(np.abs(yaw[mask, env] - command[mask, 2]).mean()) if count else None
                max_tilt = float(tilt[mask, env].max()) if count else None
                passed = count == round((end-start)/data['step_dt']) and xy <= threshold and yaw_error <= .15 and max_tilt <= 45
                windows.append({'start': start, 'end': end, 'xy_error': xy, 'yaw_error': yaw_error, 'max_tilt_degrees': max_tilt, 'passed': bool(passed)})
            episode = data['episodes'][env]
            row['windows'] = windows
            row['passed'] = bool(episode['time_out'] and not episode['terminated'] and episode['length_steps'] == round(20/data['step_dt']) and all(w['passed'] for w in windows) and all(s['stable_window_start_delay_s'] is not None and s['stable_window_start_delay_s'] <= 2 for s in stops))
        rows.append(row)
    if 'stop_restart' in path.stem:
        from behavior_acceptance import assess_behavior
        expected = assess_behavior(data, 'stop_restart')
        assert [r['passed'] for r in rows] == [r['passed'] for r in expected['rows']]
    results.append({'source': path.relative_to(OUT).as_posix(), 'source_sha256': hashlib.sha256(path.read_bytes()).hexdigest(), 'rows': rows})
report = {'note': 'Post-hoc diagnostics only. Search the entire stop phase to explain failures; the original 2 s acceptance threshold remains unchanged. Delay denotes start of a qualifying continuous 0.5 s window.', 'results': results}
(OUT / 'behavior_diagnostics.json').write_text(json.dumps(report, indent=2), encoding='utf-8')
print('Saved diagnostics for', len(results), 'conditions')
