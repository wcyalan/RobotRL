"""Independently audit saved G1 group evaluation; no simulator or policy load.

Reads the frozen inputs, JSON results and all per-step metrics in place. Uses
only NumPy and the Python standard library. Never rewrites server artifacts.
Example: python audit_saved_evaluation.py --run /path/to/run_003
"""
import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path

import numpy as np


def require(condition, message):
    if not condition:
        raise ValueError(message)


def read(path):
    return json.loads(path.read_text(encoding='utf-8'))


def longest_run(mask):
    edges = np.diff(np.r_[False, mask, False].astype(np.int8))
    starts, ends = np.flatnonzero(edges == 1), np.flatnonzero(edges == -1)
    return int(max(ends - starts, default=0))


def audit_episode(episode, metrics, case, thresholds, dt):
    require(dt > 0, 'Nonpositive policy timestep')
    horizon = round(case['duration_s'] / dt)
    require(metrics.shape == (episode['steps'], 3), 'Metric shape differs from recorded length')
    require(np.isfinite(metrics).all(), 'Nonfinite metric')
    require(0 < len(metrics) <= horizon, 'Invalid episode length')
    require(abs(episode['duration_s'] - len(metrics) * dt) < 1e-9, 'Duration mismatch')
    full = len(metrics) == horizon and not episode['early_termination']
    require(len(episode['windows']) == len(case['measurement_windows']), 'Window count mismatch')
    for saved, (start, end, kind) in zip(episode['windows'], case['measurement_windows']):
        require((saved['start_s'], saved['end_s'], saved['kind']) == (start, end, kind), 'Window definition changed')
        values = metrics[round(start / dt):round(end / dt)]
        expected = round((end - start) / dt)
        require(saved['samples'] == len(values) and saved['expected_samples'] == expected, 'Window sample count mismatch')
        if len(values):
            means = values.mean(axis=0)
            np.testing.assert_allclose(means, saved['means'], rtol=0, atol=1e-10)
        else:
            require(saved['means'] is None, 'Empty window has means')
            means = np.full(3, np.inf)
        prefix = 'stopped' if kind == 'stopped' else 'moving'
        limits = [thresholds['stopped_xy_speed_m_s' if prefix == 'stopped' else 'moving_xy_error_m_s'],
                  thresholds['stopped_yaw_speed_rad_s' if prefix == 'stopped' else 'moving_yaw_error_rad_s'],
                  thresholds['mean_tilt_deg']]
        passed = len(values) == expected and bool(np.all(means <= limits))
        require(saved['passed'] == passed, 'Window decision mismatch')
        full = full and passed
    passed, diagnostic = full, None
    if 'stop_time_s' in case:
        stop = case['stop_time_s']
        end = next(end for start, end, command in case['segments'] if start == stop and command == [0, 0, 0])
        values = metrics[round(stop / dt):round(end / dt)]
        xy_ok = values[:, 0] <= thresholds['stopped_xy_speed_m_s']
        yaw_ok = values[:, 1] <= thresholds['stopped_yaw_speed_rad_s']
        both = xy_ok & yaw_ok
        length = round(thresholds['stop_stable_window_s'] / dt)
        require(length > 0 and abs(length * dt - thresholds['stop_stable_window_s']) < 1e-9, 'Nonintegral stability window')
        sums = np.r_[0, np.cumsum(both, dtype=np.int64)]
        starts = np.flatnonzero(sums[length:] - sums[:-length] == length)
        settle = float((starts[0] + length) * dt) if len(starts) else None
        saved_settle = episode['stop_stable_time_s']
        require((saved_settle is None) == (settle is None), 'Stability presence mismatch')
        if settle is not None:
            require(abs(saved_settle - settle) < 1e-9, 'Stability time mismatch')
        passed = full and settle is not None and settle <= thresholds['stop_stable_by_s']
        diagnostic = dict(longest_stable_seconds=longest_run(both) * dt,
                          xy_violation_fraction=float(np.mean(~xy_ok)) if len(values) else None,
                          yaw_violation_fraction=float(np.mean(~yaw_ok)) if len(values) else None,
                          stable_time_s=settle)
    require(episode['episode_passed'] == passed and episode['pilot_episode_passed'] == passed, 'Episode decision mismatch')
    require(episode['full_episode_and_windows_passed'] == full, 'Full-window decision mismatch')
    return bool(passed), diagnostic


def audit(run):
    run = run.resolve()
    manifest = read(run / 'reference/inputs.json')
    for relative, expected in manifest.items():
        path = (run / relative).resolve()
        try:
            path.relative_to(run)
        except ValueError as error:
            raise ValueError('Manifest path escapes run directory') from error
        require(hashlib.sha256(path.read_bytes()).hexdigest() == expected, f'Frozen input changed: {relative}')
    plan, summary = read(run / 'plan.json'), read(run / 'summary.json')
    protocol = plan['protocol']
    require(not plan['smoke'] and summary['completed'] and not summary['smoke'], 'Not a completed full evaluation')
    require(plan['episodes_per_group'] == protocol['num_envs'] == 32, 'Unexpected group size')
    require(plan['training_seeds'] == protocol['training_seeds'] == [0, 1], 'Training seed mismatch')
    require(plan['eval_seeds'] == protocol['eval_seeds'] == [20000, 21000], 'Evaluation seed mismatch')
    require(set(plan['conditions']) == set(protocol['conditions']) and len(plan['conditions']) == 6, 'Condition mismatch')
    require(protocol['minimum_pass_count'] == 29, 'Acceptance threshold changed')
    frozen_interface = read(run / 'reference/model/interface.json')
    original_groups = {(g['training_seed'], g['eval_seed'], g['condition']): g for g in summary['groups']}
    require(len(original_groups) == len(summary['groups']) == 24, 'Missing or duplicate summary groups')
    groups, episodes, steps, early, traced = [], 0, 0, 0, 0
    for seed in plan['training_seeds']:
        for eval_seed in plan['eval_seeds']:
            for condition in plan['conditions']:
                key = seed, eval_seed, condition
                folder = run / f'seed_{seed}/eval_{eval_seed}/{condition}'
                data = read(folder / 'summary.json')
                require((data['training_seed'], data['eval_seed'], data['condition']) == key, 'Group identity mismatch')
                require(data['completed'] and not data['smoke'], 'Incomplete group')
                require(len(data['episodes']) == data['num_episodes'] == 32, 'Missing episodes')
                require(data['step_dt'] == frozen_interface['policy_dt'], 'Policy timestep changed')
                passed_count, diagnostics = 0, []
                with np.load(folder / 'trajectory.npz', allow_pickle=False) as traces:
                    for lane, episode in enumerate(data['episodes']):
                        state = plan['initial_states'][str(eval_seed)][lane]
                        require((episode['lane'], episode['initial_pose'], episode['noise_seed']) ==
                                (state['lane'], state['pose'], state['noise_seed']), 'Initial-state record changed')
                        metrics = traces[f'lane_{lane}_metrics']
                        passed, diagnostic = audit_episode(episode, metrics, protocol['conditions'][condition], protocol['thresholds'], data['step_dt'])
                        if lane in plan['trace_lanes']:
                            qpos, qvel = traces[f'lane_{lane}_qpos'], traces[f'lane_{lane}_qvel']
                            x, y, yaw = state['pose']
                            np.testing.assert_allclose(qpos[0, :3], [x, y, frozen_interface['initial_height']], atol=1e-10, rtol=0)
                            np.testing.assert_allclose(qpos[0, 3:7], [np.cos(yaw/2), 0, 0, np.sin(yaw/2)], atol=1e-10, rtol=0)
                            np.testing.assert_allclose(qvel[0], 0, atol=1e-10, rtol=0)
                            traced += 1
                        passed_count += passed
                        episodes += 1
                        steps += len(metrics)
                        early += int(episode['early_termination'])
                        if diagnostic is not None:
                            diagnostics.append(diagnostic)
                require(passed_count == data['pass_count'] == original_groups[key]['pass_count'], 'Pass count mismatch')
                group_passed = passed_count >= protocol['minimum_pass_count']
                require(group_passed == data['group_passed'] == original_groups[key]['group_passed'], 'Group decision mismatch')
                group = dict(training_seed=seed, eval_seed=eval_seed, condition=condition,
                             pass_count=passed_count, group_passed=group_passed)
                if diagnostics:
                    group['stop_diagnostics'] = {
                        'longest_stable_seconds_min_median_max': np.quantile([d['longest_stable_seconds'] for d in diagnostics], [0, .5, 1]).tolist(),
                        'mean_xy_violation_fraction': float(np.mean([d['xy_violation_fraction'] for d in diagnostics])),
                        'mean_yaw_violation_fraction': float(np.mean([d['yaw_violation_fraction'] for d in diagnostics])),
                        'stable_window_reached': sum(d['stable_time_s'] is not None for d in diagnostics)}
                groups.append(group)
    require(episodes == summary['episodes_completed'] and early == summary['early_terminations'], 'Total count mismatch')
    require(len(groups) == summary['groups_completed'] and sum(g['group_passed'] for g in groups) == summary['groups_passed'], 'Group total mismatch')
    return dict(audited_at_utc=datetime.now(timezone.utc).isoformat(), run=str(run),
                audit_passed=True, frozen_files_verified=len(manifest), groups_verified=len(groups),
                episodes_verified=episodes, policy_steps_verified=steps, representative_initial_states_verified=traced,
                early_terminations=early, groups_passed=sum(g['group_passed'] for g in groups),
                groups=groups,
                limits=['Read-only replay of saved metrics, not a new dynamics simulation.',
                        'No independent remeasurement of source-runtime physics in this audit.',
                        'Stop-axis violation fractions describe saved errors; they do not establish a causal mechanism.'])


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(audit(args.run), indent=2, allow_nan=False))
