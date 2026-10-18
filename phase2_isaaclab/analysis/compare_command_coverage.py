"""Compare paired command-coverage evaluations without changing acceptance criteria."""
import argparse
import hashlib
import json
from pathlib import Path

from behavior_acceptance import assess_behavior

ROOT = Path(__file__).resolve().parents[2]
parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('--directory', type=Path, default=ROOT / 'experiments/phase2_go2_command_coverage')
parser.add_argument('--control-directory', type=Path, help='Control models with run_001=seed0 and run_002=seed1; default uses official baseline mapping')
args = parser.parse_args()
OUT = args.directory.resolve()
results = []
missing = []
for seed, run, baseline in [(0, 'run_001', 'run_002'), (1, 'run_002', 'run_003')]:
    for group in ['baseline', 'coverage']:
        model_dir = ROOT / 'experiments/phase2_go2_baseline' / baseline if group == 'baseline' else OUT / run
        if group == 'baseline' and args.control_directory:
            model_dir = args.control_directory.resolve() / run
        models = list(model_dir.glob('logs/**/model_299.pt'))
        for behavior in ['turn_left', 'turn_right', 'stop_restart', 'forward']:
            path = OUT / run / f'{group}_{behavior}.json'
            if not path.exists():
                missing.append(str(path.relative_to(OUT)))
                continue
            assert len(models) == 1, f'Missing or ambiguous checkpoint: {model_dir}'
            data = json.loads(path.read_text(encoding='utf-8'))
            assert data['sha256'] == hashlib.sha256(models[0].read_bytes()).hexdigest()
            assert data['eval_seed'] == 40000 and len(data['episodes']) == 32
            result = {'seed': seed, 'group': group, 'behavior': behavior,
                      'source_sha256': hashlib.sha256(path.read_bytes()).hexdigest(),
                      'model_sha256': data['sha256'], 'scenario': data['scenario'],
                      'xy_error': data['mean_xy_velocity_error_m_s'],
                      'yaw_error': data['mean_yaw_velocity_error_rad_s'],
                      'termination_rate': data['termination_rate'],
                      'mean_length_seconds': data['mean_length_seconds']}
            if behavior != 'forward':
                assert data['scenario']['behavior'] == behavior
                result.update(assess_behavior(data, behavior))
            else:
                assert data['scenario']['name'] == 'nominal' and data['forward_speed_command_m_s'] == .5
            results.append(result)
comparisons = []
for seed in [0, 1]:
    for behavior in ['turn_left', 'turn_right', 'stop_restart', 'forward']:
        paired = {r['group']: r for r in results if r['seed'] == seed and r['behavior'] == behavior}
        if len(paired) < 2:
            continue
        a, b = paired['baseline'], paired['coverage']
        assert a['scenario'] == b['scenario']
        comparisons.append({'seed': seed, 'behavior': behavior,
                            'baseline_passed': a.get('passed_episodes'), 'coverage_passed': b.get('passed_episodes'),
                            'baseline_xy_error': a['xy_error'], 'coverage_xy_error': b['xy_error'],
                            'xy_error_delta_coverage_minus_baseline': b['xy_error']-a['xy_error'],
                            'baseline_yaw_error': a['yaw_error'], 'coverage_yaw_error': b['yaw_error'],
                            'baseline_termination_rate': a['termination_rate'], 'coverage_termination_rate': b['termination_rate']})
report = {'complete': not missing, 'completed_evaluations': len(results), 'required_evaluations': 16,
          'missing': missing, 'comparisons': comparisons, 'results': results,
          'note': 'Two training seeds; one evaluation seed. Forward regression reported descriptively. No threshold changes; incomplete runs do not establish success.'}
(OUT / 'comparison.json').write_text(json.dumps(report, indent=2), encoding='utf-8')
print(json.dumps({k:v for k,v in report.items() if k != 'results'}, indent=2))
