"""Apply unchanged behavior criteria to held-out evaluation seeds."""
import argparse
import hashlib
import json
from pathlib import Path
from behavior_acceptance import assess_behavior

ROOT = Path(__file__).resolve().parents[2]
parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('--profile', choices=['turn', 'stop_sequence', 'extended'], default='turn')
profile = parser.parse_args().profile
family = 'phase2_go2_stop_sequence' if profile == 'stop_sequence' else 'phase2_go2_turn_coverage'
OUT = ROOT / 'experiments' / family / 'validation_001'
if profile == 'extended':
    family = 'phase2_go2_stop_sequence_extended'
    OUT = ROOT / 'experiments' / family
eval_seeds = [43000, 44000] if profile != 'turn' else [41000, 42000]
behaviors = ['turn_left', 'turn_right', 'stop_restart'] + (['forward'] if profile != 'turn' else [])
required = len(eval_seeds) * 2 * 2 * len(behaviors)
rows, missing = [], []
for eval_seed in eval_seeds:
    for seed, run, old in [(0, 'run_001', 'run_002'), (1, 'run_002', 'run_003')]:
        for group in ['coverage', 'baseline']:
            model_dir = ROOT / 'experiments' / ('phase2_go2_turn_coverage' if group == 'coverage' else 'phase2_go2_baseline') / (run if group == 'coverage' else old)
            if profile == 'stop_sequence':
                model_dir = ROOT / 'experiments' / (family if group == 'coverage' else 'phase2_go2_turn_coverage') / run
            if profile == 'extended':
                model_dir = ROOT / 'experiments' / (family if group == 'coverage' else 'phase2_go2_stop_sequence') / run
            model = model_dir / 'model_extended.pt' if profile == 'extended' and group == 'coverage' else next(model_dir.glob('logs/**/model_299.pt'))
            if not model.exists():
                missing.extend(f'{run}/eval_{eval_seed}/{group}_{b}.json' for b in behaviors)
                continue
            model_sha = hashlib.sha256(model.read_bytes()).hexdigest()
            if profile == 'extended' and group == 'coverage':
                audit = json.loads((model_dir / 'resume_audit.json').read_text())
                parent = next((ROOT / 'experiments/phase2_go2_stop_sequence' / run).glob('logs/**/model_299.pt'))
                assert audit['parent_sha256'] == hashlib.sha256(parent.read_bytes()).hexdigest()
                assert audit['final_sha256'] == model_sha and audit['additional_iterations'] == 300
                assert audit['configs_match_parent'] and audit['optimizer_restored']
            for behavior in behaviors:
                path = OUT / f'eval_{eval_seed}/seed_{seed}/{group}_{behavior}.json'
                if profile == 'extended':
                    path = OUT / run / f'eval_{eval_seed}/{group}_{behavior}.json'
                if not path.exists():
                    missing.append(path.relative_to(OUT).as_posix())
                    continue
                data = json.loads(path.read_text(encoding='utf-8'))
                assert data['sha256'] == model_sha and data['eval_seed'] == eval_seed
                reference_path = ROOT / 'experiments' / family / run / f'{group}_{behavior}.json'
                if profile == 'extended':
                    reference_path = ROOT / 'experiments/phase2_go2_stop_sequence' / run / f'coverage_{behavior}.json'
                reference = json.loads(reference_path.read_text(encoding='utf-8'))
                assert data['scenario'] == reference['scenario']
                if behavior == 'forward':
                    assert data['forward_speed_command_m_s'] == .5 and len(data['episodes']) == 32
                    result = {'passed_episodes': None, 'note': 'Forward tracking reported descriptively; no added pass threshold'}
                else:
                    result = assess_behavior(data, behavior)
                rows.append({'eval_seed': eval_seed, 'training_seed': seed, 'group': group,
                             'behavior': behavior, 'model_sha256': model_sha,
                             'source_sha256': hashlib.sha256(path.read_bytes()).hexdigest(),
                             'xy_error': data['mean_xy_velocity_error_m_s'],
                             'yaw_error': data['mean_yaw_velocity_error_rad_s'],
                             'termination_rate': data['termination_rate'], **result})
report = {'complete': not missing, 'completed_conditions': len(rows), 'required_conditions': required,
          'missing': missing, 'results': rows}
(OUT / 'acceptance.json').write_text(json.dumps(report, indent=2), encoding='utf-8')
print('Completed', len(rows), '/', required, 'conditions')
for r in rows:
    print(r['eval_seed'], r['training_seed'], r['group'], r['behavior'], r['passed_episodes'], '/32', 'terminated', r['termination_rate'])
