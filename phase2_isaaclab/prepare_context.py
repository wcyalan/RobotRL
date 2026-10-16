"""Build a minimal flat context for one selected remote job."""
import argparse
import hashlib
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
EVAL = ['evaluation/evaluate_go2.py']
AUDIT = ['evaluation/friction_audit.py']
DELAY = ['extensions/delay/action_delay.py']
BASE = ['training/baseline_config.py', *EVAL]
DR = ['training/train_mass_friction.py', *AUDIT]
JOBS = {
    'train_baseline': ('baseline', BASE),
    'train_seeds': ('baseline', [*BASE, 'jobs/baseline/train_baseline.sh']),
    'smoke_mass_friction': ('domain_randomization', DR),
    'train_mass_friction': ('domain_randomization', [*DR, *EVAL]),
    'evaluate_robustness': ('diagnostics', EVAL),
    'audit_friction': ('diagnostics', [*EVAL, *AUDIT]),
    'evaluate_low_friction': ('diagnostics', [*EVAL, *AUDIT]),
    'replay_failure': ('diagnostics', EVAL),
    'evaluate_delay': ('delay', [*EVAL, *DELAY]),
    'replay_delay_failure': ('delay', [*EVAL, *DELAY]),
    'train_delay_randomized': ('delay', [*EVAL, *DELAY,
        'extensions/delay/train_delay_randomized.py',
        'extensions/delay/random_action_delay.py',
        'extensions/delay/verify_delay_training.py']),
}

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--job', required=True, choices=sorted(JOBS))
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    folder, dependencies = JOBS[args.job]
    relative = [f'jobs/{folder}/{args.job}.sh', 'configs/train_baseline.yaml', *dependencies]
    files = [HERE/p for p in relative]
    assert all(p.is_file() for p in files)
    assert len({p.name for p in files}) == len(files)
    destination = args.output.resolve()
    destination.mkdir(parents=True, exist_ok=False)
    manifest = {'job': args.job, 'files': {}}
    for source in files:
        data = source.read_bytes().replace(b'\r\n', b'\n')
        (destination/source.name).write_bytes(data)
        manifest['files'][source.name] = {'source':source.relative_to(HERE).as_posix(),
                                        'sha256':hashlib.sha256(data).hexdigest()}
    (destination/'context_manifest.json').write_text(json.dumps(manifest,indent=2),encoding='utf-8')
    print(f'Prepared {args.job}: {len(files)} files in {destination}')

if __name__ == '__main__':
    main()
