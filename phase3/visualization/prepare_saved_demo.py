"""Extract only the fixed demonstration lanes, frozen model and render meshes.

No policies, training checkpoints or full experiment synchronization. Run where
run_003 is already available. NumPy is required; the source is read-only.
"""
import argparse
import hashlib
import json
from pathlib import Path
import shutil

import numpy as np

CONDITIONS = ('stand', 'forward_low', 'forward_medium', 'turn_left', 'turn_right', 'stop_restart')


def read(path):
    return json.loads(path.read_text(encoding='utf-8'))


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def prepare(run, output):
    run, output = run.resolve(), output.resolve()
    if output.exists():
        raise FileExistsError(output)
    frozen = read(run/'reference/inputs.json')
    for relative, expected in frozen.items():
        source = (run/relative).resolve()
        source.relative_to(run)
        if digest(source) != expected:
            raise ValueError(f'Frozen input changed: {relative}')
    plan, total = read(run/'plan.json'), read(run/'summary.json')
    if plan['smoke'] or not total['completed'] or total['episodes_completed'] != 768:
        raise ValueError('Requires completed formal run_003')
    output.mkdir(parents=True)
    (output/'model').mkdir()
    (output/'meshes').mkdir()
    (output/'traces').mkdir()
    provenance = {}
    for name in ('g1.xml', 'interface.json'):
        source = run/'reference/model'/name
        shutil.copyfile(source, output/'model'/name)
        provenance[str(source.relative_to(run))] = digest(source)
    for source in sorted((run/'reference/meshes').glob('*.obj')):
        shutil.copyfile(source, output/'meshes'/source.name)
        provenance[str(source.relative_to(run))] = digest(source)
    selection = dict(eval_seed=20000, lane=0, training_seeds=[0, 1],
                     selection_rule='fixed before rendering; no best-episode search',
                     protocol=plan['protocol'], groups={}, episodes={})
    for seed in (0, 1):
        for condition in CONDITIONS:
            folder = run/f'seed_{seed}/eval_20000/{condition}'
            summary = read(folder/'summary.json')
            episode = summary['episodes'][0]
            if episode['lane'] != 0 or episode['early_termination']:
                raise ValueError('Fixed representative lane is missing or incomplete')
            key = f'{seed}/{condition}'
            selection['episodes'][key] = episode
            selection['groups'][key] = {k:summary[k] for k in ('pass_count','num_episodes','group_passed')}
            source = folder/'trajectory.npz'
            with np.load(source, allow_pickle=False) as data:
                arrays = {k:data[f'lane_0_{k}'] for k in ('qpos','qvel','metrics','command')}
            np.savez_compressed(output/f'traces/seed_{seed}_{condition}.npz', **arrays)
            provenance[str(source.relative_to(run))] = digest(source)
            provenance[str((folder/'summary.json').relative_to(run))] = digest(folder/'summary.json')
    (output/'plan.json').write_text(json.dumps(selection, indent=2)+'\n', encoding='utf-8')
    manifest = dict(source_run=str(run), source_plan_sha256=digest(run/'plan.json'),
                    source_input_manifest_sha256=digest(run/'reference/inputs.json'),
                    source_files=provenance,
                    files={p.relative_to(output).as_posix():digest(p) for p in sorted(output.rglob('*')) if p.is_file()},
                    excluded=['policy weights', 'training checkpoints', 'other lanes', 'other evaluation seeds'])
    (output/'manifest.json').write_text(json.dumps(manifest, indent=2)+'\n', encoding='utf-8')
    print(json.dumps({'prepared':str(output), 'files':len(manifest['files']),
                      'bytes':sum(p.stat().st_size for p in output.rglob('*') if p.is_file())}))


if __name__ == '__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args=parser.parse_args()
    prepare(args.run, args.output)
