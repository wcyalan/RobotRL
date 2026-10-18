"""Archive completed Go2 scratch data; verify every byte before removing files."""
import argparse
import hashlib
import json
import zipfile
from pathlib import Path

p = argparse.ArgumentParser()
p.add_argument('--root', required=True)
p.add_argument('--contexts', required=True)
p.add_argument('--apply', action='store_true')
a = p.parse_args()
root = Path(a.root).resolve()
assert root.name in ('RobotRL', 'RobotRL_kitless')
contexts = (root / a.contexts).resolve()
contexts.relative_to(root)
archive = root / 'archives/maintenance_003'
assert not archive.exists(), 'Archive must be new'
trees = [d for d in contexts.iterdir() if d.is_dir()] if contexts.exists() else []
files = [f for f in contexts.iterdir() if f.is_file()] if contexts.exists() else []
smokes = []
for family in (root / 'experiments').glob('phase2_go2_*'):
    for d in family.rglob('*'):
        if d.is_dir() and d.name in ('smoke', 'smoke_001'):
            if not any(x in d.parents for x in smokes):
                smokes.append(d)
    # Preserve initial/final models and every extended-training screened checkpoint.
    files.extend(f for f in family.rglob('model_*.pt')
                 if f.name in {'model_50.pt', 'model_100.pt', 'model_150.pt', 'model_200.pt', 'model_250.pt'})
trees += smokes
for d in trees:
    files.extend(f for f in d.rglob('*') if f.is_file())
files = sorted(set(files))
for f in files:
    f.resolve().relative_to(root)
    assert not any(x.is_symlink() for x in [f, *f.parents])
records = [{'path': f.relative_to(root).as_posix(), 'bytes': f.stat().st_size,
            'sha256': hashlib.sha256(f.read_bytes()).hexdigest()} for f in files]
report = {'reason': 'Go2 project closure: completed contexts, smoke tests and unused intermediate checkpoints.',
          'file_count': len(files), 'source_bytes': sum(r['bytes'] for r in records),
          'removed_trees': [d.relative_to(root).as_posix() for d in trees], 'files': records}
if a.apply:
    archive.mkdir(parents=True, exist_ok=False)
    (archive / 'archive_script.py').write_bytes(Path(__file__).read_bytes())
    zpath = archive / 'completed_scratch.zip'
    with zipfile.ZipFile(zpath, 'x', zipfile.ZIP_DEFLATED) as z:
        for f, r in zip(files, records):
            z.write(f, r['path'])
    with zipfile.ZipFile(zpath) as z:
        for r in records:
            assert hashlib.sha256(z.read(r['path'])).hexdigest() == r['sha256']
    report['archive_bytes'] = zpath.stat().st_size
    (archive / 'manifest.json').write_text(json.dumps(report, indent=2), encoding='utf-8')
    # All resolved targets are within the named workspace; no recursive deletion.
    for f, r in zip(files, records):
        assert hashlib.sha256(f.read_bytes()).hexdigest() == r['sha256']
        f.unlink()
    for d in trees:
        for sub in sorted((x for x in d.rglob('*') if x.is_dir()), key=lambda x: len(x.parts), reverse=True):
            sub.rmdir()
        d.rmdir()
    for d in smokes:
        d.mkdir()
        (d / 'README.md').write_text('# Archived smoke test\n\nCompleted preflight artifacts were moved to '
            '`archives/maintenance_003/completed_scratch.zip`. Restore entries at their original paths '
            'using the SHA256 manifest in that archive directory. Formal training and evaluation remain in place.\n', encoding='utf-8')
print(json.dumps({k: v for k, v in report.items() if k != 'files'}, indent=2))
