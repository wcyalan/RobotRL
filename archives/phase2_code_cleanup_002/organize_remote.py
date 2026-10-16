"""One-time, non-destructive server migration with file hash verification."""
from pathlib import Path
import hashlib
import json
import subprocess
import tarfile

root = Path('/labdata2/congyuwang/RobotRL_kitless').resolve()
assert str(root) == '/labdata2/congyuwang/RobotRL_kitless'
listing = subprocess.check_output(['det','command','list'], text=True)
for line in listing.splitlines():
    cells = [c.strip() for c in line.split('|')]
    if len(cells) >= 5 and cells[1] == 'congyuwang':
        assert cells[3] == 'TERMINATED', 'Active task: '+cells[0]
archive = root/'archives/reorganization_001'
assert not archive.exists() or not any(archive.iterdir()), 'Do not rerun completed or partial migration'
archive.mkdir(parents=True, exist_ok=True)
(root/'experiments').mkdir(exist_ok=True)
(root/'contexts').mkdir(exist_ok=True)
(root/'code').mkdir(exist_ok=True)
plans = []
for name in ['phase2_go2_baseline','phase2_go2_robustness','phase2_go2_delay_training','phase2_go2_mass_friction']:
    plans.append((root/name, root/'experiments'/name))
plans.append((root/'run_010', root/'experiments/phase2_go2_baseline/run_001'))
for i in range(1,10):
    plans.append((root/f'run_{i:03d}', archive/'legacy_runs'/f'run_{i:03d}'))
for p in sorted(root.iterdir()):
    if p.is_dir() and (p.name.endswith('_tasks') or p.name in ['tasks','delay_failure_001','mass_friction_full_001','mass_friction_smoke_001']):
        plans.append((p, archive/'task_contexts'/p.name))
for src,dst in plans:
    assert root in src.resolve().parents and root in dst.resolve().parents
    assert src.is_dir() and not src.is_symlink() and not dst.exists(), (src,dst)

def hashes(directory):
    result = {}
    for p in sorted(directory.rglob('*')):
        assert not p.is_symlink(), 'Unexpected symlink: '+str(p)
        if p.is_file():
            h=hashlib.sha256()
            with p.open('rb') as stream:
                for block in iter(lambda:stream.read(1024*1024), b''):h.update(block)
            result[p.relative_to(directory).as_posix()] = h.hexdigest()
    return result

# Back up source text independently; model/result bytes are preserved by rename and verified.
with tarfile.open(archive/'source_before.tar.gz','w:gz') as tar:
    for src,_ in plans:
        for p in src.rglob('*'):
            if p.is_file() and p.suffix in ['.py','.sh','.yaml','.md']:
                tar.add(p, arcname=p.relative_to(root))
records=[]
for src,dst in plans:
    before=hashes(src)
    record={'old':src.relative_to(root).as_posix(),'new':dst.relative_to(root).as_posix(),'sha256':before,'status':'planned'}
    records.append(record)
    (archive/'manifest.json').write_text(json.dumps(records,indent=2))
    dst.parent.mkdir(parents=True, exist_ok=True)
    src.rename(dst)
    assert hashes(dst)==before, 'Integrity failure: '+str(dst)
    record['status']='moved_and_verified'
    (archive/'manifest.json').write_text(json.dumps(records,indent=2))
    print('VERIFIED',record['old'],'->',record['new'],len(before),flush=True)
(root/'README.md').write_text("""# RobotRL server workspace

- code/phase2_isaaclab/: maintained source, same layout as the local repository.
- experiments/: formal experiment groups, including checkpoint/source snapshots.
- contexts/: newly prepared minimal job packages; create a fresh named directory for each job.
- archives/reorganization_001/: old task packages, early run_001 through run_009, and migration hashes.
- assets/: cached robot assets; cache/: runtime compiler caches.

Old run_010 is now experiments/phase2_go2_baseline/run_001.
Historical source snapshots and metadata keep their original paths for provenance.
Use the manifest to resolve historical paths. Run current jobs from a prepared context,
not by executing old source snapshots unchanged. No compatibility symlinks were created.
Do not rerun historical fixed-output jobs against existing results.
""",encoding='utf8')
print('REMOTE_REORGANIZATION_PASSED',len(records),flush=True)
