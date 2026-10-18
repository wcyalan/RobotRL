import argparse, hashlib, json, zipfile
from pathlib import Path
p=argparse.ArgumentParser();p.add_argument('--root',required=True);p.add_argument('--context-dir',required=True);a=p.parse_args()
root=Path(a.root).resolve(); contexts=(root/a.context_dir).resolve();contexts.relative_to(root)
assert root.name=='RobotRL' or root.name=='RobotRL_kitless'
names=['command_coverage_001','dr_outlier_001','go2_behavior_001','replay_stop_failure_001','replay_stop_failure_002','stop_stability_001','turn_coverage_001','turn_validation_001']
if a.context_dir=='.task_context':
 names += [x.name for x in contexts.iterdir() if x.is_dir() and (x.name.startswith('reorg_check_') or x.name in ['phase2_layout_check_001','reorganized_source','code_sync_behavior_001'])]
dirs=[]; files=[]
for name in names:
 d=contexts/name
 if not d.exists():continue
 assert d.is_dir() and not d.is_symlink() and d.resolve().parent==contexts
 dirs.append(d);files.extend(x for x in d.rglob('*') if x.is_file())
for family in ['phase2_go2_command_coverage','phase2_go2_stop_stability']:
 for run in ['run_001','run_002']:
  d=root/'experiments'/family/run
  assert (d/'completed.txt').exists(),str(d)
  files.extend(x for x in (d/'logs').rglob('model_*.pt') if x.name not in ['model_0.pt','model_299.pt'])
files=sorted(set(files)); archive=root/'archives/maintenance_001';archive.mkdir(parents=True,exist_ok=False)
records=[]
for f in files:
 f.resolve().relative_to(root)
 assert not f.is_symlink()
 records.append({'path':f.relative_to(root).as_posix(),'bytes':f.stat().st_size,'sha256':hashlib.sha256(f.read_bytes()).hexdigest()})
zpath=archive/'completed_contexts_and_intermediate_checkpoints.zip'
with zipfile.ZipFile(zpath,'x',zipfile.ZIP_DEFLATED) as z:
 for f,r in zip(files,records):z.write(f,r['path'])
with zipfile.ZipFile(zpath) as z:
 for r in records:assert hashlib.sha256(z.read(r['path'])).hexdigest()==r['sha256']
report={'reason':'User requested ongoing cleanup; archive completed duplicate contexts and intermediate checkpoints of rejected rounds. Keep final/initial checkpoints, metrics and per-experiment source snapshots.','files':records,'file_count':len(records),'source_bytes':sum(r['bytes'] for r in records),'archive_bytes':zpath.stat().st_size,'removed_contexts':[d.name for d in dirs]}
(archive/'manifest.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
(archive/'archive_script.py').write_bytes(Path(__file__).read_bytes())
for f,r in zip(files,records):
 assert hashlib.sha256(f.read_bytes()).hexdigest()==r['sha256'];f.unlink()
for d in dirs:
 for sub in sorted([x for x in d.rglob('*') if x.is_dir()],key=lambda x:len(x.parts),reverse=True):sub.rmdir()
 d.rmdir()
print(json.dumps({k:v for k,v in report.items() if k!='files'}))
