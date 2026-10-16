from pathlib import Path
import hashlib,json,subprocess
root=Path('/labdata2/congyuwang/RobotRL_kitless')
source=root/'code/phase2_isaaclab'
manifest=json.loads((source/'source_manifest.json').read_text())
for name,digest in manifest.items():
    assert hashlib.sha256((source/name).read_bytes()).hexdigest()==digest,name
for script in (source/'jobs').rglob('*.sh'):
    subprocess.run(['bash','-n',str(script)],check=True)
out=root/'contexts/validation_mass_friction_001'
subprocess.run(['python3',str(source/'prepare_context.py'),'--job','train_mass_friction','--output',str(out)],check=True)
assert len(list(out.glob('*.py')))==3
out.rename(root/'archives/reorganization_001/validated_minimal_context')
print('REMOTE_SOURCE_AND_BASH_CHECK_PASSED',len(manifest))
