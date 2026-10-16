"""Stage a flat, self-contained Determined task context from organized sources."""
import argparse
import hashlib
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True, help='New directory; existing directories are never overwritten.')
    args = parser.parse_args()
    destination = args.output.resolve()
    files = [p for p in HERE.glob('*.py') if p.name != 'prepare_context.py']
    files += list((HERE/'jobs').glob('*.sh')) + list((HERE/'configs').glob('*.yaml'))
    assert len({p.name for p in files}) == len(files), 'Duplicate context filenames'
    destination.mkdir(parents=True, exist_ok=False)
    manifest = {}
    for source in sorted(files):
        # Linux shell entrypoints must have LF line endings.
        content = source.read_bytes().replace(b'\r\n', b'\n')
        (destination/source.name).write_bytes(content)
        manifest[source.name] = {'source':str(source.relative_to(HERE)), 'sha256':hashlib.sha256(content).hexdigest()}
    (destination/'context_manifest.json').write_text(json.dumps(manifest,indent=2),encoding='utf-8')
    print(f'Prepared {len(files)} files in {destination}')

if __name__ == '__main__':
    main()
