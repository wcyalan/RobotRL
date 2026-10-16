"""Fetch only the Go2 visual model and its license from MuJoCo Menagerie."""
from pathlib import Path
import urllib.request
import xml.etree.ElementTree as ET
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json

root = Path(__file__).resolve().parents[1] / 'assets' / 'unitree_go2'
root.mkdir(parents=True, exist_ok=True)
base = 'https://raw.githubusercontent.com/google-deepmind/mujoco_menagerie/main/unitree_go2/'
def fetch(name):
    target = root / name
    target.parent.mkdir(parents=True, exist_ok=True)
    with urllib.request.urlopen(base + name, timeout=90) as response:
        target.write_bytes(response.read())
    return name
fetch('go2.xml')
names = ['LICENSE', 'README.md'] + ['assets/' + e.attrib['file'] for e in ET.parse(root / 'go2.xml').findall('.//mesh') if 'file' in e.attrib]
with ThreadPoolExecutor(max_workers=4) as pool:
    print(list(pool.map(fetch, names)))
(root / 'provenance.json').write_text(json.dumps({'source': base, 'sha256': {n: hashlib.sha256((root / n).read_bytes()).hexdigest() for n in ['go2.xml'] + names}}, indent=2))
