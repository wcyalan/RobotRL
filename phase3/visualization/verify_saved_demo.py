"""Decode the full delivered MP4 and check its timeline and selected frames."""
from pathlib import Path
import argparse
import hashlib
import json

import imageio.v2 as imageio
import numpy as np
from PIL import Image

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('--output', type=Path, required=True)
root = parser.parse_args().output
video = root / 'g1_multicommand_replay.mp4'
metadata = json.loads((root / 'metadata.json').read_text(encoding='utf-8'))
if (root / 'verification.json').exists():
    raise FileExistsError('Refusing to overwrite existing verification.json')
assert metadata['completed'] and not metadata['preview_only']
assert metadata['frame_count'] == 3250
reader = imageio.get_reader(video)
details = reader.get_meta_data()
assert tuple(details['size']) == (1280, 720), details
assert abs(details['fps'] - 25) < 1e-6, details
assert abs(details['duration'] - 130) < .05, details
sample_frames = {int(t * 25): i for i, t in enumerate((10, 30, 50, 70, 90, 118))}
sheet = Image.new('RGB', (1280, 1080))
count = 0
for index, frame in enumerate(reader):
    assert frame.shape == (720, 1280, 3), (index, frame.shape)
    if index in sample_frames:
        assert np.std(frame[120:600]) > 10, 'Blank scene'
        position = sample_frames[index]
        sheet.paste(Image.fromarray(frame).resize((640, 360)),
                    ((position % 2) * 640, (position // 2) * 360))
    count += 1
reader.close()
assert count == 3250, count
sheet.save(root / 'decoded_samples.png')
result = dict(passed=True, decoded_frames=count, fps=details['fps'],
              size=details['size'], duration_s=details['duration'],
              video_bytes=video.stat().st_size,
              video_sha256=hashlib.sha256(video.read_bytes()).hexdigest(),
              check='Full sequential decode, dimensions, fps, duration, selected nonblank scenes')
(root / 'verification.json').write_text(json.dumps(result, indent=2) + '\n', encoding='utf-8')
print(json.dumps(result), flush=True)
