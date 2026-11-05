"""Prepare VLM-selected DD target warps for the official Tile ControlNet."""
import hashlib
import json
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
HERE = Path(__file__).resolve().parent


def sha(path):
    h = hashlib.sha256()
    with path.open('rb') as stream:
        for chunk in iter(lambda: stream.read(8 << 20), b''):
            h.update(chunk)
    return h.hexdigest()


for label, baseline in [('case12_a', 'vlm_dd_balanced_case12_a'),
                        ('case13_b', 'vlm_dd_balanced_case13_b')]:
    record = json.loads((ROOT / baseline / 'stage3/records/route_a/depthdirector/stage3/vlm_selected.json').read_text())
    directory = ROOT / baseline / 'stage3/raw/route_a/depthdirector/stage3/vlm_selected'
    matches = list(directory.glob('*/preprocess/*/warp.mp4'))
    assert len(matches) == 1, matches
    source = matches[0]
    out = HERE / f'tile_controlnet_{label}'
    out.mkdir(exist_ok=True)
    cap = cv2.VideoCapture(str(source))
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    video = out / 'control.mp4'
    writer = cv2.VideoWriter(str(video), cv2.VideoWriter_fourcc(*'mp4v'), 15, (width, height))
    assert writer.isOpened()
    counts = []
    while True:
        okay, bgr = cap.read()
        if not okay:
            break
        valid = bgr.max(axis=2) > 20
        assert valid.any()
        # Match case15's DD-warp hole fill; source warp remains an estimate.
        fill = np.median(bgr[valid], axis=0).astype(np.uint8)
        filled = bgr.copy()
        filled[~valid] = fill
        rgb = Image.fromarray(cv2.cvtColor(filled, cv2.COLOR_BGR2RGB))
        small = rgb.resize((width // 4, height // 4))
        blurred = Image.fromarray(cv2.GaussianBlur(np.asarray(small), (5, 5), sigmaX=2))
        control = blurred.resize((width, height))
        writer.write(cv2.cvtColor(np.asarray(control), cv2.COLOR_RGB2BGR))
        counts.append({'valid_fraction': float(valid.mean()), 'fill_bgr': fill.tolist()})
    cap.release()
    writer.release()
    assert len(counts) == record['native_frames'], (label, len(counts), record['native_frames'])
    manifest = {'label': label, 'baseline': baseline, 'source_warp': str(source),
                'source_sha256': sha(source), 'control_video': str(video),
                'control_sha256': sha(video), 'frames': len(counts), 'source_fps': fps,
                'control_fps': 15, 'width': width, 'height': height,
                'mean_valid_fraction': float(np.mean([x['valid_fraction'] for x in counts])),
                'method': 'Local hole-fill of DD estimated target-view warp with per-frame nonblack median; then official TheDenk Tile README resize by 4, 5x5 Gaussian sigma 2, resize back. Not target-view ground truth.',
                'official_source': 'https://github.com/TheDenk/wan2.2-controlnet',
                'frame_diagnostics': counts}
    (out / 'manifest.json').write_text(json.dumps(manifest, indent=2) + '\n')
    print(label, video, video.stat().st_size, manifest['mean_valid_fraction'], flush=True)
