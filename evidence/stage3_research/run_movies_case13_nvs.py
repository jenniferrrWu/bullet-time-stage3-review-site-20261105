"""Run official MoVieS modules on case13 with the VLM target camera, 37 output times.

The model/backbone/renderers and checkpoint are upstream. Static camera packaging,
subject-depth orbit registration, and 37-frame orchestration are local adapters.
"""
import hashlib
import json
import os
import sys
from pathlib import Path

import cv2
import numpy as np
import torch
from safetensors.torch import load_file

ROOT = Path(__file__).resolve().parents[1]
HERE = Path(__file__).resolve().parent
MOVIES = ROOT / 'paper_sources/MoVieS'
WEIGHT = ROOT / 'paper_sources/MoVieS-weights/movies_ckpt.safetensors'
INPUT = MOVIES / 'resources/DAVIS/case13_b_static.npz'
MASK = HERE / 'nlf_case13_b_target_silhouette/source_mask_00.png'
OUT = HERE / 'movies_case13_b_nvs_37'
EXPECTED_SHA = 'e0b0f448f7474e0beb7ca86be8e2f22d35398459a6c41cd5297ee26617536df0'


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(8 << 20), b''):
            h.update(block)
    return h.hexdigest()


def main():
    assert sha(WEIGHT) == EXPECTED_SHA
    OUT.mkdir(exist_ok=True)
    os.chdir(MOVIES)
    sys.path.insert(0, str(MOVIES))
    from src.options import opt_dict
    from src.models import SplatRecon
    from extensions.vggt.vggt.models.vggt import VGGT
    # The full MoVieS checkpoint replaces every persistent VGGT-derived weight.
    # Construct the official VGGT architecture locally to avoid a redundant hub download.
    VGGT.from_pretrained = classmethod(lambda cls, *args, **kwargs: cls())

    sample = np.load(INPUT)
    images = torch.from_numpy(sample['images']).float().unsqueeze(0)
    camera = torch.from_numpy(sample['C2W']).float().unsqueeze(0)
    intr = torch.from_numpy(sample['fxfycxcy']).float().unsqueeze(0)
    assert images.shape == (1, 13, 3, 294, 518)
    assert camera.shape == (1, 13, 4, 4)
    assert intr.shape == (1, 13, 4)
    input_t = torch.linspace(0, 1, 13).unsqueeze(0)
    output_t = torch.linspace(0, 1, 37).unsqueeze(0)
    model = SplatRecon(opt_dict['movies'], load_lpips=False)
    incompatible = model.load_state_dict(load_file(str(WEIGHT)), strict=True)
    assert not incompatible.missing_keys and not incompatible.unexpected_keys
    model.eval().to('cuda')
    with torch.inference_mode(), torch.autocast('cuda', dtype=torch.bfloat16):
        backbone, motions, moving_splats = model.backbone(
            images.to('cuda', dtype=torch.bfloat16),
            camera.to('cuda', dtype=torch.bfloat16),
            intr.to('cuda', dtype=torch.bfloat16),
            input_t.to('cuda', dtype=torch.bfloat16),
            output_t.to('cuda', dtype=torch.bfloat16),
            frames_chunk_size=8,
        )
    assert motions is not None and motions.shape[1] == 37, tuple(motions.shape) if motions is not None else None
    z = model.gs_renderer.depth_activation(backbone['depth']).float()[0, 0, 0].detach().cpu().numpy()
    mask = cv2.imread(str(MASK), cv2.IMREAD_GRAYSCALE)
    assert mask is not None
    mask = cv2.resize(mask[:, 10:630], (518, 294), interpolation=cv2.INTER_NEAREST) > 0
    good = np.isfinite(z) & (z > 0) & mask
    assert good.sum() > 300
    subject_depth = float(np.median(z[good]))
    assert 0.05 < subject_depth < 100, subject_depth
    meta = json.loads(INPUT.with_suffix('.json').read_text())
    relative = np.array(meta['target_C2W_relative_to_source'], dtype=np.float32)
    rotation = relative[:3, :3]
    target_camera = np.eye(4, dtype=np.float32)
    target_camera[:3, :3] = rotation
    pivot = np.array([0, 0, subject_depth], dtype=np.float32)
    target_camera[:3, 3] = pivot - rotation @ pivot
    target_camera = torch.from_numpy(target_camera).unsqueeze(0).unsqueeze(0).to('cuda', dtype=torch.bfloat16)
    camera = camera.to('cuda', dtype=torch.bfloat16)
    intr = intr.to('cuda', dtype=torch.bfloat16)
    previews = {'source_reconstruction': [], 'target_view': []}
    with torch.inference_mode(), torch.autocast('cuda', dtype=torch.bfloat16):
        for t in range(37):
            backbone['offset'] = motions[:, t, :, :3]
            if moving_splats is not None:
                backbone.update(moving_splats[t])
            for name, pose in [('source_reconstruction', camera[:, :1]), ('target_view', target_camera)]:
                rendered = model.gs_renderer.render(backbone, camera, intr, pose, intr[:, :1])['image']
                rgb = (rendered[0, 0].float().clamp(0, 1).permute(1, 2, 0).detach().cpu().numpy() * 255).astype(np.uint8)
                previews[name].append(rgb)
            print('render', t, flush=True)
    for name, frames in previews.items():
        assert len(frames) == 37
        writer = cv2.VideoWriter(str(OUT / f'{name}.mp4'), cv2.VideoWriter_fourcc(*'mp4v'), 15, (518, 294))
        assert writer.isOpened()
        for frame in frames:
            writer.write(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
        writer.release()
    manifest = {'method': 'Official MoVieS SplatRecon/backbone/gs_renderer and released checkpoint, local static-video camera adapter',
                'official_source': 'https://github.com/chenguolin/MoVieS',
                'weight_sha256': sha(WEIGHT), 'input_npz_sha256': sha(INPUT),
                'source_frames': 13, 'generated_frames': 37, 'resolution': [518, 294],
                'subject_depth_median_model_units': subject_depth,
                'target_camera_c2w': target_camera[0,0].float().cpu().numpy().tolist(),
                'target_view_sha256': sha(OUT / 'target_view.mp4'),
                'source_reconstruction_sha256': sha(OUT / 'source_reconstruction.mp4'),
                'not_stage2_endpoint_conditioned': True,
                'target_view_not_ground_truth': True}
    (OUT / 'manifest.json').write_text(json.dumps(manifest, indent=2) + '\n')
    print(json.dumps(manifest, indent=2), flush=True)

if __name__ == '__main__':
    main()
