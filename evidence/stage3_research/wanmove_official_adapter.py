"""Call Wan-Move's checked-out trajectory functions on DD's content latent.

The functions are imported from upstream, not copied here. Visibility filtering
and alpha blending adapt the result to DD's full-video conditioning. This is an
inference-only module transfer, not the trained Wan-Move model.
"""
import importlib.util
from pathlib import Path
import torch

SOURCE = Path(__file__).resolve().parents[1] / 'paper_sources/Wan-Move/wan/modules/trajectory.py'


def _load_upstream():
    spec = importlib.util.spec_from_file_location('wanmove_upstream_trajectory', SOURCE)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def propagate_with_upstream(content, xy, visibility, confidence, *, alpha, boxes=None):
    if content.shape[0] != 1 or content.shape[2:] != (5, 80, 44):
        raise ValueError(f'Unexpected DD latent shape {tuple(content.shape)}')
    if xy.shape[0] != 17:
        raise ValueError(f'Expected 17 video-frame tracks, got {xy.shape}')
    if alpha == 0:
        return content, {'modified_latent_fraction': 0., 'reliable_tracks': 0}
    upstream = _load_upstream()
    device = content.device
    tracks = torch.as_tensor(xy, dtype=torch.float32, device=device)
    visible = torch.as_tensor(visibility * confidence > .5, dtype=torch.bool, device=device)
    visible = visible & visible[:1]
    if boxes is not None:
        boxes = torch.as_tensor(boxes, dtype=torch.float32, device=device)
        visible &= ((tracks[..., 0] >= boxes[:, None, 0]) &
                    (tracks[..., 0] <= boxes[:, None, 2]) &
                    (tracks[..., 1] >= boxes[:, None, 1]) &
                    (tracks[..., 1] <= boxes[:, None, 3]))
    eligible = visible[[0, 4, 8, 12, 16]].all(dim=0).nonzero().flatten()
    reliable = int(eligible.numel())
    if reliable < 8:
        raise ValueError(f'Only {reliable} reliable full-window tracks')
    # CoW returns one query per 448x224 pixel; cap upstream's Python loop.
    selected = eligible[torch.linspace(0, reliable - 1, min(2048, reliable), device=device).long()]
    tracks = tracks[:, selected]
    visible = visible[:, selected]
    torch.manual_seed(20261105)
    # Upstream converts pixel tracks to DD's five latent-grid positions.
    # Its feature map is not used: DD has no matching trained embedding input.
    _, positions = upstream.create_pos_feature_map(
        tracks, visible, [4, 16, 16], 1280, 704, 16,
        track_num=tracks.shape[1], device=device, dtype=torch.float32)
    official = upstream.replace_feature(content.clone(), positions[None])
    output = content + alpha * (official - content)
    if not torch.equal(output[:, :, :1], content[:, :, :1]):
        raise AssertionError('Wan-Move adapter changed first latent')
    delta = (output - content).abs()
    return output, {
        'reliable_tracks': reliable,
        'selected_tracks': int(selected.numel()),
        'modified_latent_fraction': float((delta[:, :, 1:] > 0).float().mean()),
        'mean_abs_feature_change': float(delta[:, :, 1:].float().mean()),
        'upstream_source': str(SOURCE),
    }
