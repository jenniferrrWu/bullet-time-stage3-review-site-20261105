"""Use Motion-I2V's official ForwardWarp module with measured CoW flow.

This is an inference-only DD conditioning adapter. It does not claim to run
Motion-I2V's pretrained flow predictor or motion-augmented temporal attention.
"""
import ast
from pathlib import Path
import torch
import torch.nn as nn
import torch.nn.functional as F

SOURCE = Path(__file__).resolve().parents[1] / 'paper_sources/Motion-I2V/animation/models/forward_motion_module.py'


def _upstream_forward_warp():
    source = ast.parse(SOURCE.read_text())
    classes = [node for node in source.body if isinstance(node, ast.ClassDef) and node.name == 'ForwardWarp']
    assert len(classes) == 1
    namespace = {'torch': torch, 'nn': nn, 'F': F}
    exec(compile(ast.Module(body=classes, type_ignores=[]), str(SOURCE), 'exec'), namespace)
    return namespace['ForwardWarp']()


def propagate_with_forward_warp(content, xy, visibility, confidence, *, alpha, boxes=None):
    if content.shape[0] != 1 or content.shape[2:] != (5, 80, 44):
        raise ValueError(f'Unexpected DD latent shape {tuple(content.shape)}')
    if alpha == 0:
        return content, {'modified_latent_fraction': 0.}
    device = content.device
    xy = torch.as_tensor(xy, device=device, dtype=torch.float32).reshape(17, 224, 448, 2)
    valid = torch.as_tensor(visibility * confidence > .5, device=device).reshape(17, 224, 448)
    valid = valid & valid[:1]
    if boxes is not None:
        boxes = torch.as_tensor(boxes, device=device, dtype=torch.float32)
        valid &= ((xy[..., 0] >= boxes[:, None, None, 0]) &
                  (xy[..., 0] <= boxes[:, None, None, 2]) &
                  (xy[..., 1] >= boxes[:, None, None, 1]) &
                  (xy[..., 1] <= boxes[:, None, None, 3]))
    warp = _upstream_forward_warp()
    output = content.clone()
    stats = []
    first = content[:, :, 0].float().contiguous()
    for t in range(1, 5):
        source_t = 4 * t
        delta = xy[source_t] - xy[0]
        flow = torch.stack((delta[..., 1] / 16, delta[..., 0] / 16), dim=0)[None]
        flow = F.interpolate(flow, size=(80, 44), mode='bilinear', align_corners=False)
        support = F.interpolate(valid[source_t][None, None].float(), size=(80, 44), mode='area')
        flow = flow * (support > .5)
        warped, occupancy = warp(first, flow.contiguous())
        normalized = warped / occupancy.clamp_min(1e-5)
        gate = alpha * occupancy.clamp(0, 1) * support
        old = content[:, :, t].float()
        updated = old + gate * (normalized - old)
        if not torch.isfinite(updated).all():
            raise ValueError(f'Non-finite Motion-I2V warp at {t}')
        output[:, :, t] = updated.to(content.dtype)
        stats.append({'latent_frame': t, 'source_frame': source_t,
                      'support_fraction': float((gate > .1).float().mean()),
                      'mean_abs_feature_change': float((updated - old).abs().mean())})
    assert torch.equal(output[:, :, :1], content[:, :, :1])
    return output, {'frames': stats, 'upstream_source': str(SOURCE)}
