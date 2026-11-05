"""Use the official NLF v0.3.2 TorchScript estimator for case13 target-view geometry.

Only camera registration and rasterization are local glue. Projected geometry is an
estimate, not a target-view label or a full NLF video model.
"""
import csv
import hashlib
import json
from pathlib import Path

import cv2
import numpy as np
import torch
import torchvision  # registers torchvision::nms for the official TorchScript

ROOT = Path(__file__).resolve().parents[1]
HERE = Path(__file__).resolve().parent
MODEL = ROOT / 'paper_sources/NLF-weights/nlf_l_multi_0.3.2.patch6.torchscript'
SOURCE = ROOT / 'vlm_dd_balanced_case13_b/action.mp4'
RECORD = ROOT / 'vlm_dd_balanced_case13_b/stage3/records/route_a/depthdirector/stage3/vlm_selected.json'
ANCHOR_MASK = HERE / 'case13_lama_background/mask.png'
OUT = HERE / 'nlf_case13_b_target_silhouette'
OUT.mkdir(exist_ok=True)


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def bbox(mask):
    ys, xs = np.where(mask > 0)
    return np.array([xs.min(), ys.min(), xs.max(), ys.max()], dtype=np.float64)


def silhouette(uv, width=640, height=352):
    points = np.round(uv).astype(np.int32)
    valid = (points[:, 0] >= -8) & (points[:, 0] < width + 8) & (points[:, 1] >= -8) & (points[:, 1] < height + 8)
    mask = np.zeros((height, width), np.uint8)
    for x, y in points[valid]:
        cv2.circle(mask, (int(x), int(y)), 3, 255, -1)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((7, 7), np.uint8))
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    for contour in contours:
        if cv2.contourArea(contour) > 20:
            cv2.drawContours(mask, [contour], -1, 255, -1)
    return mask


record = json.loads(RECORD.read_text())
meta = Path(record['metadata_csv'])
row = next(csv.DictReader(meta.open()))
anchor = (meta.parent / row['input_image']).resolve()
camera = json.loads(Path(record['camera_path_file']).read_text())
source_c2w = np.array(camera['source_c2w'], dtype=np.float64)
target_c2w = np.array(camera['target_c2w'][0], dtype=np.float64)
relative = np.linalg.inv(target_c2w) @ source_c2w
rotation = relative[:3, :3]
anchor_image = cv2.resize(cv2.imread(str(anchor)), (640, 352))
anchor_mask = cv2.resize(cv2.imread(str(ANCHOR_MASK), cv2.IMREAD_GRAYSCALE), (640, 352), interpolation=cv2.INTER_NEAREST)
anchor_box = bbox(anchor_mask)

model = torch.jit.load(str(MODEL), map_location='cuda').eval()
cap = cv2.VideoCapture(str(SOURCE))
frames = []
while True:
    ok, image = cap.read()
    if not ok:
        break
    frames.append(cv2.resize(image, (640, 352)))
cap.release()
assert len(frames) == 37, len(frames)

predictions = []
misses = []
for index, frame in enumerate(frames):
    batch = torch.from_numpy(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB).copy()).permute(2, 0, 1).unsqueeze(0).to('cuda')
    with torch.inference_mode():
        result = model.detect_smpl_batched(batch)
    candidates = result['vertices3d'][0]
    if not len(candidates):
        misses.append(index)
        predictions.append(None)
        print('miss', index, flush=True)
        continue
    boxes = result['boxes'][0].detach().cpu().numpy()
    selected = int(np.argmax((boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])))
    verts = candidates[selected].detach().cpu().numpy().astype(np.float32)
    joints = result['joints3d'][0][selected].detach().cpu().numpy().astype(np.float32)
    uv = result['vertices2d'][0][selected].detach().cpu().numpy().astype(np.float32)
    predictions.append((verts, joints, uv))
    print('frame', index, 'detections', len(candidates), flush=True)
assert predictions[0] is not None
assert len(misses) <= 3, misses
for index in misses:
    nearest = min((i for i, p in enumerate(predictions) if p is not None), key=lambda i: abs(i-index))
    predictions[index] = predictions[nearest]

first_verts, first_joints, first_uv = predictions[0]
q = first_verts[:, :2] / first_verts[:, 2:3]
fx, cx = np.linalg.lstsq(np.c_[q[:, 0], np.ones(len(q))], first_uv[:, 0], rcond=None)[0]
fy, cy = np.linalg.lstsq(np.c_[q[:, 1], np.ones(len(q))], first_uv[:, 1], rcond=None)[0]
reprojected = np.c_[fx * q[:, 0] + cx, fy * q[:, 1] + cy]
rmse = float(np.sqrt(np.mean(np.sum((reprojected - first_uv) ** 2, axis=1))))
assert rmse < 0.05, rmse
K = [float(fx), float(fy), float(cx), float(cy)]
pivot = first_joints[0].astype(np.float64)

def project_rotated(vertices):
    rotated = (vertices.astype(np.float64) - pivot) @ rotation.T + pivot
    z = np.maximum(rotated[:, 2], 100.0)
    return np.c_[fx * rotated[:, 0] / z + cx, fy * rotated[:, 1] / z + cy]

first_rotated_uv = project_rotated(first_verts)
raw_box = np.array([
    np.percentile(first_rotated_uv[:, 0], 1), np.percentile(first_rotated_uv[:, 1], 1),
    np.percentile(first_rotated_uv[:, 0], 99), np.percentile(first_rotated_uv[:, 1], 99),
])
raw_center = (raw_box[:2] + raw_box[2:]) / 2
anchor_center = (anchor_box[:2] + anchor_box[2:]) / 2
raw_size = raw_box[2:] - raw_box[:2]
anchor_size = anchor_box[2:] - anchor_box[:2]
scale = float(min(anchor_size / raw_size))
assert 0.2 < scale < 5, (scale, raw_box, anchor_box)

output = OUT / 'trajectory.mp4'
writer = cv2.VideoWriter(str(output), cv2.VideoWriter_fourcc(*'mp4v'), 30000/1001, (1280, 704))
assert writer.isOpened()
coverages = []
for index, (verts, joints, uv) in enumerate(predictions):
    target_uv = (project_rotated(verts) - raw_center) * scale + anchor_center
    mask = silhouette(target_uv)
    coverages.append(float((mask > 0).mean()))
    control = np.zeros((352, 640, 3), np.uint8)
    control[mask > 0] = (30, 30, 132)
    writer.write(cv2.resize(control, (1280, 704), interpolation=cv2.INTER_NEAREST))
    if index in (0, 9, 18, 27, 36):
        cv2.imwrite(str(OUT / f'target_mask_{index:02d}.png'), mask)
        overlay = cv2.addWeighted(anchor_image, .7, np.dstack([mask, np.zeros_like(mask), np.zeros_like(mask)]), .3, 0)
        cv2.imwrite(str(OUT / f'target_overlay_{index:02d}.jpg'), overlay)
writer.release()
source_mask = silhouette(first_uv)
cv2.imwrite(str(OUT / 'source_mask_00.png'), source_mask)
cv2.imwrite(str(OUT / 'source_overlay_00.jpg'), cv2.addWeighted(frames[0], .7, np.dstack([source_mask, np.zeros_like(source_mask), np.zeros_like(source_mask)]), .3, 0))
first_mask = cv2.imread(str(OUT / 'target_mask_00.png'), cv2.IMREAD_GRAYSCALE) > 0
first_ref = anchor_mask > 0
intersection = np.logical_and(first_mask, first_ref).sum()
union = np.logical_or(first_mask, first_ref).sum()
manifest = {
    'method': 'Official NLF v0.3.2 TorchScript detect_smpl_batched, original DD camera relative yaw; local projection/rasterization/first-frame registration',
    'official_source': 'https://github.com/isarandi/nlf',
    'official_release': 'https://github.com/isarandi/nlf/releases/tag/v0.3.2',
    'model_sha256_local': sha(MODEL),
    'source_sha256': sha(SOURCE),
    'anchor_sha256': sha(anchor),
    'camera_sha256': sha(record['camera_path_file']),
    'trajectory_sha256': sha(output),
    'frames': len(frames), 'missing_detections': misses,
    'K': K, 'nlf_source_projection_rmse_px': rmse,
    'target_first_frame_bbox_iou_mask': float(intersection / union),
    'anchor_box_half_resolution': anchor_box.tolist(),
    'nlf_rotated_raw_box_half_resolution': raw_box.tolist(),
    'registration_scale': scale,
    'mask_coverage_mean': float(np.mean(coverages)),
    'not_target_view_ground_truth': True,
    'not_official_nlf_video_pipeline': True,
}
(OUT / 'manifest.json').write_text(json.dumps(manifest, indent=2) + '\n')
print(json.dumps(manifest, indent=2), flush=True)
