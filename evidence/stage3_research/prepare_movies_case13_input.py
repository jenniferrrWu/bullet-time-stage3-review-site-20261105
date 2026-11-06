"""Adapt the fixed-camera 37-frame case13 source for official MoVieS inference."""
import hashlib
import json
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / 'vlm_dd_balanced_case13_b/action.mp4'
CAMERA = ROOT / 'vlm_dd_balanced_case13_b/stage3/raw/route_a/depthdirector/stage3/vlm_selected/vlm_v11/preprocess/actual_camera.json'
OUT = ROOT / 'paper_sources/MoVieS/resources/DAVIS/case13_b_static.npz'
OUT.parent.mkdir(parents=True, exist_ok=True)
cap = cv2.VideoCapture(str(SOURCE))
frames=[]
while True:
    ok, frame=cap.read()
    if not ok: break
    frames.append(frame)
cap.release()
assert len(frames)==37 and all(f.shape[:2]==(704,1280) for f in frames)
indices=np.linspace(0,36,13,dtype=int)
images=[]
for i in indices:
    cropped=frames[i][:,20:1260]
    resized=cv2.resize(cropped,(518,294),interpolation=cv2.INTER_AREA)
    images.append(cv2.cvtColor(resized,cv2.COLOR_BGR2RGB).transpose(2,0,1).astype(np.float32)/255)
images=np.stack(images)
# Focal length estimated from official NLF 3D/2D output on these same source frames.
fx_source=614.7142962974289*2
fy_source=614.714296194624*2
cx_source=319.50000032653026*2-20
cy_source=175.49999993516084*2
intrinsics=np.array([fx_source/1240,fy_source/704,cx_source/1240,cy_source/704],np.float32)
C2W=np.repeat(np.eye(4,dtype=np.float32)[None],13,axis=0)
np.savez_compressed(OUT,images=images,C2W=C2W,fxfycxcy=np.repeat(intrinsics[None],13,axis=0))
cam=json.loads(CAMERA.read_text())
source=np.array(cam['source_c2w'],dtype=np.float64)
target=np.array(cam['target_c2w'][0],dtype=np.float64)
relative=np.linalg.inv(source)@target
meta={'source':str(SOURCE),'source_sha256':hashlib.sha256(SOURCE.read_bytes()).hexdigest(),
      'camera':str(CAMERA),'camera_sha256':hashlib.sha256(CAMERA.read_bytes()).hexdigest(),
      'input_indices':indices.tolist(),'input_shape':list(images.shape),'intrinsics_normalized':intrinsics.tolist(),
      'source_C2W':'identity, static observed camera','target_C2W_relative_to_source':relative.tolist(),
      'npz':str(OUT),'npz_sha256':hashlib.sha256(OUT.read_bytes()).hexdigest(),
      'not_official_DAVIS_preprocessing':True,'not_target_view_ground_truth':True}
OUT.with_suffix('.json').write_text(json.dumps(meta,indent=2)+'\n')
print(json.dumps(meta,indent=2))
