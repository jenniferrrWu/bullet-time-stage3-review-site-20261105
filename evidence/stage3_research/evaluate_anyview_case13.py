"""Stage3 whole-film review evidence for AnyView case13 B target-view generation."""
import json
from pathlib import Path
import cv2
import numpy as np

ROOT=Path(__file__).resolve().parents[1]
SOURCE=ROOT/'vlm_dd_balanced_case13_b/action.mp4'
BASELINE=ROOT/'vlm_dd_balanced_case13_b/stage3/raw/route_a/depthdirector/stage3/vlm_selected/vlm_v11/preprocess/Rotate,0,45,0,0,0/depthdirector.mp4'
ENDPOINT=ROOT/'vlm_dd_balanced_case13_b/stage3/raw/route_a/depthdirector/stage3/vlm_selected/vlm_v11/preprocess/effective_endpoint.png'
OUT=ROOT/'paper_sources/AnyView-DVS/outputs/case13_b_vlm45_torch211'

def read(path):
 cap=cv2.VideoCapture(str(path)); frames=[]
 while True:
  ok,frame=cap.read()
  if not ok:break
  frames.append(frame)
 cap.release();return frames

source=read(SOURCE);baseline=read(BASELINE);generated=read(OUT/'pred.mp4')
assert len(source)==len(baseline)==len(generated)==37,(len(source),len(baseline),len(generated))
h,w=generated[0].shape[:2]
source=[cv2.resize(f,(w,h),interpolation=cv2.INTER_AREA) for f in source]
baseline=[cv2.resize(f,(w,h),interpolation=cv2.INTER_AREA) for f in baseline]
endpoint=cv2.resize(cv2.imread(str(ENDPOINT)),(w,h),interpolation=cv2.INTER_AREA)
assert endpoint is not None
metrics={'frames':37,'width':w,'height':h,'no_target_view_ground_truth':True,
 'anyview_first_frame_mae_vs_real_stage2_endpoint':float(np.abs(generated[0].astype(np.float32)-endpoint).mean()),
 'baseline_first_frame_mae_vs_real_stage2_endpoint':float(np.abs(baseline[0].astype(np.float32)-endpoint).mean()),
 'anyview_dark_fraction_by_frame':[],
 'anyview_luma_mean_by_frame':[],
 'anyview_interframe_mae':[],
 'baseline_interframe_mae':[]}
for i,g in enumerate(generated):
 gray=cv2.cvtColor(g,cv2.COLOR_BGR2GRAY)
 metrics['anyview_dark_fraction_by_frame'].append(float((gray<15).mean()))
 metrics['anyview_luma_mean_by_frame'].append(float(gray.mean()))
 if i:
  metrics['anyview_interframe_mae'].append(float(np.abs(g.astype(np.float32)-generated[i-1]).mean()))
  metrics['baseline_interframe_mae'].append(float(np.abs(baseline[i].astype(np.float32)-baseline[i-1]).mean()))
metrics['anyview_dark_fraction_mean']=float(np.mean(metrics['anyview_dark_fraction_by_frame']))
metrics['anyview_interframe_mae_mean']=float(np.mean(metrics['anyview_interframe_mae']))
metrics['baseline_interframe_mae_mean']=float(np.mean(metrics['baseline_interframe_mae']))
(OUT/'evaluation.json').write_text(json.dumps(metrics,indent=2)+'\n')
idx=[0,6,12,18,24,30,36]
rows=[]
for name,frames in [('Source action',source),('DD Stage3 baseline',baseline),('AnyView 45deg',generated)]:
 cells=[]
 for i in idx:
  f=frames[i].copy();cv2.rectangle(f,(0,0),(290,28),(0,0,0),-1)
  cv2.putText(f,f'{name} f{i:02d}',(4,20),cv2.FONT_HERSHEY_SIMPLEX,.52,(255,255,255),1,cv2.LINE_AA)
  cells.append(f)
 rows.append(cv2.hconcat(cells))
cv2.imwrite(str(OUT/'contact.jpg'),cv2.vconcat(rows),[int(cv2.IMWRITE_JPEG_QUALITY),87])
cv2.imwrite(str(OUT/'endpoint_first.jpg'),cv2.hconcat([endpoint,baseline[0],generated[0]]),[int(cv2.IMWRITE_JPEG_QUALITY),90])
print(json.dumps({k:v for k,v in metrics.items() if not isinstance(v,list)},indent=2),flush=True)
