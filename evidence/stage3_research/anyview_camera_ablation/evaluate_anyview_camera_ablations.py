"""Whole-film camera ablation audit for AnyView case13 B."""
import json,math
from pathlib import Path
import cv2,numpy as np
ROOT=Path(__file__).resolve().parents[1]
BASE=ROOT/'paper_sources/AnyView-DVS/outputs'
SOURCE=ROOT/'vlm_dd_balanced_case13_b/action.mp4'
ENDPOINT=ROOT/'vlm_dd_balanced_case13_b/stage3/raw/route_a/depthdirector/stage3/vlm_selected/vlm_v11/preprocess/effective_endpoint.png'
OUT=ROOT/'stage3_research/anyview_case13_camera_ablation';OUT.mkdir(exist_ok=True)
def read(path):
 c=cv2.VideoCapture(str(path));a=[]
 while True:
  ok,f=c.read()
  if not ok:break
  a.append(f)
 c.release();return a
source=read(SOURCE)
folders={'sameview':'case13_b_sameview','plus45':'case13_b_vlm45_torch211','reverse45':'case13_b_reverse45'}
vid={k:read(BASE/f/'pred.mp4') for k,f in folders.items()}
assert len(source)==37 and all(len(v)==37 for v in vid.values())
h,w=vid['plus45'][0].shape[:2]
source=[cv2.resize(x,(w,h),interpolation=cv2.INTER_AREA) for x in source]
endpoint=cv2.resize(cv2.imread(str(ENDPOINT)),(w,h),interpolation=cv2.INTER_AREA)
result={'frames':37,'resolution':[h,w],'no_true_target_view':True,'comparisons':{}}
for name,frames in vid.items():
 maes=[float(np.abs(a.astype(np.float32)-b.astype(np.float32)).mean()) for a,b in zip(frames,source)]
 temporal=[float(np.abs(frames[i].astype(np.float32)-frames[i-1].astype(np.float32)).mean()) for i in range(1,37)]
 seam=float(np.abs(frames[0].astype(np.float32)-endpoint.astype(np.float32)).mean())
 result['comparisons'][name]={'source_view_mae_mean':float(np.mean(maes)),'source_view_psnr_mean':float(np.mean([10*math.log10(255**2/max(float(np.mean((a.astype(float)-b.astype(float))**2)),1e-12)) for a,b in zip(frames,source)])),'first_frame_mae_vs_real_stage2_endpoint':seam,'interframe_mae_mean':float(np.mean(temporal))}
idx=[0,6,12,18,24,30,36]
rows=[]
for name,frames in [('source',source),*vid.items()]:
 cells=[]
 for i in idx:
  f=frames[i].copy();cv2.rectangle(f,(0,0),(168,25),(0,0,0),-1);cv2.putText(f,f'{name} f{i:02d}',(4,18),cv2.FONT_HERSHEY_SIMPLEX,.5,(255,255,255),1,cv2.LINE_AA);cells.append(f)
 rows.append(cv2.hconcat(cells))
cv2.imwrite(str(OUT/'contact.jpg'),cv2.vconcat(rows),[int(cv2.IMWRITE_JPEG_QUALITY),85])
cv2.imwrite(str(OUT/'endpoint_first.jpg'),cv2.hconcat([endpoint,*[v[0] for v in vid.values()]]),[int(cv2.IMWRITE_JPEG_QUALITY),90])
(OUT/'metrics.json').write_text(json.dumps(result,indent=2)+'\n')
print(json.dumps(result,indent=2),flush=True)
