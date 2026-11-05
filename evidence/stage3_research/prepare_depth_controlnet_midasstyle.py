"""Invert DD's Spectral_r visualization back to relative depth for ControlNet."""
import csv,json
from pathlib import Path
import cv2
import matplotlib
import numpy as np
from scipy.spatial import cKDTree
ROOT=Path(__file__).resolve().parents[1];HERE=Path(__file__).resolve().parent
record=json.load(open(ROOT/'vlm_dd_v12_balanced_case15_c/stage3/records/route_a/depthdirector/stage3/vlm_selected.json'))
meta=Path(record['metadata_csv']);row=next(csv.DictReader(meta.open()));source=meta.parent/row['depth']
out=HERE/'depth_controlnet_midasstyle_case15_c';out.mkdir(exist_ok=True)
video=out/'control.mp4';indices=[]
# DD's processor/base.py maps integer depth bins through Spectral_r, then
# its writer swaps RGB/BGR. OpenCV-decoded BGR therefore matches cmap RGB.
palette=(matplotlib.colormaps['Spectral_r'](np.arange(256))[:,:3]*255).astype(np.uint8)
tree=cKDTree(palette.astype(np.float32))
cap=cv2.VideoCapture(str(source));fps=cap.get(cv2.CAP_PROP_FPS);frames=[]
while True:
 okay,bgr=cap.read()
 if not okay:break
 distance,index=tree.query(bgr.reshape(-1,3).astype(np.float32),workers=8)
 index=index.reshape(bgr.shape[:2]);distance=distance.reshape(bgr.shape[:2]);indices.append(index)
 # MiDaS style: nearer pixels brighter, invalid DD depth black.
 inv=(255-index).astype(np.uint8);inv[index<=2]=0
 frames.append(cv2.cvtColor(inv,cv2.COLOR_GRAY2BGR))
 print('frame',len(frames)-1,'palette_error_median',float(np.median(distance)),'valid_fraction',float((index>2).mean()),flush=True)
cap.release();assert len(frames)==17
writer=cv2.VideoWriter(str(video),cv2.VideoWriter_fourcc(*'mp4v'),fps,(704,1280))
assert writer.isOpened()
for frame in frames:writer.write(frame)
writer.release()
manifest={'source':str(source),'method':'Invert DD official Spectral_r depth visualization LUT, then 255-index for near-is-bright monochrome control; index<=2 invalid black. Approximate MiDaS style, not MiDaS model output.','frames':17,'fps':fps,'video':str(video),'source_code':'/public/wufeier/automatic_bullet_time/models/DepthDirector/tools/recam/processor/base.py','palette':'Spectral_r'}
(out/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
print(video,video.stat().st_size,flush=True)
