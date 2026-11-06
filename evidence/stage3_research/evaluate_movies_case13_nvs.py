"""Whole-film visual evidence for the official MoVieS case13 run."""
import json
from pathlib import Path
import cv2
import numpy as np

ROOT=Path(__file__).resolve().parents[1]
HERE=Path(__file__).resolve().parent
OUT=HERE/'movies_case13_b_nvs_37'
SOURCE=ROOT/'vlm_dd_balanced_case13_b/action.mp4'
BASELINE=ROOT/'vlm_dd_balanced_case13_b/stage3/raw/route_a/depthdirector/stage3/vlm_selected/vlm_v11/preprocess/Rotate,0,45,0,0,0/depthdirector.mp4'
ENDPOINT=ROOT/'vlm_dd_balanced_case13_b/stage3/raw/route_a/depthdirector/stage3/vlm_selected/vlm_v11/preprocess/effective_endpoint.png'


def read_video(path):
    cap=cv2.VideoCapture(str(path)); frames=[]
    while True:
        ok,fr=cap.read()
        if not ok: break
        frames.append(fr)
    cap.release()
    return frames

source=read_video(SOURCE)
baseline=read_video(BASELINE)
recon=read_video(OUT/'source_reconstruction.mp4')
target=read_video(OUT/'target_view.mp4')
assert len(source)==len(baseline)==len(recon)==len(target)==37
assert all(f.shape[:2]==(294,518) for f in recon+target)
source=[cv2.resize(f[:,20:1260],(518,294),interpolation=cv2.INTER_AREA) for f in source]
baseline=[cv2.resize(f,(518,294),interpolation=cv2.INTER_AREA) for f in baseline]
endpoint=cv2.imread(str(ENDPOINT)); assert endpoint is not None
endpoint=cv2.resize(endpoint,(518,294),interpolation=cv2.INTER_AREA)
metrics={'frames':37,'target_view_not_ground_truth':True,
         'source_reconstruction_psnr_by_frame':[],
         'source_reconstruction_mae_by_frame':[],
         'target_first_frame_mae_vs_real_stage2_endpoint':float(np.abs(target[0].astype(np.float32)-endpoint).mean()),
         'baseline_first_frame_mae_vs_real_stage2_endpoint':float(np.abs(baseline[0].astype(np.float32)-endpoint).mean())}
for a,b in zip(recon,source):
    mse=np.mean((a.astype(np.float32)-b.astype(np.float32))**2)
    metrics['source_reconstruction_psnr_by_frame'].append(float(10*np.log10(255**2/max(mse,1e-9))))
    metrics['source_reconstruction_mae_by_frame'].append(float(np.abs(a.astype(np.float32)-b).mean()))
metrics['source_reconstruction_psnr_mean']=float(np.mean(metrics['source_reconstruction_psnr_by_frame']))
metrics['source_reconstruction_mae_mean']=float(np.mean(metrics['source_reconstruction_mae_by_frame']))
(OUT/'evaluation.json').write_text(json.dumps(metrics,indent=2)+'\n')
idx=[0,6,12,18,24,30,36]
rows=[]
for title,frames in [('source video',source),('DD baseline Stage3',baseline),('MoVieS source recon',recon),('MoVieS target 45deg',target)]:
    cells=[]
    for i in idx:
        frame=frames[i].copy()
        cv2.rectangle(frame,(0,0),(330,30),(0,0,0),-1)
        cv2.putText(frame,f'{title}  f{i:02d}',(5,21),cv2.FONT_HERSHEY_SIMPLEX,.56,(255,255,255),1,cv2.LINE_AA)
        cells.append(frame)
    rows.append(cv2.hconcat(cells))
cv2.imwrite(str(OUT/'contact.jpg'),cv2.vconcat(rows),[int(cv2.IMWRITE_JPEG_QUALITY),88])
cv2.imwrite(str(OUT/'endpoint_vs_movies_first.jpg'),cv2.hconcat([endpoint,baseline[0],target[0]]),[int(cv2.IMWRITE_JPEG_QUALITY),90])
print(json.dumps({k:v for k,v in metrics.items() if not isinstance(v,list)},indent=2),flush=True)
