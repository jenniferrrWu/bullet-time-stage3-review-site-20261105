"""Construct an observed-person mask for LaMa background completion."""
import csv,json
from pathlib import Path
import cv2,numpy as np
ROOT=Path(__file__).resolve().parents[1];HERE=Path(__file__).resolve().parent
base=ROOT/'vlm_dd_balanced_case13_b';rec=json.loads((base/'stage3/records/route_a/depthdirector/stage3/vlm_selected.json').read_text())
meta=Path(rec['metadata_csv']);row=next(csv.DictReader(meta.open()));endpoint=(meta.parent/row['input_image']).resolve()
frame=cv2.imread(str(endpoint));h,w=frame.shape[:2];track=json.loads((ROOT/'pose_trial/balanced_case13_b_native_all/tracks.json').read_text())
x1,y1,x2,y2=map(int,track['frames'][0]['bbox_xyxy']);margin=35
x1=max(1,x1-margin);y1=max(1,y1-margin);x2=min(w-2,x2+margin);y2=min(h-2,y2+margin)
mask=np.zeros((h,w),np.uint8);bg=np.zeros((1,65),np.float64);fg=np.zeros((1,65),np.float64)
cv2.grabCut(frame,mask,(x1,y1,x2-x1,y2-y1),bg,fg,8,cv2.GC_INIT_WITH_RECT)
mask=np.where((mask==cv2.GC_FGD)|(mask==cv2.GC_PR_FGD),255,0).astype(np.uint8)
mask=cv2.morphologyEx(mask,cv2.MORPH_CLOSE,cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(17,17)))
mask=cv2.dilate(mask,cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(31,31)))
out=HERE/'case13_lama_background';out.mkdir(exist_ok=True)
cv2.imwrite(str(out/'endpoint.png'),frame);cv2.imwrite(str(out/'mask.png'),mask)
preview=frame.copy();preview[mask>0]=(0.55*preview[mask>0]+0.45*np.array([0,0,255])).astype(np.uint8)
cv2.rectangle(preview,(x1,y1),(x2,y2),(255,255,0),3)
cv2.imwrite(str(out/'mask_preview.jpg'),preview,[cv2.IMWRITE_JPEG_QUALITY,88])
(out/'mask_manifest.json').write_text(json.dumps({'endpoint':str(endpoint),'bbox_xyxy':[x1,y1,x2,y2],'mask_fraction':float((mask>0).mean()),'method':'Official DWPose bbox on actual Stage2 endpoint, OpenCV GrabCut then close/dilate; observed foreground mask for official LaMa inpainting. Local mask construction, no target-view ground truth.'},indent=2)+'\n')
print(out,mask.mean()/255)
