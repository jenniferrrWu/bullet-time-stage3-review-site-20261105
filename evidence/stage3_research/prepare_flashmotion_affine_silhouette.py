"""Build complete source-person silhouettes aligned to target-view boxes.

Ablation against depth-projected pseudo tracks: an affine box mapping preserves
body parts if depth/z-buffer holes corrupt the projected control map. It is an
estimated conditioning map, not target-view geometry ground truth.
"""
import json
from pathlib import Path
import cv2,numpy as np
ROOT=Path(__file__).resolve().parents[1];HERE=Path(__file__).resolve().parent
src=json.loads((ROOT/'pose_trial/v12_balanced_case15_c_source_all/tracks.json').read_text())
tgt=json.loads((ROOT/'pose_trial/v12_balanced_case15_c_target_warp_all/tracks.json').read_text())
cap=cv2.VideoCapture(src['source_video']);H,W=1280,704;frames=[];rows=[]
for t in range(17):
 ok,frame=cap.read();assert ok and frame.shape[:2]==(H,W)
 pose=src['frames'][t];a=np.array(pose['bbox_xyxy'],np.float32);b=np.array(tgt['frames'][t]['bbox_xyxy'],np.float32)
 x1,y1,x2,y2=np.rint(a).astype(int);x1=max(0,x1);y1=max(0,y1);x2=min(W,x2);y2=min(H,y2)
 mask=np.full((H,W),cv2.GC_BGD,np.uint8);mask[y1:y2,x1:x2]=cv2.GC_PR_FGD
 keys=pose['keypoints_xy'];accepted=pose['accepted']
 for j,k in ((5,6),(5,7),(7,9),(6,8),(8,10),(5,11),(6,12),(11,12),(11,13),(13,15),(12,14),(14,16)):
  if accepted[j] and accepted[k]:cv2.line(mask,tuple(np.rint(keys[j]).astype(int)),tuple(np.rint(keys[k]).astype(int)),cv2.GC_FGD,9)
 for j in (5,6,11,12):
  if accepted[j]:cv2.circle(mask,tuple(np.rint(keys[j]).astype(int)),9,cv2.GC_FGD,-1)
 cv2.grabCut(frame,mask,None,np.zeros((1,65),np.float64),np.zeros((1,65),np.float64),3,cv2.GC_INIT_WITH_MASK)
 fg=np.uint8((mask==cv2.GC_FGD)|(mask==cv2.GC_PR_FGD))*255
 fg=cv2.morphologyEx(fg,cv2.MORPH_CLOSE,cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(9,9)))
 scale_x=(b[2]-b[0])/(a[2]-a[0]);scale_y=(b[3]-b[1])/(a[3]-a[1]);M=np.float32([[scale_x,0,b[0]-scale_x*a[0]],[0,scale_y,b[1]-scale_y*a[1]]])
 aligned=cv2.warpAffine(fg,M,(W,H),flags=cv2.INTER_NEAREST)
 c=np.zeros((H,W,3),np.uint8);c[aligned>0]=(73,76,124);frames.append(c)
 rows.append({'frame':t,'source_mask_pixels':int((fg>0).sum()),'target_mask_pixels':int((aligned>0).sum()),'source_box':a.tolist(),'target_box':b.tolist()})
 print(t,rows[-1]['target_mask_pixels'],flush=True)
cap.release();out=HERE/'flashmotion_affine_silhouette_case15_c';out.mkdir(exist_ok=True)
v=cv2.VideoWriter(str(out/'trajectory.mp4'),cv2.VideoWriter_fourcc(*'mp4v'),15,(W,H))
for f in frames:v.write(f)
v.release();cv2.imwrite(str(out/'contact.png'),np.concatenate([cv2.resize(frames[i],(176,320)) for i in (0,4,8,12,16)],axis=1))
(out/'manifest.json').write_text(json.dumps({'method':'DWPose-seeded source GrabCut silhouette, affine-aligned per frame to actual DD target-warp DWPose box','not_target_view_geometry_ground_truth':True,'source_video':src['source_video'],'target_warp_video':tgt['source_video'],'rows':rows},indent=2)+'\n')
print(out,flush=True)
