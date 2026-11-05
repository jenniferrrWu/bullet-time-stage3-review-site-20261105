"""Make a target-view silhouette-control video for official FlashMotion.

CoW source points are projected by DD depth/camera. Source segmentation here is
GrabCut with DWPose seeds; the output is an estimated control map, not GT.
"""
import json
from pathlib import Path
import cv2
import numpy as np
ROOT=Path(__file__).resolve().parents[1]
HERE=Path(__file__).resolve().parent
manifest=json.loads((HERE/'case15_c_dense_tracks/manifest.json').read_text())
source_video=Path(manifest['source'])
source=np.load(HERE/'case15_c_dense_tracks/dense_tracks.npz')
target=np.load(HERE/'case15_c_target_view_pseudotracks/tracks.npz')
poses=json.loads((ROOT/'pose_trial/v12_balanced_case15_c_source_all/tracks.json').read_text())['frames']
cap=cv2.VideoCapture(str(source_video))
W,H=704,1280
assert source['xy'].shape==(17,100352,2) and target['xy'].shape==(17,100352,2)
frames=[];rows=[]
for t in range(17):
 ok,frame=cap.read();assert ok and frame.shape[:2]==(H,W)
 pose=poses[t];b=np.array(pose['bbox_xyxy']).astype(int);x1,y1,x2,y2=b
 x1=max(0,x1);y1=max(0,y1);x2=min(W,x2);y2=min(H,y2)
 mask=np.full((H,W),cv2.GC_BGD,np.uint8)
 mask[y1:y2,x1:x2]=cv2.GC_PR_FGD
 keys=pose['keypoints_xy'];accepted=pose['accepted']
 edges=[(5,6),(5,7),(7,9),(6,8),(8,10),(5,11),(6,12),(11,12),(11,13),(13,15),(12,14),(14,16)]
 for j,k in edges:
  if accepted[j] and accepted[k]:
   a=tuple(np.rint(keys[j]).astype(int));z=tuple(np.rint(keys[k]).astype(int))
   cv2.line(mask,a,z,cv2.GC_FGD,thickness=9)
 for j in (5,6,11,12):
  if accepted[j]:cv2.circle(mask,tuple(np.rint(keys[j]).astype(int)),9,cv2.GC_FGD,-1)
 bg=np.zeros((1,65),np.float64);fg=np.zeros((1,65),np.float64)
 cv2.grabCut(frame,mask,None,bg,fg,3,cv2.GC_INIT_WITH_MASK)
 srcmask=np.uint8((mask==cv2.GC_FGD)|(mask==cv2.GC_PR_FGD))
 srcmask=cv2.morphologyEx(srcmask,cv2.MORPH_CLOSE,cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(9,9)))
 sx=source['xy'][t];tx=target['xy'][t];valid=target['visibility'][t] & np.isfinite(sx).all(axis=1) & np.isfinite(tx).all(axis=1)
 si=np.flatnonzero(valid);u=np.clip(np.rint(sx[si,0]).astype(int),0,W-1);v=np.clip(np.rint(sx[si,1]).astype(int),0,H-1)
 si=si[srcmask[v,u]>0]
 dest=np.zeros((H,W),np.uint8)
 u=np.clip(np.rint(tx[si,0]).astype(int),0,W-1);v=np.clip(np.rint(tx[si,1]).astype(int),0,H-1)
 dest[v,u]=255
 dest=cv2.morphologyEx(dest,cv2.MORPH_CLOSE,cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(11,11)))
 dest=cv2.dilate(dest,cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(5,5)))
 count,labels,stats,_=cv2.connectedComponentsWithStats(dest)
 if count>1:
  largest=1+np.argmax(stats[1:,cv2.CC_STAT_AREA]);dest=np.uint8(labels==largest)*255
 contours,_=cv2.findContours(dest,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
 cv2.drawContours(dest,contours,-1,255,cv2.FILLED)
 control=np.zeros((H,W,3),np.uint8);control[dest>0]=(73,76,124)
 frames.append(control)
 rows.append({'frame':t,'source_grabcut_pixels':int(srcmask.sum()),'projected_person_points':len(si),'target_mask_pixels':int((dest>0).sum())})
 print(t,rows[-1],flush=True)
cap.release()
out=HERE/'flashmotion_target_silhouette_case15_c';out.mkdir(exist_ok=True)
writer=cv2.VideoWriter(str(out/'trajectory.mp4'),cv2.VideoWriter_fourcc(*'mp4v'),15,(W,H))
for f in frames:writer.write(f)
writer.release()
sel=[0,4,8,12,16]
contact=np.concatenate([cv2.resize(frames[i],(176,320)) for i in sel],axis=1)
cv2.imwrite(str(out/'contact.png'),contact)
(out/'manifest.json').write_text(json.dumps({'method':'DWPose-seeded GrabCut of source person; CoW source pixel tracks projected into DD target camera; dense points dilated and largest silhouette filled; colored like official FlashMotion example mask','official_flashmotion_source_commit':'998f46a26c5814ffd1b88b7965294c7476558697','not_ground_truth':True,'source_video':str(source_video),'rows':rows},indent=2)+'\n')
assert min(r['target_mask_pixels'] for r in rows)>500
print('saved',out,flush=True)
