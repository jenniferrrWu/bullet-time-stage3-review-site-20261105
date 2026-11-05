"""Source-person silhouettes aligned to observed case13 DD target-view boxes."""
import hashlib,json
from pathlib import Path
import cv2,numpy as np
ROOT=Path(__file__).resolve().parents[1];HERE=Path(__file__).resolve().parent
src=json.loads((ROOT/'pose_trial/balanced_case13_b_source_all/tracks.json').read_text())
tgt=json.loads((ROOT/'pose_trial/balanced_case13_b_native_all/tracks.json').read_text())
H,W=704,1280;cap=cv2.VideoCapture(src['source_video']);frames=[];rows=[];last_target=None
for t in range(37):
 ok,frame=cap.read();assert ok and frame.shape[:2]==(H,W)
 obs=src['frames'][t];a=np.array(obs['bbox_xyxy'],np.float32)
 observed=tgt['frames'][t]['bbox_xyxy']
 if observed is not None:last_target=np.array(observed,np.float32)
 assert last_target is not None
 b=last_target
 x1,y1,x2,y2=np.rint(a).astype(int);x1=max(0,x1);y1=max(0,y1);x2=min(W,x2);y2=min(H,y2)
 mask=np.full((H,W),cv2.GC_BGD,np.uint8);mask[y1:y2,x1:x2]=cv2.GC_PR_FGD
 keys=obs['keypoints_xy'];accepted=obs['accepted']
 for j,k in ((5,6),(5,7),(7,9),(6,8),(8,10),(5,11),(6,12),(11,12),(11,13),(13,15),(12,14),(14,16)):
  if accepted[j] and accepted[k]:cv2.line(mask,tuple(np.rint(keys[j]).astype(int)),tuple(np.rint(keys[k]).astype(int)),cv2.GC_FGD,9)
 for j in (5,6,11,12):
  if accepted[j]:cv2.circle(mask,tuple(np.rint(keys[j]).astype(int)),9,cv2.GC_FGD,-1)
 cv2.grabCut(frame,mask,None,np.zeros((1,65),np.float64),np.zeros((1,65),np.float64),3,cv2.GC_INIT_WITH_MASK)
 fg=np.uint8((mask==cv2.GC_FGD)|(mask==cv2.GC_PR_FGD))*255
 fg=cv2.morphologyEx(fg,cv2.MORPH_CLOSE,cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(9,9)))
 sx=(b[2]-b[0])/(a[2]-a[0]);sy=(b[3]-b[1])/(a[3]-a[1]);M=np.float32([[sx,0,b[0]-sx*a[0]],[0,sy,b[1]-sy*a[1]]])
 aligned=cv2.warpAffine(fg,M,(W,H),flags=cv2.INTER_NEAREST)
 color=np.zeros((H,W,3),np.uint8);color[aligned>0]=(73,76,124);frames.append(color)
 rows.append({'frame':t,'target_box_observed':observed is not None,'source_mask_pixels':int((fg>0).sum()),'target_mask_pixels':int((aligned>0).sum()),'target_box':b.tolist()})
 print(t,rows[-1]['target_mask_pixels'],flush=True)
cap.release();out=HERE/'flashmotion_affine_silhouette_case13_b';out.mkdir(exist_ok=True)
video=out/'trajectory.mp4';writer=cv2.VideoWriter(str(video),cv2.VideoWriter_fourcc(*'mp4v'),15,(W,H));assert writer.isOpened()
for im in frames:writer.write(im)
writer.release();cv2.imwrite(str(out/'contact.jpg'),np.concatenate([cv2.resize(frames[i],(320,176)) for i in (0,9,18,27,36)],axis=1),[cv2.IMWRITE_JPEG_QUALITY,88])
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
m={'method':'DWPose-seeded source GrabCut silhouettes affine-aligned to observed DD baseline target-view boxes; last observed box carried after frame31 disappearance; estimated control, not true target view geometry','source_video':src['source_video'],'baseline_target_video':tgt['source_video'],'trajectory_video':str(video),'trajectory_sha256':sha(video),'frames':37,'unobserved_target_box_frames':[x['frame'] for x in rows if not x['target_box_observed']],'rows':rows}
(out/'manifest.json').write_text(json.dumps(m,indent=2)+'\n');print(video,m['trajectory_sha256'])
