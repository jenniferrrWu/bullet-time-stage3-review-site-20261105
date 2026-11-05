"""Prepare real DD target-view warp as TTM reference and its visible-person mask."""
import hashlib,json
from pathlib import Path
import cv2,numpy as np
ROOT=Path(__file__).resolve().parents[1];HERE=Path(__file__).resolve().parent
warppath=Path(json.loads((ROOT/'pose_trial/v12_balanced_case15_c_target_warp_all/tracks.json').read_text())['source_video'])
silpath=HERE/'flashmotion_affine_silhouette_case15_c/trajectory.mp4'
out=HERE/'ttm_targetwarp_case15_c';out.mkdir(exist_ok=True)
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
a=cv2.VideoCapture(str(warppath));b=cv2.VideoCapture(str(silpath));W,H=704,1280
v=cv2.VideoWriter(str(out/'reference.mp4'),cv2.VideoWriter_fourcc(*'mp4v'),15,(W,H));m=cv2.VideoWriter(str(out/'person_mask.mp4'),cv2.VideoWriter_fourcc(*'mp4v'),15,(W,H));rows=[];contact=[]
for t in range(17):
 ok,frame=a.read();ok2,sil=b.read();assert ok and ok2 and frame.shape[:2]==(H,W) and sil.shape[:2]==(H,W)
 valid=(frame.max(axis=2)>20).astype(np.uint8)
 body=(sil.max(axis=2)>20).astype(np.uint8)
 body=cv2.morphologyEx(body,cv2.MORPH_CLOSE,cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(9,9)))
 force=body*valid
 nonblack=frame[(valid>0)&(body==0)]
 fill=np.median(nonblack,axis=0).astype(np.uint8) if len(nonblack) else np.array([158,163,157],np.uint8)
 ref=frame.copy();ref[valid==0]=fill
 mask=np.repeat((force*255)[:,:,None],3,axis=2)
 v.write(ref);m.write(mask)
 rows.append({'frame':t,'visible_pixels':int(valid.sum()),'body_pixels':int(body.sum()),'force_pixels':int(force.sum()),'hole_pixels_filled':int((valid==0).sum()),'fill_bgr':fill.tolist()})
 if t in (0,4,8,12,16):
  small=cv2.resize(ref,(176,320));small[cv2.resize(force,(176,320),interpolation=cv2.INTER_NEAREST)>0,1]=np.maximum(small[cv2.resize(force,(176,320),interpolation=cv2.INTER_NEAREST)>0,1],100);contact.append(small)
a.release();b.release();v.release();m.release();assert not a.read()[0] and not b.read()[0]
cv2.imwrite(str(out/'contact.jpg'),np.concatenate(contact,axis=1),[cv2.IMWRITE_JPEG_QUALITY,85])
(out/'manifest.json').write_text(json.dumps({'method':'TTM motion reference from actual DD target-view warped video; black invalid pixels filled with per-frame nonbody median background; visible body mask from existing affine silhouette intersected with valid warped pixels','not_ground_truth_target_video':True,'warp_video':str(warppath),'warp_sha256':sha(warppath),'silhouette_sha256':sha(silpath),'reference_sha256':sha(out/'reference.mp4'),'mask_sha256':sha(out/'person_mask.mp4'),'rows':rows},indent=2)+'\n')
print(out,'ref',sha(out/'reference.mp4'),'mask',sha(out/'person_mask.mp4'))
