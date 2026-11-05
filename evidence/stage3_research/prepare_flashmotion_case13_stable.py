"""Stop target-box expansion after the last trustworthy baseline person frame."""
import hashlib,json
from pathlib import Path
import cv2,numpy as np
HERE=Path(__file__).resolve().parent
old=HERE/'flashmotion_affine_silhouette_case13_b';info=json.loads((old/'manifest.json').read_text())
ref=np.array(info['rows'][27]['target_box'],np.float32);assert 250<ref[2]-ref[0]<350
out=HERE/'flashmotion_affine_silhouette_case13_b_stable';out.mkdir(exist_ok=True)
cap=cv2.VideoCapture(str(old/'trajectory.mp4'));writer=cv2.VideoWriter(str(out/'trajectory.mp4'),cv2.VideoWriter_fourcc(*'mp4v'),15,(1280,704));assert writer.isOpened()
frames=[];rows=[]
for t in range(37):
 ok,frame=cap.read();assert ok
 if t>27:
  oldbox=np.array(info['rows'][t]['target_box'],np.float32)
  sx=(ref[2]-ref[0])/(oldbox[2]-oldbox[0]);sy=(ref[3]-ref[1])/(oldbox[3]-oldbox[1])
  matrix=np.float32([[sx,0,ref[0]-sx*oldbox[0]],[0,sy,ref[1]-sy*oldbox[1]]])
  frame=cv2.warpAffine(frame,matrix,(1280,704),flags=cv2.INTER_NEAREST)
 writer.write(frame);frames.append(frame)
 rows.append({'frame':t,'original_target_box':info['rows'][t]['target_box'],'effective_target_box':(ref.tolist() if t>27 else info['rows'][t]['target_box']),'person_mask_pixels':int((frame.max(axis=2)>20).sum())})
cap.release();writer.release()
cv2.imwrite(str(out/'contact.jpg'),np.concatenate([cv2.resize(frames[i],(320,176)) for i in (0,9,18,27,36)],axis=1),[cv2.IMWRITE_JPEG_QUALITY,88])
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
manifest={'method':'Original DWPose+GrabCut source-person affine silhouettes for frames0-27; later target boxes carried from trustworthy frame27 rather than inflated/absent detections. This is local control-map repair, not official FlashMotion behavior.','original_trajectory_sha256':sha(old/'trajectory.mp4'),'trajectory_video':str(out/'trajectory.mp4'),'trajectory_sha256':sha(out/'trajectory.mp4'),'trusted_last_frame':27,'rows':rows,'not_target_view_ground_truth':True}
(out/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n');print(out/'trajectory.mp4',manifest['trajectory_sha256'])
