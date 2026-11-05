"""Repair projected silhouette rows absent due depth/occlusion holes.

This is an ablation input to the official pretrained FlashMotion adapter, not
an official segmentation method or multiview ground truth.
"""
import json
from pathlib import Path
import cv2,numpy as np
H=Path(__file__).resolve().parent
A=H/'flashmotion_affine_silhouette_case15_c/trajectory.mp4'
P=H/'flashmotion_target_silhouette_case15_c/trajectory.mp4'
ca,cp=cv2.VideoCapture(str(A)),cv2.VideoCapture(str(P));frames=[];rows=[]
for t in range(17):
 oka,a=ca.read();okp,p=cp.read();assert oka and okp
 am=a[:,:,2]>35;pm=p[:,:,2]>35
 ac=am.sum(axis=1);pc=pm.sum(axis=1)
 use_affine=(pc<.65*ac) & (ac>6)
 repaired=pm.copy();repaired[use_affine]=am[use_affine]
 repaired=cv2.morphologyEx(np.uint8(repaired)*255,cv2.MORPH_CLOSE,cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(7,7)))
 c=np.zeros_like(a);c[repaired>0]=(73,76,124);frames.append(c)
 rows.append({'frame':t,'affine_pixels':int(am.sum()),'projected_pixels':int(pm.sum()),'repaired_pixels':int((repaired>0).sum()),'replaced_rows':int(use_affine.sum())})
 print(t,rows[-1],flush=True)
ca.release();cp.release();out=H/'flashmotion_repaired_silhouette_case15_c';out.mkdir(exist_ok=True)
w=cv2.VideoWriter(str(out/'trajectory.mp4'),cv2.VideoWriter_fourcc(*'mp4v'),15,(704,1280))
for f in frames:w.write(f)
w.release();cv2.imwrite(str(out/'contact.png'),np.concatenate([cv2.resize(frames[i],(176,320)) for i in (0,4,8,12,16)],axis=1))
(out/'manifest.json').write_text(json.dumps({'method':'Projected CoW/DD silhouette; rows with less than 65% of affine silhouette occupancy replaced by source GrabCut silhouette aligned to target DWPose box','not_ground_truth':True,'inputs':[str(A),str(P)],'rows':rows},indent=2)+'\n')
print(out,flush=True)
