"""Validate source DWPose joints reprojected through the exact DD Pi3X source/target cameras."""
import json
from pathlib import Path
import numpy as np
import cv2
W=Path('/root/bullet_time_research_20261102')
root=W/'vlm_dd_v12_balanced_case15_c'
G=root/'shared_geometry'
K=np.load(G/'K.npy'); P=np.load(G/'pose.npy'); D=np.load(G/'depth.npy',mmap_mode='r')
rec=json.loads((root/'stage3/records/route_a/depthdirector/stage3/vlm_selected.json').read_text())
cam=json.loads(Path(rec['camera_path_file']).read_text())
base=np.array(cam['source_c2w']); target=np.array(cam['renderer_target_w2c'])
src=json.loads((W/'pose_trial/v12_balanced_case15_c_source_all/tracks.json').read_text())
warp=json.loads((W/'pose_trial/v12_balanced_case15_c_target_warp_all/tracks.json').read_text())
assert len(P)==len(target)==len(src['frames'])==len(warp['frames'])==17
H,WW=D.shape[1:3]
fx=0.5*H/np.tan(np.arctan2(K[0,1,2]*2,2*K[0,1,1])); fy=fx
# DD get_trajectory aligns relative Pi3X motion to the registered source C2W.
pi_c2w0=np.linalg.inv(P[0]);
projected=[]; summaries=[]
for t in range(17):
 c2w=base @ (P[0] @ np.linalg.inv(P[t]))
 source_xy=np.array(src['frames'][t]['keypoints_xy'],float)
 source_ok=np.array(src['frames'][t]['accepted'],bool)
 target_xy=np.array(warp['frames'][t]['keypoints_xy'],float)
 target_ok=np.array(warp['frames'][t]['accepted'],bool)
 out=[];valid=[];errors=[]
 for j,(x,y) in enumerate(source_xy):
  if not source_ok[j] or not np.isfinite([x,y]).all() or x<1 or x>WW-2 or y<1 or y>H-2:
   out.append([None,None]);valid.append(False);continue
  z=float(D[t,int(round(y)),int(round(x)),0]);
  if not np.isfinite(z) or z<=0:
   out.append([None,None]);valid.append(False);continue
  cs=np.array([(x-WW/2)/fx*z,(y-H/2)/fy*z,z,1.]); world=c2w@cs; ct=target[t]@world
  if ct[2]<=0:
   out.append([None,None]);valid.append(False);continue
  u=WW/2+fx*ct[0]/ct[2];v=H/2+fy*ct[1]/ct[2]
  good=bool(0<=u<WW and 0<=v<H)
  out.append([float(u),float(v)]);valid.append(good)
  if good and target_ok[j]:errors.append(float(np.linalg.norm(np.array([u,v])-target_xy[j])))
 projected.append({'frame':t,'xy':out,'valid':valid,'warp_detected_joints':int(target_ok.sum()),'matched_joints':len(errors),'median_error_px':float(np.median(errors)) if errors else None,'mean_error_px':float(np.mean(errors)) if errors else None})
 summaries+=errors
result={'method':'Exact DD source camera alignment and target W2C with Pi3X depth sampled at source DWPose joint pixels; pinhole matching DD mesh intrinsics','source_tracks':str(W/'pose_trial/v12_balanced_case15_c_source_all/tracks.json'),'warp_tracks':str(W/'pose_trial/v12_balanced_case15_c_target_warp_all/tracks.json'),'matched_joint_count':len(summaries),'median_reprojection_vs_warp_dwpose_error_px':float(np.median(summaries)) if summaries else None,'mean_reprojection_vs_warp_dwpose_error_px':float(np.mean(summaries)) if summaries else None,'per_frame':projected,'caveat':'Warp DWPose is not ground truth; occlusion, depth errors and missing warped pixels can cause mismatch. No semantic-control improvement inferred.'}
out=W/'stage3_research/case15_source_pose_reprojection_validation.json';out.write_text(json.dumps(result,indent=2))
print(json.dumps({'matched_joint_count':result['matched_joint_count'],'median_error_px':result['median_reprojection_vs_warp_dwpose_error_px'],'frames':[(x['frame'],x['matched_joints'],x['median_error_px']) for x in projected]},indent=2))
