"""Diagnostic: generated CoW point motion vs DD geometry-projected pseudo tracks.

This is not MVTrack4Gen's trained cross-view correspondence head or loss. The
projected tracks are pseudo labels, not synchronized target-view ground truth.
"""
import json
import sys
from pathlib import Path
import cv2
import numpy as np
import torch
from scipy.spatial import cKDTree
HERE=Path(__file__).resolve().parent
ROOT=HERE.parent
repo=Path('/public/wufeier/automatic_bullet_time/models/WorldinWorld_multigpu_trial/third_party/CoWTracker')
checkpoint=Path('/public/wufeier/automatic_bullet_time/models/WorldinWorld_multigpu_trial/checkpoints/cowtracker/cowtracker_model.pth')
sys.path.insert(0,str(repo))
from cowtracker import CoWTracker

tr=np.load(HERE/'case15_c_target_view_pseudotracks/tracks.npz')
source=np.load(HERE/'case15_c_dense_tracks/dense_tracks.npz')
src_boxes=np.array([f['bbox_xyxy'] for f in json.load(open(ROOT/'pose_trial/v12_balanced_case15_c_source_all/tracks.json'))['frames']])
tgt_box=np.array(json.load(open(ROOT/'pose_trial/v12_balanced_case15_c_target_warp_all/tracks.json'))['frames'][0]['bbox_xyxy'])
sxy=source['xy'];p=tr['xy'];pv=tr['visibility']
initial_person=(sxy[0,:,0]>=src_boxes[0,0])&(sxy[0,:,0]<=src_boxes[0,2])&(sxy[0,:,1]>=src_boxes[0,1])&(sxy[0,:,1]<=src_boxes[0,3])
initial_target=(p[0,:,0]>=tgt_box[0])&(p[0,:,0]<=tgt_box[2])&(p[0,:,1]>=tgt_box[1])&(p[0,:,1]<=tgt_box[3])
ids=np.flatnonzero(initial_person&initial_target&pv.all(axis=0))
assert len(ids)>=512,len(ids)
ids=ids[np.linspace(0,len(ids)-1,min(2048,len(ids))).astype(int)]
pt=p[:,ids]
print('selected stable target-view pseudo tracks',len(ids),flush=True)

cases={
 'zero_control':HERE/'wanmove_dense_case15_c_all_0p0/result/depthdirector.mp4',
 'wanmove_person_035':HERE/'wanmove_dense_case15_c_person_0p35/result/depthdirector.mp4',
 'wanmove_person_070':HERE/'wanmove_dense_case15_c_person_0p7/result/depthdirector.mp4',
 'wanmove_all_035':HERE/'wanmove_dense_case15_c_all_0p35/result/depthdirector.mp4',
 'motion_i2v_forwardwarp_person_035':HERE/'motion_i2v_forwardwarp_case15_c_person_0p35/result/depthdirector.mp4'
}
model=CoWTracker.from_checkpoint(str(checkpoint),device='cuda',dtype=torch.float16)
result={}
for name,path in cases.items():
    cap=cv2.VideoCapture(str(path));frames=[]
    while True:
        ok,bgr=cap.read()
        if not ok:break
        frames.append(cv2.cvtColor(cv2.resize(bgr,(448,224)),cv2.COLOR_BGR2RGB))
    cap.release();assert len(frames)==17
    video=torch.from_numpy(np.stack(frames)).permute(0,3,1,2).to('cuda',torch.float16)
    with torch.no_grad(),torch.amp.autocast('cuda',dtype=torch.float16): pred=model.forward(video=video,queries=None)
    xy=pred['track'][0].float().cpu().numpy().reshape(17,-1,2)
    vis=(pred['vis'][0]*pred['conf'][0]).float().cpu().numpy().reshape(17,-1)
    xy[...,0]*=704/448;xy[...,1]*=1280/224
    dist,query=cKDTree(xy[0]).query(pt[0],k=1)
    match=dist<5
    assert match.sum()>=512,(name,match.sum())
    gen=xy[:,query]
    visible=(vis[:,query]>.5)&match[None]
    delta_gen=gen-gen[:1];delta_pseudo=pt-pt[:1]
    errors=np.linalg.norm(delta_gen-delta_pseudo,axis=2)
    errs=np.where(visible,errors,np.nan)
    frame_medians=np.nanmedian(errs,axis=1)
    result[name]={'matched_initial_points':int(match.sum()),
                  'valid_fraction_all_frames':float(np.mean(visible)),
                  'median_trajectory_displacement_error_px':float(np.nanmedian(errs)),
                  'mean_trajectory_displacement_error_px':float(np.nanmean(errs)),
                  'frame_median_error_px':[float(x) for x in frame_medians],
                  'last_frame_median_error_px':float(frame_medians[-1])}
    print(name,result[name],flush=True)
summary={'method':'CoW-generated target-view tracks compared with geometry-projected source CoW pseudo tracks',
         'not_mvtrack4gen':True,'not_ground_truth':True,
         'joint_projection_vs_target_warp_median_error_px':12.338196227102635,
         'source_query_count':int(len(ids)),'comparison':result,
         'caveat':'CoW errors on faded images, depth and visibility errors, and synthetic target warp confound this diagnostic. Do not select a Stage3 winner from this number alone.'}
(HERE/'case15_target_view_track_diagnostic.json').write_text(json.dumps(summary,indent=2)+'\n')
