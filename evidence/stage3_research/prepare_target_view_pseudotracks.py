"""Project official CoW source tracks through DD depth/cameras into target view.

These are geometry-derived pseudo correspondences, not synchronized multi-view
measurements or MVTrack4Gen training labels. A z-buffer removes obvious
occlusions. The joint reprojection validation is stored separately.
"""
import hashlib
import json
from pathlib import Path
import cv2
import numpy as np
ROOT=Path(__file__).resolve().parents[1]
HERE=Path(__file__).resolve().parent
case=ROOT/'vlm_dd_v12_balanced_case15_c'
G=case/'shared_geometry'
tracks=np.load(HERE/'case15_c_dense_tracks/dense_tracks.npz')
xy=tracks['xy'].astype(np.float32)
visibility=tracks['visibility']*tracks['confidence']>.5
K=np.load(G/'K.npy');P=np.load(G/'pose.npy');D=np.load(G/'depth.npy',mmap_mode='r')
record=json.loads((case/'stage3/records/route_a/depthdirector/stage3/vlm_selected.json').read_text())
camera=json.loads(Path(record['camera_path_file']).read_text())
base=np.asarray(camera['source_c2w']);targets=np.asarray(camera['renderer_target_w2c'])
H,W=D.shape[1:3]
assert xy.shape==(17,100352,2) and D.shape[:3]==(17,1280,704) and len(targets)==17
fx=.5*H/np.tan(np.arctan2(K[0,1,2]*2,2*K[0,1,1]));fy=fx
out_xy=np.full_like(xy,np.nan);out_z=np.full(xy.shape[:2],np.nan,np.float32)
out_valid=np.zeros(xy.shape[:2],bool)
rows=[]
for t in range(17):
    x,y=xy[t,:,0],xy[t,:,1]
    src_ok=visibility[t] & visibility[0] & np.isfinite(x) & np.isfinite(y) & (x>=1)&(x<W-1)&(y>=1)&(y<H-1)
    xi=np.clip(np.rint(x).astype(np.int32),0,W-1);yi=np.clip(np.rint(y).astype(np.int32),0,H-1)
    z=np.asarray(D[t,yi,xi,0],np.float32)
    src_ok &= np.isfinite(z)&(z>0)
    cs=np.stack(((x-W/2)/fx*z,(y-H/2)/fy*z,z,np.ones_like(z)),axis=-1)
    c2w=base@(P[0]@np.linalg.inv(P[t]))
    target=(cs@c2w.T)@targets[t].T
    tz=target[:,2]
    u=W/2+fx*target[:,0]/np.where(tz>0,tz,1)
    v=H/2+fy*target[:,1]/np.where(tz>0,tz,1)
    valid=src_ok&np.isfinite(u)&np.isfinite(v)&(tz>0)&(u>=0)&(u<W)&(v>=0)&(v<H)
    ui=np.clip(np.rint(u).astype(np.int32),0,W-1);vi=np.clip(np.rint(v).astype(np.int32),0,H-1)
    zbuf=np.full(H*W,np.inf,np.float32)
    np.minimum.at(zbuf,vi[valid]*W+ui[valid],tz[valid].astype(np.float32))
    zbuf=cv2.erode(zbuf.reshape(H,W),np.ones((3,3),np.uint8)).reshape(-1)
    with np.errstate(over='ignore', invalid='ignore'):
        seen=valid & (tz<=zbuf[vi*W+ui]*1.05+.05)
    out_xy[t,seen,0]=u[seen];out_xy[t,seen,1]=v[seen]
    out_z[t,seen]=tz[seen]
    out_valid[t]=seen
    rows.append({'frame':t,'source_visible_points':int(src_ok.sum()),'target_in_frame_points':int(valid.sum()),
                 'target_after_zbuffer_points':int(seen.sum())})
out=HERE/'case15_c_target_view_pseudotracks'
out.mkdir(exist_ok=True)
np.savez_compressed(out/'tracks.npz',xy=out_xy,depth=out_z,visibility=out_valid)
validation=json.loads((HERE/'case15_source_pose_reprojection_validation.json').read_text())
manifest={'method':'CoW source tracks projected with actual DD Pi3X depth and renderer target camera, 3x3 z-buffer visibility',
          'not_multiview_ground_truth':True,'not_mvtrack4gen_loss_or_model':True,
          'source_tracks_sha256':hashlib.sha256((HERE/'case15_c_dense_tracks/dense_tracks.npz').read_bytes()).hexdigest(),
          'joint_projection_vs_warp_median_error_px':validation['median_reprojection_vs_warp_dwpose_error_px'],
          'joint_projection_matched_points':validation['matched_joint_count'],
          'canvas_wh':[W,H],'frames':rows,
          'caveat':'Depth error, disocclusion and synthetic warp prevent treating these as supervised target-view tracks.'}
(out/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
print(json.dumps({'mean_source_visible':np.mean([r['source_visible_points'] for r in rows]),
                  'mean_target_after_zbuffer':np.mean([r['target_after_zbuffer_points'] for r in rows]),
                  'joint_projection_median_error_px':manifest['joint_projection_vs_warp_median_error_px']}))
