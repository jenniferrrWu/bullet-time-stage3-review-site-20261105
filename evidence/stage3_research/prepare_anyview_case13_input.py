"""Package real case13 B and VLM-selected DD camera as an official AnyView unified scene."""
import hashlib
import json
import shutil
from pathlib import Path
import cv2
import numpy as np

ROOT=Path(__file__).resolve().parents[1]
SOURCE=ROOT/'vlm_dd_balanced_case13_b/action.mp4'
CAMERA=ROOT/'vlm_dd_balanced_case13_b/stage3/raw/route_a/depthdirector/stage3/vlm_selected/vlm_v11/preprocess/actual_camera.json'
OUT=ROOT/'paper_sources/AnyView-DVS/data/case13_b_vlm45'
(OUT/'rgb').mkdir(parents=True,exist_ok=True)
(OUT/'lowdim').mkdir(parents=True,exist_ok=True)
cap=cv2.VideoCapture(str(SOURCE))
count=int(cap.get(cv2.CAP_PROP_FRAME_COUNT));width=int(cap.get(cv2.CAP_PROP_FRAME_WIDTH));height=int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT));fps=float(cap.get(cv2.CAP_PROP_FPS));cap.release()
assert (count,width,height)==(37,1280,704),(count,width,height)
shutil.copy2(SOURCE,OUT/'rgb/cam1.mp4')
cam=json.loads(CAMERA.read_text())
source_c2w=np.array(cam['source_c2w'],np.float32)
target_c2w=np.array(cam['target_c2w'],np.float32)
assert target_c2w.shape==(37,4,4)
source_c2w=np.repeat(source_c2w[None],37,axis=0)
# Official NLF 3D/2D intrinsics estimate from the same real source frames, scaled to 1280x704.
focal=614.7142962974289*2
K=np.array([[focal,0,639.0000006530605],[0,focal,350.9999998703217],[0,0,1]],np.float32)
for name,poses in [('cam1',source_c2w),('cam0',target_c2w)]:
    np.savez_compressed(OUT/'lowdim'/f'{name}.npz',camera=np.array([name]*37),
        timestep=np.arange(37,dtype=np.int64),
        intrinsics=np.repeat(K[None],37,axis=0),extrinsics=poses)
meta={'info':{'name':'case13_b_vlm45','storage':'videos'},'cameras':['cam0','cam1'],
      'num_frames':37,'framerate':fps,'resolution':[704,1280],
      'rgb':{'extension':'mp4'},'extrinsics':{'transform':'cam2world'},
      'specific':{'roles':{'input':'cam1','target':'cam0'},'scale_factor':0.125,
                  'source':{'dataset':'custom_static_dance'}}}
(OUT/'metadata.json').write_text(json.dumps(meta,indent=2)+'\n')
provenance={'official_method':'https://github.com/TRI-ML/AnyView-DVS',
            'input_source':str(SOURCE),'input_sha256':hashlib.sha256(SOURCE.read_bytes()).hexdigest(),
            'camera':str(CAMERA),'camera_sha256':hashlib.sha256(CAMERA.read_bytes()).hexdigest(),
            'intrinsics_source':'official NLF estimate on same case13 source, rescaled 2x',
            'intrinsics':K.tolist(),'scale_factor':0.125,
            'target_camera_is_actual_VLM_DD_45_degree':True,
            'stage2_endpoint_not_model_condition':True,
            'no_target_view_ground_truth':True}
(OUT/'provenance.json').write_text(json.dumps(provenance,indent=2)+'\n')
print('prepared',OUT,'frames',count,'source_fps',fps,flush=True)
