"""Use the actual Stage2 endpoint as target-view fill for missing DD warp pixels."""
import csv,hashlib,json
from pathlib import Path
import cv2,numpy as np
from PIL import Image
ROOT=Path(__file__).resolve().parents[1];HERE=Path(__file__).resolve().parent
base=ROOT/'vlm_dd_balanced_case13_b';record=json.loads((base/'stage3/records/route_a/depthdirector/stage3/vlm_selected.json').read_text())
meta=Path(record['metadata_csv']);row=next(csv.DictReader(meta.open()));endpoint_path=(meta.parent/row['input_image']).resolve()
source=next((base/'stage3/raw/route_a/depthdirector/stage3/vlm_selected').glob('*/preprocess/*/warp.mp4'))
out=HERE/'tile_controlnet_case13_b_endpointfill';out.mkdir(exist_ok=True)
endpoint=cv2.imread(str(endpoint_path));assert endpoint.shape==(704,1280,3)
cap=cv2.VideoCapture(str(source));writer=cv2.VideoWriter(str(out/'control.mp4'),cv2.VideoWriter_fourcc(*'mp4v'),15,(1280,704));assert writer.isOpened()
examples=[];rows=[];i=0
while True:
 ok,warp=cap.read()
 if not ok:break
 assert warp.shape==endpoint.shape
 valid=warp.max(axis=2)>20
 composite=np.where(valid[:,:,None],warp,endpoint)
 image=Image.fromarray(cv2.cvtColor(composite,cv2.COLOR_BGR2RGB));small=image.resize((320,176))
 blurred=Image.fromarray(cv2.GaussianBlur(np.asarray(small),(5,5),sigmaX=2));full=blurred.resize((1280,704))
 writer.write(cv2.cvtColor(np.asarray(full),cv2.COLOR_RGB2BGR))
 if i in (0,9,18,27,36):
  left=cv2.resize(warp,(320,176));right=cv2.resize(composite,(320,176));examples.extend([left,right])
 rows.append(float(valid.mean()));i+=1
cap.release();writer.release();assert i==37
contact=np.concatenate([np.concatenate(examples[k:k+2],axis=1) for k in range(0,len(examples),2)],axis=0)
cv2.imwrite(str(out/'contact.jpg'),contact,[cv2.IMWRITE_JPEG_QUALITY,88])
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
m={'method':'DD target-view warp invalid pixels filled from actual Stage2 endpoint at identical VLM target camera, then official Tile downscale 4/Gaussian 5x5 sigma 2/upscale; local preprocessing, not official Tile method','source_warp':str(source),'source_sha256':sha(source),'actual_stage2_endpoint':str(endpoint_path),'endpoint_sha256':sha(endpoint_path),'control_video':str(out/'control.mp4'),'control_sha256':sha(out/'control.mp4'),'frames':i,'mean_warp_valid_fraction':float(np.mean(rows)),'endpoint_f0_person_can_ghost_at_later_frames':True,'not_target_view_ground_truth':True}
(out/'manifest.json').write_text(json.dumps(m,indent=2)+'\n');print(out/'control.mp4',m['control_sha256'])
