"""Fill sparse DD target warp with LaMa-completed actual Stage2 background."""
import hashlib,json
from pathlib import Path
import cv2,numpy as np
from PIL import Image
ROOT=Path(__file__).resolve().parents[1];HERE=Path(__file__).resolve().parent
base=ROOT/'vlm_dd_balanced_case13_b';source=next((base/'stage3/raw/route_a/depthdirector/stage3/vlm_selected').glob('*/preprocess/*/warp.mp4'))
background_path=HERE/'case13_lama_background/background.png';background=cv2.imread(str(background_path));assert background.shape==(704,1280,3)
out=HERE/'tile_controlnet_case13_b_lamafill';out.mkdir(exist_ok=True)
cap=cv2.VideoCapture(str(source));writer=cv2.VideoWriter(str(out/'control.mp4'),cv2.VideoWriter_fourcc(*'mp4v'),15,(1280,704));assert writer.isOpened()
valids=[];examples=[];i=0
while True:
 ok,warp=cap.read()
 if not ok:break
 valid=warp.max(axis=2)>20
 filled=np.where(valid[:,:,None],warp,background)
 pil=Image.fromarray(cv2.cvtColor(filled,cv2.COLOR_BGR2RGB));small=pil.resize((320,176))
 blur=Image.fromarray(cv2.GaussianBlur(np.asarray(small),(5,5),sigmaX=2));full=blur.resize((1280,704))
 writer.write(cv2.cvtColor(np.asarray(full),cv2.COLOR_RGB2BGR))
 if i in (0,9,18,27,36):examples.append(cv2.resize(filled,(320,176)))
 valids.append(float(valid.mean()));i+=1
cap.release();writer.release();assert i==37
cv2.imwrite(str(out/'contact.jpg'),np.concatenate(examples,axis=0),[cv2.IMWRITE_JPEG_QUALITY,88])
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
m={'method':'DD target-view warp invalid pixels filled with official LaMa-inpainted actual Stage2 endpoint background; official TheDenk Tile resize by 4/Gaussian 5x5 sigma 2/upscale. LaMa use and hole fill are local adaptations, not official Tile end-to-end pipeline.','source_warp':str(source),'source_sha256':sha(source),'lama_background':str(background_path),'lama_background_sha256':sha(background_path),'control_video':str(out/'control.mp4'),'control_sha256':sha(out/'control.mp4'),'frames':i,'mean_warp_valid_fraction':float(np.mean(valids)),'not_target_view_ground_truth':True}
(out/'manifest.json').write_text(json.dumps(m,indent=2)+'\n');print(out/'control.mp4',m['control_sha256'])
