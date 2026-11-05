"""Remove DD-warp ghosts outside the visible person, then apply official tile blur."""
import json,hashlib
from pathlib import Path
import cv2
import numpy as np
from PIL import Image
HERE=Path(__file__).resolve().parent;ref=HERE/'ttm_targetwarp_case15_c/reference.mp4';mask=HERE/'ttm_targetwarp_case15_c/person_mask.mp4';out=HERE/'tile_controlnet_case15_c_clean';out.mkdir(exist_ok=True)
a=cv2.VideoCapture(str(ref));b=cv2.VideoCapture(str(mask));fps=a.get(cv2.CAP_PROP_FPS);frames=[]
while True:
 ok,frame=a.read();mk,matte=b.read()
 if not ok:break
 assert mk and frame.shape==matte.shape
 # The case15 studio has a nearly uniform background. Sample only far corners.
 corners=np.concatenate([frame[:100,:100].reshape(-1,3),frame[:100,-100:].reshape(-1,3),frame[-100:,:100].reshape(-1,3),frame[-100:,-100:].reshape(-1,3)])
 background=np.median(corners,axis=0).astype(np.float32)
 alpha=(cv2.cvtColor(matte,cv2.COLOR_BGR2GRAY).astype(np.float32)/255).clip(0,1)
 alpha=cv2.GaussianBlur(alpha,(7,7),1.5)[:,:,None]
 clean=(frame.astype(np.float32)*alpha+background[None,None,:]*(1-alpha)).clip(0,255).astype(np.uint8)
 image=Image.fromarray(cv2.cvtColor(clean,cv2.COLOR_BGR2RGB));w,h=image.size
 image=image.resize((w//4,h//4));image=Image.fromarray(cv2.GaussianBlur(np.asarray(image),(5,5),sigmaX=2));image=image.resize((w,h))
 frames.append(cv2.cvtColor(np.asarray(image),cv2.COLOR_RGB2BGR))
a.release();b.release();assert len(frames)==17
video=out/'control.mp4';writer=cv2.VideoWriter(str(video),cv2.VideoWriter_fourcc(*'mp4v'),fps,(w,h));assert writer.isOpened()
for x in frames:writer.write(x)
writer.release()
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
manifest={'source':str(ref),'person_mask':str(mask),'control_video':str(video),'frames':17,'fps':fps,'source_sha256':sha(ref),'mask_sha256':sha(mask),'control_video_sha256':sha(video),'method':'Local visible-person matte removes DD target-warp ghosts outside body; background filled with corner median, then exact TheDenk README tile blur resize/5x5 Gaussian/resize. Estimated input, not ground truth.'}
(out/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n');print(video,video.stat().st_size)
