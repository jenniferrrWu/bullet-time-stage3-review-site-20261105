"""Apply the official TheDenk README tile-control preprocessing to DD target warp."""
import json
from pathlib import Path
import cv2
import numpy as np
from PIL import Image
HERE=Path(__file__).resolve().parent
source=HERE/'ttm_targetwarp_case15_c/reference.mp4';out=HERE/'tile_controlnet_case15_c';out.mkdir(exist_ok=True)
cap=cv2.VideoCapture(str(source));fps=cap.get(cv2.CAP_PROP_FPS);frames=[]
while True:
 okay,bgr=cap.read()
 if not okay:break
 image=Image.fromarray(cv2.cvtColor(bgr,cv2.COLOR_BGR2RGB));w,h=image.size
 image=image.resize((w//4,h//4))
 image=Image.fromarray(cv2.GaussianBlur(np.asarray(image),(5,5),sigmaX=2))
 image=image.resize((w,h))
 frames.append(cv2.cvtColor(np.asarray(image),cv2.COLOR_RGB2BGR))
cap.release();assert len(frames)==17 and (w,h)==(704,1280)
video=out/'control.mp4';writer=cv2.VideoWriter(str(video),cv2.VideoWriter_fourcc(*'mp4v'),fps,(w,h));assert writer.isOpened()
for frame in frames:writer.write(frame)
writer.release()
manifest={'source':str(source),'control_video':str(video),'frames':17,'fps':fps,'method':'Official TheDenk tile-control README: resize by 4, cv2 GaussianBlur kernel 5 sigma 2, resize back. Input is local DD target-view warp with missing pixels filled, not ground truth.','official_source':'https://github.com/TheDenk/wan2.2-controlnet#for-tile-controlnet-do-processing-like-this'}
(out/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
print(video,video.stat().st_size)
