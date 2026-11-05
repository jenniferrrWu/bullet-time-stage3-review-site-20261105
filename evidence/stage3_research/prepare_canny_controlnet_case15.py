"""Run unchanged official controlnet_aux CannyDetector on case15 target warp."""
import ast,hashlib,json,warnings
from pathlib import Path
import cv2,numpy as np
from PIL import Image
HERE=Path(__file__).resolve().parent;AUX=HERE/'.venv-controlnet/lib/python3.12/site-packages/controlnet_aux'
scope={'cv2':cv2,'np':np,'Image':Image,'warnings':warnings}
for path,names in [(AUX/'util.py',{'HWC3','resize_image'}),(AUX/'canny/__init__.py',{'CannyDetector'})]:
 tree=ast.parse(path.read_text());body=[node for node in tree.body if isinstance(node,(ast.FunctionDef,ast.ClassDef)) and node.name in names]
 assert {node.name for node in body}==names
 module=ast.Module(body=body,type_ignores=[]);ast.fix_missing_locations(module);exec(compile(module,str(path),'exec'),scope)
processor=scope['CannyDetector']()
source=HERE/'ttm_targetwarp_case15_c/reference.mp4';out=HERE/'canny_controlnet_case15_c';out.mkdir(exist_ok=True)
cap=cv2.VideoCapture(str(source));w=int(cap.get(cv2.CAP_PROP_FRAME_WIDTH));h=int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
video=out/'control.mp4';writer=cv2.VideoWriter(str(video),cv2.VideoWriter_fourcc(*'mp4v'),15,(w,h));assert writer.isOpened()
frames=0
while True:
 ok,bgr=cap.read()
 if not ok:break
 pil=Image.fromarray(cv2.cvtColor(bgr,cv2.COLOR_BGR2RGB));edge=processor(pil).resize((w,h))
 writer.write(cv2.cvtColor(np.asarray(edge),cv2.COLOR_RGB2BGR));frames+=1
cap.release();writer.release();assert frames==17 and (w,h)==(704,1280)
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
m={'source_warp_reference':str(source),'source_sha256':sha(source),'control_video':str(video),'control_sha256':sha(video),'frames':frames,'width':w,'height':h,'official_processor':'controlnet_aux.CannyDetector 0.0.10 defaults, exact installed official source class and helpers','official_source':'https://github.com/huggingface/controlnet_aux','local_adaptation':'input DD target warp holes already filled by TTM reference preprocessing; resize official Canny result to DD dimensions','not_target_view_ground_truth':True}
(out/'manifest.json').write_text(json.dumps(m,indent=2)+'\n');print(video,sha(video))
