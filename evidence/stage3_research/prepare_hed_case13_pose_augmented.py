"""Official HED map of LaMa-filled DD target warp with estimated NLF target person."""
import ast,hashlib,json,os,warnings
from pathlib import Path
import cv2,numpy as np,torch
from einops import rearrange
from PIL import Image
ROOT=Path(__file__).resolve().parents[1];HERE=Path(__file__).resolve().parent
AUX=HERE/'.venv-controlnet/lib/python3.12/site-packages/controlnet_aux'
WEIGHT=ROOT/'paper_sources/HED-annotator-weights/ControlNetHED.pth'
assert WEIGHT.stat().st_size==29444406
scope={'cv2':cv2,'np':np,'Image':Image,'warnings':warnings,'torch':torch,'rearrange':rearrange,'os':os}
for path,names in [(AUX/'util.py',{'HWC3','resize_image','safe_step','nms'}),(AUX/'hed/__init__.py',{'DoubleConvBlock','ControlNetHED_Apache2','HEDdetector'})]:
 tree=ast.parse(path.read_text());body=[n for n in tree.body if isinstance(n,(ast.FunctionDef,ast.ClassDef)) and n.name in names]
 assert {n.name for n in body}==names;module=ast.Module(body=body,type_ignores=[]);ast.fix_missing_locations(module);exec(compile(module,str(path),'exec'),scope)
processor=scope['HEDdetector'].from_pretrained(str(WEIGHT.parent)).to('cuda')
record=json.loads((ROOT/'vlm_dd_balanced_case13_b/stage3/records/route_a/depthdirector/stage3/vlm_selected.json').read_text())
matches=list((ROOT/'vlm_dd_balanced_case13_b/stage3/raw/route_a/depthdirector/stage3/vlm_selected').glob('*/preprocess/*/warp.mp4'));assert len(matches)==1
warp=matches[0];traj=HERE/'nlf_case13_b_target_silhouette/trajectory.mp4';bg=HERE/'case13_lama_background/background.png'
out=HERE/'hed_controlnet_case13_b_pose';out.mkdir(exist_ok=True)
a=cv2.VideoCapture(str(warp));b=cv2.VideoCapture(str(traj));w=int(a.get(3));h=int(a.get(4));background=cv2.resize(cv2.imread(str(bg)),(w,h))
writer=cv2.VideoWriter(str(out/'control.mp4'),cv2.VideoWriter_fourcc(*'mp4v'),15,(w,h));assert writer.isOpened()
count=0
while True:
 ok,frame=a.read();ok2,pose=b.read()
 if not ok and not ok2:break
 assert ok and ok2
 valid=frame.max(axis=2)>20
 scene=frame.copy();scene[~valid]=background[~valid]
 mask=(pose[:,:,2]>70).astype(np.uint8)
 mask=cv2.morphologyEx(mask,cv2.MORPH_CLOSE,np.ones((7,7),np.uint8)); mask=cv2.GaussianBlur(mask,(0,0),2)>0.25
 # Dark person proxy creates boundary; it is estimated geometry, not real target pixels.
 scene[mask]=(np.array([75,75,75],np.uint8)*.7+scene[mask]*.3).astype(np.uint8)
 edge=processor(Image.fromarray(cv2.cvtColor(scene,cv2.COLOR_BGR2RGB))).resize((w,h))
 writer.write(cv2.cvtColor(np.asarray(edge),cv2.COLOR_RGB2BGR));count+=1
 if count in (1,19,37):cv2.imwrite(str(out/f'control_{count-1:02d}.jpg'),cv2.cvtColor(np.asarray(edge),cv2.COLOR_RGB2BGR))
a.release();b.release();writer.release();assert count==37
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
m={'frames':37,'control_sha256':sha(out/'control.mp4'),'source_warp_sha256':sha(warp),'nlf_silhouette_sha256':sha(traj),'lama_background_sha256':sha(bg),'hed_model_sha256':sha(WEIGHT),'official_module':'controlnet_aux.HEDdetector from installed official source and pretrained weight','local_adaptation':'DD target warp holes filled by LaMa background; NLF estimated target person darkened before HED','not_target_view_ground_truth':True}
(out/'manifest.json').write_text(json.dumps(m,indent=2)+'\n');print(json.dumps(m,indent=2),flush=True)
