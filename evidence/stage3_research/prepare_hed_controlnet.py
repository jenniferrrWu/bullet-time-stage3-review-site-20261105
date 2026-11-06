"""Apply installed official controlnet_aux HED detector to DD target-view warps."""
import ast,hashlib,json,os,warnings
from pathlib import Path
import cv2,numpy as np,torch
from einops import rearrange
from PIL import Image
ROOT=Path(__file__).resolve().parents[1];HERE=Path(__file__).resolve().parent
AUX=HERE/'.venv-controlnet/lib/python3.12/site-packages/controlnet_aux'
WEIGHT=ROOT/'paper_sources/HED-annotator-weights/ControlNetHED.pth'
EXPECTED='5ca93762ffd68a29fee1af9d495bf6aab80ae86f08905fb35472a083a4c7a8fa'
def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for b in iter(lambda:f.read(8<<20),b''):h.update(b)
 return h.hexdigest()
assert WEIGHT.stat().st_size==29444406 and sha(WEIGHT)==EXPECTED
scope={'cv2':cv2,'np':np,'Image':Image,'warnings':warnings,'torch':torch,'rearrange':rearrange,'os':os}
for path,names in [(AUX/'util.py',{'HWC3','resize_image','safe_step','nms'}),(AUX/'hed/__init__.py',{'DoubleConvBlock','ControlNetHED_Apache2','HEDdetector'})]:
 tree=ast.parse(path.read_text());body=[n for n in tree.body if isinstance(n,(ast.FunctionDef,ast.ClassDef)) and n.name in names]
 assert {n.name for n in body}==names,(path,[n.name for n in body]);module=ast.Module(body=body,type_ignores=[]);ast.fix_missing_locations(module);exec(compile(module,str(path),'exec'),scope)
processor=scope['HEDdetector'].from_pretrained(str(WEIGHT.parent)).to('cuda').netNetwork.eval()
processor=scope['HEDdetector'](processor)
for label,baseline in [('case12_a','vlm_dd_balanced_case12_a'),('case13_b','vlm_dd_balanced_case13_b'),('case15_c','vlm_dd_v12_balanced_case15_c')]:
 record=json.loads((ROOT/baseline/'stage3/records/route_a/depthdirector/stage3/vlm_selected.json').read_text())
 if label=='case15_c':source=HERE/'ttm_targetwarp_case15_c/reference.mp4'
 else:
  matches=list((ROOT/baseline/'stage3/raw/route_a/depthdirector/stage3/vlm_selected').glob('*/preprocess/*/warp.mp4'))
  assert len(matches)==1,matches;source=matches[0]
 out=HERE/f'hed_controlnet_{label}';out.mkdir(exist_ok=True)
 cap=cv2.VideoCapture(str(source));w=int(cap.get(cv2.CAP_PROP_FRAME_WIDTH));h=int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
 writer=cv2.VideoWriter(str(out/'control.mp4'),cv2.VideoWriter_fourcc(*'mp4v'),15,(w,h));assert writer.isOpened()
 count=0;valids=[]
 while True:
  ok,bgr=cap.read()
  if not ok:break
  valid=bgr.max(axis=2)>20;assert valid.any();fill=np.median(bgr[valid],axis=0).astype(np.uint8)
  bgr=bgr.copy();bgr[~valid]=fill
  pil=Image.fromarray(cv2.cvtColor(bgr,cv2.COLOR_BGR2RGB))
  edge=processor(pil).resize((w,h))
  writer.write(cv2.cvtColor(np.asarray(edge),cv2.COLOR_RGB2BGR));valids.append(float(valid.mean()));count+=1
 cap.release();writer.release();assert count==record['native_frames'],(label,count,record['native_frames'])
 m={'label':label,'source_warp':str(source),'source_sha256':sha(source),'control_video':str(out/'control.mp4'),'control_sha256':sha(out/'control.mp4'),'frames':count,'resolution':[h,w],'valid_fraction_mean':float(np.mean(valids)),'official_processor_source':'https://github.com/huggingface/controlnet_aux','official_processor_weight_source':'https://huggingface.co/lllyasviel/Annotators/blob/main/ControlNetHED.pth','official_processor_weight_sha256':EXPECTED,'official_processor':'controlnet_aux HEDdetector defaults detect_resolution=512/image_resolution=512 loaded from installed source','local_adaptation':'fill DD warp holes with median then resize detector output to DD dimensions','not_target_view_ground_truth':True}
 (out/'manifest.json').write_text(json.dumps(m,indent=2)+'\n');print(label,count,m['valid_fraction_mean'],flush=True)
