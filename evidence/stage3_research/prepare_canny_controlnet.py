"""Run the unchanged official controlnet_aux CannyDetector on DD target warps."""
import ast
import hashlib
import json
import warnings
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

ROOT=Path(__file__).resolve().parents[1]
HERE=Path(__file__).resolve().parent
AUX=HERE/'.venv-controlnet/lib/python3.12/site-packages/controlnet_aux'


def sha(path):
 h=hashlib.sha256()
 with path.open('rb') as f:
  for data in iter(lambda:f.read(8<<20),b''):h.update(data)
 return h.hexdigest()

scope={'cv2':cv2,'np':np,'Image':Image,'warnings':warnings}
for path,names in [(AUX/'util.py',{'HWC3','resize_image'}),(AUX/'canny/__init__.py',{'CannyDetector'})]:
 tree=ast.parse(path.read_text())
 body=[node for node in tree.body if isinstance(node,(ast.FunctionDef,ast.ClassDef)) and node.name in names]
 assert {node.name for node in body}==names
 module=ast.Module(body=body,type_ignores=[])
 ast.fix_missing_locations(module)
 exec(compile(module,str(path),'exec'),scope)
processor=scope['CannyDetector']()
for label,baseline in [('case12_a','vlm_dd_balanced_case12_a'),('case13_b','vlm_dd_balanced_case13_b')]:
 record=json.loads((ROOT/baseline/'stage3/records/route_a/depthdirector/stage3/vlm_selected.json').read_text())
 directory=ROOT/baseline/'stage3/raw/route_a/depthdirector/stage3/vlm_selected'
 matches=list(directory.glob('*/preprocess/*/warp.mp4'))
 assert len(matches)==1
 source=matches[0]
 out=HERE/f'canny_controlnet_{label}';out.mkdir(exist_ok=True)
 cap=cv2.VideoCapture(str(source));w=int(cap.get(cv2.CAP_PROP_FRAME_WIDTH));h=int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
 writer=cv2.VideoWriter(str(out/'control.mp4'),cv2.VideoWriter_fourcc(*'mp4v'),15,(w,h));assert writer.isOpened()
 rows=[]
 while True:
  ok,frame=cap.read()
  if not ok:break
  valid=frame.max(axis=2)>20
  assert valid.any()
  fill=np.median(frame[valid],axis=0).astype(np.uint8)
  image=frame.copy();image[~valid]=fill
  pil=Image.fromarray(cv2.cvtColor(image,cv2.COLOR_BGR2RGB))
  # TheDenk CLI calls the official detector with its defaults. The official
  # pipeline subsequently resizes control images to the generation size.
  edge=processor(pil).resize((w,h))
  writer.write(cv2.cvtColor(np.asarray(edge),cv2.COLOR_RGB2BGR))
  rows.append(float(valid.mean()))
 cap.release();writer.release()
 assert len(rows)==record['native_frames']
 m={'label':label,'source_warp':str(source),'source_sha256':sha(source),'control_video':str(out/'control.mp4'),
    'control_sha256':sha(out/'control.mp4'),'frames':len(rows),'width':w,'height':h,
    'mean_valid_fraction':float(np.mean(rows)),'official_processor':'controlnet_aux.CannyDetector 0.0.10 default thresholds 100/200 and detect/image resolution 512; unmodified class and helper functions loaded from installed official source',
    'official_processor_source':'https://github.com/huggingface/controlnet_aux','local_adaptation':'DD target warp holes filled with nonblack per-frame median before official CannyDetector; output resized to DD generation size',
    'not_target_view_ground_truth':True}
 (out/'manifest.json').write_text(json.dumps(m,indent=2)+'\n')
 print(label,sha(out/'control.mp4'),len(rows),m['mean_valid_fraction'],flush=True)
