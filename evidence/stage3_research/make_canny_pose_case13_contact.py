"""Compact full-film contact sheet for case13 pose-augmented Canny trial."""
import cv2,json,numpy as np
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];HERE=Path(__file__).resolve().parent
record=json.loads((ROOT/'vlm_dd_balanced_case13_b/stage3/records/route_a/depthdirector/stage3/vlm_selected.json').read_text())
rows=[('Source motion',Path(record['actual_source_condition_video'])),('DD first version',Path(record['native_video'])),('Canny warp',HERE/'canny_controlnet_case13_b_weight_0p5/result/depthdirector.mp4'),('Canny + LaMa/NLF',HERE/'canny_controlnet_case13_b_pose_weight_0p5/result/depthdirector.mp4'),('Tile + LaMa',HERE/'tile_controlnet_case13_b_lamafill_weight_0p2/result/depthdirector.mp4')]
frames=[0,9,18,27,36];out=[]
for label,path in rows:
 cap=cv2.VideoCapture(str(path));cells=[]
 for i in frames:
  cap.set(cv2.CAP_PROP_POS_FRAMES,i);ok,im=cap.read();assert ok,(label,path,i)
  im=cv2.resize(im,(320,176));cv2.putText(im,f'{label} / f{i}',(5,20),cv2.FONT_HERSHEY_SIMPLEX,.45,(255,255,255),1,cv2.LINE_AA);cells.append(im)
 cap.release();out.append(cv2.hconcat(cells))
cv2.imwrite(str(HERE/'case13_b_canny_pose_contact.jpg'),cv2.vconcat(out),[int(cv2.IMWRITE_JPEG_QUALITY),82])
print(HERE/'case13_b_canny_pose_contact.jpg')
