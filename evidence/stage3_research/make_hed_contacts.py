"""Full-film sampled contacts for official HED ControlNet trials."""
import cv2,json
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];HERE=Path(__file__).resolve().parent
cases={
'case13_b':('vlm_dd_balanced_case13_b',[0,9,18,27,36],[('Canny warp','canny_controlnet_case13_b_weight_0p5'),('Canny + LaMa/NLF','canny_controlnet_case13_b_pose_weight_0p5'),('HED warp','hed_controlnet_case13_b_weight_0p5'),('HED + LaMa/NLF','hed_controlnet_case13_b_pose_weight_0p5'),('HED + LaMa','hed_controlnet_case13_b_lama_weight_0p5'),('HED + affine','hed_controlnet_case13_b_affine_weight_0p5'),('Tile + LaMa','tile_controlnet_case13_b_lamafill_weight_0p2')]),
'case12_a':('vlm_dd_balanced_case12_a',[0,8,16,24,32],[('Canny warp','canny_controlnet_case12_a_weight_0p5'),('HED warp','hed_controlnet_case12_a_weight_0p5'),('Tile warp','tile_controlnet_case12_a_weight_0p5')]),
'case15_c':('vlm_dd_v12_balanced_case15_c',[0,4,8,12,16],[('Canny warp','canny_controlnet_case15_c_weight_0p5'),('HED warp','hed_controlnet_case15_c_weight_0p5'),('Tile person','tile_controlnet_case15_c_warp_person_weight_0p5')])}
for label,(base,indices,variants) in cases.items():
 record=json.loads((ROOT/base/'stage3/records/route_a/depthdirector/stage3/vlm_selected.json').read_text())
 rows=[('Source',Path(record['actual_source_condition_video'])),('DD first version',Path(record['native_video']))]
 rows += [(name,HERE/f'{folder}/result/depthdirector.mp4') for name,folder in variants]
 tiles=[];portrait=label!='case13_b';size=(180,320) if portrait else (320,176)
 for title,path in rows:
  assert path.is_file(),path;cap=cv2.VideoCapture(str(path));cells=[]
  for i in indices:
   cap.set(cv2.CAP_PROP_POS_FRAMES,i);ok,im=cap.read();assert ok,(title,i)
   im=cv2.resize(im,size);cv2.putText(im,f'{title} f{i}',(5,20),cv2.FONT_HERSHEY_SIMPLEX,.37,(255,255,255),1,cv2.LINE_AA);cells.append(im)
  cap.release();tiles.append(cv2.hconcat(cells))
 out=HERE/f'{label}_hed_controlnet_contact.jpg';cv2.imwrite(str(out),cv2.vconcat(tiles),[int(cv2.IMWRITE_JPEG_QUALITY),75]);print(out,out.stat().st_size)
