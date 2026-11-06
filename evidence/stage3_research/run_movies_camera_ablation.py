"""Ablate only camera yaw/pivot on official MoVieS first-frame splats."""
import json
import os
import sys
from pathlib import Path
import cv2
import numpy as np
import torch
from safetensors.torch import load_file

ROOT=Path(__file__).resolve().parents[1]
MOVIES=ROOT/'paper_sources/MoVieS'
WEIGHT=ROOT/'paper_sources/MoVieS-weights/movies_ckpt.safetensors'
INPUT=MOVIES/'resources/DAVIS/case13_b_static.npz'
OUT=ROOT/'stage3_research/movies_case13_b_camera_ablation'
OUT.mkdir(exist_ok=True)
os.chdir(MOVIES);sys.path.insert(0,str(MOVIES))
from src.options import opt_dict
from src.models import SplatRecon
from extensions.vggt.vggt.models.vggt import VGGT
VGGT.from_pretrained=classmethod(lambda cls,*args,**kwargs: cls())
sample=np.load(INPUT)
images=torch.from_numpy(sample['images']).float().unsqueeze(0).to('cuda',dtype=torch.bfloat16)
camera=torch.from_numpy(sample['C2W']).float().unsqueeze(0).to('cuda',dtype=torch.bfloat16)
intr=torch.from_numpy(sample['fxfycxcy']).float().unsqueeze(0).to('cuda',dtype=torch.bfloat16)
model=SplatRecon(opt_dict['movies'],load_lpips=False)
model.load_state_dict(load_file(str(WEIGHT)),strict=True)
model.eval().to('cuda')
with torch.inference_mode(),torch.autocast('cuda',dtype=torch.bfloat16):
    backbone,motions,moving_splats=model.backbone(images,camera,intr,torch.linspace(0,1,13).unsqueeze(0).to('cuda',dtype=torch.bfloat16),torch.tensor([[0.]],device='cuda',dtype=torch.bfloat16),frames_chunk_size=8)
backbone['offset']=motions[:,0,:,:3]
if moving_splats is not None:backbone.update(moving_splats[0])
subject_depth=0.1533549726009369
specs=[(-45,1),(-15,1),(0,1),(5,1),(15,1),(30,1),(45,1),(45,.5),(45,2),(45,4)]
results=[]; cells=[]
with torch.inference_mode(),torch.autocast('cuda',dtype=torch.bfloat16):
    for yaw,factor in specs:
        theta=np.deg2rad(yaw)
        rotation=np.array([[np.cos(theta),0,np.sin(theta)],[0,1,0],[-np.sin(theta),0,np.cos(theta)]],np.float32)
        pose=np.eye(4,dtype=np.float32);pose[:3,:3]=rotation
        pivot=np.array([0,0,subject_depth*factor],np.float32)
        pose[:3,3]=pivot-rotation@pivot
        pose_t=torch.from_numpy(pose).unsqueeze(0).unsqueeze(0).to('cuda',dtype=torch.bfloat16)
        data=model.gs_renderer.render(backbone,camera,intr,pose_t,intr[:,:1])
        frame=(data['image'][0,0].float().clamp(0,1).permute(1,2,0).cpu().numpy()*255).astype(np.uint8)
        alpha=data['alpha'][0,0,0].float().cpu().numpy()
        coverage=float(np.mean(alpha>0.5))
        results.append({'yaw_degrees':yaw,'pivot_depth_factor':factor,'alpha_coverage_gt_0p5':coverage,'mean_rgb':float(frame.mean())})
        name=f'yaw{yaw:+d}_pivot{factor:g}'
        cv2.imwrite(str(OUT/f'{name}.jpg'),cv2.cvtColor(frame,cv2.COLOR_RGB2BGR))
        cell=cv2.cvtColor(frame,cv2.COLOR_RGB2BGR)
        cv2.rectangle(cell,(0,0),(330,28),(0,0,0),-1)
        cv2.putText(cell,f'yaw {yaw:+d}  pivot {factor:g}  cov {coverage:.2f}',(4,20),cv2.FONT_HERSHEY_SIMPLEX,.54,(255,255,255),1,cv2.LINE_AA)
        cells.append(cell)
        print(name,coverage,flush=True)
cv2.imwrite(str(OUT/'contact.jpg'),cv2.vconcat([cv2.hconcat(cells[:5]),cv2.hconcat(cells[5:])]),[int(cv2.IMWRITE_JPEG_QUALITY),88])
(OUT/'results.json').write_text(json.dumps({'official_source':'https://github.com/chenguolin/MoVieS','only_first_frame':True,'result':results},indent=2)+'\n')
