"""Run native DD Stage3 with FlashMotion's official pretrained ResNet adapter.

Official Wan22ControlResNet and its SlowAdapter weights are used verbatim.
Only the integration hook maps its per-block outputs into DD's identical-size
Wan2.2 5B blocks, preserving DD's camera, Stage2 endpoint, and source input.
The silhouette control map is geometry-derived and estimated, not multiview GT.
"""
import argparse,csv,hashlib,json,os,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
HERE=Path(__file__).resolve().parent
PROJECT=Path('/public/wufeier/automatic_bullet_time')
DD=PROJECT/'models/DepthDirector'
FLASH=ROOT/'paper_sources/FlashMotion'
WEIGHTS=ROOT/'paper_sources/FlashMotion-weights/SlowAdapter/ResNet/model.pt'
TRAJECTORIES={'projected':HERE/'flashmotion_target_silhouette_case15_c/trajectory.mp4',
              'affine':HERE/'flashmotion_affine_silhouette_case15_c/trajectory.mp4',
              'repaired':HERE/'flashmotion_repaired_silhouette_case15_c/trajectory.mp4'}
WEIGHT_SHA='df28d58ba918ae38da458f48c2110804a9113cf72f60e71d3c75032077798d58'
def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def main():
 ap=argparse.ArgumentParser();ap.add_argument('--alpha',type=float,required=True);ap.add_argument('--gpu',required=True);ap.add_argument('--map-kind',choices=list(TRAJECTORIES),required=True)
 a=ap.parse_args();assert 0<=a.alpha<=1
 traj=TRAJECTORIES[a.map_kind]
 out=HERE/('flashmotion_resnet_case15_c_'+a.map_kind+'_alpha_'+str(a.alpha).replace('.','p'))
 os.environ.update(CUDA_VISIBLE_DEVICES=a.gpu,ABT_EXPERIMENT_ROOT=str(out),ABT_OBSERVE_NATIVE_NOISE='1',OMP_NUM_THREADS='4',HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1',IMAGEIO_FFMPEG_EXE='/usr/bin/ffmpeg')
 if os.environ.get('FLASHMOTION_BOOT')!='1':
  os.environ.update(FLASHMOTION_BOOT='1',LD_PRELOAD='/usr/lib/x86_64-linux-gnu/libstdc++.so.6',LD_LIBRARY_PATH=str(ROOT/'runtime_libs/usr/lib/x86_64-linux-gnu')+':'+os.environ.get('LD_LIBRARY_PATH',''))
  os.execvpe(sys.executable,[sys.executable,*sys.argv],os.environ)
 assert sha(WEIGHTS)==WEIGHT_SHA
 assert traj.is_file()
 out.mkdir(parents=True,exist_ok=False)
 os.chdir(DD)
 for p in (DD,DD/'tools/recam',PROJECT/'scripts',ROOT,FLASH):sys.path.insert(0,str(p))
 import cv2,numpy as np,torch,types
 from PIL import Image
 try:import modelscope
 except ModuleNotFoundError:
  offline=types.ModuleType('modelscope')
  def unexpected(*args,**kwargs):raise RuntimeError('Unexpected ModelScope download')
  offline.snapshot_download=unexpected;sys.modules['modelscope']=offline
 from diffsynth.pipelines.wan_video_new import WanVideoUnit_ConcatVideoEmbedder
 from wan22.modules.model import Wan22ControlResNet
 baseline=ROOT/'vlm_dd_v12_balanced_case15_c'
 record=json.loads((baseline/'stage3/records/route_a/depthdirector/stage3/vlm_selected.json').read_text())
 old_metadata=Path(record['metadata_csv']);row=next(csv.DictReader(old_metadata.open()))
 for key in ('input_image','input','cond','depth','mask'):
  row[key]=str((old_metadata.parent/row[key]).resolve())
  assert Path(row[key]).is_file()
 assert row['prompt']==record['prompt']
 row['output']='result/output.mp4'
 metadata=out/'metadata_test_all.csv'
 with metadata.open('w') as f:
  w=csv.DictWriter(f,fieldnames=list(row));w.writeheader();w.writerow(row)
 (out/'depthdirector_stage3.yaml').write_text((baseline/'stage3/depthdirector_stage3.yaml').read_text())
 (out/'camera.json').write_bytes(Path(record['camera_path_file']).read_bytes())
 manifest={'method':'Official FlashMotion Wan22ControlResNet SlowAdapter pretrained weights injected after each DD Wan2.2 5B block; target-view silhouette control map',
           'flashmotion_source_commit':'998f46a26c5814ffd1b88b7965294c7476558697',
           'official_checkpoint_sha256':WEIGHT_SHA,'official_checkpoint_path':str(WEIGHTS),
           'trajectory_sha256':sha(traj),'trajectory_manifest':str(traj.parent/'manifest.json'),'map_kind':a.map_kind,
           'not_official_flashmotion_end_to_end_pipeline':True,
           'trajectory_is_geometry_pseudolabel':True,'alpha':a.alpha,'gpu':a.gpu,
           'baseline_sha256':record['native_video_sha256'],
           'effective_input_sha256':{key:sha(row[key]) for key in ('input_image','input','depth','mask')},
           'postdecode_frame_replacement':False,'state':'inference_running'}
 (out/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
 original=WanVideoUnit_ConcatVideoEmbedder.process
 calls=[]
 def process(self,pipe,concat_video,tiled,tile_size,tile_stride):
  result=original(self,pipe,concat_video,tiled,tile_size,tile_stride)
  latent=result.get('concat_latent')
  if latent is None:return result
  assert latent.shape==(1,48,5,80,44),tuple(latent.shape)
  cap=cv2.VideoCapture(str(traj));pil=[]
  while True:
   ok,bgr=cap.read()
   if not ok:break
   pil.append(Image.fromarray(cv2.cvtColor(bgr,cv2.COLOR_BGR2RGB)))
  cap.release();assert len(pil)==17
  trajectory=pipe.preprocess_video(pil)
  with torch.inference_mode():
   traj_latent=pipe.vae.encode(trajectory,device=pipe.device,tiled=tiled,tile_size=tile_size,tile_stride=tile_stride).to(dtype=pipe.torch_dtype,device=pipe.device)
   assert traj_latent.shape==latent.shape,tuple(traj_latent.shape)
   control=Wan22ControlResNet(in_dim=48,dim=3072,num_layers=30)
   state=torch.load(WEIGHTS,map_location='cpu',weights_only=True)['controlnet']
   control.load_state_dict(state,strict=True)
   control=control.to(device=pipe.device,dtype=pipe.torch_dtype).eval()
   states=control(traj_latent,seq_len=5*40*22)
   assert len(states)==30 and all(s.shape==(1,4400,3072) for s in states),[tuple(s.shape) for s in states]
   assert len(pipe.dit.blocks)==30 and pipe.dit.dim==3072
   for idx,block in enumerate(pipe.dit.blocks):
    def hook(module,inputs,output,idx=idx):
     assert output.ndim==3 and output.shape[1:]==(8800,3072),(idx,output.shape)
     # DD concatenates generated five latent frames first, source five second.
     # FlashMotion's adapter conditions only the generated sequence.
     updated=output.clone()
     updated[:,:4400,:]+=a.alpha*states[idx].to(dtype=output.dtype,device=output.device)
     return updated
    block.register_forward_hook(hook)
   calls.append({'trajectory_latent_shape':list(traj_latent.shape),
                 'num_adapter_states':len(states),'state_mean_abs':float(torch.stack([s.float().abs().mean() for s in states]).mean()),
                 'state_max_abs':float(torch.stack([s.float().abs().max() for s in states]).max()),
                 'first_content_latent_unchanged':bool(torch.equal(latent[:,:,:1],result['concat_latent'][:,:,:1]))})
  return result
 WanVideoUnit_ConcatVideoEmbedder.process=process
 import run_stage23_depthdirector_stage3 as native
 native.OUT=out
 paths={'metadata':metadata,'camera':out/'camera.json','native':out/'result/depthdirector.mp4','endpoint':Path(row['input_image']),'audit':out/'native_endpoint_audit.json'}
 native.infer('vlm_selected',paths)
 assert len(calls)==1
 manifest.update(state='completed',native_video=str(paths['native']),native_sha256=sha(paths['native']),call=calls[0])
 (out/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
 print(json.dumps({'output':str(out),'sha256':manifest['native_sha256'],'call':calls[0]}),flush=True)
if __name__=='__main__':main()
