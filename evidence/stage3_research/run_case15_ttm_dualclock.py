"""Transfer TTM's official dual-clock denoising update into native DD Stage3.

Runs the actual DD 5B sampler and its native Stage2 endpoint pin. The temporal
mask conversion is executed from the official TTM Git source; only glue for the
DD scheduler and checkpoint layout is local. TTM's official Wan example uses
I2V 14B, so this is an algorithm transfer, not its published end-to-end run.
"""
import argparse,ast,csv,hashlib,json,os,sys,types
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];HERE=Path(__file__).resolve().parent
PROJECT=Path('/public/wufeier/automatic_bullet_time');DD=PROJECT/'models/DepthDirector';TTM=ROOT/'paper_sources/TTM';REF=HERE/'ttm_targetwarp_case15_c'
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def main():
 ap=argparse.ArgumentParser();ap.add_argument('--strong-index',type=int,required=True);ap.add_argument('--gpu',required=True);a=ap.parse_args();assert 1<=a.strong_index<50
 out=HERE/f'ttm_targetwarp_case15_c_strong_{a.strong_index}'
 os.environ.update(CUDA_VISIBLE_DEVICES=a.gpu,ABT_EXPERIMENT_ROOT=str(out),ABT_OBSERVE_NATIVE_NOISE='1',OMP_NUM_THREADS='4',HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1',IMAGEIO_FFMPEG_EXE='/usr/bin/ffmpeg')
 if os.environ.get('TTM_BOOT')!='1':
  os.environ.update(TTM_BOOT='1',LD_PRELOAD='/usr/lib/x86_64-linux-gnu/libstdc++.so.6',LD_LIBRARY_PATH=str(ROOT/'runtime_libs/usr/lib/x86_64-linux-gnu')+':'+os.environ.get('LD_LIBRARY_PATH',''))
  os.execvpe(sys.executable,[sys.executable,*sys.argv],os.environ)
 out.mkdir(parents=True,exist_ok=False);os.chdir(DD)
 for p in (DD,DD/'tools/recam',PROJECT/'scripts',ROOT,TTM):sys.path.insert(0,str(p))
 import cv2,numpy as np,torch,torch.nn.functional as F
 from PIL import Image
 try:import modelscope
 except ModuleNotFoundError:
  offline=types.ModuleType('modelscope')
  def unexpected(*args,**kwargs):raise RuntimeError('Unexpected ModelScope download')
  offline.snapshot_download=unexpected;sys.modules['modelscope']=offline
 from diffsynth.pipelines.wan_video_new import WanVideoUnit_InputVideoEmbedder
 from diffsynth.schedulers.flow_match import FlowMatchScheduler
 from pipelines.utils import load_video_to_tensor
 # Execute the actual official TTM mask converter without importing its
 # Diffusers 14B pipeline, which cannot load DD's Wan2.2 TI2V 5B checkpoint.
 source=(TTM/'pipelines/wan_pipeline.py').read_text();tree=ast.parse(source)
 cls=next(x for x in tree.body if isinstance(x,ast.ClassDef) and x.name=='WanImageToVideoTTMPipeline')
 fn=next(x for x in cls.body if isinstance(x,ast.FunctionDef) and x.name=='convert_rgb_mask_to_latent_mask')
 fn.decorator_list=[];module=ast.Module(body=[fn],type_ignores=[]);ast.fix_missing_locations(module);scope={'torch':torch,'F':F};exec(compile(module,str(TTM/'pipelines/wan_pipeline.py'),'exec'),scope)
 convert_mask=scope['convert_rgb_mask_to_latent_mask']
 ref_tensor=load_video_to_tensor(str(REF/'reference.mp4'))
 mask_tensor=load_video_to_tensor(str(REF/'person_mask.mp4'))
 assert ref_tensor.shape==(1,3,17,1280,704),ref_tensor.shape
 assert mask_tensor.shape==(1,3,17,1280,704),mask_tensor.shape
 frames=[Image.fromarray((ref_tensor[0,:,i].permute(1,2,0).numpy()*255).round().clip(0,255).astype(np.uint8)) for i in range(17)]
 mask_t1=(mask_tensor[0].permute(1,0,2,3).amax(dim=1,keepdim=True)>.5).float()
 mask=convert_mask(types.SimpleNamespace(vae_scale_factor_temporal=4,vae_scale_factor_spatial=16),mask_t1).permute(0,2,1,3,4).contiguous()
 assert mask.shape==(1,1,5,80,44),mask.shape
 mask[:,:,0]=0
 baseline=ROOT/'vlm_dd_v12_balanced_case15_c';record=json.loads((baseline/'stage3/records/route_a/depthdirector/stage3/vlm_selected.json').read_text())
 old_metadata=Path(record['metadata_csv']);row=next(csv.DictReader(old_metadata.open()))
 for key in ('input_image','input','cond','depth','mask'):
  row[key]=str((old_metadata.parent/row[key]).resolve());assert Path(row[key]).is_file()
 assert row['prompt']==record['prompt'];row['output']='result/output.mp4'
 metadata=out/'metadata_test_all.csv'
 with metadata.open('w') as f:w=csv.DictWriter(f,fieldnames=list(row));w.writeheader();w.writerow(row)
 (out/'depthdirector_stage3.yaml').write_text((baseline/'stage3/depthdirector_stage3.yaml').read_text())
 (out/'camera.json').write_bytes(Path(record['camera_path_file']).read_bytes())
 manifest={'method':'Official TTM Wan dual-clock denoising logic transferred to native DD Wan2.2 TI2V 5B scheduler','official_source':'https://github.com/time-to-move/TTM','official_commit':'0e1c62fc4a38af148ca53b5535241e10e85cfc01','official_function_executed':'WanImageToVideoTTMPipeline.convert_rgb_mask_to_latent_mask','official_reference_loader_executed':'pipelines.utils.load_video_to_tensor','tweak_index':0,'strong_index':a.strong_index,'not_official_14b_pipeline':True,'reference_is_estimated_target_warp':True,'reference_sha256':sha(REF/'reference.mp4'),'mask_sha256':sha(REF/'person_mask.mp4'),'baseline_sha256':record['native_video_sha256'],'effective_input_sha256':{k:sha(row[k]) for k in ('input_image','input','depth','mask')},'gpu':a.gpu,'state':'inference_running','postdecode_frame_replacement':False}
 (out/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
 holder={'steps':0,'forced_steps':0,'mask_fraction':float(mask.float().mean())}
 old_input=WanVideoUnit_InputVideoEmbedder.process;old_step=FlowMatchScheduler.step
 def input_process(self,pipe,input_video,noise,tiled,tile_size,tile_stride,vace_reference_image):
  assert input_video is None and vace_reference_image is None
  pipe.load_models_to_device(['vae'])
  video=pipe.preprocess_video(frames)
  ref_latents=pipe.vae.encode(video,device=pipe.device,tiled=tiled,tile_size=tile_size,tile_stride=tile_stride).to(dtype=pipe.torch_dtype,device=pipe.device)
  assert ref_latents.shape==noise.shape==(1,48,5,80,44),(ref_latents.shape,noise.shape)
  holder['ref']=ref_latents;holder['fixed_noise']=noise.clone();holder['mask']=mask.to(device=pipe.device,dtype=pipe.torch_dtype)
  latents=pipe.scheduler.add_noise(ref_latents,noise,pipe.scheduler.timesteps[0])
  holder['same_initial_noise']=bool(torch.equal(latents,noise))
  return {'latents':latents}
 def step(self,model_output,timestep,sample,*args,**kwargs):
  result=old_step(self,model_output,timestep,sample,*args,**kwargs)
  if 'ref' not in holder:return result
  idx=holder['steps'];holder['steps']+=1
  if idx<a.strong_index:
   if idx+1<len(self.timesteps):
    ref_noisy=self.add_noise(holder['ref'],holder['fixed_noise'],self.timesteps[idx+1])
   else:ref_noisy=holder['ref']
   m=holder['mask'];result=result*(1-m)+ref_noisy.to(result.dtype)*m;holder['forced_steps']+=1
  return result
 WanVideoUnit_InputVideoEmbedder.process=input_process;FlowMatchScheduler.step=step
 import run_stage23_depthdirector_stage3 as native
 native.OUT=out
 paths={'metadata':metadata,'camera':out/'camera.json','native':out/'result/depthdirector.mp4','endpoint':Path(row['input_image']),'audit':out/'native_endpoint_audit.json'}
 native.infer('vlm_selected',paths)
 assert holder['steps']==50 and holder['forced_steps']==a.strong_index,holder
 assert holder['same_initial_noise']
 manifest.update(state='completed',native_video=str(paths['native']),native_sha256=sha(paths['native']),call={k:v for k,v in holder.items() if k not in ('ref','fixed_noise','mask')})
 (out/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
 print('completed',manifest['native_sha256'],manifest['call'],flush=True)
if __name__=='__main__':main()
