"""Apply the GitHub-released LaMa TorchScript model via the cloned wrapper."""
import hashlib,json,os,sys
from pathlib import Path
import numpy as np
from PIL import Image
ROOT=Path(__file__).resolve().parents[1];HERE=Path(__file__).resolve().parent
REPO=ROOT/'paper_sources/simple-lama-inpainting';WEIGHT=REPO/'big-lama.pt'
assert WEIGHT.stat().st_size==205803670
os.environ['LAMA_MODEL']=str(WEIGHT)
sys.path.insert(0,str(REPO))
import torch
from simple_lama_inpainting import SimpleLama
folder=HERE/'case13_lama_background';source=Image.open(folder/'endpoint.png').convert('RGB');mask=Image.open(folder/'mask.png').convert('L')
model=SimpleLama(device=torch.device('cuda:0'))
inpainted=model(source,mask)
a=np.asarray(source);b=np.asarray(inpainted);m=np.asarray(mask)>0
out=np.where(m[:,:,None],b,a).astype(np.uint8)
Image.fromarray(out).save(folder/'background.png')
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
manifest={'method':'Official LaMa TorchScript release through cloned simple-lama-inpainting SimpleLama wrapper; local DWPose+GrabCut observed-person mask; preserve unmasked pixels from actual Stage2 endpoint','wrapper_source':'https://github.com/enesmsahin/simple-lama-inpainting','wrapper_commit':'0790c455160991a6a1417723a68d3d48e3c859e5','upstream_lama_source':'https://github.com/advimman/lama','model_release':'https://github.com/enesmsahin/simple-lama-inpainting/releases/tag/v0.1.0','model_sha256':sha(WEIGHT),'input_endpoint_sha256':sha(folder/'endpoint.png'),'mask_sha256':sha(folder/'mask.png'),'background_sha256':sha(folder/'background.png'),'mask_fraction':float(m.mean()),'not_target_view_ground_truth':True}
(folder/'lama_manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
print(folder/'background.png',manifest['background_sha256'],flush=True)
