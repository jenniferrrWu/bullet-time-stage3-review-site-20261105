"""Build same-view and mirrored-yaw AnyView scenes from verified case13 B input."""
import json,shutil
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[1]
BASE=ROOT/'paper_sources/AnyView-DVS/data/case13_b_vlm45'
with np.load(BASE/'lowdim/cam1.npz') as z:
    k=z['intrinsics'];source=z['extrinsics'];t=z['timestep']
with np.load(BASE/'lowdim/cam0.npz') as z:
    target=z['extrinsics']
assert source.shape==target.shape==(37,4,4)
assert np.allclose(source,source[0],atol=1e-6)
rel=np.linalg.inv(source[0])@target[0]
scenes={'sameview':source.copy(),'reverse45':np.repeat((source[0]@np.linalg.inv(rel))[None],37,axis=0)}
for label,poses in scenes.items():
    out=BASE.parent/f'case13_b_{label}'
    (out/'rgb').mkdir(parents=True,exist_ok=True)
    (out/'lowdim').mkdir(parents=True,exist_ok=True)
    shutil.copy2(BASE/'rgb/cam1.mp4',out/'rgb/cam1.mp4')
    shutil.copy2(BASE/'lowdim/cam1.npz',out/'lowdim/cam1.npz')
    np.savez_compressed(out/'lowdim/cam0.npz',camera=np.array(['cam0']*37),timestep=t,intrinsics=k,extrinsics=poses)
    meta=json.loads((BASE/'metadata.json').read_text())
    meta['info']['name']=f'case13_b_{label}'
    (out/'metadata.json').write_text(json.dumps(meta,indent=2)+'\n')
    (out/'provenance.json').write_text(json.dumps({'derived_from':str(BASE),'ablation':label,'source_to_original_target':rel.tolist(),'not_stage2_endpoint_conditioned':True},indent=2)+'\n')
    theta=np.degrees(np.arccos(np.clip((np.trace((np.linalg.inv(source[0])@poses[0])[:3,:3])-1)/2,-1,1)))
    print(label,'degrees',theta,'relative_translation',(np.linalg.inv(source[0])@poses[0])[:3,3].tolist(),flush=True)
