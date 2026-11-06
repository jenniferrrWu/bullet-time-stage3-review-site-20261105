"""Audit four official HED ControlNet full-film trials against fixed VLM baselines."""
import hashlib,json,os,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT));os.chdir(ROOT)
from pose_trial.compare_tracks import compare
from person_contrast_proxy import measure
def load(p):return json.loads(Path(p).read_text())
def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for b in iter(lambda:f.read(8<<20),b''):h.update(b)
 return h.hexdigest()
cases=[('case12_a','vlm_dd_balanced_case12_a','balanced_case12_a'),('case13_b','vlm_dd_balanced_case13_b','balanced_case13_b'),('case13_b_pose','vlm_dd_balanced_case13_b','balanced_case13_b'),('case13_b_lama','vlm_dd_balanced_case13_b','balanced_case13_b'),('case13_b_affine','vlm_dd_balanced_case13_b','balanced_case13_b'),('case15_c','vlm_dd_v12_balanced_case15_c','v12_balanced_case15_c')]
rows={}
for label,base,prefix in cases:
 record=load(ROOT/base/'stage3/records/route_a/depthdirector/stage3/vlm_selected.json')
 old_audit=load(record['sampling_audit'])
 src_track=ROOT/'pose_trial'/f'{prefix}_source_all/tracks.json';src=load(src_track);src_video=Path(src['source_video'])
 old_track=ROOT/'pose_trial'/f'{prefix}_native_all/tracks.json'
 trial=HERE/f'hed_controlnet_{label}_weight_0p5';new_video=trial/'result/depthdirector.mp4';new_track=ROOT/'pose_trial'/f'hed_controlnet_{label}_native_all/tracks.json'
 manifest=load(trial/'manifest.json');audit=load(trial/'native_endpoint_audit.json')
 assert manifest['state']=='completed' and manifest['native_sha256']==sha(new_video)
 assert audit['initial_noise']['sha256']==old_audit['initial_noise']['sha256']
 assert audit['endpoint_pixel_sha256']==old_audit['endpoint_pixel_sha256']
 items={}
 for name,video,track in [('baseline',Path(record['native_video']),old_track),('hed',new_video,new_track)]:
  t=load(track);assert len(t['frames'])==record['native_frames']
  pose=compare(src,t);contrast=measure(src_video,src_track,video,track)
  items[name]={'video':str(video),'sha256':sha(video),'limb_motion_error':pose['limb_mean_motion_delta_mae_body_units'],'core_motion_error':pose['core_mean_motion_delta_mae_body_units'],'person_contrast_trend_over_source':contrast['comparison']['gray_std_generated_trend_over_source_trend'],'detected_frames':pose['native_detected_frames']}
  if name=='hed':
   (trial/'pose_compare.json').write_text(json.dumps(pose,indent=2)+'\n');(trial/'contrast_compare.json').write_text(json.dumps(contrast,indent=2)+'\n')
 rows[label]={'frames':record['native_frames'],'same_stage2_endpoint_and_noise':True,'comparisons':items}
result={'rows':rows,'note':'DWPose cross-view 2D and contrast are diagnostic proxies; whole-film fidelity and movement decide quality.'}
(HERE/'hed_controlnet_summary.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n')
for label,row in rows.items():
 print(label,'frames',row['frames'],'limb',row['comparisons']['baseline']['limb_motion_error'],'=>',row['comparisons']['hed']['limb_motion_error'],'contrast',row['comparisons']['baseline']['person_contrast_trend_over_source'],'=>',row['comparisons']['hed']['person_contrast_trend_over_source'],'detected',row['comparisons']['hed']['detected_frames'])
