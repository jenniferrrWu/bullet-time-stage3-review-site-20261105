"""Compare native full-film Depth ControlNet outputs with selected VLM Stage3 baselines."""
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
cases={'case12_a':('vlm_dd_balanced_case12_a','balanced_case12_a'),
       'case13_b':('vlm_dd_balanced_case13_b','balanced_case13_b')}
rows={}
for label,(baseline_root,pose_prefix) in cases.items():
 record=load(ROOT/baseline_root/'stage3/records/route_a/depthdirector/stage3/vlm_selected.json')
 source_tracks=ROOT/'pose_trial'/f'{pose_prefix}_source_all/tracks.json'
 old_tracks=ROOT/'pose_trial'/f'{pose_prefix}_native_all/tracks.json'
 new_dir=HERE/f'depth_controlnet_{label}_midasstyle_weight_0p5'
 new_video=new_dir/'result/depthdirector.mp4'
 new_tracks=HERE/f'dwpose_depth_controlnet_{label}_0p5/tracks.json'
 manifest=load(new_dir/'manifest.json');audit=load(new_dir/'native_endpoint_audit.json');old_audit=load(record['sampling_audit'])
 assert manifest['state']=='completed' and manifest['native_sha256']==sha(new_video)
 assert audit['initial_noise']['sha256']==old_audit['initial_noise']['sha256']
 assert audit['endpoint_pixel_sha256']==old_audit['endpoint_pixel_sha256']
 src=load(source_tracks);src_video=Path(src['source_video'])
 case={}
 for name,video,track in [('baseline',Path(record['native_video']),old_tracks),('controlnet',new_video,new_tracks)]:
  t=load(track);assert len(t['frames'])==record['native_frames']
  pose=compare(src,t);contrast=measure(src_video,source_tracks,video,track)
  case[name]={'video':str(video),'sha256':sha(video),
              'limb_motion_error':pose['limb_mean_motion_delta_mae_body_units'],
              'core_motion_error':pose['core_mean_motion_delta_mae_body_units'],
              'right_wrist_source_range':pose['per_joint'].get('right_wrist',{}).get('source_xy_range_body_units'),
              'right_wrist_generated_range':pose['per_joint'].get('right_wrist',{}).get('native_xy_range_body_units'),
              'person_contrast_trend_over_source':contrast['comparison']['gray_std_generated_trend_over_source_trend'],
              'detected_frames':pose['native_detected_frames']}
  (new_dir/f'{name}_pose_compare.json').write_text(json.dumps(pose,indent=2)+'\n')
  (new_dir/f'{name}_contrast_compare.json').write_text(json.dumps(contrast,indent=2)+'\n')
 rows[label]={'frames':record['native_frames'],'same_stage2_endpoint_and_noise':True,'baseline_root':baseline_root,'comparisons':case}
result={'rows':rows,'caveat':'2D cross-view DWPose and contrast diagnostics; visual whole-film review decides overall quality.'}
out=HERE/'depth_controlnet_crosscase_summary.json';out.write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n');print(json.dumps(result,ensure_ascii=False,indent=2))
