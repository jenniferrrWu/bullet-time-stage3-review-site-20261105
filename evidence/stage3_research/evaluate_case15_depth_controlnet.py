"""Full-film diagnostics for pretrained depth ControlNet against native DD controls."""
import hashlib,json,sys,os
from pathlib import Path
HERE=Path(__file__).resolve().parent;ROOT=HERE.parent
sys.path.insert(0,str(ROOT))
from pose_trial.compare_tracks import compare
from person_contrast_proxy import measure

def load(path):return json.loads(Path(path).read_text())
def sha(path):
 h=hashlib.sha256()
 with Path(path).open('rb') as f:
  for b in iter(lambda:f.read(8<<20),b''):h.update(b)
 return h.hexdigest()

def main():
 os.chdir(ROOT)
 entries={
  'zero_control':('wanmove_dense_case15_c_all_0p0','dwpose_zero_control'),
  'flashmotion_repaired_1':('flashmotion_resnet_case15_c_repaired_alpha_1p0','dwpose_flashmotion_repaired'),
  'depth_controlnet_color_0p8':('depth_controlnet_case15_c_weight_0p8','dwpose_depth_controlnet_0p8'),
  'depth_controlnet_midasstyle_0p8':('depth_controlnet_case15_c_midasstyle_weight_0p8','dwpose_depth_controlnet_midasstyle_0p8'),
  'depth_controlnet_midasstyle_0p5':('depth_controlnet_case15_c_midasstyle_weight_0p5','dwpose_depth_controlnet_midasstyle_0p5')}
 source_tracks=ROOT/'pose_trial/v12_balanced_case15_c_source_all/tracks.json'
 source=load(source_tracks);source_video=Path(source['source_video'])
 rows={}
 for name,(trial,track_name) in entries.items():
  folder=HERE/trial;manifest=load(folder/'manifest.json');video=folder/'result/depthdirector.mp4';track_path=HERE/track_name/'tracks.json';tracks=load(track_path)
  track_video=Path(tracks['source_video']);track_video=track_video if track_video.is_absolute() else ROOT/track_video
  assert len(tracks['frames'])==17 and track_video.resolve()==video.resolve()
  assert manifest['state']=='completed' and sha(video)==manifest['native_sha256']
  pose=compare(source,tracks)
  contrast=measure(source_video,source_tracks,video,track_path)
  wrist=pose['per_joint'].get('right_wrist',{})
  audit=load(folder/'native_endpoint_audit.json')
  rows[name]={'video':str(video),'sha256':sha(video),'initial_noise_sha256':audit['initial_noise']['sha256'],
              'endpoint_pixel_sha256':audit['endpoint_pixel_sha256'],
              'core_motion_error':pose['core_mean_motion_delta_mae_body_units'],
              'limb_motion_error':pose['limb_mean_motion_delta_mae_body_units'],
              'right_wrist_source_range':wrist.get('source_xy_range_body_units'),
              'right_wrist_generated_range':wrist.get('native_xy_range_body_units'),
              'person_contrast_trend_over_source':contrast['comparison']['gray_std_generated_trend_over_source_trend'],
              'detected_frames':pose['native_detected_frames']}
  (folder/'pose_compare.json').write_text(json.dumps(pose,indent=2)+'\n')
  (folder/'contrast_compare.json').write_text(json.dumps(contrast,indent=2)+'\n')
 assert len({x['initial_noise_sha256'] for x in rows.values()})==1
 assert len({x['endpoint_pixel_sha256'] for x in rows.values()})==1
 assert len({sha(HERE/trial/'camera.json') for trial,_ in entries.values()})==1
 assert len({json.dumps(load(HERE/trial/'manifest.json')['effective_input_sha256'],sort_keys=True) for trial,_ in entries.values()})==1
 output={'case':'case15 C','frames':17,'same_camera_inputs_noise_endpoint':True,'rows':rows,
         'caveat':'DWPose 2D and person-box contrast are diagnostics; full-film visual review determines acceptance.'}
 path=HERE/'case15_depth_controlnet_summary.json';path.write_text(json.dumps(output,ensure_ascii=False,indent=2)+'\n')
 print(json.dumps(output,ensure_ascii=False,indent=2))
if __name__=='__main__':main()
