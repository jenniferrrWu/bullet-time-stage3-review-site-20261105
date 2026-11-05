"""Run paired native DD Stage3 tests with dense first-frame feature propagation."""
import argparse
import csv
import hashlib
import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
HERE = Path(__file__).resolve().parent
PROJECT = Path('/public/wufeier/automatic_bullet_time')
DD = PROJECT / 'models/DepthDirector'


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--alpha', type=float, choices=[0., .35, .7], required=True)
    parser.add_argument('--region', choices=['person', 'all'], required=True)
    parser.add_argument('--track-space', choices=['source', 'target'], default='source')
    parser.add_argument('--gpu', required=True)
    args = parser.parse_args()
    stem = 'wanmove_dense_case15_c_' if args.track_space == 'source' else 'wanmove_targettrack_case15_c_'
    output = HERE / (stem + args.region + '_' + str(args.alpha).replace('.', 'p'))
    os.environ.update(CUDA_VISIBLE_DEVICES=args.gpu, ABT_EXPERIMENT_ROOT=str(output),
                      ABT_OBSERVE_NATIVE_NOISE='1', OMP_NUM_THREADS='4', HF_HUB_OFFLINE='1',
                      TRANSFORMERS_OFFLINE='1', IMAGEIO_FFMPEG_EXE='/usr/bin/ffmpeg')
    if os.environ.get('DENSE_PROPAGATION_BOOT') != '1':
        os.environ.update(DENSE_PROPAGATION_BOOT='1',
                          LD_PRELOAD='/usr/lib/x86_64-linux-gnu/libstdc++.so.6',
                          LD_LIBRARY_PATH=str(ROOT / 'runtime_libs/usr/lib/x86_64-linux-gnu') + ':' + os.environ.get('LD_LIBRARY_PATH', ''))
        os.execvpe(sys.executable, [sys.executable, *sys.argv], os.environ)
    output.mkdir(parents=True, exist_ok=False)
    os.chdir(DD)
    for path in (DD, DD / 'tools/recam', PROJECT / 'scripts', ROOT):
        sys.path.insert(0, str(path))

    import numpy as np
    import torch
    # DD is run strictly offline from verified local checkpoints. Its optional
    # ModelScope downloader is imported at module load even when unused.
    try:
        import modelscope  # noqa: F401
    except ModuleNotFoundError:
        import types
        offline = types.ModuleType('modelscope')
        def _unexpected_download(*args, **kwargs):
            raise RuntimeError('Unexpected ModelScope download in offline DD experiment')
        offline.snapshot_download = _unexpected_download
        sys.modules['modelscope'] = offline
    from diffsynth.pipelines.wan_video_new import WanVideoUnit_ConcatVideoEmbedder
    from stage3_research.wanmove_official_adapter import propagate_with_upstream

    baseline = ROOT / 'vlm_dd_v12_balanced_case15_c'
    record = json.loads((baseline / 'stage3/records/route_a/depthdirector/stage3/vlm_selected.json').read_text())
    old_metadata = Path(record['metadata_csv'])
    row = next(csv.DictReader(old_metadata.open()))
    for key in ('input_image', 'input', 'cond', 'depth', 'mask'):
        row[key] = str((old_metadata.parent / row[key]).resolve())
        if not Path(row[key]).is_file():
            raise FileNotFoundError(row[key])
    assert row['prompt'] == record['prompt']
    source_track_file = HERE / 'case15_c_dense_tracks/dense_tracks.npz'
    track_manifest = json.loads((source_track_file.parent / 'manifest.json').read_text())
    assert hashlib.sha256(Path(row['input']).read_bytes()).hexdigest() == track_manifest['source_sha256']
    track_file = (source_track_file if args.track_space == 'source'
                  else HERE / 'case15_c_target_view_pseudotracks/tracks.npz')
    tracks = np.load(track_file)
    confidence = tracks['confidence'] if args.track_space == 'source' else np.ones_like(tracks['visibility'])
    boxes = None
    if args.region == 'person':
        pose_name = ('v12_balanced_case15_c_source_all' if args.track_space == 'source'
                     else 'v12_balanced_case15_c_target_warp_all')
        joints = json.loads((ROOT / 'pose_trial' / pose_name / 'tracks.json').read_text())
        assert len(joints['frames']) == 17
        boxes = np.array([frame['bbox_xyxy'] for frame in joints['frames']], dtype=np.float32)
        if not np.isfinite(boxes).all():
            raise ValueError('Missing source-view person boxes')
    row['output'] = 'result/output.mp4'
    metadata = output / 'metadata_test_all.csv'
    with metadata.open('w') as f:
        writer = csv.DictWriter(f, fieldnames=list(row))
        writer.writeheader(); writer.writerow(row)
    (output / 'depthdirector_stage3.yaml').write_text((baseline / 'stage3/depthdirector_stage3.yaml').read_text())
    (output / 'camera.json').write_bytes(Path(record['camera_path_file']).read_bytes())
    manifest = {
        'method': 'Wan-Move official create_pos_feature_map and replace_feature into DD concat_video condition',
        'wanmove_source_commit': '80c58a7d2ad175fa82a4d57f79f2a1415317dcfa',
        'not_official_wanmove_weights_or_training': True,
        'alpha': args.alpha, 'region': args.region, 'gpu': args.gpu,
        'track_space': args.track_space,
        'target_tracks_are_geometry_pseudolabels': args.track_space == 'target',
        'baseline_sha256': record['native_video_sha256'],
        'track_manifest': str(track_file.parent / 'manifest.json'),
        'track_file_sha256': hashlib.sha256(track_file.read_bytes()).hexdigest(),
        'effective_input_sha256': {key: hashlib.sha256(Path(row[key]).read_bytes()).hexdigest()
                                   for key in ('input_image', 'input', 'depth', 'mask')},
        'postdecode_frame_replacement': False, 'state': 'inference_running',
    }
    (output / 'manifest.json').write_text(json.dumps(manifest, indent=2) + '\n')

    original = WanVideoUnit_ConcatVideoEmbedder.process
    calls = []
    def process(self, pipe, concat_video, tiled, tile_size, tile_stride):
        result = original(self, pipe, concat_video, tiled, tile_size, tile_stride)
        latent = result.get('concat_latent')
        if latent is None:
            return result
        assert latent.shape[2:] == (5, 80, 44), latent.shape
        modified, stats = propagate_with_upstream(latent, tracks['xy'], tracks['visibility'],
                                                  confidence, alpha=args.alpha, boxes=boxes)
        if args.alpha == 0:
            assert torch.equal(modified, latent)
        else:
            assert not torch.equal(modified, latent)
        result['concat_latent'] = modified
        calls.append({'shape': list(latent.shape), 'stats': stats,
                      'first_latent_exact': bool(torch.equal(modified[:, :, :1], latent[:, :, :1]))})
        return result
    WanVideoUnit_ConcatVideoEmbedder.process = process

    import run_stage23_depthdirector_stage3 as native
    native.OUT = output
    paths = {'metadata': metadata, 'camera': output / 'camera.json',
             'native': output / 'result/depthdirector.mp4',
             'endpoint': Path(row['input_image']), 'audit': output / 'native_endpoint_audit.json'}
    native.infer('vlm_selected', paths)
    assert len(calls) == 1
    digest = hashlib.sha256(paths['native'].read_bytes()).hexdigest()
    control = {}
    if args.alpha == 0:
        import cv2
        historical = Path(record['native_video'])
        old_cap, new_cap = cv2.VideoCapture(str(historical)), cv2.VideoCapture(str(paths['native']))
        errors = []
        while True:
            old_ok, old_frame = old_cap.read()
            new_ok, new_frame = new_cap.read()
            assert old_ok == new_ok, 'Control frame count differs from historical baseline'
            if not old_ok: break
            assert old_frame.shape == new_frame.shape
            errors.append(float(np.abs(old_frame.astype(np.float32) - new_frame).mean()))
        old_cap.release(); new_cap.release()
        old_audit = next((baseline / 'stage3').rglob('native_endpoint_audit.json'))
        prior = json.loads(old_audit.read_text())
        current = json.loads(paths['audit'].read_text())
        assert prior['initial_noise']['sha256'] == current['initial_noise']['sha256']
        assert prior['endpoint_pixel_sha256'] == current['endpoint_pixel_sha256']
        assert len(errors) == 17 and np.mean(errors) < 1.0, errors
        control = {'historical_sha256_exact': digest == record['native_video_sha256'],
                   'historical_decoded_frame_mae_0_255': float(np.mean(errors)),
                   'historical_same_initial_noise_and_endpoint': True,
                   'encoder_note': 'Original uses FFmpeg Lavf61; current system FFmpeg Lavf58. Compare decoded frames.'}
    manifest.update(state='completed', native_video=str(paths['native']),
                    native_sha256=digest, call=calls[0], zero_control=control)
    (output / 'manifest.json').write_text(json.dumps(manifest, indent=2) + '\n')
    print(json.dumps({'output': str(output), 'sha256': digest, 'stats': calls[0]['stats']}), flush=True)


if __name__ == '__main__':
    main()
