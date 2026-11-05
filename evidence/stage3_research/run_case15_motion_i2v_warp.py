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
    parser.add_argument('--gpu', required=True)
    args = parser.parse_args()
    output = HERE / ('motion_i2v_forwardwarp_case15_c_' + args.region + '_' + str(args.alpha).replace('.', 'p'))
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
    from stage3_research.motion_i2v_official_warp_adapter import propagate_with_forward_warp

    baseline = ROOT / 'vlm_dd_v12_balanced_case15_c'
    record = json.loads((baseline / 'stage3/records/route_a/depthdirector/stage3/vlm_selected.json').read_text())
    old_metadata = Path(record['metadata_csv'])
    row = next(csv.DictReader(old_metadata.open()))
    for key in ('input_image', 'input', 'cond', 'depth', 'mask'):
        row[key] = str((old_metadata.parent / row[key]).resolve())
        if not Path(row[key]).is_file():
            raise FileNotFoundError(row[key])
    assert row['prompt'] == record['prompt']
    track_file = HERE / 'case15_c_dense_tracks/dense_tracks.npz'
    track_manifest = json.loads((track_file.parent / 'manifest.json').read_text())
    assert hashlib.sha256(Path(row['input']).read_bytes()).hexdigest() == track_manifest['source_sha256']
    tracks = np.load(track_file)
    boxes = None
    if args.region == 'person':
        joints = json.loads((ROOT / 'pose_trial/v12_balanced_case15_c_source_all/tracks.json').read_text())
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
        'method': 'Motion-I2V official ForwardWarp with CoW-measured flow into DD concat_video condition',
        'motion_i2v_source_commit': 'bc787e167610a2c6f4d569eca71fb6a26e5b8c56',
        'not_official_motion_i2v_flow_predictor_or_temporal_attention': True,
        'alpha': args.alpha, 'region': args.region, 'gpu': args.gpu,
        'baseline_sha256': record['native_video_sha256'],
        'source_track_manifest': str(track_file.parent / 'manifest.json'),
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
        modified, stats = propagate_with_forward_warp(latent, tracks['xy'], tracks['visibility'],
                                                  tracks['confidence'], alpha=args.alpha, boxes=boxes)
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
    if args.alpha == 0:
        assert digest == record['native_video_sha256'], (digest, record['native_video_sha256'])
    manifest.update(state='completed', native_video=str(paths['native']),
                    native_sha256=digest, call=calls[0], zero_control_reproduced=(args.alpha == 0))
    (output / 'manifest.json').write_text(json.dumps(manifest, indent=2) + '\n')
    print(json.dumps({'output': str(output), 'sha256': digest, 'stats': calls[0]['stats']}), flush=True)


if __name__ == '__main__':
    main()
