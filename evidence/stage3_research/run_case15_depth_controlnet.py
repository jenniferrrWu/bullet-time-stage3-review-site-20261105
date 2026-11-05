"""Inject TheDenk's official pretrained 5B depth ControlNet into native DD Stage3."""
import argparse
import ast
import csv
import hashlib
import json
import os
import sys
import types
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
HERE = Path(__file__).resolve().parent
PROJECT = Path('/public/wufeier/automatic_bullet_time')
DD = PROJECT / 'models/DepthDirector'
SOURCE = ROOT / 'paper_sources/wan2.2-controlnet'
WEIGHTS = ROOT / 'paper_sources/wan2.2-controlnet-weights/depth'
CHECKPOINT = WEIGHTS / 'diffusion_pytorch_model.safetensors'
WEIGHT_SHA = 'adff10028ef3462f9df9afe57d12cebba85003b553df9c862776716c7af4c054'
SOURCE_COMMIT = 'c8b3b70d27048882140dc419b7491c4d63e283f6'


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(8 << 20), b''):
            h.update(chunk)
    return h.hexdigest()


def official_prepare_frames(pil, torch, np):
    """Execute the unmodified official prepare_frames normalization (same-size frames)."""
    path = SOURCE / 'wan_t2v_controlnet_pipeline.py'
    tree = ast.parse(path.read_text())
    function = next(x for x in tree.body if isinstance(x, ast.FunctionDef) and x.name == 'prepare_frames')
    module = ast.Module(body=[function], type_ignores=[])
    ast.fix_missing_locations(module)
    scope = {'torch': torch, 'np': np}
    exec(compile(module, str(path), 'exec'), scope)
    return scope['prepare_frames'](pil, (1280, 704), do_resize=False, do_crop=False).permute(0, 2, 1, 3, 4).contiguous()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--gpu', required=True)
    parser.add_argument('--weight', type=float, default=0.8)
    parser.add_argument('--control-kind', choices=('dd_color', 'midasstyle'), default='dd_color')
    args = parser.parse_args()
    assert 0 < args.weight <= 1
    assert CHECKPOINT.stat().st_size == 691979056 and sha(CHECKPOINT) == WEIGHT_SHA
    out = HERE / ('depth_controlnet_case15_c_' + ('midasstyle_' if args.control_kind == 'midasstyle' else '') + 'weight_' + str(args.weight).replace('.', 'p'))
    os.environ.update(CUDA_VISIBLE_DEVICES=args.gpu, ABT_EXPERIMENT_ROOT=str(out), ABT_OBSERVE_NATIVE_NOISE='1', OMP_NUM_THREADS='4', HF_HUB_OFFLINE='1', TRANSFORMERS_OFFLINE='1', IMAGEIO_FFMPEG_EXE='/usr/bin/ffmpeg')
    if os.environ.get('DEPTH_CONTROLNET_BOOT') != '1':
        os.environ.update(DEPTH_CONTROLNET_BOOT='1', LD_PRELOAD='/usr/lib/x86_64-linux-gnu/libstdc++.so.6', LD_LIBRARY_PATH=str(ROOT / 'runtime_libs/usr/lib/x86_64-linux-gnu') + ':' + os.environ.get('LD_LIBRARY_PATH', ''))
        os.execvpe(sys.executable, [sys.executable, *sys.argv], os.environ)
    out.mkdir(parents=True, exist_ok=False)
    os.chdir(DD)
    for path in (DD, DD / 'tools/recam', PROJECT / 'scripts', ROOT, SOURCE):
        sys.path.insert(0, str(path))
    # Keep Diffusers 0.35 from this venv first, then reuse DD's established dependencies.
    sys.path.append(str(HERE / '.venv/lib/python3.12/site-packages'))
    import cv2
    import numpy as np
    import torch
    import torch.nn.functional as F
    from PIL import Image
    # Diffusers 0.35 passes enable_gqa=False to SDPA; host torch 2.4 lacks
    # that keyword. The official ControlNet uses equal Q/KV head counts.
    native_sdpa = F.scaled_dot_product_attention
    def compatible_sdpa(*values, enable_gqa=False, **options):
        assert not enable_gqa
        return native_sdpa(*values, **options)
    F.scaled_dot_product_attention = compatible_sdpa
    try:
        import modelscope
    except ModuleNotFoundError:
        offline = types.ModuleType('modelscope')
        def unexpected(*_args, **_kwargs):
            raise RuntimeError('Unexpected ModelScope download')
        offline.snapshot_download = unexpected
        sys.modules['modelscope'] = offline
    from diffsynth.pipelines.wan_video_new import WanVideoUnit_ConcatVideoEmbedder
    from wan_controlnet import WanControlnet
    baseline = ROOT / 'vlm_dd_v12_balanced_case15_c'
    record = json.loads((baseline / 'stage3/records/route_a/depthdirector/stage3/vlm_selected.json').read_text())
    old_metadata = Path(record['metadata_csv'])
    row = next(csv.DictReader(old_metadata.open()))
    for key in ('input_image', 'input', 'cond', 'depth', 'mask'):
        row[key] = str((old_metadata.parent / row[key]).resolve())
        assert Path(row[key]).is_file()
    assert row['prompt'] == record['prompt']
    control_source = Path(row['depth']) if args.control_kind == 'dd_color' else HERE / 'depth_controlnet_midasstyle_case15_c/control.mp4'
    assert control_source.is_file()
    row['output'] = 'result/output.mp4'
    metadata = out / 'metadata_test_all.csv'
    with metadata.open('w') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(row))
        writer.writeheader()
        writer.writerow(row)
    (out / 'depthdirector_stage3.yaml').write_text((baseline / 'stage3/depthdirector_stage3.yaml').read_text())
    (out / 'camera.json').write_bytes(Path(record['camera_path_file']).read_bytes())
    manifest = {'method': 'TheDenk official pretrained Wan2.2 TI2V 5B depth ControlNet injected into DD Wan blocks',
                'official_source': 'https://github.com/TheDenk/wan2.2-controlnet',
                'official_commit': SOURCE_COMMIT, 'official_checkpoint_sha256': WEIGHT_SHA,
                'not_official_end_to_end_pipeline': True, 'depth_is_DD_estimate': True,
                'weight': args.weight, 'guidance_steps': 40, 'stride': 3, 'gpu': args.gpu,
                'control_kind': args.control_kind, 'control_video': str(control_source), 'control_video_sha256': sha(control_source),
                'baseline_sha256': record['native_video_sha256'],
                'effective_input_sha256': {key: sha(row[key]) for key in ('input_image', 'input', 'depth', 'mask')},
                'postdecode_frame_replacement': False, 'state': 'inference_running'}
    (out / 'manifest.json').write_text(json.dumps(manifest, indent=2) + '\n')
    original = WanVideoUnit_ConcatVideoEmbedder.process
    holder = {'calls': 0, 'active_steps': 0, 'injected_blocks': 0}

    def process(self, pipe, concat_video, tiled, tile_size, tile_stride):
        result = original(self, pipe, concat_video, tiled, tile_size, tile_stride)
        latent = result.get('concat_latent')
        assert latent is not None and latent.shape == (1, 48, 5, 80, 44), tuple(latent.shape) if latent is not None else None
        capture = cv2.VideoCapture(str(control_source))
        frames = []
        while True:
            okay, bgr = capture.read()
            if not okay:
                break
            frames.append(Image.fromarray(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)))
        capture.release()
        assert len(frames) == 17
        control_video = official_prepare_frames(frames, torch, np).to(device=pipe.device, dtype=pipe.torch_dtype)
        assert control_video.shape == (1, 3, 17, 1280, 704)
        controlnet = WanControlnet.from_pretrained(str(WEIGHTS), torch_dtype=pipe.torch_dtype).to(pipe.device).eval()
        assert len(controlnet.blocks) == 6 and len(pipe.dit.blocks) == 30
        holder.update(control_video=control_video, controlnet=controlnet, states=None)
        old_model_fn = pipe.model_fn

        def model_fn(*fn_args, **kw):
            index = holder['calls']
            holder['calls'] += 1
            step = index  # DD trainer fixes CFG=1, so there is one model call per step.
            if step < 40:
                with torch.inference_mode():
                    states = controlnet(hidden_states=kw['latents'].to(controlnet.dtype),
                                        timestep=kw['timestep'].to(controlnet.dtype),
                                        encoder_hidden_states=kw['context'].to(controlnet.dtype),
                                        controlnet_states=control_video,
                                        return_dict=False)[0]
                assert len(states) == 6 and all(x.shape == (1, 4400, 3072) for x in states)
                holder['states'] = states
                holder['active_steps'] += 1
            else:
                holder['states'] = None
            hooks = []
            if holder['states'] is not None:
                for j, state in enumerate(holder['states']):
                    block_idx = j * 3
                    def hook(_module, _inputs, output, state=state):
                        assert output.ndim == 3 and output.shape[1:] == (8800, 3072), tuple(output.shape)
                        updated = output.clone()
                        updated[:, :4400, :] += args.weight * state.to(device=output.device, dtype=output.dtype)
                        holder['injected_blocks'] += 1
                        return updated
                    hooks.append(pipe.dit.blocks[block_idx].register_forward_hook(hook))
            try:
                return old_model_fn(*fn_args, **kw)
            finally:
                for hook in hooks:
                    hook.remove()
        pipe.model_fn = model_fn
        return result

    WanVideoUnit_ConcatVideoEmbedder.process = process
    import run_stage23_depthdirector_stage3 as native
    native.OUT = out
    paths = {'metadata': metadata, 'camera': out / 'camera.json',
             'native': out / 'result/depthdirector.mp4', 'endpoint': Path(row['input_image']),
             'audit': out / 'native_endpoint_audit.json'}
    native.infer('vlm_selected', paths)
    assert holder['calls'] == 50 and holder['active_steps'] == 40 and holder['injected_blocks'] == 240, {k: v for k, v in holder.items() if isinstance(v, int)}
    manifest.update(state='completed', native_video=str(paths['native']), native_sha256=sha(paths['native']),
                    call={k: v for k, v in holder.items() if isinstance(v, int)},
                    control_input_sha256=sha(control_source))
    (out / 'manifest.json').write_text(json.dumps(manifest, indent=2) + '\n')
    print('completed', manifest['native_sha256'], manifest['call'], flush=True)


if __name__ == '__main__':
    main()
