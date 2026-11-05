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
FLASH = ROOT / 'paper_sources/FlashMotion'
FLASH_WEIGHT = ROOT / 'paper_sources/FlashMotion-weights/SlowAdapter/ResNet/model.pt'
FLASH_TRAJ = HERE / 'flashmotion_repaired_silhouette_case15_c/trajectory.mp4'
FLASH_SHA = 'df28d58ba918ae38da458f48c2110804a9113cf72f60e71d3c75032077798d58'


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
    return scope['prepare_frames'](pil, (pil[0].height, pil[0].width), do_resize=False, do_crop=False).permute(0, 2, 1, 3, 4).contiguous()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--gpu', required=True)
    parser.add_argument('--weight', type=float, default=0.8)
    parser.add_argument('--control-kind', choices=('dd_color', 'midasstyle'), default='dd_color')
    parser.add_argument('--baseline-root', default='vlm_dd_v12_balanced_case15_c')
    parser.add_argument('--label', default='case15_c')
    parser.add_argument('--flash-alpha',type=float,default=0.3)
    args = parser.parse_args()
    assert 0 < args.weight <= 1
    assert CHECKPOINT.stat().st_size == 691979056 and sha(CHECKPOINT) == WEIGHT_SHA
    assert sha(FLASH_WEIGHT) == FLASH_SHA and FLASH_TRAJ.is_file()
    out = HERE / ('depth_flash_combo_' + args.label + '_cn' + str(args.weight).replace('.', 'p') + '_flash' + str(args.flash_alpha).replace('.', 'p'))
    os.environ.update(CUDA_VISIBLE_DEVICES=args.gpu, ABT_EXPERIMENT_ROOT=str(out), ABT_OBSERVE_NATIVE_NOISE='1', OMP_NUM_THREADS='4', HF_HUB_OFFLINE='1', TRANSFORMERS_OFFLINE='1', IMAGEIO_FFMPEG_EXE='/usr/bin/ffmpeg')
    if os.environ.get('DEPTH_CONTROLNET_BOOT') != '1':
        os.environ.update(DEPTH_CONTROLNET_BOOT='1', LD_PRELOAD='/usr/lib/x86_64-linux-gnu/libstdc++.so.6', LD_LIBRARY_PATH=str(ROOT / 'runtime_libs/usr/lib/x86_64-linux-gnu') + ':' + os.environ.get('LD_LIBRARY_PATH', ''))
        os.execvpe(sys.executable, [sys.executable, *sys.argv], os.environ)
    out.mkdir(parents=True, exist_ok=False)
    os.chdir(DD)
    for path in (DD, DD / 'tools/recam', PROJECT / 'scripts', ROOT, SOURCE, FLASH):
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
    from wan22.modules.model import Wan22ControlResNet
    baseline = ROOT / args.baseline_root
    record = json.loads((baseline / 'stage3/records/route_a/depthdirector/stage3/vlm_selected.json').read_text())
    old_metadata = Path(record['metadata_csv'])
    row = next(csv.DictReader(old_metadata.open()))
    for key in ('input_image', 'input', 'cond', 'depth', 'mask'):
        row[key] = str((old_metadata.parent / row[key]).resolve())
        assert Path(row[key]).is_file()
    assert row['prompt'] == record['prompt']
    control_source = Path(row['depth']) if args.control_kind == 'dd_color' else HERE / ('depth_controlnet_midasstyle_' + args.label) / 'control.mp4'
    assert control_source.is_file()
    frame_count = int(record['native_frames'])
    capture_probe = cv2.VideoCapture(str(control_source))
    height = int(capture_probe.get(cv2.CAP_PROP_FRAME_HEIGHT))
    width = int(capture_probe.get(cv2.CAP_PROP_FRAME_WIDTH))
    capture_probe.release()
    latent_frames = (frame_count - 1) // 4 + 1
    generated_tokens = latent_frames * (height // 32) * (width // 32)
    assert frame_count % 4 == 1 and height % 32 == 0 and width % 32 == 0
    row['output'] = 'result/output.mp4'
    metadata = out / 'metadata_test_all.csv'
    with metadata.open('w') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(row))
        writer.writeheader()
        writer.writerow(row)
    (out / 'depthdirector_stage3.yaml').write_text((baseline / 'stage3/depthdirector_stage3.yaml').read_text())
    (out / 'camera.json').write_bytes(Path(record['camera_path_file']).read_bytes())
    manifest = {'method': 'Official TheDenk pretrained depth ControlNet plus official FlashMotion SlowAdapter both injected into DD Wan blocks',
                'official_source': 'https://github.com/TheDenk/wan2.2-controlnet',
                'official_commit': SOURCE_COMMIT, 'official_checkpoint_sha256': WEIGHT_SHA,
                'not_official_end_to_end_pipeline': True, 'depth_is_DD_estimate': True,
                'weight': args.weight, 'guidance_steps': 40, 'stride': 3, 'gpu': args.gpu,
                'control_kind': args.control_kind, 'control_video': str(control_source), 'control_video_sha256': sha(control_source),
                'label': args.label, 'frames': frame_count, 'height': height, 'width': width,
                'flashmotion_checkpoint_sha256': FLASH_SHA, 'flashmotion_trajectory_sha256': sha(FLASH_TRAJ), 'flash_alpha': args.flash_alpha,
                'baseline_sha256': record['native_video_sha256'],
                'effective_input_sha256': {key: sha(row[key]) for key in ('input_image', 'input', 'depth', 'mask')},
                'postdecode_frame_replacement': False, 'state': 'inference_running'}
    (out / 'manifest.json').write_text(json.dumps(manifest, indent=2) + '\n')
    original = WanVideoUnit_ConcatVideoEmbedder.process
    holder = {'calls': 0, 'active_steps': 0, 'injected_blocks': 0}

    def process(self, pipe, concat_video, tiled, tile_size, tile_stride):
        result = original(self, pipe, concat_video, tiled, tile_size, tile_stride)
        latent = result.get('concat_latent')
        assert latent is not None and latent.shape == (1, 48, latent_frames, height//16, width//16), tuple(latent.shape) if latent is not None else None
        capture = cv2.VideoCapture(str(control_source))
        frames = []
        while True:
            okay, bgr = capture.read()
            if not okay:
                break
            frames.append(Image.fromarray(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)))
        capture.release()
        assert len(frames) == frame_count
        control_video = official_prepare_frames(frames, torch, np).to(device=pipe.device, dtype=pipe.torch_dtype)
        assert control_video.shape == (1, 3, frame_count, height, width)
        # The official FlashMotion SlowAdapter runs once on the same estimated
        # target-view trajectory; its trained residuals are added to 30 DD blocks.
        flash_cap = cv2.VideoCapture(str(FLASH_TRAJ));flash_frames=[]
        while True:
            ok,bgr=flash_cap.read()
            if not ok:break
            flash_frames.append(Image.fromarray(cv2.cvtColor(bgr,cv2.COLOR_BGR2RGB)))
        flash_cap.release();assert len(flash_frames)==frame_count
        flash_video=pipe.preprocess_video(flash_frames)
        with torch.inference_mode():
            flash_latent=pipe.vae.encode(flash_video,device=pipe.device,tiled=tiled,tile_size=tile_size,tile_stride=tile_stride).to(dtype=pipe.torch_dtype,device=pipe.device)
            adapter=Wan22ControlResNet(in_dim=48,dim=3072,num_layers=30)
            adapter.load_state_dict(torch.load(FLASH_WEIGHT,map_location='cpu',weights_only=True)['controlnet'],strict=True)
            adapter=adapter.to(device=pipe.device,dtype=pipe.torch_dtype).eval()
            flash_states=adapter(flash_latent,seq_len=generated_tokens)
        assert len(flash_states)==30 and all(x.shape==(1,generated_tokens,3072) for x in flash_states)
        for i,block in enumerate(pipe.dit.blocks):
            def flash_hook(_module,_inputs,output,i=i):
                assert output.shape[1:]==(generated_tokens*2,3072)
                updated=output.clone();updated[:,:generated_tokens,:]+=args.flash_alpha*flash_states[i].to(device=output.device,dtype=output.dtype)
                return updated
            block.register_forward_hook(flash_hook)
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
                assert len(states) == 6 and all(x.shape == (1, generated_tokens, 3072) for x in states)
                holder['states'] = states
                holder['active_steps'] += 1
            else:
                holder['states'] = None
            hooks = []
            if holder['states'] is not None:
                for j, state in enumerate(holder['states']):
                    block_idx = j * 3
                    def hook(_module, _inputs, output, state=state):
                        assert output.ndim == 3 and output.shape[1:] == (generated_tokens*2, 3072), tuple(output.shape)
                        updated = output.clone()
                        updated[:, :generated_tokens, :] += args.weight * state.to(device=output.device, dtype=output.dtype)
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
