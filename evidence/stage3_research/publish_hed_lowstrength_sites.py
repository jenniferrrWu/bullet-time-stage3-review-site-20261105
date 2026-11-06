"""Publish audited HED strength ablation and an explicit testing-method note."""
import json
import shutil
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
HERE = Path(__file__).resolve().parent
DESTS = [Path('/root/bullet_time_progress_sites_20261106/dist'), Path('/root/bullet_time_review_safe_publish')]
TRIALS = [
    ('lama', 'hed_case13_b_lama_w0p2', 'HED＋LaMa 背景 · 强度 0.2',
     '37/37 帧检出人物，末段不再白屏；中段倒立动作缺失、人物局部变形，未通过。'),
    ('affine', 'hed_case13_b_affine_w0p2', 'HED＋LaMa＋二维剪影 · 强度 0.2',
     '37/37 帧检出人物，背景和人物连续性优于原版；倒立动作缺失、躯干/手臂变形，未通过。'),
]
summary = {
    'method': 'TheDenk official pretrained HED WanControlnet with local DD residual adapter; 0.2 strength for first 40 of 50 denoising steps; no retraining',
    'controls': 'Same VLM target camera, true Stage2 endpoint, source clip, effective DD inputs, initial noise and 37-frame native sampling as baseline',
    'baseline_limb_error': 0.093,
    'baseline_person_contrast_trend': 0.903,
    'whole_film_note': 'Both 0.2 runs keep person and background visible but miss the source inversion/mid-film action; neither passes overall Stage3 acceptance. Cross-view 2D pose/contrast scores are diagnostic proxies, not ground truth.',
    'trials': {},
}
for name, ident, _, _ in TRIALS:
    p = HERE / f'hed_controlnet_case13_b_{name}_weight_0p2'
    m = json.loads((p / 'manifest.json').read_text())
    a = json.loads((p / 'native_endpoint_audit.json').read_text())
    pose = json.loads((p / 'pose_compare.json').read_text())
    contrast = json.loads((p / 'contrast_compare.json').read_text())
    assert m['state'] == 'completed' and m['frames'] == 37
    assert a['endpoint_pixel_sha256'] and a['initial_noise']['sha256']
    summary['trials'][ident] = {
        'native_sha256': m['native_sha256'],
        'limb_motion_error': pose['limb_mean_motion_delta_mae_body_units'],
        'core_motion_error': pose['core_mean_motion_delta_mae_body_units'],
        'person_detected_frames': pose['native_detected_frames'],
        'person_contrast_trend': contrast['comparison']['gray_std_generated_trend_over_source_trend'],
        'same_endpoint_and_noise_as_baseline': True,
        'quality_pass': False,
    }
(HERE / 'hed_lowstrength_case13_summary.json').write_text(json.dumps(summary, ensure_ascii=False, indent=2) + '\n')

methodology = '''<div class="methodology"><h3>这些方案怎么测？需要重新训练吗？</h3><p>多数方案没有重新训练：直接调用作者 GitHub 源码和已发布的预训练权重，或在推理时替换控制图、残差和参数。两次正式的 6 步小样本训练只测试 DD 原损失与人体运动区域加权损失；留出视频和真实 case15 整片都没有恢复动作，不能把它当成充分训练或有效改进。早期 1 步测试只验证训练/加载能运行。</p><p>可接入 DD 的方案固定同一 VLM 目标相机、真实 Stage2 末帧、源动作、输入裁切、初始噪声和原生 50 步，再输出完整 17／33／37 帧视频；逐片核对首帧衔接、人物运动、外观、场景和运镜。DWPose 肢体误差与人物对比度仅作辅助，跨视角二维关节和漏检会误导。MoVieS、AnyView 是独立新视角模型，没有 DD 的 Stage2 首帧约束，因此另列，不与可接入方案等同。</p></div>'''

for dst in DESTS:
    data_path = dst / 'review_data-20261106-movies.json'
    catalog_path = dst / 'catalog-20261106-movies.json'
    data = json.loads(data_path.read_text())
    catalog = json.loads(catalog_path.read_text())
    group = next(x for x in data['comparison_groups'] if x['id'] == 'case13-depthcn')
    ids = {x['id'] for x in group['variants']}
    assert all(ident not in ids for _, ident, _, _ in TRIALS)
    for name, ident, label, verdict in TRIALS:
        source = HERE / f'hed_controlnet_case13_b_{name}_weight_0p2'
        native = source / 'result/depthdirector.mp4'
        m = json.loads((source / 'manifest.json').read_text())
        media = dst / 'media/upstream_modules' / ident
        media.mkdir(parents=True, exist_ok=True)
        for filename, scale, crf in [('stage3-fast.mp4', '480:264', '34'), ('stage3-review.mp4', '960:528', '27')]:
            subprocess.run(['ffmpeg', '-hide_banner', '-loglevel', 'error', '-y', '-i', str(native), '-vf', f'scale={scale}', '-c:v', 'libx264', '-preset', 'veryfast', '-crf', crf, '-pix_fmt', 'yuv420p', '-movflags', '+faststart', '-an', str(media / filename)], check=True)
        group['variants'].insert(0, {
            'id': ident, 'label': label, 'verdict': verdict,
            'url': f'./media/upstream_modules/{ident}/stage3-fast.mp4',
            'hd_url': f'./media/upstream_modules/{ident}/stage3-review.mp4',
            'kind': 'official_pretrained_module_local_control_adapter',
            'evidence': f'./evidence/stage3_research/{source.name}/manifest.json',
            'timing': {'fps': 15, 'frames': 37},
            'same_stage2_endpoint_and_noise_as_baseline': True,
            'training_performed': False,
        })
        catalog['variants'].append({
            'id': ident, 'case': 'case13',
            'native_url': f'./media/upstream_modules/{ident}/stage3-fast.mp4',
            'native_sha256': m['native_sha256'], 'native_bytes': native.stat().st_size,
            'review_transcode': True, 'official_pretrained_module': True,
            'native_dd_stage3': True, 'same_stage2_endpoint_and_noise_as_baseline': True,
            'training_performed': False,
        })
    evidence = ['hed_lowstrength_case13_summary.json', 'case13_b_hed_lowstrength_contact.jpg', 'publish_hed_lowstrength_sites.py']
    for name, _, _, _ in TRIALS:
        source = f'hed_controlnet_case13_b_{name}_weight_0p2'
        evidence += [f'{source}/manifest.json', f'{source}/native_endpoint_audit.json', f'{source}/pose_compare.json', f'{source}/contrast_compare.json']
    for rel in evidence:
        source = HERE / rel
        target = dst / 'evidence/stage3_research' / rel
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, target)
        catalog['assets'].append({'name': 'stage3_research/' + rel, 'url': './evidence/stage3_research/' + rel, 'type': source.suffix.lstrip('.'), 'case': 'case13', 'bytes': source.stat().st_size})
    catalog['experiments'].append({
        'id': 'hed_case13_strength_0p2_full_film', 'group': 'Stage3／官方带权重模块迁移', 'case': 'case13 B',
        'title': 'HED 强度 0.2：LaMa 背景与二维剪影',
        'method': '官方 HED ControlNet 预训练权重，强度从 0.5 降至 0.2；同相机、Stage2 末帧、源动作、噪声、50 步，整段 37 帧。无重训。',
        'result': '两版均 37/37 帧检出人物，末段连续性改善；中段倒立动作缺失、人物仍变形，整片未通过。',
        'status': '两条原生整片完成／连续性改善但动作失败',
        'evidence': ['stage3_research/hed_lowstrength_case13_summary.json', 'stage3_research/case13_b_hed_lowstrength_contact.jpg'],
    })
    catalog['counts']['native_nontrajectory_variants'] += len(TRIALS)
    catalog['counts']['upstream_module_trials'] += len(TRIALS)
    catalog['counts']['experiments'] = len(catalog['experiments'])
    catalog['counts']['evidence_files'] = len(catalog['assets'])
    data['summary'] = 'HED 0.2 的 case13 两条整片保持人物和背景连续，但漏掉倒立动作；HED 0.5 对 case12/15 有局部动作收益。所有方案仍未跨三段通过 Stage3 整片验收。'
    data_path.write_text(json.dumps(data, ensure_ascii=False, separators=(',', ':')) + '\n')
    catalog_path.write_text(json.dumps(catalog, ensure_ascii=False, separators=(',', ':')) + '\n')
    hp = dst / 'review-movies-20261106.html'
    html = hp.read_text()
    assert '这些方案怎么测？需要重新训练吗？' not in html
    html = html.replace('</section>\n<section id="pipeline"', methodology + '</section>\n<section id="pipeline"', 1)
    anchor = ' · <a href="./evidence/stage3_research/hed_controlnet_summary.json">完整指标 ↗</a>'
    links = ' · <a href="./review-movies-20261106.html?group=case13-depthcn&amp;variant=hed_case13_b_lama_w0p2#compare">LaMa 0.2 ↗</a> · <a href="./review-movies-20261106.html?group=case13-depthcn&amp;variant=hed_case13_b_affine_w0p2#compare">二维剪影 0.2 ↗</a> · <a href="./evidence/stage3_research/hed_lowstrength_case13_summary.json">强度对照 ↗</a>'
    assert anchor in html
    html = html.replace(anchor, links + anchor, 1)
    html = html.replace('case13 四种输入均失败。', 'case13 的 0.2 强度连续性改善，但倒立动作缺失；整体仍失败。')
    hp.write_text(html)
    (dst / 'index.html').write_text(html)
    css = dst / 'style.css'
    style = css.read_text()
    if '.methodology{' not in style:
        style += '\n.methodology{margin-top:20px;padding:20px 24px;border:1px solid #384251;border-radius:14px;background:#101923}.methodology h3{margin:0 0 10px;color:#e9f3ff;font-size:1.05rem}.methodology p{margin:8px 0;color:#b8c7d8;line-height:1.72}\n'
    css.write_text(style)
    print(dst, 'variants', len(catalog['variants']), 'experiments', len(catalog['experiments']), 'assets', len(catalog['assets']))
