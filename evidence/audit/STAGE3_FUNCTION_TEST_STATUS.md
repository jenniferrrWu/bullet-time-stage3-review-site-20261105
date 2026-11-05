# Stage3 功能与整体效果核验（2026-11-04）

| 检查项 | 当前结论 | 证据与限制 |
|---|---|---|
| 原生 DD 推理、有效输入和权重一致性 | 已核验，并更正输入称谓 | case12/13/15 的真实 DD 候选做过预处理文件 SHA、原生视频 SHA、帧数及官方权重检查；当前Stage3实际消费`input_image`、`input`、`depth`、`mask`，metadata里的`cond/dw_warp`未进入模型，改变它仍复现基线MP4 SHA。`dd_effective_inputs.py`已接入逐候选回评核验实际输入映射和SHA。case15 C control gain=1.0 精确复现原基线。见 `v12_balanced_case15_abc_native_integrity.json`、`v12_balanced_case15_c_control_gain_0p5_1p0_1p5_review.json`与`case15_c_warp_background_fill_review/summary.json`。 |
| Stage2 末帧到 Stage3 首帧锚定 | 已核验 | `dd_feedback.collect` 逐候选检查真实 Stage2 native endpoint 绑定；三视频所执行候选通过。 |
| Stage2→Stage3 原生首帧视觉接缝 | 已在四例测过，近似连续 | case12新A、case13旧A/新C、case15 A 的Stage2末帧PNG与原生解码逐像素相同；Stage3原生首帧相对末帧平均差1.20–1.91/255，逐对图看未见明显跳帧。只证明首帧接缝，不证明后续Stage3质量。见 `stage2_stage3_seam_audit/summary.json`。 |
| Stage3 固定目标机位 | 数值路径已核验，视觉仍需检查 | 目标相机矩阵在 Stage3 各帧恒定，Stage2 终点=Stage3 起点；模型仍可能在图像上生成人物/背景漂移。 |
| 连续动作窗口和时长 | 已核验输入；生成动作仍失败 | action.mp4 从所选高光帧连续截取，帧数合法；真实输出不保证动作跟随。case15 C 人物姿态近冻结，case13 多套输出变形/消失；新 f845/左20°/21帧有人物与屋顶连续，重投影对照下腿部延续较好，后段手臂偏源片。 |
| 源动作内容分支是否实际起作用 | 已做原生因果对照，整片仍失败 | case15 C仅把Stage3实际消费的`concat_video`源动作改成首帧冻结；另用相同无损编码重封装动态原动作作控制，逐帧解码像素与原输入完全一致，生成MP4 SHA也与原基线完全相同。冻结组与动态组其他有效输入、首帧锚定、初始噪声一致，输出逐帧平均差5.49/255且随时间增大，证明源动作进入生成并影响输出。但源片右臂横扫在两组都没跟上、后期都褪色，功能有响应不等于动作保真。见`case15_c_concat_video_motion_review/summary.json`与`stage3_contact.jpg`。 |
| 高光选帧与左右/角度/三段时长 | VLM 已能按定式输出，未证明最优 | `planner_v12.py` 输出 DD 可执行的固定 JSON。第三视频 A=f101/右30°/Stage2 4s/Stage3 25帧，仅是候选编号；同帧右15/30/45°原生消融中30°相对更好但仍后段模糊。case13 源帧可见性与动作描述还会被 VLM 误读；其全身候选 A=f75/左30°/33帧、B=f600/右45°/45帧、C=f845/左20°/21帧，真实DD回评把初选 A 改成 C。随后VLM对同f845提出D右35°/E左45°并真实执行；三案回评偏D，但相同证据的BASE左20°与D两案正反顺序回评均选BASE。在BASE/D各自证据完全相同的受控三案回评中又选D，说明加入E后的比较情境会改变赢家；当前角度不能自动裁决，仍须核整段真实视频。 |
| 第三舞蹈局部高光/角度再规划 A/D/E | 原生执行及输入链通过，整体均未通过 | A=f101右30°保持朴素背景但后段脸手软、躯干动作偏；D=f109右45°的Stage3后段发白、右腕水平运动范围相对源片被压到约31%；E=f117左30°能展开手臂，却生成源片没有的右侧帘状结构且末段抬腿/移动不充分。全帧图和源/生成密集对照见`case15_f101_local_refinements_native_review/`。VLM事前分数非质量证据；真实出片回评暂选D，但误称E源片有帘幕并夸大E人体融化，故已判证据不可靠，最终选择仍为空；三者不证明全局最佳。 |
| Stage3 整体成片质量 | **未验收通过** | 密集源/生成同刻对照同时看动作、人物完整度、背景、清晰度和机位；case15 A 相对 B/C 较好但脸手后期模糊，B 动作错位，C 褪色/冻结。不能用单帧、姿态误差或模型评分代替。新case13四套原生Stage3中C=f845相对最稳，但姿态细节不足；A后段建筑重影，B与另方案f622后段融化/背景消失。见 `v12_balanced_case15_stage3_holistic_review.json`。 |
| 人体动作轨迹插入 | 局部有效，未解决整体质量 | DWPose 目标热图门控 DD camera_encoder 时间残差；case12 f171 右肘/腕误差局部下降，case15 C 核心误差下降而肢体变差。均为真实原生 DD 对照，尚不能推广。新增case13 f845将源DWPose按真实DD相机/深度投到目标视角，beta0精确复现原生SHA；beta0.5把后段手臂平均重投影误差约41.6→38.0px，beta1回升到39.2px；三版人物/屋顶都保持，但脸手仍软。另在case15 C的**源动作内容VAE分支**加关节局部时间残差，beta0原生SHA精确复现、beta0.1/0.5同输入原生出片：腕部水平位移略增，但源动作仍没跟上、后段仍褪色。见 `case13_projected_pose_beta1_review/summary.json`、`case15_content_pose_residual_review_0p1/summary.json`、`case15_content_pose_residual_review_0p5/summary.json`。 |
| 非轨迹控制 | 对褪色有影响，动作仍失败 | case15 C camera_encoder gain 0.5/1.0/1.5：提高强度可保轮廓但右腕近冻结；空间分区人物0.75/背景1.5亦未恢复动作。case15 C将真正消费的depth视频mask外编码值填充、有效深度像素不变后原生输出改变，中段略清晰但后段仍褪色且动作未跟上，整体未过；同一DD `WarpControl`在VAE前按已有mask清零无效深度，gate0复现原MP4 SHA，gate1后段少发白但右臂仍近冻结，整体未过；case13 f622/右55°将原61帧的四路预处理视频前21帧无损截短（其中cond未被本配置消费）（解码逐像素一致），短版原生人物/屋顶较清楚但源片俯身动作仍未跟上且边缘有暗条，整体失败；时长也改变latent/noise时间维，不能称严格内部单变量。同一case15 C只调DD原生CFG：CFG2相对基线减轻后段发白但右臂动作仍近冻结，未过整体质量；CFG5则在f8–f12出现几乎全黑画面并在后段恢复，对比度代理在此错误变好，严格判失败。有效输入、初始噪声、首帧锚定一致。把源动作内容残差beta0.1与gain1.5组合，原生输出几乎复现单独gain1.5（全17帧平均RGB差0.344/255），右腕仍近冻结。内容残差beta0.1与VAE前深度遮罩组合也几乎复现遮罩单项（全17帧平均RGB差0.399/255），仍未恢复右臂动作。 另把源内容latent时间变化全局增强0.5，0强度精确复现基线；右腕水平运动幅度增加，但后段更发白、源右臂动作仍没完整跟上，整体失败。见 `v12_balanced_case15_c_control_gain_0p5_1p0_1p5_review.json`、`v12_balanced_case15_c_spatial_control_review.json`与`case13_f622_exact21_review/summary.json`、`case15_c_cfg2_native_review/summary.json`、`case15_c_cfg5_native_review/summary.json`、`case15_c_depth_background_fill_review/summary.json`、`case15_c_mask_depth_before_vae_review/summary.json`、`case15_content_mask_combo_review/summary.json`。 |
| VLM 对成片回评 | 候选内相对排序可参考，绝对评分不可信 | 曾把 C 不存在的 f24 当证据；帧号校验已加，角度三方案补充人体上半身与动作证据后两次顺序均选右30°，但对模糊/变形程度仍有误读。case13跨计划旧f396与新f845回评选新f845，却给其全5分；目标机位重投影表明末帧腿关节约10.6px、手臂约56.9px，脸手偏软。进一步给VLM同机位warp参考与放大图，它仍把beta0.5的小幅手臂变化评成全5分、整体通过；f845三角度真实回评又虚构原20°严重人体融化。f845的BASE/D严格两案正反顺序都选BASE，不能把与三案选择的差异归咎于展示顺序；控制BASE/D证据不变而加入E后仍翻转成D，且两案/三案回评分别夸大D或BASE的“人体融化”，视觉断言不可靠。见 `case13_same_view_native_vlm_feedback_audit.json`、`case13_f845_camera_refinement_vlm_feedback_audit.json`、`case13_f845_feedback_stability.json`、`case13_f845_candidate_set_effect.json`。 |

case15 A/D/E的真实DD回评虽暂选D，但该回评错误地声称E源片有帘幕，并夸大E人体融化，还高估D动作；已标`model_claims_validated:false`，最终方案字段继续为空。见`case15_f101_local_refinements_vlm_feedback_audit.json`。

所有“已核验”只表示该执行或约束成立，不表示 Stage3 最终画质通过。`dd_pipeline.py`现把单轮VLM选项标为provisional，`selected_id`和最终视频字段保持空，见`pipeline_v12_case15_final_gate_verified/pipeline_result.json`。当前目标是以真实 DD 视频的整体效果闭环筛选和迭代。

2026-11-04 同f117机位修复复验：E左30°、F左15°、G右10°均以相同源动作21帧、2.4秒Stage2完成真实DD；有效输入SHA、Stage2端点与原生输出完整性均通过。逐帧整片审查显示E有部分抬臂但虚构右侧帘状背景、末段抬腿不足；F仍有帘状背景且手臂/腿动作近冻结；G背景较干净但手臂几乎全程抱头、运镜小。VLM事前偏F不获出片支持；三案整体均失败，最终机位仍为空。见`case15_f117_camera_repair_native_review/summary.json`和各案`stage3_all_frames.jpg`。

2026-11-04 非轨迹组合负例：case15 C 全局源内容时间增益0.5 + CFG2同原生输入/噪声/权重完成，逐17帧查看f7–f10几乎黑屏、f11后闪成苍白，源右臂横扫未再现。姿态误差代理数字虽下降，不能视为改进；整体严格失败。见`case15_c_global_content_cfg2_combo_review/summary.json`和`stage3_all_frames.jpg`。

2026-11-04 case15 C 源视角零新增角度诊断：复用同17帧源内容、Pi3X几何、官方权重与初始噪声，首帧改源片首帧，目标机位固定源首帧，并重投影相应depth/mask；原生DD17帧、50次首latent约束、无后处理替帧通过。原右20°遮罩有效覆盖均值60.8%，零视角95.1%，源镜头仅转约1.20°。零视角右腕水平活动比右20°多，但仍逐帧转成错误姿态、末段发白，未复现源臂动作，整片失败。说明大角度覆盖不足会加剧问题，但不是唯一原因；锚点和条件也随零视角改变，不能称网络单变量因果。见`case15_c_zero_view_review/summary.json`及全17帧图。

2026-11-04 f117 E/F/G实际出片VLM缩图回评：三GPU看f0/10/15/20源片+各案原生片并给双腕/踝DWPose后改选E，窄义上E确实比F/G保留更多抬臂。但模型称E背景干净连续、F严重人体拉伸、G有中等动作，均与全21帧及关节数值冲突。右侧无人区域源片灰度std均值2.80、E26.07/F38.68/G2.64，强垂直边缘源片/G为0、E2.23%/F6.56%，独立支持E/F虚构帘幕。已标`model_claims_validated:false`，最终机位为空；见`case15_f117_camera_repair_vlm_feedback_audit.json`。

2026-11-04 DD训练能力核验：官方MultiCamWarp配对样本四路各81帧原生`VideoDataset`加载通过；使用官方step-10000初始化camera_encoder+DiT LoRA，在17帧单样本烟测完整完成一次前向/反向/优化，保存的新checkpoint中608/608可训练张量有变化。尚无留出样本或真实hardcase推理比较，不能称Stage3改进。见`training_probe/multicamwarp_nightclub_orbit/one_step_train_audit.json`。

2026-11-04 单步合成数据适配真实hardcase原生回评：官方step-10000初始化，MultiCamWarp一个orbit样本17帧训练一步后新相机编码器+LoRA checkpoint用于case15 C，源动作/实际Stage2末帧/depth/mask/prompt/初始噪声相同，原生50步首latent约束通过。17帧输出SHA不同但相对官方平均RGB差仅0.379/255；逐帧仍褪色、右臂动作未跟上，DWPose肢体误差0.484→0.634身体单位，整体失败。只证明训练→推理链通，不能称合成数据适配有效。见`case15_c_one_step_adapter_review/summary.json`和全帧图。

2026-11-04 固定机位配对数据方向：官方MultiCamWarp同场景orbit深度81帧、180×320；center目标视频与orbit样本原输入逐字节相同。将orbit真实视频作为新输入、center作为固定目标，首帧同机位，可形成比官方移动target更贴近Stage3的反向配对。官方相机+深度低分辨率点投影诊断前17帧覆盖f0=100%、f8=82.1%、f16=68.2%，覆盖区与固定目标RGB MAE3.89/8.50/12.40；f32/f80覆盖降至48.5%/41.2%。只是几何可行性，非原生DD网格条件、非训练效果。见`training_probe/multicamwarp_nightclub_orbit/reverse_fixed_view_proxy.json`、五时刻图及`RESEARCH_AND_PLAN.md`。

2026-11-04 固定新视角数据探针：DD原生网格以orbit源+公开低分辨率深度投到固定center，17帧四路训练MP4生成并单步训练；真case15 C只换新权重原生回评仍褪色、右臂不动，整片失败。新旧全17帧RGB差0.412/255，肢体误差0.484→0.476只是微弱代理变化。反向配对首帧源/目标同机位、仅一场景/深度4倍上采样，不能当匹配真实新角度Stage3的训练集。见`training_probe/multicamwarp_nightclub_orbit/reverse_fixed_view_17_native/`和`case15_c_fixed_view_one_step_adapter_review/summary.json`。

2026-11-04 VLM整窗细节复验：Qwen3-VL-30B看case15 A全部25帧/C全部17帧256px源生成配对，正确判C失败，却把A误判全片清晰通过。A原生f20/f24人脸放大对照显示明显糊化/重影；独立针对性VLM在人脸图上改判细节未保留。证据门据此否决A，原A/B/C三套全部`rejected_observed_failure`，最终选择仍空。见`whole_window_vlm_case15_ac/`、`case15_f101_local_refinements_native_review/A/A_face_detail_vlm_contradiction.json`与`pipeline_v12_case15_all_observed_failures_verified/pipeline_result.json`。

2026-11-04 真实固定新视角舞蹈训练探针：SJTU同步机位0→3首帧就差13.57°，真实f779–795含伸腿、上身和扇子动作。官方DD原生网格条件四路17帧已生成并编码验收，平均全画面mask67.0%、目标人体框mask67.0%，最差帧目标17关节仅11个中心有warp支撑；真实目标允许研究缺口补全，但条件并不完美。官方DD在此单样本一步训练成功（loss0.486、相机编码器8/8与LoRA600/600张量改变），只换权重回测case15 C原生17帧：平均像素差0.397/255、肢体运动误差0.484→0.479，右腕仍近冻结、人物后段继续褪色，整片仍失败。见`training_probe/sjtu_dancer/motion_779_795_native/encoded_training_validation.json`、`one_step_train_audit.json`和`case15_c_sjtu_novel_view_one_step_adapter_review/summary.json`。此方向需多场景训练/留出验证，单步不构成改进。

2026-11-04 细节反馈接入状态：`dd_pipeline.py --focused-face-review`可把哈希/帧号核验过的局部人脸VLM判断作为告警附在候选结果，不把它当整片通过或最终角度。case15 A/C有局部细节损失告警，A与整片VLM高分冲突；即使不提供人工质量证据，管线也在`decision_status`写需独立复核，最终字段null。case12侧身片段又揭示紧脸裁图可能把头发遮挡误称模糊，现用较宽头肩上下文做第三轮复测，不能说局部VLM已泛化可靠。见`pipeline_v12_case15_focused_alert_no_manual_quality/pipeline_result.json`。

2026-11-04 训练对照更正：前述各单样本一步微调的metadata未提供`input_image`，官方训练会默认拿**源视角**首帧作锚点，目标视频却是**目标视角**；因此仅证明训练/回测链路通且真实case15 C未改善，不是严格匹配Stage3的有效训练对照。三段SJTU固定新视角样本现另存`metadata_stage3_anchor.csv`及目标视频f0像素精确一致的`input_image_target_first.png`，后续训练须把`input_image`加入`data_file_keys`；原metadata保留复现。体操/跳绳17帧原生条件编码通过，但最差帧目标人体关节中心受warp支持仅10/17与8/17。当前无Stage3整体质量改进通过。

2026-11-05 正确首帧锚点两场景六步微调的**留出跳绳失败**：官方/新权重在同目标机位、同17帧、同初始噪声SHA下完成；目标右腕纵向动作幅度0.866身体单位，官方0.174、新权重0.171。整段RGB MAE4.840→4.349/255看似变好，但全17帧并排图仍显示两版大部分时间张臂站立，未跟上目标甩绳/跳跃；肢体运动误差0.27923→0.27836近乎不变。`training_probe/sjtu_multiscene/anchor_multiscene_pilot/heldout_paired_review.json`明确`quality_pass:false`。真实case15 C同条件回测仍在进行，不能先报改善。

2026-11-05 VLM细节可见性探针：v4先要求Qwen3-VL-30B判断宽头肩画面中脸是否可比，case12 f12/f24/f32识别头发/转头遮挡，纠正v3的误称；但case15 A f20/f24同时说眼鼻嘴全可见和“不可比”，自相矛盾。结构化审计已标记，仍只作告警，不能自动选角或免除整段核验。见`audit/face_visibility_probe_v4_audit.json`。

2026-11-05 真实hardcase闭环：两场景正确目标首帧六步adapter在case15 C复用原四路有效输入、Stage2端点、prompt、初始噪声和50步首latent pin，只换训练权重，原生17帧完整性通过。肢体误差0.484→0.466的局部数字变化未恢复源右臂横扫，右腕水平幅度源1.174、官方0.130、新权重0.144；后段仍褪色，整段质量失败。留出跳绳与真实hardcase均未通过，不能报告本轮Stage3改善；见`audit/case15_c_multiscene_anchor_six_step_adapter_review/summary.json`与全帧图。

2026-11-05 VLM反馈稳定性：同三例12个原生帧、相同宽头肩裁图、相同本地Qwen3-VL-30B，只收紧人脸可比较性提示，v4/v5有5个yes/no判断翻转；v4在case15 A后段还存在“五官可见”同时判“不可比”的内部矛盾。该局部探针只作待核查告警，不能自动否决、豁免模糊或裁决运镜。`audit/face_visibility_v4_v5_stability.json`。

2026-11-05 Stage3损失实验：SJTU舞蹈＋体操两场景，正确目标首帧锚点，官方step-10000、同训练随机种子，各六步官方损失与目标肢体区域/时间残差加权损失。训练首步原始损失匹配、608/608可训练权重更新。留出跳绳两版同初始噪声的17帧都把甩绳/跳跃压成近张臂站姿；加权损失版肢体运动误差0.2859→0.2880身体单位、目标整帧MAE4.716→5.103/255，未改善。真实case15 C三版原生17帧输入/噪声/锚点核验通过；两训练版仍不跟源右臂横扫、后段褪色。原损失/加权损失的肢体**总体平均**使用不同检出关节，0.661/0.463不可直接比较；同帧右肘0.454→0.455、右腕0.868→0.841，右腕水平幅度0.138→0.107（源1.174），未达到整体改善。`training_probe/sjtu_multiscene/anchor_multiscene_pilot/pose_motion_pilot_review/summary.json`、`audit/case15_c_pose_motion_pilot_review/{summary.json,common_joint_compare.json}`及全17帧图。当前没有可报为Stage3整体通过的方案。

2026-11-05 动作关键部位mask覆盖：SJTU留出跳绳全画面`dw_mask`有效66.7%，目标左腕中心仅3/17帧有效，左肘2/17，而右腕/右肘17/17；右腕虽然满覆盖，生成纵向摆幅仍仅约0.17（目标0.864身体单位）。训练舞蹈/体操目标左腕也仅8/17、6/17有效。全局覆盖不能替代局部人体warp审核，局部缺口与动作分支失效需分别处理。`training_probe/sjtu_multiscene/anchor_multiscene_pilot/joint_warp_coverage.json`。
