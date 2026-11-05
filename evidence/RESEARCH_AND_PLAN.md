# 子弹时刻：Planner 与 DD 动作诊断（2026-11-02）

目标保持两条完整主线：VLM 自主选择高光/方向/角度/各段时长并真正驱动 DD；严格排除运行错误后，在 DD 架构内实验人体轨迹和非轨迹改进。尚未完成，不能把脚本、出片或可解码当质量通过。

**当前进度（2026-11-04）：** VLM已能提出合法的高光帧、运镜方向/角度和三段时长并驱动真实DD生成，候选内回评也能改变初选，但未有用户标注的最佳帧，亦未证明全局最优。第三视频case15的A=f101/右30°/4秒/25帧只是编号；A/B/C均完成原生Stage2→3，A在这三套里相对稳，后段脸手模糊且空间运镜感弱，整体未通过。VLM同帧15°/30°/45°原生对照两轮选30°，限已测档位，绝对评分会高估。另对case13新全身候选A=f75、B=f600、C=f845及f622候选均真实生成，C=f845/左20°/21帧保持人和屋顶最稳，但手臂后段偏移；其源关节按DD深度/相机重投影到新机位后，f20腿平均差约11px、手臂约57px。旧f396与新f845跨计划模型回评选新C，却误给全5分；整体画面与用户验收仍是最终标准。Stage3原生输入/锚点/固定机位通过严格核验，人体轨迹与非轨迹增益只局部改善，尚无全面质量通过。见`audit/STAGE3_FUNCTION_TEST_STATUS.md`、`audit/case13_old396_vs_new845_vlm_feedback_audit.json`和文末最新实验。

## 已取得的证据

- 15例×2阶段重新核查了实际 metadata.csv 指向的 input.mp4，而不只是原动作路径。全部原生输出 SHA 与记录相符；Stage3 对应当前 Stage2，端点像素一致。
- 5条竖屏 case01/03/05/12/15 受到 tools/recam/utils/ioutils.py 的 depthcrafter 16:9 中心裁切，仅保留约29.6%–32.5%源面积。后续 VideoDataset 再裁到竖屏，造成首图、内容、深度画幅不一致。历史“native_verified_runs”并未核透该层，不能作为这些案例纯模型失败的依据。
- Pi3X.video_depth 还固定 width=504 后裁高于504的画面，含未定义 resize_depths；原外层横裁使这条分支未触发。内参恢复还将 fx/fy 同乘 width 比率，round到14倍数时 y 缩放应独立。
- case11/case13 两条横屏全配方复跑完成，新旧 MP4 SHA完全相同。case13抽帧确认严重融化；case11相对较好。只证明稳定复现及已查条件正确，不证明所有几何估计都准确。
- 原动作全部81帧不等于模型实际看到完整空间画面；两项必须独立审计。

## DD 接口与接入点

仓库 /public/wufeier/automatic_bullet_time/models/DepthDirector。

1. tools/recam/processor/trajectory.py:process -> Pi3X.video_depth -> warp_mono_video -> Dynamic3DMesh.build/render。
2. metadata.csv 的 input 为真正内容条件；depth/mask 是目标机位投影；input_image 为本次Stage2原生末帧。
3. diffsynth/trainers/trainer.py:prepare_data 将 input -> concat_video，depth/mask -> warp_video/warp_mask。
4. wan_video_new.py:WanVideoUnit_ConcatVideoEmbedder 用 VAE 编码到 concat_latent。
5. WanVideoUnit_WarpControl 用 VAE 分别编码深度/遮罩，拼通道后经 camera_encoder -> control_latents。
6. WanVideoUnit_ImageEmbedderFused 及 denoising loop 保持原生首latent。不保证解码首像素相同，不能生成后替帧。
7. 默认81帧/15fps/50步/CFG1；配置可写4n+1帧，但新长度必须实跑验证。输出时长和源动作覆盖范围必须分别记录。

## 研究筛选与实施选择

- [DD官方代码](https://github.com/FREDZEL2020/DepthDirector)，[论文](https://arxiv.org/abs/2601.10214)：双流分开表达内容与视角。保留该骨干和已训练深度编码器；不会把骨架RGB直接塞进深度通道并称其原生支持。
- [CamPilot相机反馈](https://arxiv.org/html/2601.16214v1)：可见区域约束提供评估思路；其3D奖励网络并非现成VLM Planner，也不是无需训练即可移植的模块。
- [同名CamPilot电影规划](https://arxiv.org/abs/2609.10943)：规划与镜头控制分离的思路有参考价值，不冒称复现其GRPO训练或指标。
- [VideoCoCo官方](https://github.com/micky-li-hd/VideoCoCo)：借鉴可执行中间方案、先检查代理表现再生成的流程。其物理代理/OmniWeaving不替换DD；我们用DD深度代理和严格计划编译。
- [CoWTracker官方](https://github.com/facebookresearch/cowtracker)，[论文](https://arxiv.org/abs/2602.04877)：现有本地完整权重，可提取密集表面点与可见性/置信度。它不是语义骨架，需要人体区域选择/姿态估计才能称人体关节轨迹。
- [Wan-Move官方](https://github.com/ali-vilab/Wan-Move)：沿轨迹传递首帧特征的思路可作为DD条件分支实验。其成功依赖训练，不能推断在DD上直接搬特征就有效；权重和模型不替换DD。
- [Motion-I2V官方](https://github.com/G-U-N/Motion-I2V)：显式运动表示与生成分离。先作为结构参考，优先复用已有跟踪器而不引入另一整套生成模型。

## 轨迹方向：两种DD内方案

T1（优先）：轨迹约束的几何时序稳定。提取人体区域/表面点轨迹，沿轨迹采样Pi3X深度，过滤遮挡/深度断层；仅优化几何深度，不训练生成权重。目标
L = Σ w(t,j)·Huber(log z(t,j) - log z_obs(t,j)) + λ Σ valid·Huber(Δ² log z(t,j))。
保留实际2D动作位移，不在固定像素上平滑跨肢体/背景。经原生K/源姿态反投影，再用Stage3固定目标机位重投影，通过原生depth/mask→camera_encoder注入。风险：加速度平滑可能抑制真实快速运动，需λ=0对照、置信度门控及不同强度。

T2：借鉴Wan-Move的轨迹特征传递，投影到新视角后的轨迹用于约束DD条件特征（优先深度控制特征，必要时小强度内容latent对照），不替换生成帧。必须先通过目标坐标/遮挡/时间压缩映射检查；未经训练的DD对此可能分布外，仅是实验假设。若需要训练才能奏效，另写冻结骨干的adapter设计与数据计划，不擅自宣称已有权重支持。

评估：先检查轨迹可见覆盖、前后向误差/深度重投影一致性，再检查生成的人体持续性、相对运动、背景和机位。不得用“更静止”冒充“动作更准确”，也不以VLM自评分证明最优。

**输入链更正（2026-11-04）：** 此项目当前Stage3生成配置`extra_inputs: "input_image,depth_cond,concat_video"`，对应模型真正接收Stage2末帧、`data["input"]`源动作内容、`data["depth"]`动态深度及`data["mask"]`遮罩；metadata中的`cond/dw_warp`不在`data_file_keys`，并未进入该原生DD模型。此前多处“同五路输入SHA”应理解为预处理文件一致性，其中`cond`只作外部目标视角几何/姿态诊断，不是模型控制输入。直接把`dw_warp`无效黑区填成背景色，17帧有效像素逐帧完全一致、无效区平均改动约165/255，原生输出仍与基线MP4 SHA完全相同，证明此路径无效；见`audit/case15_c_warp_background_fill_review/summary.json`。这不是DD运行出错，而是审计对实际输入的称谓需要纠正。后续非轨迹填充对照改在**真正消费的depth视频**上。

## 非轨迹方向

N0：修复真实输入画幅，分别重跑Stage2→Stage3。full_canvas/ 独立记录；首图/动作同一显式letterbox，几何不裁，Pi3X保持完整比例，fx/fy正确反缩放。先测case12，再测case15。
N1：case13同真实条件只改CFG1→5；N2：同条件只改LoRA强度1→0.5。两项均保留seed42/50步/81帧和噪声证据，不能与L1旧修复混计。判定不仅看融化减少，还要核对动作与新视角保留。
N3待做：源视角重建/零角度对照，分离动作编码失败与新视角合成困难；源镜头运动与固定目标相机几何检查。

## Planner v1：严格协议与执行约束

planner.py 实现两轮本地VLM：全片粗选20帧→三个候选附近精细检查→三套完整候选计划及选优。无手工补造候选、无静默夹角/截帧修正；不合法JSON/时序约束回交模型重答。当前30B替换失效的8B软链接，保留raw。

schema_version=dd-bullet-plan/1.0；主体、适用性、selected_id、三套候选、风险。每套含 highlight_frame/orbit_direction/orbit_degrees/orbit_seconds/action_end_frame/action_seconds/reason/score。Stage1从源起点至高光，长度由模型选帧决定；Stage2/3显示时长由VLM决定，原生帧全保留、显示PTS调整明确标注。Stage3源窗口17..81连续真实帧且4n+1；5..60°是初始研究范围，不是已认证能力。

未完成：JSON Schema独立文件、代理几何评分/返修、真正DD执行器接通、端到端生成与对照评估、候选优劣人工验收。不能称“已搭好整个pipeline”或“已规划出最优”。

## 目录与后续接手

工作目录 /root/bullet_time_research_20261102。
audit/input_audit.json 为15例新核查；replay/*/manifest.json 为两例复跑；full_canvas/ 为修复新链；nontrajectory/ 为单变量实验；tracks/ 为轨迹估计；planner_case13_30b/ 为新VLM调用。
每次续跑先查真实PID/GPU及输出，不能凭状态文件running重复启动。官方代码与旧结果不覆盖。

## 2026-11-03 增量进展

Planner v1三次真实30B回答均因包含端点的帧数算术错误被拒绝；原始回答保留。v1.1改由VLM选择action_num_frames（17..81的4n+1枚举），编译器推导终点、实际播放倍率；新调用运行中。独立JSON Schema已保存。dd_execution.py已实现共享几何、同一半径与数字机位接续、原生末帧、全帧显式时序拼接，尚待真实端到端验证。

case12全画幅Stage2已完成，Stage3运行中；case15 Stage2运行中。CoWTracker密集轨迹完成；筛选1653条可靠运动轨迹后进行了深度正则化（平均8.81%空间支持，log深度改动限制0.1，首帧保持），经原生渲染接入DD并运行中。这里是运动表面点，不声称语义人体骨架。原始几何重建的depth/mask/input MP4与历史SHA全部一致。

case13 CFG5与LoRA0.5均出片；同图/动作/深度/遮罩/初始噪声核对通过。实际七时刻抽帧确认CFG5彩色噪声，LoRA0.5紫色雾化，均未修复，不能报改善。audit/nontrajectory_visual_review.json留证。

## 2026-11-03 实测更新（以本节为当前状态）

- 严格复现 case11/case13 Stage3 81帧并与历史 MP4 SHA 完全一致；case13 融化是真的可复现失败。CFG5、LoRA0.5 单变量原生对照均更差；记录在 `audit/nontrajectory_visual_review.json`。
- T1首轮 CoWTracker 运动表面点深度时序稳定已经完成原生 Stage3 81帧。抽查 f0/8/20/40/60/80 与基线几乎同样严重融化，未见改善；`audit/trajectory_depth_visual_review.json`。该实验不是人体关节轨迹，不得宣称已验证关节方法。
- 全画幅竖屏 case12 Stage2+Stage3 已完成。取景比历史显著恢复，但中后段仍有人体变形，质量未过；`audit/full_canvas_case12_visual_review.json`。case15 Stage2+Stage3 已完成：Stage3 704×1280/81帧、原生首latent 50次 pin、首图为本次 Stage2末帧。六时刻与源动作对照较好，人物完整，无严重融化；仅为抽样暂定正例，缺全连续和接缝验收，且未找到其历史 Stage3 直接对照；`audit/full_canvas_case15_visual_review.json`。
- Planner v1.1 本地Qwen3-VL-30B-A3B-Thinking 真实调用与schema校验通过。case13选 A: 原片第396帧/向右30°/4s环绕/25帧动作/0.834s；编译保留源帧396..420。视觉复核显示 A 人物腿部出画，模型称“姿态完整”并错误描述左侧裁切；候选 B(622帧)静帧显示全身，故 A 不能称最佳。`audit/planner_case13_frame_review.json`。
- `dd_execution.py` 已真正执行 A 的 Stage1→Stage2→Stage3：Stage1 397帧，Stage2 原生81帧，Stage3 原生25帧；Stage3首图SHA与本次 Stage2原生末帧一致，Stage3实际相机矩阵首帧等于 Stage2终点且25帧固定，两个阶段首latent原生约束各50次且最终差0。Stage2有视差/基本冻姿，Stage3抽样动作与源25帧对应，无严重融化，但裁切仍在。见 `audit/vlm_case13_stage2_visual_review.json`、`audit/vlm_case13_stage3_visual_review.json`。
- 最终 `vlm_dd_case13_v11/complete.mp4` 503帧；Stage1/2/3分别13.247/4.000/0.834秒，容器总时长18.081秒，所有帧PTS唯一且无增删。两处接缝抽样连贯；`vlm_dd_case13_v11/timing_audit.json` 和 `audit/vlm_case13_complete_review.json`。DD执行链路已跑通，但“最佳方案”、跨案例泛化、原片全部剩余动作覆盖均未证明。
- 下一步：针对候选 B 做真实或至少几何代理对照，增加独立全身可见性校验与结果反馈；补 N3 源视角/零角度对照；T1需真实人体区域或关节轨迹、对照强度与动作一致性量化。case12/case15需全段连续复核。若完整保留高光后的整段源动作，则需设计超过81帧的连续分窗 Stage3 与锚点传播，当前25帧试验不满足这一更强范围。

## 2026-11-03 A/B 原生对照与选优修正

- 以原 VLM 给出的 B（622帧、左45°、Stage2 5.5秒、Stage3 33帧/1.101秒）作为**明确人工指定的对照**，并未修改原模型历史输出。相同 `dd_execution.py` 与官方权重完成B：Stage1 623帧、Stage2 81帧、Stage3 33帧，最终737帧/27.389秒；真实Stage2末帧锚定、数值相机终点相接、Stage3固定机位、原生首latent pin均通过。B 的 Stage2 人物完整且有视差，Stage3 中段后人物和背景严重融化，明显差于A。见 `audit/vlm_case13_candidate_comparison.json`。此对照改变帧、方向、角度、时长和动作内容，不能单因归因。
- `planner.py` 的单次“全身可见”门控有误判；双提示检查能发现A底部裁切并让VLM改选B（`planner_case13_v14/`），但B真实生成更差，证明**可见性不能硬排除**。当前代码已撤去硬排除，仅保留两轮可见性风险记录，最终选优须等DD反馈。v14是失败设计留证，不能用其B作为最终推荐。
- 10张真实DD候选输入/输出截图交给本地30B做反馈，其判断A优于B：A Stage3人体/场景/动作分别4/4/4，B分别0/0/0，理由为B中后段彻底融化；`planner_case13_dd_feedback/result.json`。人工抽帧独立支持这个相对结论，但两候选之外未比较，不能称全局最优。
- DD `dynamic_mesh.py` 的 `dw_mask` 白色=有动态网格面支撑。新增 `dd_geometry_proxy.py` 和六例统计 `audit/geometry_proxy/summary.json`：A Stage3平均有效覆盖26.1%、28%帧<20%；B17.1%、97%帧<20%；历史相对较好case11 59.3%，严重失败case13 11.5%；竖屏case12 28.0%仍有融化，case15 46.0%抽样较好。低覆盖与严重失败在这些案例相关，但不是充分条件或通用阈值；case12是反例。下一步将其作为DD代理风险特征，结合动作/遮挡/模型反馈，而非单阈值承诺质量。

## 2026-11-03 Stage3 分项核验及源镜头控制

- `dd_feedback.py` 已作为可复用候选回评脚本，对 A/B 原生输出核对计划字段、视频、端点、相机固定/相接、首帧锚定后调用本地30B评估。`planner_case13_dd_feedback_generic/result.json` 选A；只证明已执行两候选中A较好，不是全局最优，也未执行C。
- N3 固定首帧源方向的81帧原生 Stage3 正在 `nontrajectory/zero_view_case13/` 推理。此控制虽为零新增转角，却不是逐帧源视角恒等控制：原动作镜头向右跟拍，Pi3X 源相机相对首帧旋转约8.1°/23.6°/53.1°/71.6°/90.5°（f8/20/40/60/80）。固定首帧方向的网格有效画面从f0的100%降至f80的0.8%；不能据此单独归咎网络生成人物失败。
- 新增真正逐帧源位姿匹配的几何控制 `source_pose_geometry_control.py`，目标镜头=每帧对应源镜头（因此镜头在移动，不能充当最终固定镜头 Stage3）。`audit/source_pose_geometry_control.json` 中 f0/8/20/40/60/80 的有效画面分别100%/95.0%/95.5%/95.3%/96.2%/95.7%。对比说明上述几何覆盖崩塌主要来自把明显跟拍的原动作强行重投影到固定首帧视角；不支持“Pi3X深度后期全坏了”的简单结论。仍须在原生生成视频上判断画面质量，并测源相机运动风险门控/更短时间窗。
- A/B 候选各自 `shared_geometry/pose.npy` 也支持“源相机运动”作为风险特征：A 的25帧末端相对首帧旋转6.08°、平移0.736深度单位（首帧中位深度31.86）；B 的33帧末端旋转10.98°、平移4.863单位（首帧中位深度39.09）。B动作、方向、角度和时长也同时变化，故这只是风险关联，不能单因证明B融化由相机运动造成。
- N3 固定首帧视角的原生 Stage3 81帧已经出片：1280×704/81帧可解码、50次首latent pin、最终差0、官方权重不变、无解码后替帧。`audit/case13_zero_view_compare.jpg` 对照原动作和历史30°：f8已严重人物/背景涂抹，f20以后几乎全融化，质量未过。见 `audit/zero_view_case13_visual_review.json`。这否定“只要把新增30°改为0°就能救回case13”，但因为原镜头大幅跟拍，不能表述为网络在真实逐帧源视角也失败。下一步需选择源镜头低运动片段或对固定目标视角增加几何覆盖/长时窗策略。

## 2026-11-03 继续：三候选、跨视频与语义关节准备

- `dd_camera_motion.py` 从实际进入 DD 的 Pi3X `shared_geometry/pose.npy` 与深度计算相对首帧旋转/平移风险，作为诊断接入 `dd_feedback.py`；不设通用通过阈值。A=25帧末端6.08°、平移/首帧中位深度2.31%；B=33帧10.98°、12.44%；C=17帧26.06°、11.70%。A/B `collect` 原生一致性校验已复测通过。C 的大幅镜头变化是事前风险，不是成片结论。
- `planner_case13_candidate_c_override.json` 仅把原模型三候选计划的 selected_id 改为C，`vlm_dd_case13_candidate_c/` 正在执行完整原生 Stage2→Stage3。原始 VLM 计划未改写。C 源高光帧人物贴近边界，`audit/case13_abc_source.jpg` 可见上臂/下半身裁切；先前双提示可见性检查也标记 C 不完整。最终仍需看原生出片再排序。
- 第二个真实竖屏舞蹈视频 `/public/wufeier/automatic_bullet_time/inputs/formal15/new_dance/a-woman-dancing-alone-at-home-5743004.mp4`（1440×2732、560帧/25fps）已启动独立本地30B Planner 泛化实验，输出目标 `planner_case12_30b/`；尚未有可验证计划时不得称泛化成功。
- 官方 DWPose OpenCV ONNX 代码最小文件已取自官方分支；官方姿态模型与 YOLOX 检测器经 HF 官方 SHA256（分别 `724f4ff2...`、`7860ae79...`）进行分块续传，完整下载尚待验证。`pose_trial/extract_dwpose.py` 准备输出原始 COCO-WholeBody 前17个人体关节、置信度、逐帧检测框和可视化；未跑前不得称人体关节轨迹已成功，更不得称改善 DD。
- 已增加独立 `dd_plan.schema.json`（Draft 2020-12）并接入 `planner.validate`，固定顶层字段、A/B/C各恰一套、DD合法 4n+1 动作窗、角度/时长/分数范围和无多余字段；源视频相关的观察帧与结尾越界仍由动态校验负责。已用现有A计划通过、重复ID和18帧动作窗拒绝进行实际校验。该合同解决格式漂移，仍不等于规划质量最优。
- case13 VLM 候选 C 已完整原生执行并通过 `dd_feedback.collect` 一致性/帧数/相机/锚点核验。Stage2 25°运镜抽样保持人物，但源高光本身严重贴边裁切。Stage3 17帧 f0/4/8 尚可见动作，f12开始人物/背景发白溶解，f16几乎白屏，质量失败；见 `audit/case13_candidate_c_visual_review.json` 与 `audit/case13_candidate_c_stage3.jpg`。值得注意：C 的 `dw_mask` 平均有效覆盖65.9%、无帧低于20%，却仍失败，明确反证“几何覆盖率高就够了”。A/B/C 三候选本地30B原生结果回评正在运行；最终排序以实际出片和人工抽帧复核为准。
- case13 25帧短窗单变量对照的前两次预备运行已作废：直接码流截取导致末两帧解码不一致；重新渲染条件导致视频编码像素有细微差异。第三版 `short_window_case13_exact25.py` 将历史 `input/cond/depth/mask` 四路视频的前25帧全部无损重编码并逐像素比较，四路均完全一致后才开始原生 Stage3。此控制仅改变DD所见的总时间窗长度，仍待画质结论。
- 第二个竖屏样例 Planner 已产出格式合法的A/B/C三候选，选A（f171、右30°、Stage2 4秒、Stage3 25帧/1秒）；`planner.validate` 与独立 schema 均通过。源帧人工核查A全身可见、C明显裁切；`visibility.json` 对C有正确风险标记。完整原生 DD A 正在运行。其 Pi3X 源镜头25帧仅旋转1.39°、平移/首帧中位深度0.96%，可与case13大镜头运动作为对照。
- 严格短窗对照 `nontrajectory/short_window_case13_exact25_v2/` 已完成。历史 input/cond/depth/mask 四路前25帧解码后逐像素完全一致，历史 Stage2 首图、prompt、权重和 seed 参数不变；25帧1280×704原生解码、50次首latent pin且最终差0。与历史81帧同一前25源动作比较：短窗 f4 人物更清楚但动作近乎冻结，f8开始融化，f12以后基本丢失，f20/24近白屏，未救回；见 `audit/case13_short_window_exact25_review.json`。改变运行时长度也改变 latent/noise 时间维（7 vs21）和注意力上下文，不能声称内部噪声完全相同。
- 官方 DWPose 姿态ONNX（134399116字节）与YOLOX检测器ONNX（216746733字节）均已按官方文件SHA256完整校验，OpenCV DNN 可加载。对 case12 A 25帧和历史 case13 81帧逐帧自动检测/提取真实COCO身体前17关键点：前者25/25检测成功、平均16.6/17置信通过、最低15；后者81/81检测成功、平均15.06/17、最低12，踝关节由于画幅裁切覆盖仅约44%/51%。`pose_trial/..._full/tracks.json`、`audit/dwpose_*_audit.json` 和 overlay 已人工抽查坐标贴合可见四肢。这只是语义轨迹准备，尚未接入DD改善实验。
- 第二独立竖屏样例 case12 A 的原生 Stage2/Stage3 已完成；Stage2 右30°抽样有背景视差、人物基本稳，Stage3 25帧人物/房间结构完整，但动作明显跟随不足：源舞者持续俯身并抬手靠近脸，生成舞者接近起始姿态。官方DWPose同一模型在源与生成视频检测鼻子/双髋全部25帧都通过置信门槛；鼻子相对骨盆竖直位移范围按首帧肩髋长归一化，源0.636，生成0.125（约20%），作为“动作被压平”证据；见 `audit/case12_vlm_a_motion_audit.json`、`audit/case12_vlm_a_stage3.jpg`。目标视角 `dw_warp`/`dw_depth` 抽帧可见躯干/头随时间下降，不能把失败简单归因于几何条件不含动作。
- 整片封装曾因拼接最终 `-r 30` 而把末段25fps最后一帧按1/30秒计，导致总时长11.874秒而目标11.88秒；该验收正确拒绝了。已修正 `dd_execution.py` 最终 `-r` 取 Stage3 实际显示帧率，验证全部278帧、时间戳唯一和11.88秒。旧错误文件保留 `complete_rate30_duration_error.mp4`，修正版为 `complete.mp4`；画质仍未通过。
- 对 case13 C 增加 `dd_visual_artifacts.py` 逐帧近白屏对照：C末帧近白亮低饱和像素99.996%，比同帧源高66.3个百分点；A最大增量为负、B的融化不一定触发此指标。回评脚本现在额外采样最大白屏异常帧并保存诊断，依然不把单指标当通用质量判据。
- case13 A/B/C 三套原 VLM 候选已全部经同一官方 DD Stage2→Stage3执行、锚点/相机/帧数/hash 完整性核验及本地30B回评；`planner_case13_dd_feedback_abc/result.json` 选A，`selection_status.json` 明确仅为三套已执行候选内优胜，不能证明所有帧/角度的全局最优。人工复核也选A；模型对C末帧近白屏仍给 Stage3 3/5，说明VLM自评偏宽，故增加近白屏最大异常帧采样并保留人工质量门。
- 语义轨迹第一轮 DD 内插入实验 `pose_trial/feature_gain_case12.py` 已运行中：官方DWPose从 DD 目标视角 `dw_warp` 提取带置信度关节，最长3帧的有界缺口才线性补点，7个 latent 时间块形成高斯化骨架热图；在现有 `WanVideoUnit_WarpControl.process` 的 `camera_encoder` 输出上仅于关节轨迹空间乘以 `1+0.75*heatmap`，不改官方权重/输入视频/深度/遮罩/首图，也不训练。并行 alpha=0 必须字节级复现原生 case12 A 后，才比较 alpha=0.75 的动作幅度、人物/场景完整性。该推理实验没有损失函数，不能假装已训练轨迹适配器。

### 语义关节方法的训练版设计（当前未训练）

- 现有推理实验将目标视角的17关节轨迹热图输入点设在 `WanVideoUnit_WarpControl.process` 的 `camera_encoder` 输出之后、`model_fn_wan_video` 加到生成token之前；无新增权重。alpha0与case12原生基线MP4 SHA完全相同、初始噪声SHA相同、50次首latent pin一致，证明插入不改变未启用时结果。alpha0.75 的实际特征张量为 `[1,3072,7,40,22]`、平均增益1.0625、峰值约1.718。源/生成关节头部相对骨盆运动范围 0.636 / 0.125（基线）/ 0.136（增强）；对齐首帧后的头轨迹MAE 0.340→0.303，但仍离源动作很远，人物局部锐度下降、抽帧可见模糊/幽影，不能称质量通过。弱强度0.25正在原生测试。
- 若推理增益不能兼顾动作和画质，训练版应在**现有DD**冻结Wan骨干、LoRA与原 `camera_encoder`，加一个零初始化的轻量3D卷积残差适配器：输入目标视角 `[17关节热图,置信度,可见性]` 的 latent 时间块，输出与 `camera_encoder` 同形的残差特征；初始输出0保证原生基线。只在有真实目标视角视频的配对样本上训练。基础去噪损失保留DD原 `L_noise`；另在目标关节置信区域用可微 `grid_sample` 对预测清洁latent与真实目标latent加权 `L_joint`，并对沿目标关节轨迹的相邻帧latent特征变化加 `L_motion`，对低置信/遮挡关节权重置0；另用背景区域保持损失和首帧锚点约束防止运动提升变成背景融化。不能把ONNX DWPose当可微损失直接反传。
- `configs/dataset/recam.yaml` 所指DD训练数据 `../data/v20-pack/metadata_train.csv` 在本地当前路径未找到，现有15个失败/成功案例也缺真实新视角目标视频。因此训练版目前只是明确的接入/损失/数据条件设计；不能宣称已训练或已验证。不得用生成视频当自己的真值训练并报泛化改善。
- 关节增益剂量0/0.25/0.75的原生对照与全25帧DWPose评估已完成，见 `audit/case12_joint_gain_dose_review.json` 和 `audit/case12_joint_gain_dose_compare.jpg`。0.25头轨迹MAE仍0.340（基线0.340），几乎无动作改善；0.75 MAE降至0.303、f16误差0.595→0.517，但头相对骨盆运动幅度仅0.136（源0.636），抬手到脸动作未恢复，人物局部更模糊/幽影，人体区域Laplacian方差平均67.3→60.9。两档都不能报告为Stage3质量通过。alpha0 MP4与原生基线SHA完全一致，初始噪声SHA相同，保证此负/弱阳性结论不是跑错条件。
- 新增 `dd_pipeline.py` 把 plan schema校验、三候选原生执行/复用、严格native feedback与完整视频时间帧校验串为可运行入口。用现有case13 A/B/C真实产物做 `pipeline_case13_verified/pipeline_result.json` 全链审计，仍选A；输出明示仅三套VLM候选内最优、全局最优未证、人工质量仍待确认。第二视频B/C原生候选正在运行，待完整后使用同入口回评。
- 发现旧 `planner.py` 两个不同视频都给A/B/C完全相同的0.95/0.92/0.90分数，并都选A；旧提示词示例直接写 `selected_id:"A"`，存在位置与数值模板偏置，不能把其初选当独立最优证明。新增 `planner_v12.py`：先仅提出按时间排序的三套DD合法方案（不报分/不选赢家），再以两种打乱顺序 C-A-B、B-C-A 的真实高光及后续动作帧独立打分/选优，分歧时第三轮决胜；最终编译为同一个固定 `dd-bullet-plan/1.1` schema，仍需真实DD回评。case12 首个 v1.2 实验正在运行；旧VLM计划保留作对照。

## 2026-11-03 后续分项核验

- case15 全画幅 Stage3 不再只凭6帧抽样：官方DWPose对源/原生生成各81帧均检测到人体，平均有效身体关节16.81/16.80个；逐关节相对骨盆、肩髋长归一化后，核心鼻/肩/膝/踝动作相对首帧轨迹平均误差0.110身体单位，肘/腕0.221。每5帧源/生成对照共17帧，人物和动作次序保持，局部手臂姿态有偏差；接缝解码帧MAE1.23、PSNR44.70dB。原生15fps使81帧动作从源3.24秒延长至5.4秒，连续主观播放/独立人工验收仍待做；没有历史同条件Stage3直接对照。`audit/case15_full_trajectory_audit.json`。
- Planner v1.2 case12 首轮暴露图片序号歧义：输出12/17/19实际指第12/17/19张展示图，而不是源视频帧12/17/19。原输出保留在`planner_v12_case12_30b/coarse.json`，校验拒绝后增加显式映射与`coarse_interpretation.json`审计；三项映射为源帧342/485/543，继续运行，不把序号当源帧。之后的图片标签改用真实帧号。
- Planner v1.2 case13 首轮同样输出展示图片序号11/15/19；理由文本明确写明对应实际源帧622/848/1075。原始`planner_v12_case13_30b/coarse.json`保留，经显式序号映射在独立`planner_v12_case13_30b_resume/`继续。
- case12 原 VLM 候选 B/C 源动作全帧DWPose已提取：B 33/33帧有人体、平均13.73/17有效关节（倒身/头发遮挡使部分点缺失）；C 17/17帧有人体但平均仅8.41/17、最低2，首帧下肢和上身严重出界，且复杂倒身造成姿态模型明显不稳。C 的源/生成关节数值不能作为可信动作保真量化，应以实际出片、可见性和人工审核为主；overlay在`pose_trial/case12_c_source_all/overlay.jpg`。
- case12 B/C 原生Stage3预处理已完成，几何网格有效画面平均覆盖分别23.33%/60.55%，低于20%画面的帧占比分别15.15%/0%。这是生成前风险特征，不代表出片质量；尤其C源动作严重裁切/倒身，case13 C已证明覆盖高仍可白屏。生成推理尚在进行。
- case12 原VLM三候选A/B/C全部完成真实Stage2→Stage3原生DD执行，`dd_feedback.collect`逐一验证plan匹配、源/生成帧数、原生hash、实际Stage2末帧锚点、目标相机衔接及Stage3固定、源镜头位姿帧数，均`integrity=passed`。人工对照：A人体/房间稳但俯身抬手动作被压平；B转45°后首帧人物贴左边，f16融化、f24/32人物与房间大幅涂抹发白；C源高光本已严重裁切，生成不白屏但近乎定格且手臂动作不跟。三者质量均未通过；`audit/case12_abc_native_visual_review.json`及B/C contact sheet。
- case12 B源33/33帧DWPose检测到人，生成仅20/33；平均可信身体关节13.73→6.64，支持严重生成崩坏。C源/生成均17/17检出，但可信关节8.41→5.88且源倒身严重/边界裁切，不能把C的数值轨迹误差当可靠保真评分。完整VLM DD反馈仍在运行。
- Planner v1.2 两例粗选均把图片序号错当真实帧号，严格验证后映射修复；case12三轮提案均给f539配45或57帧动作，超出560帧源片；`planner_v12_case12_30b_resume/proposal_0..2.json`逐轮留存并被校验拒绝。新增`--resume-proposal`和按每个候选明确合法长度的局部VLM纠错运行中，不对非法输出静默裁剪。
- case12 A/B/C真实DD结果的本地30B回评完成，`planner_case12_dd_feedback_abc/result.json`选A，B/C Stage3人体、场景、动作均给低分，人工抽帧同意A是三项中最好。**但模型给A动作保真5/5，和官方DWPose源0.636、生成0.125的头相对骨盆运动范围及人工可见俯身未跟明显矛盾**；所以此VLM反馈只能做排序参考，不能作为动作质量验收。`dd_pipeline.py`复用三套已验证原生结果/反馈完成整条链复核，`pipeline_case12_verified/pipeline_result.json`选A，但`human_quality_acceptance=pending`，三项原生画质都未通过。
- Planner v1.2 case13经源帧序号修正、非法末尾长度重提及双高分辨率裁切核验后，输出合法`dd-bullet-plan/1.1`：A f622左30°/25帧，B f848右45°/33帧，C f1075左20°/17帧。两个打乱顺序的投票分别选B和A，加权均分B4.5>A4.3>C2.3，第三轮只看A/B高光图选B；`planner_v12_case13_30b_resume/`留完整原文/顺序/计划。此为源图 provisional，不是原生最优；模型反复说“背景镜头移动很小”无Pi3X依据。原VLM优胜f397并未进v1.2三候选，说明20帧粗选存在漏候选风险。新A/B/C已启动完整原生DD以验证。
- v1.2 case13三套新候选的真实Pi3X源相机位姿已提取：A(f622/25f)端点旋转8.35°、位移/首帧中位深度8.25%；B(f848/33f)10.35°、5.30%；C(f1075/17f)3.77°、6.06%。VLM提案/投票中“背景镜头移动很小”未经测量且与此风险不完全一致；这些数值仍非质量阈值。`audit/camera_motion_v12_case13_*.json`。
- case12局部纠错进一步证明30B在自由生成时仍会复制非法45帧，甚至在明确给出17/21两个合法选项后，用60或320 token预算输出冗长算术推理而不是固定JSON；原文分别保留`planner_v12_case12_repair/`与`planner_v12_case12_duration_choice_v2/`。后者旧进程因确定性的格式缺陷主动停止。已改为令牌前缀树约束解码：VLM只能在完整合法`{"choice":"option_N"}`响应中选择，选项映射回动作帧数，不用代码替模型静默裁切。新运行`planner_v12_case12_constrained/`待验证。
- 反馈模型动作失察复核：`dd_feedback.py`新增每候选Stage3源/生成f0对照，提示明确要求比较f0到后续头/躯干/手脚位移，独立新回评`planner_case12_dd_feedback_motionaware/result.json`仍将A动作保真给5/5且声称所有姿态匹配；与DWPose实测头相对骨盆幅度0.636→0.125及人工俯身未跟矛盾。模型仍选A相对正确，但其动作细节分不能用作验收或“最优”证明，需结构化姿态指标和人工连续审核。
- case12 VLM片尾动作窗口受限解码验证成功：在f539只允许17/21帧，模型通过token trie约束生成严格`{"choice":"option_2"}`，映射为21帧，保留原自选1.8秒；完整`proposal_duration_repaired.json`的A/B/C分别为f338/29帧、f481/37帧、f539/21帧，均不越界。其后裁切核验和双顺序投票仍在运行。
- v1.2 case13三套真实 Stage2 已原生出片，抽查终点 A/B 人物基本完整且背景有视差，C高光源帧及终点近景裁切脸/下身；`audit/v12_case13_stage2_abc.jpg`。Stage3预处理的DD网格平均有效覆盖A30.5%、B34.4%、C20.6%，低于20%覆盖的帧占比A/B0%、C58.8%。这些只是几何代理，Stage3仍在原生扩散中，不据此宣称画质。
- v1.2 case12虽经固定JSON合法化并两票一致选A(f338)，三候选f338/f481/f539双轮可见性核验均为非全身；人工同帧对比`audit/case12_old_new_source_highlights.jpg`确认旧f171主体全身可见，新A顶部越界、新B人物大半贴边、新C近墙大幅裁切。故v1.2源片粗选明显漏了更可用的早期帧，不能设为默认，更不能用其投票分证明“最优”。新增`--balanced-coarse`要求前中后三段各一候选，保持模型决定实际帧/角度/时长，两个源片独立实验`planner_balanced_case12/13`运行中。
- v1.2 case13新A/B/C三套Stage2→Stage3原生完整出片且`dd_feedback.collect`全部完整性通过（plan字段、官方原生hash、帧数、实际Stage2末帧锚定、相机终点固定）。实际Stage3：A人物存在却几乎停在初始扶头姿势，源片持续弯身放下手臂；B踢腿动作大致跟上，33/33帧生成人体被DWPose检测，后半建筑/屋顶明显变形且末帧有轻微人物涂抹；C源近景裁切，f8人物碎裂、f12/16消失，仅11/17帧生成检出人体。`audit/v12_case13_abc_native_visual_review.json`及三张对照图；三者整体质量均未验收通过。
- 官方DWPose在新A源/生成25/25均检出、核心/肢体2D相对运动误差0.377/1.181身体单位；新B源/生成33/33均检出、0.160/0.687；新C源17/17、生成11/17，源本已裁切导致关节不可靠。旧v1.1 A f397的同法抽帧实际动作较好，`audit/old_v11_case13_a_stage3.jpg`；但不同源动作和目标视角会改变2D关节误差，不能跨候选仅按该数值直接排序。
- v1.2 case13三套新计划的真实DD反馈完成，`planner_v12_case13_dd_feedback_abc/result.json`及`pipeline_v12_case13_verified/pipeline_result.json`均在三项内选B，人工多时刻也认为B相对A冻结/C消失最好。回评把A的主要问题说成“严重融化”而漏掉更关键的动作冻结，又把B后段明显建筑结构变化说成“轻微模糊”；因此相对排序尚可，绝对画质评分不可信，B仍不能报质量通过。旧v1.1 f397 A不在新三候选内，且人工同法抽帧有较稳动作/背景，故新版未证明比旧版更好或全局最优。
- 分段case12粗选确实找回早期f171，另两帧f285/f457；但一次性16类受限边界解码错误把可见全身f171判为上下左右四边都裁切（两轮相同），与源图`audit/case12_old_new_source_highlights.jpg`直接矛盾，故该轮`planner_balanced_case12_constrained/visibility.json`废弃并主动终止投票。受限解码只证明格式正确，不证明视觉语义正确。已改逐候选单图双轮短JSON与严格类型/边界一致性校验，独立续跑`planner_balanced_case12_single_image_visibility/`。
- 分段case13三候选真实Pi3X源镜头风险：A f169/25f旋转34.12°、位移/深度10.17%；B f565/37f9.63°、6.07%；C f961/49f60.82°、29.22%。VLM理由仍称A“镜头移动很小”，明显不成立。B是分段候选中源镜头风险较低者，但角度45°/37帧的真实质量仍待原生Stage3。`audit/camera_motion_balanced_case13_*.json`。
- 当前正在跑：分段case13 A/B/C完整原生DD分别`vlm_dd_balanced_case13_a/b/c`（GPU0/3/6，原生Stage2进行中），分段case12逐图可见性修复`planner_balanced_case12_single_image_visibility/`（GPU1/2）。两者均已有活进程，不可因观察超时重复启动。下一步先查completion/plan，再做Stage3抽帧、`dd_feedback.collect`、姿态量化与VLM回评。
- 分段case12逐图可见性两轮对f171矛盾：第一轮判全身，第二轮妄称头/双手/脚同时越过上/左右边界；完整高分辨率源帧`audit/case12_f171_fullres.jpg`直接证伪第二轮。相同官方DWPose在相应f171动作首帧检测框约x151..543/y340..1138（704×1280），鼻与双踝可信，均远离边界；f285检测框y10..1261贴上下边，f457鼻/双踝缺失且x14贴左边。VLM边界判断不能盲信双轮AND，源码改为两轮分歧记`uncertain`而不硬判裁切（影响后续新运行，当前活进程仍旧逻辑）。
- 分段case13三套原生Stage2已完成；`audit/balanced_case13_stage2_abc.jpg`显示A人物在运镜终点完全消失（源f169高裁切/25帧源镜头34.12°），B人物完整且有视差，C始终近景裁切。Stage3原生mask平均有效覆盖A7.3%、B13.9%、C44.2%，低于20%覆盖的帧占比80%/100%/14.3%；这仍仅为风险代理，Stage3生成尚在推理。源DWPose平均可信身体关节A8.0/17、B13.49/17、C10.59/17。
- 分段case12的f171可见性双VLM轮次互相矛盾（全身 vs 头/手/脚多边裁切）；官方DWPose原视频f171检测框远离边界且关键身体点存在，而f285/f457有边界/姿态风险。新`pose_trial/visibility_advisory.py`保守输出`no_border_crop_evidence`/`edge_or_pose_risk`/`uncertain`，`planner_v12.py`遇到VLM与姿态证据冲突时改为不确定，避免把错误边界标签当事实。复用既有VLM图像原文的`planner_balanced_case12_pose_checked/`正在重新投票。
- 分段case13三套原生Stage3已出片（最终封装/strict collect待全部完成）：A Stage2终点人物已消失，Stage3全抽样无人物且背景涂抹；B Stage2终点人物完整，但Stage3人物逐渐滑出下边，f18/27人物/建筑溶解，f36近白屏；C f12人物变淡，f24+几乎只剩灰白模糊背景。三套实质画质失败，见`audit/balanced_case13_stage3_visual_review.json`及A/B/C接触图，不能因几何覆盖C44%宣称可用。
- 分段case12原生三套正在运行；Pi3X源相机运动A f171/33f旋转2.04°、位移/深度1.61%，B f285/41f6.57°、5.78%，C f457/57f29.86°、49.86%。C源相机风险显著。`audit/camera_motion_balanced_case12_*.json`。当前源图双票冲突旧决胜仅看一张图而选C，已修代码：均分差≥0.25直接用双票加权最高，否则第三轮看完整动作快照而非单帧；修复仅影响新运行，旧输出保留作失败证据。
- 分段 case12 的旧运行 `planner_balanced_case12_pose_checked/` 在两张打乱顺序选票分歧时用了旧的仅看高光帧决胜，给 C；但两轮完整动作快照加权均分 B4.1、C3.7、A3.4。源码已改为均分优势≥0.25时按均分选，不再丢弃完整动作证据。依据**已记录原始选票**、不新增 VLM 调用的确定性重编译 `planner_balanced_case12_selection_replay/` 给 B，schema/源帧动态校验通过；这仍是源画面 provisional 选择，B 的原生 DD 与人工画质验收未完成。旧计划原件保留，以免抹掉选错的证据。
- 分段case13 A/B/C的本地30B原生回评已完成，三项 `native_integrity.json` 均`passed`，反馈选B，但其B人体/场景/动作仅3/2/2、画质远未通过。反馈将C的f48仍描述为有可见人物，与官方DWPose生成仅11/49帧检出及人工f24+近空白背景相冲突；不可把模型回评分当画质真值。`planner_balanced_case13_dd_feedback/result.json`。
- 以**各自原始固定plan**对应真实原生DD输出，跨旧v1.1 A f397、v1.2 B f848、分段B f565进行了三方案回评；三份`collect`严格完整性均通过。30B选旧A，旧A与v1.2 B均给人体/场景/动作4/4/4，分段B给0/0/0；`case13_cross_plan_finalists_v2/result.json`。人工多时刻认为旧A相对稳，v1.2 B动作有但后段建筑明显变形，分段B严重融化。这个锦标赛只比较已执行三方案，不证明全局最优，且模型把v1.2 B背景结构变形轻描淡写。
- `dd_pipeline.py`复用分段case13的三套已验原生结果和现成反馈，完成整链复核 `pipeline_balanced_case13_verified/pipeline_result.json`：selected B，三个候选平均有效网格7.3%/13.9%/44.2%，源相机旋转34.12°/9.63°/60.82°。pipeline明确`global_optimality_proven=false`、人工验收pending；结合独立画面审阅，此B只是三项失败样本中的相对优胜，不是可交付成片。跨计划对照已证旧A更好，故单轮VLM分段规划未改善结果。
- 新分段case12的三套Stage2原生视频已出：`audit/balanced_case12_stage2_abc.jpg` 抽取f0/40/80，A人物大体完整且室内有视差；B高举手首帧顶部裁切、f40/80身体与手臂贴右/上边；C高光源与Stage2均大幅裁切。Stage3预处理DD动态网格有效覆盖A36.25%、B17.57%（95.1%帧低于20%）、C34.05%（5.3%帧低于20%）；`audit/geometry_balanced_case12_{a,b,c}.json`。B是当前VLM加权选择，却在真实运镜终点和目标几何上已有明显风险；生成尚在跑，预处理统计不能代替画质结论。
- 分段case12 A/B/C全部完成原生Stage2→Stage3，`audit/balanced_case12_abc_native_integrity.json` 三项严格`collect`均passed（各自方案/源窗/原生hash、Stage2实际末帧锚定、相机、帧数）。人工抽样 `audit/balanced_case12_stage3_visual_review.json`：A 33帧人物/房间保持且有可见动作，仍有局部模糊/边缘压力，需全段量化与连续验收；B 41帧有部分动作但中后段身体出右框、头/身体变形；C 57帧f14后渐融，f28+只余褐色模糊，均未通过。两失败不是运行错误。
- 官方DWPose对新case12 A源/生成各33/33帧检测到人体，平均可信身体关节16.58/16.18；相对骨盆、肩髋尺度归一后核心鼻/肩/膝/踝动作轨迹MAE 0.172身体单位，肘腕0.313，鼻相对位移范围源[0.200,0.779]、生成[0.283,0.942]。对比旧右30°/25帧A，同指标核心0.223、肢体0.134且头位移显著压平。新A核心运动更接近，但肢体误差更高、动作长度和新视角均不同，不能单数值声称整体优于旧A；`audit/balanced_case12_a_motion_compare.json`、`audit/case12_old_a_motion_compare_same_metric.json`。B/C的生成人体逐帧检测和VLM回评仍在跑。
- 除轨迹输入外，启动case13旧A的**单变量运镜角度对照**：同f396、右转、4秒、25动作帧、官方DD参数，仅右30°→15°；独立合法plan `nontrajectory/case13_old_a_angle15_plan.json`，GPU5真实Stage2→Stage3在跑。此为人工消融，不是模型自动选出的15°；目标检验降低角度能否减轻裁切或动作/背景异常。
- case12新A/B/C完整回评`planner_balanced_case12_dd_feedback/result.json`选A（A人体/场景/动作4/5/4，B2/4/1，C1/3/0）；三项严格native校验passed。A每隔2帧的17张原生连续抽帧`audit/balanced_case12_a_stage3_2frame_grid.jpg`显示人物与房间基本稳定、头身/腿持续变化，但脚边缘压力和手臂动作偏差仍在；B生成人体41/41帧被DWPose检出但平均可信关节仅6.83/17，C检出45/57、可信关节仅1.44/17，检测器可误认融化轮廓。`audit/balanced_case12_stage3_visual_review.json`。模型相对排序与人工一致，但A的动作4/5仍不能替代量化/连续验收。
- `dd_pipeline.py`复用新case12三套真实native结果及回评完成 `pipeline_balanced_case12_verified/pipeline_result.json`，最终选A、原计划源画面选择B被DD结果反馈推翻；输出只称三候选内优胜，`global_optimality_proven=false`、人工验收pending。已启动旧右30°/25帧A与新左35°/33帧A的跨计划原生回评，避免只在新三候选内比较。
- 旧case12 A与新A的跨计划回评 `case12_cross_plan_a_finalists/result.json` **不可信**：同一个新A在三候选回评得人体/场景/动作4/5/4，跨计划却得2/3/1并声称“f0躯干严重融化、钢琴被桌子替代、动作冻结”；17张每隔2帧原生截图明确显示新A f0人体完整、随帧俯身/步态变化，新侧视角露出的桌子不等于场景替换。它又给旧A动作5/5，和旧源/生成头相对骨盆幅度0.636→0.125相冲突。模型跨计划选旧A不能作为最优证明；旧A整体更锐利/完整，新A核心动作更丰富但手臂偏差/边缘压力更大，尚需更可靠的主观连续验收与多目标权衡。
- 可复用的原生Stage2终点构图预警 `stage2_endpoint_screen.py` 已实现并在case12新A/B/C实际81帧Stage2末帧测试：A官方DWPose 17/17、`no_border_crop_evidence`，B14/17且人物框触及右/上/下边、`edge_or_pose_risk`，C5/17触及左/上边、`edge_or_pose_risk`；每项保存实际Stage2视频SHA与结构化诊断 `audit/balanced_case12_{a,b,c}_stage2_screen/screen.json`。此为早期预警，不自动验收/拒绝，不证明Stage3画质；本例与后续B/C失败吻合。
- case13旧A右15°单变量对照的真实Stage2抽帧 `audit/case13_old_a_angle15_vs30_stage2.jpg`，15°终点人物/腿更居中、背景运镜幅度更小。目标DD动态网格覆盖同25帧从30°的26.09%（28%帧低于20%）提高到15°的55.26%（无帧低于20%），`audit/geometry_case13_old_a_angle15.json` 与 `audit/geometry_case13_old_a_angle30_recomputed.json`；这仅是生成前几何条件，Stage3真实画质仍在推理，不能预先报改善。
- case13旧A 15°运行在中途被本轮外部中断：进程不存在，Stage3 record仍标`running`且原生输出视频不存在，不能解释为DD失败。已把未完成的Stage3目录完整归档为`nontrajectory/case13_old_a_angle15_native/stage3_interrupted_attempt_20261103/`，写`stage3_restart_audit.json`，沿用已完成且不变的Stage2锚点重启独立Stage3推理/封装。后续只对重启的完整产物做原生一致性与画质结论。
- 可复用原生Stage2终点screen另在case13旧30°、15°、v1.2 B、分段B上实际运行：旧30°与15°均17/17关节、bbox底边余量仅0.4%（都是边缘风险）；v1.2 B 17/17、底边4.7%属`uncertain`；分段B13/17、底边0.4%属边缘风险。`audit/case13_{old30,angle15,v12b,balancedb}_stage2_screen/screen.json`。15°几何网格覆盖改善并未消除人物脚贴底边的构图风险，不能推断最终出片一定改善。
- 轨迹主线已同步启动第二种**DD内原生**推理实验 `pose_trial/temporal_pose_residual_case12.py`，针对旧case12 A动作被压平的同一25帧窗口。仍使用官方DWPose从实际DD目标视角`dw_warp`提取17关节，置信门控、仅补最长3帧的有界缺口、映射到7个Wan latent时间块；插在现有`WanVideoUnit_WarpControl.process`的`camera_encoder`输出后。不同于上一轮静态空间增益，本轮只在关节热区增强相对第0时间块的**时变控制特征残差** `C'_t=C_t+beta*H_t*(C_t-C_0)`，第0时间块保持，输入视频/权重/相机/首图不改。beta0与0.5分别GPU0/1原生运行，先要求beta0与旧baseline视频SHA相同，再比较源/生成全帧姿态误差与人物/场景清晰度。此为无需训练的轨迹条件消融，不假称学习版损失已验证。
- 为防本地30B跨候选回评把动作冻结/融化看错，`dd_feedback.judge`新增可选的**独立官方DWPose量化证据**输入（保持旧调用兼容），`compare_plan_finalists.py --motion-evidence label,compare_tracks.json`严格检查标签与实际动作帧数，逐候选提供源/生成检出率、有效关节、核心/肢体运动误差与鼻相对骨盆运动范围。已在case12旧A vs 新A跨计划回评启动新一轮有证据对照；要看模型是否改变自相矛盾的旧A动作5/5和新A“冻结”结论。2D跨视角数值只是诊断，不是3D保真或最终画质真值。
- case13旧A右15°单变量对照重启后完整出片、`dd_feedback.collect`严格完整性`passed`，见`audit/case13_old_a_angle15_native_integrity.json`。相同25动作帧与官方DD，网格平均覆盖26.09%→55.26%，但原生Stage3七时刻对照`audit/case13_old_a_angle15_vs30_stage3.jpg`显示15°后段左侧白蓝漂浮伪影、末帧手臂方向不如30°贴源。官方DWPose两版生成均25/25有人，15°/30°核心关节2D运动误差0.823/0.666、肢体0.907/0.530身体单位（跨新视角指标只是代理，不能单凭它排序）。结论：更小角度明显改善几何覆盖，却**未证明成片质量改善**；`audit/case13_old_a_angle15_vs30_review.json`。
- 第二轮语义轨迹原生消融 `pose_trial/temporal_pose_residual_case12.py` 已出片：beta0视频SHA与原生case12旧A**完全相同**，beta0.5与beta0的五路DD输入SHA一致、初始噪声SHA一致、官方权重未动、50次首latent pin一致。beta0.5改变输出且官方DWPose生成25/25检出，核心/肢体2D运动误差基线0.223/0.134→0.220/0.120，鼻运动误差0.469→0.466、竖直幅度0.128→0.134（源0.829）；改善极小，源手到脸动作仍未复现，不能称Stage3质量通过。`audit/case12_temporal_pose_residual_review.json`与三行对照图。
- 旧case12 A vs 新A的跨计划回评在加入官方DWPose原生量化证据后仍选旧A，但旧A动作评分只从5降4，仍称0.13/0.83的头竖直幅度“仅略微降低”；又把新A不同视角出现的边桌说成“钢琴被替换”，并指称f0严重融化，与原生逐帧图不符。`case12_cross_plan_a_motion_evidence/result.json`。额外数值提示未修复VLM的跨视角视觉判断错误，这一回评器不能独立承担最终“最佳方案”验收。
- 全链入口 `dd_pipeline.py` 现支持显式 `--planner-implementation v12 --balanced-coarse`，可从第三方源视频自动走新版VLM粗选、合法计划、三候选原生DD与回评；旧v11默认保持兼容，`--plan`仍能审核既有计划。参数校验、py_compile和复用case12三套原生结果的router smoke均通过（输出A、`planner_implementation=supplied_plan`）。新版规划器质量仍是实验态，不能因入口可运行就宣称会自动选全局最佳。
- 已在第三个真实舞蹈源 `inputs/formal15/new_dance/woman-dancing-7570258.mp4` 独立启动v1.2分段VLM Planner泛化实验，计划输出 `planner_v12_balanced_case15/`。待原始提案/可见性/投票合法再考虑真实DD；当前不能报任何case15新计划结果。
- 为把语义轨迹法扩到更难的case13 v1.2 B，先对DD实际目标视角`dw_warp`全33帧运行官方DWPose前置可观测性检查：仅15/33帧检出人体，平均可信关节2.88/17，`pose_trial/v12_case13_b_target_warp_all/tracks.json`。因此该片段无法提供可靠17关节条件；已拒绝直接做“关节增强有效/无效”的推理对照，避免把缺失轨迹误写成模型负结论。下一步选择关节可观测但动作偏差仍明显的新case12 A作为第二真实片段；其DD目标视角关节提取正在进行。
- 对12条已完整原生执行的case12/case13方案逐一运行`stage2_endpoint_screen.py`，每项记录实际Stage2视频SHA和官方DWPose端点关节；汇总`audit/stage2_screen_matrix/summary.json`：8条`edge_or_pose_risk`、3条`uncertain`、1条`no_border_crop_evidence`。它能提前提示多条裁切/人物消失风险；但case12旧A端点无裁切仍有动作压平，case13 v1.2 A/B端点不明显裁切仍分别动作冻结/背景扭曲。样本是诊断挑选且多数失败，没有已校准的准确率或通用画质阈值，不能把Stage2 screen作为单一接收门。
- 轨迹残差法在第二真实片段case12新A（f171左35°/33帧）已完成beta0/0.5原生DD对照：beta0 MP4 SHA与该候选原生Stage3完全一致，两组五路输入SHA/初始噪声SHA相同、官方权重不变、50次首latent pin；实际控制张量`[1,3072,9,40,22]`、关节热区约17.3%。官方DWPose两组生成均33/33有人，核心动作2D误差0.1723→0.1609、肘腕0.3127→0.2303、鼻0.1896→0.1670身体单位；人物框Laplacian方差均值41.55→43.04，改动主要在人物区域（RGB MAE人内2.87、人外0.76）。`audit/balanced_case12_a_temporal_residual_review.json`及源/基线/增强三行图。此窗口有可重复的局部动作指标改善且抽帧结构保持，**但仍有脚边缘压力和精确手势偏差，未通过整体画质验收**；不能把一例正向指标泛化。旧case12 A同方法改善极小，说明效果依片段而变，后续需更多可靠关节片段、连续观看和独立人工评价。
- 第三视频case15的新v1.2分段VLM计划现已合法产出：A f101右30°/4秒、25动作帧/1秒，B f271左45°/6秒、33帧/1.32秒，C f577右20°/2.5秒、17帧/0.68秒；两轮打乱顺序投票均选A，均分4.775/3.05/3.5，三处高光DWPose均17/17且源人物未贴边。**但其A理由声称“两脚腾空、双臂上举的跳跃”与实际源f101（一脚着地、一腿抬起、双臂未上举）明显不符**，见`audit/case15_v12_balanced_source_abc.jpg`及源复核JSON。三候选已分别GPU0/1/2启动真实Stage2→Stage3 DD；未出片前不接受VLM的A最优判断。
- 对第二片段轨迹正向信号做了**共同有效帧重算**：源/基线/增强三者同一帧均有可信关节才计入；源第0帧左肘/左腕缺失，所以原“肘腕平均”实际只覆盖**右肘和右腕**，不能宣称两臂均改善。右肘/腕共同帧的平均误差0.3068→0.2303身体单位（右肘32/33帧，右腕33/33帧），核心7关节仍0.1723→0.1609；`audit/balanced_case12_a_temporal_residual_common_frame_compare.json`。此约束缩小了正向结论范围，但右臂改善在相同有效帧上依然成立。
- 第三视频case15新A/B/C真实DD已进入Stage2，三者Pi3X实际源动作窗口相机位姿已产出：A 25帧端点旋转0.263°、平移/首帧中位深度0.425%；B33帧1.761°、2.544%；C17帧1.201°、0.744%。`audit/camera_motion_v12_balanced_case15_{a,b,c}.json`。都远低于case13多组大幅跟拍的风险，是检验“低源相机运动但不同目标运镜方案”的较好对照；仍不能由低源相机运动推断生成画质必好。

- 第三视频case15新A/B/C三套原生DD Stage2→Stage3已全部完成，`dd_feedback.collect`哈希/帧数/Stage2端点锚定/固定机位均通过。源/原生各10个均匀时刻对照及全帧官方DWPose见`audit/v12_balanced_case15_stage3_holistic_review.json`与`audit/v12_balanced_case15_{a,b,c}_dense.jpg`。A 25/25人体检出、关节17/17，动作大体跟随但后期脸手发糊，纯色背景使30°运镜空间感弱；B 33/33检出但中后段人物发虚，源后半段挥臂未跟上，肢体2D动作误差0.619身体单位；C 17/17检出但人物逐渐淡入背景、手臂近冻结，肢体误差0.483。C核心2D误差仅0.088却明显画质失败，说明姿态代理不能代替整体视觉验收。可说A在本次已执行三套里相对更强，不能说用户标注了最佳帧或A是全局最优；VLM f101“两脚腾空”理由已证伪，角度仍须由VLM综合实际出片反馈选择。
- 第三视频case15原生回评已选A，但其A五项全5分并称“所有帧精确匹配原片”与密集画面复核的后期脸手模糊/局部姿态偏差不符，且回评只输入了少数抽帧。只采纳A在本次三候选内相对较强的排序，不采纳满分或全局最优宣称；`planner_v12_balanced_case15_dd_feedback/result.json`，人工整段抽样复核见`audit/v12_balanced_case15_stage3_holistic_review.json`。
- 第三视频case15整链`dd_pipeline.py`复用三套完整原生DD结果及回评成功，`pipeline_v12_balanced_case15_verified/pipeline_result.json`选A，结果明确`global_optimality_proven=false`、`human_quality_acceptance=pending`。本次排序仅限模型提出且已执行的三套，不能解释成用户标注最佳帧或绝对角度最优。
- 第三视频case15 C的轨迹主线已完成另一组官方DD原生对照：目标视角DW warp官方DWPose 17/17检出、平均15.35/17可信关节，但人物右侧有大块黑洞、右臂关节可能跨空洞误估。beta0重跑MP4 SHA与原生C完全相同；beta0.5与基线五路输入SHA相同、官方权重不变。beta0.5核心2D动作误差0.088→0.067，却使整体肢体误差0.483→0.609；右腕0.841→0.752、右肘0.450→0.467，且人物后期仍淡入背景、未复现源挥臂。`audit/v12_balanced_case15_c_semantic_residual_review.json`与三行对照图。说明case12新A可见右臂局部改善未跨片段稳定泛化，也不能凭关节可检测/核心代理好看就认为Stage3整体合格。
- 对第三视频case15回评加密到每候选5个均匀Stage3时刻并约束“只能评价所见帧”；640px版本因显存不足中止，降至448px后成功。复测仍选A并给其五项全5分、否认后期模糊；更严重的是对只有f0..f16的C引用不存在的“f24”，`planner_v12_balanced_case15_dd_feedback_dense448/result.json`。这说明单靠增帧提示未解决VLM的证据幻觉。当前自动选择仅作候选内排序，不能自动发质量通过/角度全局最优结论；应在回评器加入帧序号引用校验和独立画面证据门控。
- 为防自动回评的客观帧号幻觉进入选择，`dd_feedback.validate`现核对各候选回评所引用Stage3/action帧是否在本候选动作窗口内（Stage2 f80例外），`compare_plan_finalists.py`及`dd_pipeline.py`复用相同校验。旧case15回评通过，新加密回评被准确拒绝：C只有f0..f16却引用f24；原始有效回评仍可在全链入口复核通过，`pipeline_v12_balanced_case15_reference_guard_verified/`。这是证据有效性门，不代表模型评分本身正确；A满分仍需人工否定。
- 新增`person_contrast_proxy.py`按官方DWPose逐帧人体框计算灰度对比度、Laplacian边缘方差，并用原动作的首尾趋势归一化，保留视频SHA/逐帧值，不设置通用通过阈值。case15新A/B/C的末段相对源的对比度保留比约0.759/0.742/0.657，边缘方差趋势比约0.974/0.857/0.554；C褪色最明显，增强版C改善到0.742/0.676但仍未救回动作/视觉。历史case15全画幅相对较好输出为0.849/1.500。然而历史case13严重融化却为1.002/1.439，**明确假阴性**：背景纹理/伪边缘和检测框会污染代理。`audit/*_person_contrast.json`，只能做成片风险提示，不能作为Stage3整体画质硬门。
- 新的非轨迹单变量对照case15 C测试DD `camera_encoder`控制强度1.0与0.5，固定同一Stage2终点、五路输入SHA/官方权重/seed；gain1.0出片SHA与原生C完全相同。gain0.5从f4开始人物更快褪色，f12/f16几乎消失，源挥臂未恢复，人物框相对源的首尾灰度对比度趋势从原生0.657降到0.283、边缘趋势0.554降到0.225；整体效果明确倒退。`audit/v12_balanced_case15_c_control_gain_review.json`与三行对照。简单“减弱warp控制就能防褪色”被本例否定，不排除更有针对性的控制方法。
- `pose_trial/compare_tracks.py`修正了严重掉检时先前产生NaN的缺失髋肩/关节聚合逻辑：只算源/生成均有有限坐标的共同帧，缺少可比关节返回null并记录核心/肢体有效关节数，写JSON时禁止NaN。case15 C原生指标保持一致；gain0.5人体检出16/17、核心误差0.088→0.143，肢体聚合因匹配帧改变不可直接比较。
- 对第三视频VLM A的角度开展**同一高光帧f101、同一Stage1/25帧动作/4秒Stage2/1秒Stage3**单变量原生DD对照：VLM原案右30°，人工反事实右15°和45°。三套Stage1/action/anchor SHA完全一致，实际Stage2旋转分别15.000/30.000/45.000°，严格DD hash/端点/固定Stage3机位/25帧均通过。Stage3全图有效网格75.7%/57.4%/44.9%；官方DWPose源-生成2D肢体动作误差0.536/0.113/0.219身体单位。10时刻逐帧画面：15°后半段手臂掉下而源双手在头部，45°侧视更明显但人物朝向和手臂后段漂移，30°动作相对更贴源但脸手后段模糊。人工在这三个**已测试的右转角度**中倾向30°，不证明全角度/方向最优，也不代表用户标注。`audit/case15_f101_angle15_30_45_holistic_review.json`及三套密集接触图。
- 同帧三角度原生VLM回评的一个展示顺序选R15，称其f24手臂仅“略不精确”，与源双手在头、R15双臂下落的放大图冲突；另一个并发进程以SIGTERM结束无结果，不能当投票。`angle_review_case15_f101_order453015/result.json`和`audit/case15_f101_angle_upper_body_crops_preview.jpg`。已加可选官方DWPose人体上半身放大图和全帧动作证据到跨计划回评，并改为单进程重测；必须先核验其解释，不能因模型选择就写最终角度。
- 第三视频同帧角度复测完成：在官方DWPose全帧动作证据和源/成片f12/f24上半身放大图输入下，VLM以R15→R30→R45及反向R45→R30→R15两次独立顺序均选**R30**；输出通过帧号合法校验，`audit/case15_f101_angle_refinement_status.json`。这与人工密集对照的相对动作保真方向一致，说明此已测试三角度里原VLM角度30°有支持；但模型仍夸大15°/45°“严重融化”、低估30°后期脸手模糊，一份理由还把0.2185说得比0.5357更高，不能采纳其绝对画质分或宣称全局最佳/人眼验收通过。
- 2026-11-04续：case15 C同目标机位非轨迹对照新增相机编码器control gain=1.5，和0.5/1.0三组全部是原生DD输出；五个输入SHA相同、官方权重未改、gain1.0原样复现基线SHA。1.5的后段人物清晰度确有提升：DWPose框内灰度对比度末/初趋势相对源0.832（基线0.657、gain0.5为0.283），边缘趋势0.979（基线0.554、0.5为0.225）。但源右腕运动范围约[1.174,0.978]身长单位，1.5生成仅[0.135,0.052]，f8/12/16手臂几乎停住；整体Stage3仍失败。量化和视觉审查在`audit/v12_balanced_case15_c_control_gain_0p5_1p0_1p5_review.json`，对照图`audit/case15_c_control_gain_1p5_{early,late}.jpg`。结论限于该候选：均匀control gain可改变褪色程度，不能单独解决动作延续。
- case13高光召回：三份历史coarse响应的**真实源帧**并集为f169/396/565/622/848/961/1075；早期v12返回的11/15/19是“已展示图片序号”，已映射成622/848/1075。以源视频全片每5帧DWPose提出额外姿态/动作峰，初版含f175及f575→580等检测跳变或裁切；现加相邻两侧>=14可信关节、上边界未裁切门控后，20帧召回库见`planner_candidate_bank/case13_pose_augmented_bank.json`，仍保留旧f396。接触图`planner_candidate_bank/case13_bank_part{1,2,3,4}.jpg`。这只是候选召回，不能按姿态峰值断言高光最好；VLM正在看原片帧+后续动作做三级候选短名单，最终须用真实Stage3整体画面复核。
- 非轨迹第二种原生DD控制对照：case15 C按同一目标warp的官方DWPose人物框在camera_encoder latent时空网格作柔和空间分区，人物gain0.75、背景gain1.5，其余五输入SHA和基线完全相同，官方权重不改。17帧成片人物对比度/边缘末段相对源0.827/0.865（基线0.657/0.554），但核心姿态运动误差0.176（基线0.088），右腕生成运动范围仍仅[0.119,0.228]身长单位（源[1.174,0.978]），源动作跟不上。`audit/v12_balanced_case15_c_spatial_control_review.json`和`audit/case15_c_spatial_control_{early,late}.jpg`；画面更清晰不等于Stage3整体通过。
- case13源高光库扩展了独立构图安全峰f75/f600/f845；当前22帧库保留旧f396，`planner_candidate_bank/case13_pose_augmented_bank.json`。初轮VLM把明显仅上身近景f1075错误描述成“全身完整”，见`audit/case13_source_bank_vlm_visibility_error.json`。加DWPose原片可见性预筛后，VLM分组选f280/f430/f622/f848，最终仍取f280/f430/f622；但原片800px复核f280站立脚越过下边、f430双脚越过下边，VLM却错误声称f430头发/右指越过上/右边界。f622本人完整，但Planner把双手靠近头部的姿态说成“弯腰伸臂”。这些错误在`audit/case13_bank_planner_source_visual_audit.json`，说明源帧VLM文字和候选内票选未达到自动“最佳帧”标准。此版Planner的右转55°/61帧f622 C仅是待验真实DD提案；正在原生Stage2→3执行。另启严格全身可见的源短名单[75,600,605,622,845,848] VLM复测，仍不等于最终质量结论。
- case13严格全身可见候选VLM源片短名单已完成：只允许每5帧DWPose在上下边界保留余量且>=14可信关节的原帧，六个候选f75/600/605/622/845/848，模型选f75/600/845，`planner_candidate_bank/case13_vlm_shortlist_strict_fullbody/result.json`。这是源片proposal，不是最佳帧；原片查看f845是抬膝而非模型描述的“高踢”，f75手臂靠近脸而非双臂高举，模型仍会夸张语义。源帧图`audit/case13_fullbody_source_f{75,600,845}.jpg`。新名单正在独立生成角度/方向/时长计划。
- case13前一名单f622计划的61帧真实Pi3X源相机轨迹已出：窗口末端相对起点旋转40.29°、平移/初始深度33.12%，`audit/case13_bank_c_source_camera_motion.json`。这对固定Stage3机位是高风险信号，但不据此提前判画质；其原生DD Stage2→Stage3仍在运行。
- 为防Planner从单帧臆断“源镜头静止”，新增`planner_candidate_bank/source_background_motion_proxy.py`，用源片五时刻ORB/RANSAC背景特征位移和端点匹配数量作**建议性**镜头风险证据，并接入后续`planner_v12.py`双顺序候选投票（当前已在运行的进程仍用加载时的旧版代码）。实测旧case13 f397/25帧端点559内点、位移画面对角线7.56%；新f622/61帧端点仅8内点，配合真实Pi3X源相机40.29°高风险；case15纯色背景端点匹配不足，不能硬判镜头运动。`audit/case13_old_f397_25_orb_pan_proxy.json`、`audit/case13_bank_f622_61_orb_pan_proxy.json`、`audit/case15_a_action_25_orb_pan_proxy.json`。此量不评价生成质量，不设通用阈值。
- 严格全身名单Planner第一轮投票仍把f845“头顶裁切”当作降分证据，然而f845原图头部上方有明显空白，官方DWPose框顶部余量35.6%；`visibility.json`已把两轮矛盾标为uncertain，模型却引用其中错误半句。`planner_v12.py`后续版本已将**互相矛盾的裁切原话从投票输入中剔除**，只给“证据冲突，请看像素”，同时保留完整原始审计；再配源背景位移代理。当前在跑的进程仍是加载时旧逻辑，其票选不能当最佳证明。
- 严格全身名单的完整VLM计划`planner_v12_case13_fullbody_shortlist/plan.json`：A f75左30°/33帧，B f600右45°/45帧，C f845左20°/21帧；模型暂选A但双票不一致，且把f75跑步步态、f845抬膝描述成“高踢”，对C又引用已标uncertain的虚假“头顶裁切”。`selection_status.json`明确provisional。三个VLM指定参数的原生DD各在GPU0/1/6运行，另f622右55°/61帧在GPU3运行。`audit/evaluate_case13_native_runs.py`已备好，待完成后严格检查native hash/帧数/Stage2端点/固定机位，自动导出Stage3十时刻源片与成片对照及DWPose/对比度诊断；视觉验收仍需人工逐帧。四个运行目录分别`vlm_dd_case13_fullbody_{a,b,c}`与`vlm_dd_case13_bank_c`，勿在进程运行时重复启动。
- **旧方案帧号更正**：历史v1.1 case13 A的权威`vlm_dd_case13_v11/planner_plan.json`是暂停源帧**f396**，动作窗f396..f420，非部分早期笔记口头写的f397。已按真实窗重算ORB端点背景位移7.591%（`audit/case13_old_f396_25_orb_pan_proxy.json`），十时刻图重标为`audit/case13_old_f396_stage3_dense.jpg`。之前f397/25的代理是相邻窗口，不能作为该原生输出的精确输入证据。
- case13严格全身候选A/B/C和另一f622方案**四条原生DD Stage2→Stage3均完成**。`audit/case13_fullbody_abc_native_review/summary.json`、`audit/case13_bank_c_native_review/summary.json`中的`dd_feedback.collect`对原生SHA/帧数/实际Stage2端点锚定/固定目标相机全部passed。密集十时刻图逐一复核：A f75左30°/33帧人体在，但中段类似拱身姿态停留、后段大楼变成半透明重叠块；B f600右45°/45帧f10起人体急剧融化，后段只剩苍白漂浮物，背景消失；C f845左20°/21帧人物、屋顶保持，抬膝逐步前伸，但手臂/手部与原片仍有偏差、脸手偏软；额外f622右55°/61帧生成动作和源片背离、后段人体蜷成空中团且屋顶消失。完整视觉结论`audit/case13_new_four_native_holistic_review.json`，不以DWPose/对比度代理代替。
- case13新全身三套原生回评`planner_v12_case13_fullbody_dd_feedback/result.json`选**C**，与人工密集图的候选内相对强弱一致；但模型给C五项全5分并称“动作精确匹配”、给A场景5分且漏掉后段建筑重影，故绝对评分/质量通过断言不可信，见`audit/case13_fullbody_vlm_feedback_audit.json`。整链`dd_pipeline.py`复用三套已验原生DD与合法回评，`pipeline_v12_case13_fullbody_verified/pipeline_result.json`选C，明确`global_optimality_proven=false`、`human_quality_acceptance=pending`。这不是用户标注“最佳帧”，只是模型提案加真实成片反馈后的三项内选择。
- 正与旧v1.1 A **f396**右30°/25帧真实成片做跨计划VLM回评，输入两项原生DD、官方DWPose动作证据和上半身裁图，目录`case13_old396_vs_new845_native_vlm_review/`。人工密集图初见：旧f396手臂动作更有变化但上方背景出现黑色浮块、底部构图裁脚；新f845屋顶/人更完整稳定但20°运镜幅度小且手臂姿态不完全吻合。尚不能宣称新C跨历史全局最好。

- case13旧f396右30°/25帧与新f845左20°/21帧的**跨计划原生DD回评**完成，模型选新f845，但给新方案五项全5分、声称手臂腿脸精确匹配。放大源/生成f10/f20明确见新方案手臂在头旁而源片已前伸，f20源腿向图右水平伸、生成腿在图左边缘；旧方案的手臂动作较接近源片，后段大楼却生黑色浮块。视觉证据`audit/case13_fullbody_c_person_crops.jpg`、`audit/case13_old_f396_stage3_dense.jpg`；审计`audit/case13_old396_vs_new845_vlm_feedback_audit.json`。可称新f845在这两套的人物/屋顶连续性更好，不可称满分、整体验收或全局最佳。
- case13新C的Stage3姿态改进并行探索：已用**DD实际Pi3X相对源相机、目标W2C与逐帧深度**把源DWPose关节投到目标机位。与目标warp中可识别的249个同关节，中位重投影差9.12px；f7/9/10等局部误差较大，故不能认定所有关节准确。脚本`audit/project_case13_source_pose_to_target.py`、结果`audit/case13_source_pose_reprojection_validation.json`。正在做原生DD时间残差beta0复现检查，才会决定是否继续beta0.5实验；这只是条件构造验证，尚无生成质量结论。
- **跨视角动作判断修正**：新case13 f845左20°成片中腿出现在画面左侧、原片同刻腿在右侧，这不能直接判动作错；根据DD实际深度/源与目标机位将源DWPose关节重投影后，f20腿关节对生成平均约10.6px，手臂关节约56.9px，且手臂后段偏差持续增大。`audit/case13_fullbody_c_target_view_pose_error.json`。前述跨计划回评审计已改为此更准确结论；VLM“全部精确匹配/全5分”依然站不住，但新C腿部动作确有较好的几何支持。
- case13 f845新轨迹原生实验完成：源DWPose关节按DD实际Pi3X深度与目标机位重投影，仅在关节热区增强`camera_encoder`时变残差。beta0视频SHA与原生基线**完全相同**，beta0.5五路输入SHA、官方权重相同而原生输出不同。全21帧官方DWPose加密集图/人物放大图：后7帧手臂目标机位误差41.62→37.99px，f20上抬手更前伸；屋顶/人物持续存在，但脸手仍软、对比度趋势1.044→0.962，Stage3整体不能称通过。beta0.5的官方DWPose在f10/f20把双膝误配到站立腿，故腿部/总姿态误差数字不可信，不拿来宣称提升或恶化。`audit/case13_projected_pose_residual_review/summary.json`、`stage3_dense.jpg`、`person_crops_small.jpg`。
- 四例真实Stage2→Stage3**首帧视觉接缝**检查：case12新A、case13旧A/新C、case15 A 的Stage2末帧PNG与原生解码逐像素一致；Stage3 f0平均RGB像素差1.20–1.91/255、PSNR38.7–44.9dB，双帧图无明显跳变。`audit/stage2_stage3_seam_audit/summary.json`、`seams_contact_small.jpg`。此只核首帧，不替代后段画质。
- 为检验VLM跨视角误读，给本地30B同时看case13 f845的**同目标机位**warp、原生baseline与源关节残差beta0.5的f0/10/15/20及f15/20人物放大图；模型仍给beta0.5人体/场景/动作/细节/构图全5分并宣布整体通过，还声称构图和屋顶变好。密集图只见抬手略向前，后段手臂投影误差仍约38px、脸手软、人物对比度趋势下降，且相机和Stage2末帧完全相同，不能采其满分或质量通过。`audit/case13_same_view_native_vlm_feedback_audit.json`。同视角参考不足以使该VLM绝对评分可靠；只把相对偏好作辅助，并保留人工/量化整体门槛。
- 同源同DD条件下继续源关节投影残差beta1.0原生对照，官方权重/五路输入SHA与beta0/0.5一致、输出SHA各不相同。后7帧抬手目标机位误差beta0/0.5/1.0分别41.62/37.99/39.18px，不随强度单调改善；人物灰度对比趋势相对源1.044/0.962/0.979。完整21帧图和f10/15/20放大图显示三者均保人和屋顶，beta1手略前移但脸手仍软，Stage3整体尚未通过；beta0.5的DWPose双膝误检不计入全身总分。`audit/case13_projected_pose_beta1_review/summary.json`。
- case13 f845保持同一21帧动作窗，向VLM展示真实Stage2/Stage3后请它提出两套新运镜，不人工定角度：D右35°/2.5秒，E左45°/3.5秒，模型事前暂偏D。两份参数通过DD schema/时序校验，原生DD分别GPU2/3已启动；尚无画质结论。模型理由中“E多0.4秒Stage2让Stage3覆盖更多帧”是错的，Stage3两案都固定21源帧，见`audit/case13_f845_camera_refinement_proposal_audit.json`。
- case15 C（手臂近冻结/人物褪色）新增**DD内容分支**原生对照：`WanVideoUnit_ConcatVideoEmbedder`的源动作VAE latent在DWPose源关节热区增强时间差，和相机控制分支前轮独立。实际metadata `input.mp4`与源action.mp4逐17帧同尺寸像素均差约2.10/255，热图坐标对应；beta0原生视频SHA与基线完全一致，beta0.1/0.5五路输入SHA相同、官方权重不变、视频各不同。官方DWPose右腕源运动范围[1.174,0.978]身体单位，基线[0.124,0.100]，beta0.1[0.234,0.250]、beta0.5[0.451,0.099]，有局部位移变化，但17帧图中源手臂横扫仍未再现、人物后段依然褪色；beta0.5肢体综合误差0.483→0.620且对比度趋势无改善。整体Stage3失败，不把关节数字小变化说成成功。`audit/case15_content_input_alignment.json`、`audit/case15_content_pose_residual_review_{0p1,0p5}/summary.json`。
- 轨迹+非轨迹**组合**原生DD对照：case15 C源内容latent残差beta0.1加相机控制gain1.5，与各自单项及基线五路输入SHA相同、官方权重不变；组合零强度beta0/gain1完全复现原生SHA。组合与单独gain1.5在完整17帧平均RGB差仅0.344/255、右腕运动范围分别[0.137,0.052]与[0.135,0.052]身体单位，而源为[1.174,0.978]。人物轮廓较基线清楚，但手臂仍近冻结，说明该简单叠加未把内容残差的局部运动增益保留下来。`audit/case15_content_camera_combo_review/summary.json`与`stage3_contact.jpg`。
- case13 f845**VLM运镜再规划闭环**：保留同一f845..865源动作SHA，VLM给D右35°/2.5秒、E左45°/3.5秒，与原C左20°/3.1秒三案都完成原生Stage2→3；三项native哈希/端点/帧数/相机严格passed。全21帧比较：原20°人物/屋顶最稳、腿末段更贴源，但运镜弱且手臂偏差；D右35°视差明显、后段手臂投影误差约41.6→27.1px，但腿f20误差约10.6→30.3px、身体中段前蜷、楼面有淡色重建块；E左45°有巨大右侧建筑重影且动作弱。D/E目标warp人体检测16/21与2/21，后者仅22个投影匹配关节，E几何动作数字不可靠。VLM事前和真实回评都选D，但回评虚称原20°“严重融化”、高估D腿部保真；因此D只是相对候选，不是整体质量/全局最优证明。`audit/case13_f845_camera_refinement_native_review/summary.json`、`audit/case13_f845_camera_refinement_vlm_feedback_audit.json`、`planner_v12_case13_f845_camera_refinement/refinement_result.json`。

- case13 f845同源同动作窗的角度回评增加**严格BASE/D两案正反顺序**：两次使用相同标注原生图、人物放大图、目标机位姿态数值和原生执行记录，仅交换候选呈现位置，均选BASE左20°（`audit/case13_f845_feedback_stability.json`）。因此三案回评选D与两案选BASE的差异不能归因于此次展示顺序；候选集合和额外目标机位证据同时变化，机制未分离。两案反馈又夸大D“严重融化”和BASE动作精确度，故只保留窄范围相对偏好。原生视频的整体审查仍显示BASE运镜弱/后段手臂不准，D运镜更明显但腿/楼面和人体姿态有问题，E建筑重影，三者都未验收通过。
- Stage3非轨迹**严格短时长对照**完成：case13 f622/右55°原61帧输入、条件、深度、遮罩四路预处理视频前21帧全部解码逐像素相同（其中条件/cond未被此DD配置实际消费），原Stage2终点/官方权重保持，短21帧原生DD输出通过50次首latent pin、最终锚定差0、21帧解码。六时刻源/长版前21/短版并排图`audit/case13_f622_exact21_review/stage3_comparable_grid.jpg`：短版人物/屋顶较长版清楚，长版在前段便发灰且人持续后仰；短版虽从后仰转为屈膝和两臂外伸，源片却持续向前俯身、手臂靠头并下探，动作仍明显错，边缘有暗条。DWPose检出三者各21/21；同角度对比的跨视角2D核心误差0.469→0.424、肢体误差0.269→1.404，仅作辅助。时长改变时间latent/noise维与注意力上下文，不能称完全同内部噪声；结论是缩短时窗缓解局部发灰，却没有修好整体Stage3。`audit/case13_f622_exact21_review/summary.json`。
- Stage3非轨迹CFG实测：第三舞蹈case15 C f577/右20°/17帧，复用原DD五路输入、prompt、Stage2真实末帧、官方权重与初始噪声，只将原生`cfg_scale`从1改为5（实际pipeline调用记录确认），首latent仍50步固定且最终差0。原生视频完整解码；九时刻图`audit/case15_c_cfg5_native_review/stage3_contact.jpg`及逐帧亮度`frame_luminance.json`显示f8–f12几乎全黑（灰度均值约5.2–5.8/255，>99.9%像素<32），其余帧亮度闪变，手臂仍近冻结。CFG1基线逐渐发白但无黑屏；CFG5不是修复。人物对比度趋势指标却从0.657升到1.316，证明该代理可被亮度闪变误导，整体视觉审查优先。较低CFG2原生剂量对照正在运行，不提前推断。`audit/case15_c_cfg5_native_review/summary.json`。
- 为避免把单轮VLM偏好误报为最终最佳，`dd_pipeline.py`保留候选视频可复核输出，同时新增`provisional_selected_id`、`final_selected_id:null`、`final_decision_ready:false`及明确的`decision_status`；`selection_scope`改为“已执行三候选的单轮VLM相对偏好”。使用第三舞蹈视频A/B/C三份完整原生DD和原有回评复用运行`pipeline_v12_case15_provisional_verified/pipeline_result.json`验证：A仍为候选内暂偏，最终方案字段为空、质量验收未由该管线证明。此仅修正状态契约，不改善模型画质或证明最优。
- 相同case15 C原生DD CFG2剂量对照完成：五路输入、prompt、Stage2末帧、初始噪声、官方权重与CFG1相同，实际运行调用为CFG2且50次首latent锚定；17帧原生视频完整解码。九时刻图`audit/case15_c_cfg2_native_review/stage3_contact.jpg`显示后段人物/背景较CFG1少发白、无CFG5黑屏；画面均值f0/f8/f16为167.2/170.5/175.0，CFG1为167.2/174.6/188.1。DWPose右腕水平位移范围0.124→0.271身体单位，但源为1.174；视觉上源右臂横扫而生成仍近初始姿态。CFG2仅局部改善外观，**整体Stage3动作和画质未通过**。`audit/case15_c_cfg2_native_review/summary.json`。
- f845角度裁决新增**候选集合受控复评**：BASE/D各自的标注图、人物放大图、DWPose/目标机位数值及原生DD记录，与已完成的BASE/D两案正序回评逐项完全一致；仅新增E及由两案变三案的比较提示。同BASE/D证据下，两案正反顺序均选BASE，三案却选D，`audit/case13_f845_candidate_set_effect.json`证实候选集合/比较情境影响赢家，不能归因为展示顺序，也不能孤立归因于“E画面”或提示长度中的某一项。三案新回评又虚称BASE“人体融化”、D“没有明显瑕疵”，与完整21帧真实画面不符；见`audit/case13_f845_threeway_same_evidence_vlm_audit.json`。因此当前VLM角度裁决未稳定，`refinement_result.json`最终角度保持null，三套Stage3整体均未通过。两次模型显存不足的运行分别归档，只有第三次五GPU完整JSON可用于此结论。
- 真正消费的`depth/dw_depth`无效区编码值填充原生对照完成：case15 C 17帧中保持mask、源内容、Stage2末帧、prompt、官方权重、初始噪声和有效深度像素逐帧完全一致，只把mask外的伪彩深度值改成同帧上方有效区域的中位编码值（不是RGB背景色）；原生MP4 SHA从`324c48…`变为`48f5ab…`，验证修改进入DD。九时刻完整对照`audit/case15_c_depth_background_fill_review/stage3_contact.jpg`中段人物略清楚、后段仍褪色，源片横扫身体的右臂未跟上；DWPose肢体误差0.483→0.547身体单位，不能称改善整体动作。对照此前只改未消费`cond/dw_warp`后MP4 SHA不变，进一步确认Stage3有效输入链。见`audit/case15_c_depth_background_fill_review/summary.json`。
- `dd_pipeline.py`最终方案契约再收紧：此前虽加`final_selected_id:null`，仍保留`selected_id:A`字段，容易被下游当作最终结果。现把`selected_id`和`selected_complete_video`均置null，候选内回评保留在`provisional_selected_id`/`provisional_complete_video`，`final_decision_ready:false`。使用第三舞蹈视频三套真实DD和回评复用跑通`pipeline_v12_case15_final_gate_verified/pipeline_result.json`，断言最终字段为空、暂选为A、成片质量由此管线不作验收。没有修改原生视频。
- 有效输入核验已进入`dd_feedback.collect`：新增`dd_effective_inputs.py`逐候选检查Stage3生成配置的`data_file_keys`、官方`extra_inputs`/训练模块映射、四路实际消费媒体路径及SHA、Stage2末帧SHA和prompt一致性，并把`cond_consumed:false`显式放入回评证据。case15 A/C与case13 f845 C三个原生候选均通过，使用新核验复用第三舞蹈完整A/B/C整链`pipeline_v12_case15_effective_inputs_verified/pipeline_result.json`通过；最终字段仍为空。正在对`WarpControl.process`做mask先于VAE编码的原架构内门控实验，先跑gate0精确基线，再跑gate1；它与前述直接改深度视频不同，保留原媒体文件不变。
- `WarpControl.process`遮罩前置实验的gate0已完成原生DD零强度对照：新运行`nontrajectory/case15_c_mask_depth_before_vae_gate0/`的Stage3视频SHA与case15 C原版完全相同`324c48f…`，实际wrapper调用1次、官方权重/媒体文件未改、首帧固定50步；证明接入点本身不引入差异。gate1现已启动：复用相同的真实DD深度/遮罩/源动作/Stage2端点，只在传给原始`WanVideoUnit_WarpControl.process`前把遮罩无效的深度编码像素置零，有效像素断言保持一致；待原生出片后再评价整段。
- `dd_visual_artifacts.py`除白屏外新增逐帧黑暗像素占比相对源片异常，`dd_feedback.collect`记录黑屏峰、`judge`在超出源片10个百分点时额外采样该帧（采样触发，不是质量通用阈值）。case15 C CFG5被抓到f7，生成灰度<32像素约99.989%、相对源片黑暗占比+95.7个百分点；CFG1/CFG2峰值均为负，不会增加无关帧。此修补避免VLM只看常规分位帧和白屏峰时漏掉中段黑屏；仍须看真实全片，不把检测器当质量通过判据。
- DD架构内`WarpControl.process`无效深度**VAE前遮罩门控**完整原生对照：gate0 MP4 SHA精确复现case15 C基线`324c48f…`；gate1复用同媒体文件SHA、初始噪声、官方权重与Stage2末帧，原处理前按已有mask将17帧约601万无效深度像素置零，有效像素逐帧不变，原生MP4 SHA变`13d40ae…`，50步首latent锚定通过。九时刻视觉图`audit/case15_c_mask_depth_before_vae_review/stage3_contact.jpg`显示后段人物比基线少发白；灰度均值f0/f8/f16基线167.2/174.6/188.1、门控167.2/172.5/181.0。生成右臂仍没跟上源横扫，肢体运动误差0.483→0.480仅近乎持平，整段质量**未通过**。说明无效深度编码进入VAE会影响外观，但遮罩前置单独不能恢复动作。`audit/case15_c_mask_depth_before_vae_review/summary.json`。
- 轨迹+非轨迹第二组组合：case15 C在DD原源内容分支关节时间残差beta0.1基础上，加`WarpControl`深度VAE前遮罩门控；与两单项和基线全部使用相同原媒体SHA、Stage2末帧、初始噪声、官方权重，两个hook均实际执行1次、17帧深度无效区被清零，原生出片SHA各不同。五行源/基线/内容/门控/组合时序图`audit/case15_content_mask_combo_review/stage3_contact.jpg`：组合后段较基线清楚，但右臂仍近冻结；组合右腕位移范围[0.163,0.062]身体单位、肢体误差0.482，与门控单项[0.164,0.071]/0.480近同，而内容单项原本[0.234,0.250]。组合与门控单项逐帧解码RGB平均差仅0.399/255；说明关节内容残差的局部增益未在组合中保留，整体Stage3仍失败。`audit/case15_content_mask_combo_review/summary.json`。
- 第三舞蹈视频的VLM**局部高光+运镜再规划**：给本地30B看f85/93/101/109/117与各自后续f+8/f+20源帧，以及原A f101右30°真实Stage2/3图，要求它自己在四个非101局部帧中选两帧和左右/角度/Stage2秒数/Stage3帧长。产出D=f109右45°/3.2秒/25帧，E=f117左30°/2.4秒/21帧，两份都通过DD源片动态校验，已启动完整原生DD运行。它又将f109单腿抬起误述为“双脚腾空”，并说f117后续动作“很小”，但源f137手臂已打开；事前低模糊/轨道与动作一致等均无实拍证据，见`audit/case15_f101_local_highlight_refinement_proposal_audit.json`和`audit/case15_f101_local_highlight_source_grid.jpg`。不能把D预选分0.75当最佳或成片通过，待真实Stage2/3全段复评。

- Stage3源动作内容分支原生因果对照已完成：case15 C把实际被`concat_video`消费的17帧源动作分别无损重封装动态原帧和首帧冻结，动态重封装逐帧解码像素与原输入完全一致，DD生成MP4 SHA也与历史基线完全一致`324c48…`。冻结组仅动作输入变，其他有效媒体SHA、prompt、Stage2末帧、官方权重、初始噪声SHA和50步首latent pin一致，原生MP4 SHA变`f166c6…`，与动态组17帧平均RGB差5.49/255、末帧12.80/255。这证明内容分支**实际影响**推理，不能说源动作没送进去；九时刻图显示两组都没跟上源右臂横扫且后段发白、脸手变糊，整段Stage3依然失败。`audit/case15_c_concat_video_motion_review/summary.json`、`stage3_contact.jpg`。

- 第三舞蹈VLM局部高光/运镜D/E已完成原生DD，分别与旧A对照。三者Stage2→3帧数、实际消费媒体SHA、真实Stage2端点、固定目标机位与原生输出哈希全部通过；D=f109右45°/25帧，E=f117左30°/21帧。逐帧人工检查：A普通背景保持但Stage3后段脸手变软、躯干动作有偏差；D Stage2角度有变化，但Stage3右腕水平运动范围源0.465、生成0.144身体单位，后段人物/背景持续变淡且脸手模糊；E Stage3手臂后期展开较明显，但源片末段的抬腿/身体移动未完整跟上，Stage2及Stage3右侧生成源画面不存在的高大帘状结构，场景一致性失败。E对比度趋势1.20或D姿态2D误差较小都不能代替整段判定；A/D/E目前全部未通过整体质量，亦不能认定任何一个是全局最佳。见`audit/case15_f101_local_refinements_native_review/summary.json`、`stage2_all.jpg`及各候选`stage3_all_frames.jpg`。真实出片VLM回评在运行，需逐条审它的视觉断言。

- A/D/E三套真实DD回评由本地30B选D（Stage3人体4/5、动作4/5），但逐句核查发现它说“E源片有帘幕”与原源片相反，E连续21帧全身可辨却被夸成“全程严重融化”；对D后段人物/背景发白及右腕水平运动压缩也未充分扣分。因此该轮VLM的`preferred_id:D`只保留为**未通过证据核查的候选内暂偏**，`refinement_result.json`保持`final_selected_id:null`、`final_decision_ready:false`。三套整体质量全不通过、全局最佳无证明。见`audit/case15_f101_local_refinements_vlm_feedback_audit.json`和`planner_v12_case15_f101_local_highlight_refinement/native_feedback_ade/result.json`。

- Stage3非轨迹**源内容latent全局时间残差增益**原生对照：case15 C保持真实Stage2末帧、源动作/深度/mask有效媒体SHA、官方权重与初始噪声相同，在`WanVideoUnit_ConcatVideoEmbedder.process`输出后只对第1–4时间块用`C_t←C_t+0.5(C_t−C_0)`。0强度hook MP4 SHA精确复现基线`324c48…`，0.5输出不同`d193ac…`，50步首latent固定且f0内容latent不变。全17帧源/基线/增益图`audit/case15_c_global_content_motion_gain_review/stage3_all_frames.jpg`：右腕水平运动范围从0.130增到0.519身体单位，但源为1.174，右臂横扫仍未完整复现；后段人脸/身体及背景更发白，人物灰度对比度趋势0.658→0.549。肢体2D误差0.484→0.462只能算局部数字改善，**整段画质失败**。`summary.json`。

## 2026-11-04 真实配对数据与DD内训练可行性

官方[MultiCamWarp](https://huggingface.co/datasets/Fred24/MultiCamWarp)实际仓库树已核：`video_data/<environment>/scene_.../<trajectory>/`下有同场景`input.mp4`、`target.mp4`、`dw_depth.mp4`、`dw_mask.mp4`和`camera_params.json`，不是只靠网页viewer的video/label两列判断。下载单个NightClub/MedievalGirl/orbit样本：四路均1280×720、30fps、81帧，相机参数85条；以DD官方`data_file_keys=input,video,depth,mask`生成单行训练CSV，并用原生`VideoDataset`成功读出四路各81帧、1280×704。`training_probe/multicamwarp_nightclub_orbit/data_feasibility.json`和`sample_contact.jpg`留证。这仅证明**数据接入可行**，尚未训练、不能推断真实舞蹈改善；相机参数85条与81帧视频需弄清对齐关系，训练当前不消费相机JSON。

下一步适配训练保持DD骨干：使用官方`configs/train/lora.yaml`相机编码器可训练+DiT LoRA的接口，按场景拆分训练/验证，选择人物动作明显和目标机位覆盖接近hardcase的样本，先做17/21帧对齐窗口与官方扩散损失基线，再比较仅相机编码器/LoRA以及带可靠目标视角人体轨迹约束的版本。验证必须回到真实case13/15的Stage3全帧动作、人物、场景和机位，不以合成训练loss或单帧代理宣称成功。需要先补数据对齐、分布与训练资源审计，不能把单个样本冒充可用训练集。

## 2026-11-04 DD内适配训练单步实证

用已核对的MultiCamWarp单行配对CSV、原生`training/train.py`与官方`step-10000.safetensors`初始化，配置17帧/704×1280/单GPU/1样本/1 epoch，仅做训练路径烟测。训练日志显示LoRA 600键与相机编码器8个张量加载；前向+反向+AdamW步骤成功，loss=0.039419，输出`step-1.safetensors`。相对官方权重，608/608个可训练张量发生非零改变，官方文件SHA保持不变；见`training_probe/multicamwarp_nightclub_orbit/one_step_train_audit.json`与`one_step_train.log`。这不证明质量改善，也不是可泛化训练集。
下一实验需按`scene_...`而非单视频划分训练/验证，覆盖多个人体动作与目标视角，严格构造同起点的17/21帧input/target/depth/mask窗口；先比较官方与适配模型在留出合成视角的整片结果，再在真实case13/15用相同候选/输入/seed原生DD推理，审动作、场景、清晰度和机位。若合成提升但真实失败，要记录域差，不以训练loss替代Stage3验收。

配对相机再核：同场景`center`与`orbit`两份camera_params.json各85姿态，f0位置/旋转完全相同，f20目标相对中心约36.9场景单位、pitch14.0°/yaw8.16°，f80约66.3单位、pitch11.38°/yaw25.60°；源/目标视频f0解码RGB平均差4.33/255，f20差48.61/255，符合相机起点相接、随后真实新视角的结构。视频仅81帧，额外4个姿态记录不得用于监督。见样本`data_feasibility.json`。

训练分布限制：已下载的MultiCamWarp `orbit`目标相机f0..80持续运动（如yaw由约-87.5°到-61.9°），而本工程Stage3目标机位固定。该样本支持DD原有相机视角训练链烟测，但不等于Stage3固定机位成对监督。后续需筛选/生成固定目标视角的同步视频，或设计移动机位训练到固定机位的受控迁移实验，并以真实hardcase整片检验。

单步checkpoint真实Stage3回环：在case15 C完全复用四路实际条件、Stage2末帧、prompt和初始噪声，仅切换官方权重到新adapter，原生17帧DD完成。视频SHA变化但全17帧平均RGB差0.379/255，人物仍褪色、右臂仍近冻结，肢体误差0.484→0.634，未修复。见`audit/case15_c_one_step_adapter_review/summary.json`。下一步需解决数据分布：MultiCamWarp已有轨迹中orbit/double/rand/pan/rotate/zoom均不是连续17帧严格固定新机位（center固定但不提供新视角），不能把现有单orbit样本当Stage3固定机位监督。

## 2026-11-04 固定目标机位配对样本反向可行性

下载并核对同一官方MultiCamWarp场景orbit源视角真实深度`depth.npz`：81帧连续时间、180×320×1、float16，等于720×1280视频空间各4倍下采样；`center/target.mp4`与已取得的`orbit/input.mp4`逐字节相同。可以构造一个**反向配对问题**：以orbit真实视频为输入、center真实视频为固定目标；两相机f0完全重合，之后orbit移动而center保持不动。这与Stage3“首帧锚定固定目标机位”形式更近，但相对机位轨迹、动作与真实舞蹈hardcase仍有域差。

用官方UE相机转换、orbit公开深度和320×180点投影做几何诊断（**不是**DD的`nvdiffrast`三角网格渲染，更不是可训练的`dw_depth/dw_mask`）：f0/8/16/32/80点覆盖1.000/0.821/0.682/0.485/0.412；覆盖区RGB对固定center真值的MAE为3.89/8.50/12.40/18.27/19.13（0–255），同像素未重投影源/目标MAE为3.89/37.44/43.31/54.31/51.91。前17帧可用覆盖下降但投影在已覆盖区显著对齐，支持反向配对**研究可行**；f32/80遮挡和空洞增长，长窗不可直接当高覆盖训练样本。具体代码、数据和五时刻图见`training_probe/multicamwarp_nightclub_orbit/reverse_fixed_view_proxy.py`、`reverse_fixed_view_proxy.json`、`reverse_fixed_view_proxy.jpg`。下一步须用DD等价网格渲染生成并视觉核对训练条件、按场景拆分多样本训练/留出验证，才可试固定视角微调。当前单步orbit适配在真实case15 C失败，不应称新方案已改善Stage3。

## 2026-11-04 成片证据门与最终方案状态

`dd_quality_gate.py`已作为保守后验筛查接入`dd_pipeline.py --quality-evidence`：已观测到整片失败、动作冻结或场景凭空新增时否决候选；证据缺失标`unverified`；门本身不自动宣告任何视频质量通过。第三舞蹈f117的E/F/G均被已观测失败否决；原A/B/C中B/C被整片失败否决，A因缺完整接受证据保持未验证。实际管线输出`pipeline_v12_case15_quality_evidence_status_verified/pipeline_result.json`中VLM候选内暂偏A，但`selected_id`、`final_selected_id`和`quality_screen_selected_id`均为null，`final_decision_ready:false`。这避免把A误写成用户标注最佳帧或最终最优，也明确当前VLM回评还不具备独立验收Stage3的能力。证据见`audit/case15_f117_quality_gate_result.json`及`audit/case15_abc_quality_evidence.json`。

## 2026-11-04 固定机位原生训练探针及真实回环

使用DD工程自带环境的`Dynamic3DMesh.render`/`nvdiffrast`（`open3d`在该调用路径仅有未用到的导入，因环境缺`libEGL.so.1`而以空模块绕过），将公开MultiCamWarp同场景orbit真实RGB+公开低分辨率metric深度重投影到固定center机位。720×1280原生网格f8/f16有效mask为83.09%/70.56%，覆盖区RGB对真实center目标MAE7.565/10.821（0–255），不重投影直接同像素MAE38.776/44.709。由此生成17帧四路`input/target/dw_depth/dw_mask` MP4并人工核对五时刻接触图；四路均17帧/30fps/1280×720，原生mask覆盖均值83.294%、末帧70.557%。源码、记录、视频在`training_probe/multicamwarp_nightclub_orbit/render_reverse_fixed_view_training.py`和`reverse_fixed_view_17_native/`。公开深度仅180×320，4倍双线性上采样会模糊边界；原生几何通过不等于高质量训练条件。

这份反向样本经DD原生`training/train.py`以官方step-10000初始化做17帧单样本一步训练，loss=0.149662，camera_encoder 8/8与DiT LoRA 600/600张量有非零更新，见`reverse_fixed_view_17_native/one_step_train_audit.json`。仅换此新权重、保持真实case15 C四路有效输入、Stage2实际端点、prompt、初始噪声一致的原生17帧推理完成；新旧片全帧RGB平均差0.412/255，肢体运动误差0.484→0.476身体单位，视觉上仍后段褪色、右臂横扫未复现，整体质量失败。见`audit/case15_c_fixed_view_one_step_adapter_review/summary.json`及密集三行对照。单步微小指标变化不支持质量提升。

关键分布差：反向orbit→center样本虽然目标机位固定，f0的源和目标相机完全相同；真实Stage3通常f0已是Stage2转到的新机位，相对源片有明显角度。前17帧orbit源相对center的yaw从0增长到5.38°、pitch到-12.05°，且样本仅一场景。因此下一训练数据需同步的**固定多机位新视角**真实目标视频以及可靠源深度。官方[AIST++下载说明](https://google.github.io/aistplusplus_dataset/download.html)与[AIST Dance DB结构](https://aistdancedb.ongaaccel.jp/database_structure/)显示九机位固定舞蹈及相机内外参，是动作与机位分布更匹配的候选；但[官方条款](https://aistdancedb.ongaaccel.jp/terms_of_use/)要求使用前提交申请表、限学术研究、禁止未经授权再分发，当前只调研，未下载或训练。其官方数据未提供原生逐帧深度，仍需可靠估计/重建并验证同步及warp条件。另一官方[GCD/Kubric-4D](https://gcd.cs.columbia.edu/)有16固定机位的RGB-D动态数据，但tiny子集48GB且非舞蹈主域，可作几何预训练或对照，不能替代舞蹈目标域。

## 2026-11-04 VLM整窗反馈与细节漏判实测

本地Qwen3-VL-30B分别按时间顺序看第三舞蹈原生A全25帧和C全17帧源/生成配对缩图（256px最大边）。它将严重动作错误、后段褪色的C判`whole_video_quality=fail`，但把A判`clear_details=pass`和`whole_video_quality=pass`，声称f13–24没有细节损失。用相同原生视频的DWPose鼻子/肩宽定位f0/12/20/24人脸，200px配对放大图中A生成f20/f24脸部明显糊化/重影，源片仍清晰；Laplacian方差辅助代理f0源/生成8.435/7.861，f24为15.639/5.378。针对这张图的独立VLM窄任务又正确判`late_face_detail_retained=no`并指出f20/f24模糊/重影。见`audit/whole_window_vlm_case15_ac/{A,C}.json`、`audit/case15_f101_local_refinements_native_review/A/A_face_detail_vlm_contradiction.json`及`audit/case15_a_face_crop_vlm_result.json`。该代理受视角/姿态/编码影响，视觉放大对照是主要证据；一轮窄任务成功不证明泛化。

`dd_quality_gate.py`现将独立确认的后段人脸细节损失作为否决项，且任意明确`observed_failure=true`或`quality_pass=false`都会否决，避免矛盾证据漏判。用case15三套已执行原生视频和VLM回评复用整链，`pipeline_v12_case15_all_observed_failures_verified/pipeline_result.json`中A仍只是`provisional_selected_id`，A/B/C均被真实失败证据否决，`quality_screen_selected_id`/`selected_id`/`final_selected_id`均null。下一步VLM角度反馈应采用全窗粗筛+针对人物脸手/背景/动作的独立局部审查，再提出新候选并原生重跑；仅喂更多缩图会继续漏掉细节。

可复用的 `dd_face_detail.py` 从源/生成 DWPose 轨迹按早中晚帧定位脸部，核验视频 SHA，输出配对裁图与 Laplacian 辅助值。`dd_face_detail_vlm.py` 用同一 30B 模型分别复查 case15 A 与 C，二者均给出 `late_face_detail_retained=no` 并指出各自后段模糊；记录在 `audit/face_detail_vlm_case15_AC/result.json`。这支持把局部细节复核单列为质量证据，但仍需跨视频验证，不能以此宣称 VLM 已能稳定验收整片。

## 2026-11-04 固定新视角舞蹈配对数据探针

官方 [SJTU Free Viewpoint RGB-D Video Dataset](https://medialab.sjtu.edu.cn/post/free-viewpoint-rgb-d-video-dataset/) 含同步固定多机位人体运动、标定 RGB 和深度；其 [官方仓库](https://github.com/sjtu-medialab/Free-Viewpoint-RGB-D-Video-Dataset) 说明深度编码与相机公式。已取得“A Dancer”机位 0/3 的完整 RGB 和官方标定参数，文件大小/SHA 见 `training_probe/sjtu_dancer/rgb_download_manifest.json`；机位 0 深度仅做稀疏本地探针，取文件索引和前 3 MiB，**不是完整下载或可用于后续时间段的文件**。三路前 17 帧均能解码。按标定 0→3 首帧即相差 13.57°，可补前一 MultiCamWarp 反向样本首帧无新视角的分布缺口。尚未把深度灰度按官方公式转换、完成原生 DD 网格投影或与机位 3 真值逐帧对齐，故目前只证明数据读取/相机布局可行，**不证明训练条件有效或 Stage3 改进**。官方数据页限定学术研究用途，研究探针不得当可再分发训练包。

## 2026-11-04 固定多机位数据的原生 DD 条件核验与局部反馈接入

SJTU “A Dancer”机位0/3两路RGB各1317帧、25fps、1920×1080且前17帧同步可解码；机位0深度前17帧可解码，按官方灰度公式映射为40–150标定单位。0→3相机首帧旋转13.57°，满足固定新视角形式，但前17帧舞者近静止，不能当复杂动作训练窗。全段RGB人体区域8帧间隔像素变化扫描的非重叠高运动窗包括f669–685与f779–795；人工五时刻对照确认f779–795有明显伸腿、上身变化和扇子挥动，两机位同步可见。扫描记录与图在`training_probe/sjtu_dancer/rgb_motion_scan.json`、`moving_windows_contact.jpg`；像素变化只是筛选建议，不是姿态真值。f779–795的深度稀疏取数仍在进行，未验证该窗逐帧深度或几何。

使用官方DD `Dynamic3DMesh.render`/nvdiffrast，以官方标定和深度公式重投影f0/8/16到机位3：320×180原始深度的有效mask约39.7%/39.1%/36.6%，已覆盖区RGB对目标真值MAE约11.8/11.8/11.8（0–255），同位置未投影源图约17.9/17.7/18.2，说明方向大致对但条件稀疏。源机位恒等重投影mask也仅约63.2%/62.1%/59.2%，表明原深度纹理和DD三角网格筛选自身会丢很多面。f8的320×180中值5滤波把恒等mask62.1%→83.6%、新机位mask39.1%→62.9%，但新机位已覆盖区MAE11.76→12.30；覆盖不是唯一目标。在真实DD接近尺寸1280×720，原始新机位mask28.2%，先降到320×180做中值5再升回来的mask67.1%、人物附近ROI约55.5%，已覆盖区MAE12.17→13.05，视觉上仍有明显人物背后空洞。结果及原生图见`training_probe/sjtu_dancer/native_fixed_view_320x180.json`、`identity_depth_diagnostic.json`、`depth_filter_ablation.json`、`highres_depth_probe_1280.json`/`.jpg`。尚未有动态窗17帧条件质量，不能训练或宣称改善Stage3。

VLM侧新增`dd_focused_face_review.py`并以`dd_pipeline.py --focused-face-review`接入原生候选结果；验证源动作/Stage3视频SHA、抽帧范围和可选裁图SHA，局部“细节损失”只作待独立确认告警，绝不当自动验收或全局最佳证明。复用case15原生A/B/C与既有回评的`pipeline_v12_case15_focused_face_verified/pipeline_result.json`中，A/C为`face_detail_loss_alert`，B缺局部证据；A与整片高人体分矛盾，最终选择仍null。旧一轮裁图结果未记录当时裁图SHA，故`contact_hash_verified:false`已显式标记；后续`dd_face_detail_vlm.py`会在模型查看前后核对并记录裁图SHA。篡改原生视频SHA、越界帧、裁图SHA的三项负测均拒绝。

SJTU动态舞蹈f779–795已用DD原生`Dynamic3DMesh.render`生成真正同步的17帧固定新视角配对四路`input/target/dw_depth/dw_mask`（另存`dw_warp`诊断）。机位0→3首帧即相差13.57°，25fps、1280×720；全部编码流17帧、逐帧可解码，源/目标抽样与官方原帧重编码差小于5/255。实际编码mask平均67.041%、最低65.859%；目标机位DWPose 17/17帧均17个关节检出，人物检测框mask均值67.021%、最低62.054%，最差帧仅11/17关节中心有warp支撑。已覆盖区warp对真目标平均MAE13.137，直接同像素源/目标18.578；背景及伸展肢体仍有明显黑洞。数据、逐帧统计和五时刻图见`training_probe/sjtu_dancer/motion_779_795_native/{manifest.json,encoded_training_validation.json,four_streams_contact.jpg}`。这是带真目标的可行性样本，不是可泛化训练集。

官方DD `training/train.py`以该样本17帧/704×1280、官方step-10000初始化完成单GPU一次前反向与AdamW，loss=0.486342，camera_encoder 8/8及DiT LoRA 600/600张量相对官方权重均非零变化，官方文件SHA不变；`motion_779_795_native/one_step_train_audit.json`。只切该checkpoint、真实case15 C四路有效输入/Stage2末帧/prompt/初始噪声均同基线的原生DD 17帧回测已完成：平均新旧RGB差0.397/255，肢体运动误差0.4841→0.4791身体单位、源右腕横向范围1.174而生成0.130→0.122，后段人物继续褪色、源右臂横扫仍未恢复。原生完整性通过、整段视觉验收失败；`audit/case15_c_sjtu_novel_view_one_step_adapter_review/{summary.json,stage3_dense.jpg}`。单步数字微变不支持Stage3改善。下一步转向按场景拆分多样本训练/留出验证，避免重复单样本一步即回测的低信息增益。

已取得SJTU“A Broadcast Gymnast”和“Skipping Rope”各机位0/3完整RGB，用同一25fps官方多机位采集环境构造潜在跨场景训练/留出对照。前者366帧，源RGB动作建议窗f299–315；后者423帧，跳绳建议窗f52–68。两组动作五时刻图与扫描见`training_probe/sjtu_multiscene/{a_broadcast_gymnast,skipping_rope}/`。深度稀疏窗口仍在按官方HEVC关键帧及HTTP Range核验取数，未出原生几何或训练结果；`Three Dancers` RGB源约55/75MB，暂未下载。官方数据页限学术研究、非商业，所有探针保持本地研究用途。

局部VLM第二轮把“头部转向/头发遮挡”写进提示后，在case15 A/C仍指出后期实际模糊；但case12新A的生成f24/f32脸被头发/姿态遮住，它仍把不可直接比较的五官称为“极度模糊”，说明仅两列紧脸图仍会误判跨视角可见性。结果见`audit/face_detail_vlm_visibility_v2_three_cases/result.json`。现增加源/生成各自较宽的头肩上下文图并做第三轮复测；该VLM结果始终只是告警，不自动否决或通过。现有`dd_pipeline.py --focused-face-review`对已观测视频做hash/帧号核验后可附加此告警，但最终方案保持null。

## 2026-11-04 迭代语义与 Stage3 训练首帧条件更正

当前`dd_pipeline.py`一次提交执行VLM提案A/B/C各自的Stage1→Stage2→Stage3，然后用真实原生视频做一轮相对回评；它没有在回评后自动生成下一批候选、重跑并以整片质量收敛的循环。过去f845改角、f117换角和f101局部高光D/E属于人工组织的多轮实验，不是证明系统会自动反复运行或找到最优。单个被采用的候选在生成时是依次跑完三阶段；只有换高光/运镜或失败需要重试时才生成新候选并运行相应链路。第三视频A只是候选编号，原A/B/C各有真实整片失败证据，最终方案仍为空。

复核官方`WanTrainingModule.prepare_data`发现：`extra_inputs`含`input_image`，若metadata不含已加载的`input_image`，则直接用`data['input'][0]`（源相机首帧）作锚点。此前MultiCamWarp、SJTU等单步微调样本的metadata仅有`input,video,depth,mask`，所以它们证明了优化器和checkpoint回测可以运行，但首帧条件**不严格匹配**本工程Stage3的“Stage2已转到目标机位的末帧”。此前所有单步真实case15 C回测也均未通过整体质量，不能称适配改善。现保留原metadata以复现实验，另生成`metadata_stage3_anchor.csv`及`input_image_target_first.png`，并明确训练时须设`data_file_keys: "input,video,depth,mask,input_image"`。对A Dancer f779–795、A Broadcast Gymnast f299–315、Skipping Rope f52–68三段，PNG从目标视频f0解码生成且像素逐点相同；源/目标f0平均RGB差分别17.75、19.34、16.96/255，确认旧回退锚点不是可忽略的同图差异。`training_probe/sjtu_multiscene/prepare_stage3_anchor_metadata.py`及各样本`stage3_anchor_audit.json`留证。后续多场景训练/留出验证必须使用新metadata；当前仍无质量改善结论。

新两场景17帧官方DD网格条件已生成并编码/同步验收：体操窗口f299–315全画面mask均值62.4%、目标人体框56.7%、目标关节中心最少10/17受warp覆盖；跳绳窗口f52–68分别66.7%、62.5%、最少8/17。覆盖区域RGB对真实机位3的平均MAE，体操warp13.82、源同坐标19.50，跳绳warp12.70、源同坐标17.89（0–255）。二者构成可研究的新角度配对，但空洞和人体覆盖不足明显，仍需整片留出验证；见各自`fixed_view_*_native/encoded_training_validation.json`。

## 2026-11-05 正确首帧多场景训练及成片回评限制

训练锚点修正后，以“A Dancer”f779–795和“A Broadcast Gymnast”f299–315各17帧为训练场景，严格按场景留出“Skipping Rope”f52–68。官方`training/train.py`使用`data_file_keys=input,video,depth,mask,input_image`、目标机位视频f0像素精确相同的`input_image_target_first.png`、官方step-10000初始化、camera_encoder+DiT LoRA、LR1e-5、3轮共6步。训练返回0并生成183MiB的`step-6.safetensors`，SHA `67d288d05736c780a80c5de73d6cbac7179464a086011fbd8bd3b06f179f4032`；与官方权重608/608键一致，608张量改变，其中相机8张量改变。配置和权重核验见`training_probe/sjtu_multiscene/anchor_multiscene_pilot/{train.yaml,train_metadata.csv,checkpoint_audit.json}`。仅证明受控优化完成，尚不能从训练loss或权重变化推断生成质量；留出跳绳的官方/适配整段DD推理已启动，待全帧比较。

局部VLM人脸反馈v3在case15 A/C仍报告后段模糊，且与本地全帧人工观察一致；但在case12侧身头发遮挡片段，即使同页增加源/生成头肩上下文，仍将f24/f32遮挡误说成“不是头发或转头造成的模糊”，甚至错误宣称按提示排除了紧脸图。`audit/face_detail_vlm_context_v3_three_cases/result.json`保留原回答；`audit/face_landmark_visibility_case12_A.json`显示其generated f24/f32脸部点位中位响应比源低约0.26/0.29，而case15 A同帧人脸仍可见、响应差较小。模型响应不是已标定的遮挡分类器，不设自动阈值。`dd_pipeline.py --focused-face-review`继续只给告警，不能把v3的“no”独立当整片失败或最终选角证据；必须看真实连续片段和对应较宽画面。多模态质量判断不精确的外部研究参照：[Q-Bench-Video官方](https://github.com/Q-Future/Q-Bench-Video)；运动评估需区分幅度、平滑、完整度：[VMBench官方](https://github.com/AMAP-ML/VMBench)。

人体轨迹训练的新参考：[MVTrack4Gen论文](https://arxiv.org/html/2606.26087)以跨视角点轨迹和注意力对应损失联合训练，但其[项目页](https://cvlab-kaist.github.io/MVTrack4Gen/)目前标代码未发布，且其reference/target作为并列token进入注意力；DD的源动作在`concat_video`内容分支、目标扩散与depth/mask控制分支不同，不能照搬该损失位置。若进入此方向，须先验证DD可提取的源/目标对应特征，再在多视角真值点轨迹上设计辅助头和遮挡权重，不能用当前只有17帧×2场景的样本下结论。非轨迹参考：[MV-Performer论文](https://arxiv.org/html/2510.07190v1)展示视角相关法线和深度修正对人体新视角的作用；其模型/权重不同于DD，优先用SJTU目标真值检验DD现有Pi3X深度、人体边界、投影空洞，再决定是否训练额外法线条件，而非直接替换主模型。

## 2026-11-05 留出跳绳整段同噪声复验与VLM可见性探针

两场景正确目标首帧锚点训练的step-6权重与官方step-10000做**按场景留出**对照：Skipping Rope f52–68的固定机位3真实目标视频17帧。首次官方`training/test.py`默认`seed=None`，两次进程无法证明同初始噪声，已保留为探索输出、不拿来做因果比较。随后`seeded_official_test.py`只将官方生成器的随机种子固定为20261105，两个独立进程打印的初始噪声SHA同为`60012b9d76186f71ee97d04301048634a7a7ab76783f0528f9040a1ac9e0d204`，其余仍走官方`training/test.py`。两版17帧1280×704原生输出均完整。相对真实目标，整片RGB MAE官方4.840→适配4.349/255，目标人体框12.642→12.437/255，但此数字被静态摄影棚背景和近似保持首帧动作误导。全17帧源/目标/官方/适配并排图显示目标持续甩绳、抬臂/放臂与跳跃，两生成版大部分时间保持张臂站姿，动作未跟上。官方DWPose在目标/官方/适配各17帧均检出17关节；肢体运动误差0.27923→0.27836身体单位近乎不变，右腕纵向运动幅度目标0.866、官方0.174、适配0.171；左腕0.699/0.104/0.108。按**整体动作+人物+场景+清晰度**判定两版均失败，不能将较小像素误差称Stage3改善。证据`training_probe/sjtu_multiscene/anchor_multiscene_pilot/{heldout_paired_review.json,heldout_paired_metrics.json,heldout_paired_all_frames_small.jpg,pose_target_vs_base.json,pose_target_vs_adapted.json}`。真实case15 C只换该step-6权重的同输入/噪声原生回测已启动，仍待结果；这次留出失败已表明当前两场景六步微调不是有效修复。

把局部细节VLM分拆成“先看脸是否真可比较”的v4探针后，同一Qwen3-VL-30B在case12侧身f12/f24/f32明确指出头发与转头遮住嘴，这与v3无根据断言“不是遮挡、是模糊”相冲突；但在case15 A f20/f24又同时输出“眼鼻嘴都可见”和“不可比较/头发遮挡”，产生内部矛盾；case15 C后段均判脸可见。所有原图、源/原生视频SHA核对，逐例结果与结构化自相矛盾审计见`audit/face_visibility_probe_v4/`和`audit/face_visibility_probe_v4_audit.json`。因此拆步骤在一个已知假阳例有帮助，但未通过跨例可靠性检验；不能让它自动豁免case15 A真实后段糊化或自动裁决最终角度。VLM选角继续以真实整段生成、独立证据核查和可弃权状态为准。

2026-11-05 真实case15 C六步多场景adapter回测也完成：复用官方原生17帧动作/depth/mask/Stage2末帧/prompt/初始噪声，只有checkpoint从官方step-10000换成正确目标首帧两场景训练step-6；50步首latent pin、最终差0，哈希/帧数核验通过。新旧逐17帧平均RGB差0.441/255；肢体运动误差0.484→0.466身体单位略降，但源右腕水平范围1.174，生成0.130→0.144，仍约八分之一；对比度趋势0.657→0.642稍差。源/官方/适配密集图和适配全17帧图均显示源右臂横扫没有再现，后段人和背景继续褪色。整体`quality_pass:false`，见`audit/case15_c_multiscene_anchor_six_step_adapter_review/{summary.json,stage3_dense.jpg,adapter_all_frames.jpg}`。结合留出跳绳失败，当前六步训练只改动权重与少量像素，没有提供Stage3动作保真改善。不能以两个训练场景推出“训练方向无效”；现阶段更有价值的是提升人体关键部位的warp有效覆盖、扩大多动作/多视角数据和训练时显式运动监督，再按留出整段验证。

2026-11-05 VLM人脸可比较性稳定性复测：在三例共12个相同原生视频帧、相同宽头肩裁图和相同Qwen3-VL-30B上，仅收紧提示词，v4→v5有5个yes/no判断翻转，包含case15 A后期f24从“不可比”改“可比”；详见`audit/face_visibility_v4_v5_stability.json`。v4自身在case15 A有“眼鼻嘴可见”却判“不可比”的内部矛盾。该局部VLM探针尚未校准，不能用自动可见性结论豁免真实后期模糊、否决候选或裁决最终角度；整段画面与独立证据优先。

2026-11-05 Stage3运动区域训练对照已启动：使用SJTU同步固定机位舞蹈与体操两段、正确目标首帧`input_image`锚点、官方step-10000初始化，随机种子20261105与同一训练CSV分别训练官方原损失和目标视角人体肢体区域加权损失。目标DWPose映到DD裁切后的5×44×80 latent网格，训练时把噪声预测残差的肢体区域MSE与相邻时间残差差分加入官方损失；不改DD推理骨干。热图与目标人体对齐图见`training_probe/sjtu_multiscene/anchor_multiscene_pilot/posemask_{dancer,gymnast}/overlay.jpg`，代码`pose_trial/{target_pose_latent_heatmap,train_motion_loss_pilot}.py`。这只是**针对目标人体时空区域强调误差的扩散损失**，不是直接预测关节坐标的轨迹损失；效果必须用留出跳绳及真实case15 C相同噪声的原生整段回测，当前未出结果。

2026-11-05 正确目标首帧锚点、同训练随机种子的六步**官方损失 vs 目标肢体区域/时间残差加权损失**留出测试完成。608/608权重张量在两版均相对官方更新、两版彼此608/608不同；训练第一步原损失0.0378522016 vs 新损失中的原损失重算0.0378522030，确认同初始化/数据/噪声，对照记录`training_probe/sjtu_multiscene/anchor_multiscene_pilot/pose_motion_training_audit.json`。留出Skipping Rope两版原生17帧同初始噪声SHA，目标/基线/加权损失逐17帧并排图显示：真实人物持续甩绳、抬放双臂和跳跃，两生成版都大多保持张臂站姿。加权损失版相对基线整帧目标MAE4.716→5.103/255、目标人体框12.518→12.621，肢体运动误差0.2859→0.2880身体单位，右腕纵向幅度0.176→0.174，真实目标0.864。整体质量仍失败，见`training_probe/sjtu_multiscene/anchor_multiscene_pilot/pose_motion_pilot_review/summary.json`及全17帧图。这个小数据六步设置没有支持显式运动区域损失可修复Stage3；真实case15 C同输入/同噪声两版回测已出片，待定量与全段最终记录。

2026-11-05 同一训练损失对照的真实case15 C回测也完成：官方、六步原损失、六步人体区域加权损失均原生17帧，Stage2末帧、四路有效输入、prompt、初始噪声和50次首latent锚定逐项核对，三片噪声SHA相同。全17帧源/三版并排图显示两训练版仍没复现源右臂横扫，后段人物/背景继续变淡，整体`quality_pass:false`。DWPose“全肢体平均误差”原损失0.661、加权0.463看似大降，**但两版进入平均的关节集合不同**（原损失只检出右肘/右腕，加权另纳入仅4帧的左肘），不能把两数直接比较。在相同有效帧上，右肘误差0.454→0.455、右腕0.868→0.841身体单位；右腕水平动作范围反而0.138→0.107，源1.174。故本轮未证明Stage3动作改善，局部数字变化不足抵消整片失败；见`audit/case15_c_pose_motion_pilot_review/{summary.json,common_joint_compare.json,stage3_dense.jpg}`和全17帧图。下一实验应先解决人体关键区域warp稀疏、多动作多视角训练数据及更直接的动作监督，而不是继续重复两样本六步的小剂量试验。

2026-11-05 按目标视角DWPose关节逐帧采样DD原生`dw_mask`，解释为何全画面mask均值不能代表动作监督质量：留出跳绳17帧全画面有效覆盖66.7%，但左肘/左腕关节中心仅2/17、3/17帧落在有效网格，左踝4/17；右肘/右腕为17/17。训练舞蹈左腕8/17、体操左腕6/17。17×17像素邻域平均有效率留出左腕19.3%、右腕99.6%。代码及逐关节表`training_probe/sjtu_multiscene/anchor_multiscene_pilot/{audit_joint_warp_coverage.py,joint_warp_coverage.json}`。这支持“稀疏人体关键部位warp是数据瓶颈之一”，但**不是完整解释**：留出右腕有满覆盖却仍近冻结（目标纵向幅度0.864、两训练版约0.176/0.174身体单位）。因此下一改进应同时处理人体局部几何/遮挡条件与模型对源动作的利用，不能只提高全画面覆盖率。
