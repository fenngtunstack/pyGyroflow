# PyGyroFlow 优化推进路线（P0–P4）

> 日期: 2026-08-14 | 前置: docs/07-optimization-report.md
> 状态标记: [ ] 未开始 / [~] 进行中 / [x] 已完成 / [-] 已验证（含测试）

## 0 结论先行

07 报告之后的核心判断：**数值内核已过硬（frame_transform、4/6 积分器、平滑、horizon、10 个畸变模型），真正的短板是"最后一公里"——数据在正确的算法之间没有接通，算出来的东西在渲染端被丢弃。**

本轮按以下五档推进，优先级依据是"单位工作量的正确性收益"：接线类改动（几十到几百行）优先于新算法、新覆盖。

| 档 | 主题 | 核心问题 |
|----|------|----------|
| P0 | 端到端正确性 | 同步偏移未作用于渲染；RS 矩阵被丢弃；畸变模型硬编码；音频丢失 |
| P1 | 同步子系统产品化 | 真实的 RollingShutterSync 是死代码，实际入口是降级的 1-D 模长互相关 |
| P2 | 渲染性能 | 每帧全量重算（dict 拷贝/mgrid/畸变）；逐行 Python 循环 |
| P3 | 数值与 GPU 遗留 | complementary 重写、VQF 航向对齐、GPU 全 0 三根因 |
| P4 | 覆盖面与产品化 | telemetry 仅 2 种格式；镜头库零数据；keyframes 不消费；GUI 无预览 |

---

## P0 — 端到端正确性（本轮主攻）

多为"已算出但没用上"的接线类缺陷，用父目录现成的 `GX010045.MP4` / DJI 视频可立即验证。

### P0-1 渲染时间戳接入真实 pts + sync offsets

- 现状: `manager.py` `stabilize_frame` 用 `frame_idx * 1000/fps` 均匀时间戳，忽略 `gyro_source` 的 sync offsets（`source.py:79-92` offsets/`offset_at_video_timestamp`），也丢弃 FfmpegProcessor 解出的真实 pts（`ffmpeg_processor.py` 算好了没传下去）。变帧率视频变换时刻错误；autosync 结果对渲染无效。
- 方案: 渲染循环把真实 pts（毫秒）传入 `stabilize_frame`；`get_frame_transform`/`get_quat_at_timestamp` 路径统一过 `offset_at_video_timestamp` 修正。
- [x] 状态（2026-08-14）: `ComputeParams.sync_offsets_adjusted` 新字段；frame_transform 的全部四元数查询（中心/平滑/逐行）统一走 `GyroSource.offset_at_video_timestamp` 修正（镜像上游 `quat_at_timestamp`）；渲染回调用真实 pts。测试: `TestSyncOffsets`（偏移 500ms 时 t=1000 的变换 ≡ 无偏移时 t=500 的变换）。

### P0-2 CPU 路径逐行应用卷帘快门矩阵

- 现状: `cpu_undistort.py:204-210` 在 `matrix_count > 1` 时只取 `matrices[matrix_count // 2]`（中心行），frame_transform 逐行算出的最多 1080 个行矩阵全部丢弃 → **CPU 输出实际无 RS 校正**，而逐行计算成本照付。
- 方案: 按行分桶——输出图像按行分组到对应行矩阵，组内向量化重映射；保持与逐行语义一致。
- [x] 状态（2026-08-14）: 重写为上游 `undistort_coord` 语义——中心矩阵试探 → 逐像素估算源行/列 → 逐像素 gather 行矩阵再映射（比"每输出行取均值矩阵"的近似更精确）。stmap 导出器同步复用。测试: `test_rolling_shutter_uses_per_row_matrices`（每行输出 ≡ 单独用该行矩阵的渲染结果）。

### P0-3 cpu_undistort 分派全部畸变模型

- 现状: `cpu_undistort.py:57-68` 硬编码 opencv_fisheye 的 Kannala-Brandt 公式（只用 k1[0..3]），`distortion_models/` 的 10 个逐字移植模型只服务于 GPU shader 和 stmap。非 fisheye 镜头的 CPU 去畸变是错的。
- 方案: 按 `kernel_params.distortion_model` 分派到 distortion_models 的模型实现（先向量化），fisheye 快路径保留。
- [x] 状态（2026-08-14）: `DistortionModelBase` 新增 `distort_points`/`undistort_points` 批量 API；10 个模型全部实现向量化 distort（与标量版**逐位一致**，diff=0）；fisheye 另有向量化 Newton undistort（diff<1.6e-10）。cpu_undistort 按 `FrameTransform.distortion_model_name` 分派；顺带接入 `r_limit`（manager 经 `radial_distortion_limit()` 计算并缓存）、`lens_correction_amount` 混合、`translation2d`（此前全部被忽略）。**顺带修复存量 bug**: 采样器 `p10`/`p01` 像素取反——`p10 = frame[y1c, x0c]` 配的权重却是 x 方向的 `w10 = fx*(1-fy)`，任何亚像素平移都采错邻域（重写前就存在，从未被测试捕获）；另修复 poly3 `undistort_point` k1=0 除零。测试: `TestDistortPointsParity`（10 模型）、`TestCpuUndistort`（identity/亚像素/RS/分派/r_limit/lca 混合）。

### P0-4 渲染输出接通音频

- 现状: `AudioResampler.copy_audio` 实现完整但全库零调用，CLI 输出无声。
- 方案: manager.render 完成视频轨后调用 copy_audio；提供 `--no-audio` 开关。
- [x] 状态（2026-08-14）: 音频两阶段 API（`prepare_audio_streams` 在首包前建流 + `mux_audio` 视频趟后拷包）——**必须**在容器头写入前建流，否则 PyAV 抛 `Cannot rebase to zero time`；适配 PyAV 17 的 `add_stream_from_template`；mux 前 seek 回文件头（视频 demux 已读 EOF，否则 0 包拷贝、轨道被静默丢弃——即"日志说 copied 实则无声"的假成功）。**顺带修复存量 bug**: `process_frames` 在 `thread_type="AUTO"` 多线程解码下从不 flush 解码器，438 帧 GoPro 片段丢 13 帧（435 可解码帧只输出 422）。真实视频验证: 视频 435/435 帧、音频 682/682 帧（AAC stream copy）。CLI 加 `--no-audio`。

### P0-5 GoPro GPMF 轴序/时间戳模型

- 现状（原）: 轴序硬编码 "ZXY"；时间戳假设"每 DEVC 块约 1 秒均匀分布"。
- 方案: 按上游推导轴序（ORIN+ORIO→MTRX→IMUO，机型特例，无标签→None）；时间轴保持 STMP+每秒一块模型（实测 STMP 间隔 ~1s 成立）。
- [x] 状态（2026-08-15 完结，经 ground truth 验证）: ① 轴序对齐上游；② **发现并移植 CORI×IORI 每帧姿态四元数流**（上游 Hero9+ 主稳定数据源，与 raw GYRO 路径并存，quaternions 非空时 gyro_source 自动走 integration_method=0 直用）；③ SROT 卷帘快门时间解析（26ms）+ 移除 manager 对 GoPro readout 的强制清零。GoPro 四元数与上游真值归一化后逐位一致。遗留：raw GYRO 路径的均匀时间戳模型仍未改（被 CORI 路径取代后优先级降低）；运行时 guess_imu_orientation 未实现。

### P0-6 端到端回归测试

- 现状: `tests/fixtures/` 为空目录，无任何真实/合成视频资产；CPU 渲染管线从未有自动化端到端验证（07 报告 §5.1）。
- 方案: 程序化合成 1–2 秒嵌 GoPro GPMF 的小 mp4（KLV 写入器从解析端逆向），断言 load_video → render 全链输出帧数/尺寸/与逐帧 stabilize_frame 一致。父目录真实视频做手动冒烟（不入库，>50MB）。
- [x] 状态（2026-08-14）: `tests/test_e2e.py` 新增 25 个测试——合成 GPMF/mp4 构造器（box 写入器 + GPMF KLV 编码，含 GoPro 真实的 `struct_size=1×repeat` 容器编码）、10 模型向量化一致性、cpu_undistort 语义六项、sync offsets、真实编解码渲染（帧数守恒/音频拷贝/--no-audio/稳定输出≠直通）、自动同步恢复已知延迟。

---

## P1 — 同步子系统产品化

### P1-1 接通 RollingShutterSync 真实偏移搜索

- 现状: `find_offset/rs_sync.py:124-382` 有完整的逐点四元数角度误差最小化实现（逐行 RS 时间戳、去畸变到单位球、粗到细 4 轮细化），**但是死代码**；实际被 `find_time_offset`/autosync 调用的 `find_offset_rs_sync()`（:415-513）只是 1-D 模长互相关（自身 docstring 承认降级）。`visual_features.py:80-88` 同样降级。
- 方案: `find_time_offset` 切换到 RollingShutterSync；保留降级版作 fallback（无镜头参数时）。
- [x] 状态（2026-08-14）: `AutosyncProcess.run(quaternions=...)` 在 offset_method=2 且有四元数时走真实 RS 搜索（从 PoseEstimator 保留的逐点匹配建 track），失败回退互相关。`_compute_cost` 向量化重写（批量 slerp/四元数乘/旋转，与标量版相对差 <4e-8，**~1000× 提速**——2 秒视频搜索从 240s → 4s）。

### P1-2 同步接入 manager/CLI

- 现状: `synchronization/` 整个子系统未被 manager.py/cli 引用；GUI Auto Sync 弹"未实现"。OptimSync（最佳同步区间选取）孤立。
- 方案: manager 加 `synchronize()` 步骤（AutosyncProcess + OptimSync 可选），结果写回 gyro_source offsets；CLI 加 `--autosync` 开关。与 P0-1 联动后同步结果才真正生效。
- [x] 状态（2026-08-14，OptimSync 区间选取仍未接）: `StabilizationManager.synchronize()`——抽取灰度帧（pts 时间戳、抽帧+降采样到 ≤480 宽）、陀螺角速度信号（raw_imu 或四元数差分兜底）、运行 AutosyncProcess、`gyro.set_offset()` 写回（经 P0-1 立即作用于渲染）。CLI `--autosync`。验证: 合成已知 200ms 延迟视频恢复误差 <25ms（`TestAutoSync`）；真实 GoPro 素材 28s 完成，找到 -254.7ms 偏移（量级与 IMU/视频时长差相符；绝对准确性待 P0-5 GPMF 时间轴修正后复验）。GUI 接线仍未做。

---

## P2 — 渲染性能（P0-6 回归测试兜底后进行）

- **P2-1 cv2.remap 采样 [x]（2026-08-15）**: `cpu_undistort.py:202` 每帧重建 np.mgrid 全坐标网格；畸变部分与帧无关，可预计算成 map；采样用 4 次全帧 fancy-index，opencv 已在依赖中却未用 cv2.remap。预计数倍。
- **P2-2 每帧 ComputeParams 全量重建**: `manager.py:536-537` 每帧 `dict(self.gyro.quaternions)` 全量拷贝（6000 项 × 900 帧 ≈ 540 万次条目拷贝）+ fovs/camera_matrix 每帧再拷。渲染开始前构建一次，循环内只更新逐帧量。
- **P2-3 frame_transform 逐行循环向量化**: 每行 slerp + 3×3 求逆，批量插值 + 批量求逆，预计 5-10×（07 报告只做到 1.75×）。
- **P2-4 fov_iterative 排序缓存**: `fov_iterative.py:133-134` 每帧两次全量 `sorted()` 且不传预排序 keys——frame_transform 修过的同款 bug 在此复发，一行修复。
- **P2-5 渲染流水线**: demux→decode→callback→encode 单线程；YUV→rgb24→YUV 双重色彩转换。生产者/消费者线程 + 保持 YUV 处理是后话。

## P3 — 数值与 GPU 遗留（穿插进行）

- **P3-1 VQF 初始航向对齐**: 结构完整（1072 行完整移植），仅初始航向差 ~24°（max_err 0.34，xfail 跟踪）。两个 xfail 中性价比最高。
- **P3-2 complementary 论文版重写**: Python 是 8 行简化版镜像 vs Rust 论文 V1/V2（600 行），需按 "Keeping a Good Attitude" 重写。
- **P3-3 GPU 全 0 三根因**: ① coeffs 插值表恒全 0（`backend.py:270-275`，shader 双线性权重全 0）② `pixel_value_limit` 从未设置（wgsl `min(sum, 0)` 恒黑）③ `output_stride` 默认 0（所有行覆写第 0 行）。另有 CPU fallback 函数签名错误（`backend.py:349-366`，一调用就 TypeError）。代码级可修，最终验证需真实 GPU（本机仅 llvmpipe）。
- **P3-4 fov_iterative 接入畸变模型**: `_undistort_points_simple` 无镜头畸变（docstring 自认），鱼眼镜头自适应 zoom 会留黑边/过度裁切。

## P4 — 覆盖面与产品化（按需排期）

- **P4-1 镜头库数据** [x]（2026-08-15 完成）: 官方 `profiles.cbor.gz`（9811 档案）入包 `pygyroflow/resources/camera_presets/`，database.py 加包内候选路径（加载 12409 含展开）；GoPro MINF 机型提取 + `_try_auto_load_lens_profile`（品牌守卫 + 宽高比优先）实测自动命中 HERO12 4k 8:7 Wide（帧级 1.11×→1.22×）；DJI 内嵌档案优先不受影响；229 测试过。
- **P4-2 telemetry 格式扩展**: 仅 GoPro GPMF + DJI（`parser.py:47-51` 只认 .mp4/.mov + gpmd/djmd）。上游 telemetry-parser 支持 20+ 种；Sony/Insta360/Betaflight/Runcam 全缺，外部日志文件无入口。
- **P4-3 keyframes 接入渲染**: `frame_transform.py:260-267` 把 video_rotation/fov_scale/zoom center 等在函数开头读成常量，逐帧动画全部失效；keyframes/manager.py 模块本身完整。
- **P4-4 horizon gravity 数据流**: `horizon.py:274` gravity 分支不可达（manager 不传 grav/use_grav）；`use_gravity_vectors` 无人消费。
- **P4-5 smoothing 缺 forward.rs**: 上游五种平滑算法（default/plain/fixed/forward/none），Python 缺 forward 前向预测。
- **P4-6 GUI 可用化**: VideoWidget.set_frame 就绪只差解码循环；导出在 UI 线程同步跑会冻结；Max zoom/Sync offset 等滑条未连信号；Auto Sync 空壳（与 P1-2 联动）。
- **P4-7 相机自动识别**: `_try_auto_load_lens_profile` 显式 no-op。
- **P4-8 GPMF 镜头信息提取**: FileMetadata.lens_profile 对 GoPro 恒为空。

---

## 推进顺序

1. **P0-1/2/3/4 + P0-6**（接线类，CLI 输出从"理论上通"变"可验证正确"，真实视频立即可验）
2. **P0-5**（GPMF 时间轴，为 P1 精度打底）
3. **P1-1 → P1-2**（Gyroflow 招牌功能，目前等于没有）
4. **P2**（有 P0-6 兜底后重构安全）
5. **P3-1/P3-3 穿插**，P4 按需

## 验证命令

```bash
python -m pytest tests/ -q          # 全量（基线: 201 passed, 5 skipped, 3 xfailed）
python -m pytest tests/test_e2e.py -v  # 本轮新增端到端
python -m pygyroflow ..\GX010045.MP4 -o out.mp4 --smoothness 0.5   # 真实视频冒烟
```

## 进度日志

| 日期 | 项 | 结果 |
|------|----|------|
| 2026-08-14 | 路线文档建立 | 本文档 |
| 2026-08-14 | P0-1~P0-4 + P0-6 | 全部完成；顺带修 3 个存量 bug（采样器 p10/p01 互换、解码器 flush 丢帧、poly3 除零）；测试 201 → 226 passed |
| 2026-08-14 | P1-1 + P1-2 | RS 真实搜索接通 + 向量化 1000×；`manager.synchronize()` + CLI `--autosync`；OptimSync 区间选取与 GUI 接线遗留 |
| 2026-08-14 | **真实视频效果验证（P0 管线）** | 发现稳定无效：GoPro/DJI 输出抖动均高于输入。经闭环合成测试 + 独立 Rust 参照（`msgyro-stabilization`，新 harness `ft_compare`）定位 **frame_transform 无 RS 分支移植 bug**，修复后与 Rust 参照一致至 1e-5；golden 重生成，226 测试全过 |
| 2026-08-14 | **效果验证第二轮（3 个新语义修复）** | 补修 smoothed 校正量转换缺失、fov_iterative 消费语义、（第一轮已修的）无 RS 公式后，渲染链经约定匹配闭环证真（roll 0.000px）；真实素材仍差→根因定位到 telemetry 数据层（P0-5 升级为 blocker），48 置换校准工具就绪 |
| 2026-08-15 | **P0-5 轴序修复 + DJI 连续性补齐** | ① GoPro 轴序：删除硬编码 "ZXY"，按上游 ORIN+ORIO→MTRX→IMUO 推导（移植 `orientations_to_matrix`/`mtrx_to_orientation` + HERO6/7Silver 机型特例），Hero8+ 无标签 → None（Gyroflow 靠运行时 guess 兜底）；② DJI 四元数双覆盖连续性翻转（`inv`，跳变>1.5 取逆，上游有 Python 无）补齐。227 测试全过 |
| 2026-08-15 | **真实素材 A/B 枚举（诚实边界）** | 分辨率一致的帧级实验（480 全链）+ 相位相关位移度量：GoPro/DJI 输出 \|disp\| 均为输入 ~2×。穷举方向翻转×时间轴缩放×S 共轭共 8 变体无一实现运动相消；校正与画面运动相关性 ~0.35。**根因未完全定位**——最强嫌疑在"绝对/相对姿态语义 + flip"与真实积分链的配合（合成闭环跳过了积分器入口）。已排除：渲染公式（合成闭环锁定）、轴序（已修）、分辨率/度量伪影、假 sync offset、adaptive zoom、积分器选择（3 种同差）、DJI 连续性（本素材无跳变）。**建议下一步：编译上游 gyroflow-cli 对同一素材出 ground truth 输出/四元数流，逐层 diff** |
| 2026-08-15 | **Ground-truth 对照修复（tp-dump 路线）** | 建立上游 telemetry-parser(2f4218b) 的独立 dump 工具（`D:/tmp/tp-dump`，本地盘编译），对两素材产出真值。**修复 3 个数据层根因**：① GoPro 缺失 CORI×IORI 每帧姿态四元数流（上游 Hero9+ 的主稳定数据源，438 个 vs 旧路径 2886 raw 样本自积分）——已移植（SCAL 32767、x 取反、帧率时间戳），与真值归一化后逐位一致；② DJI 四元数乘法顺序反（上游 `y180⊗(raw⊗m1)`，Python `(raw⊗m1)⊗y180` 致 y/z 符号反）——已修，真值 parity 5/5；③ GoPro SROT 卷帘快门时间（26ms）未解析且被 manager 强制清零——已修。顺带修 default_algo smoothness=0 除零。测试 229 passed（新增 parity fixture 回归） |
| 2026-08-15 | **全片渲染终验（修复后）** | GoPro 435/435 帧 + 音频 682、DJI 753/753 + 音频 1178 全部完整。量化: **GoPro** 帧级 \|disp\| 1.11× 改善（此前 0.61×），med-jitter 0.79×；**DJI** med-jitter **1.64×** 改善、med\|ω\| 1.39× 改善（此前全部 <1×）。两素材首次全部呈现真实稳定信号。剩余差距（诚实边界）: fov_iterative 开启时引入呼吸（0.81×，遗留项）；无真实镜头标定（默认镜头 f=0.7×宽）；DJI 云台机四元数含机身运动（23°/s vs 画面 5.7°/s，上游同样如此，非移植 bug） |
| 2026-08-15 | **官方 Gyroflow 输出对比（用户提供真值）** | `DJI_..._stabilized.mp4`（Gyroflow 官方导出）: med\|ω\| **2.74°/s（7.3×）**、med-jitter 12×、pixdiff 2.1× vs 我们 1.4×。逆向分析定位剩余缺口：**DJI clip_meta 镜头档案未提取**（上游从文件内生成官方镜头档案: focal=770、畸变系数 k1-k4、readout=9.11ms；我们一直用默认镜头+readout=0）。已移植 `get_lens_profile` 语义 + sensor_readout/digital_focal_length/distortion_coefficients/sensor_fps 提取（值与 tp-dump 真值精确一致），并修复 DJI 四元数时间戳模型（上游 `((i-offset)/len)·vsync + frame/fps_ratio` 精确公式，旧"帧内均匀"模型 25 秒累计漂移 0.33 秒）。修复后帧级 DJI med\|ω\| 压缩 **3.44×**（此前 1.4×），全片渲染复验进行中 |
| 2026-08-15 | **终验（vs 官方输出）** | 全片重渲（镜头档案+readout+精确时间戳）: 我们 **med\|ω\| 6.43（3.1×）/ med-jitter 3.68× / pixdiff 1.82×**，官方 2.74（7.3×）/ 12× / 2.09×。与官方差距从 5× 收敛到 ~2×。剩余差距主要构成: ① 平滑强度（官方 Default 算法对慢运动几乎全锁，我们保留更多——DefaultAlgo 移植的参数语义可继续对照上游）；② 官方默认开启 horizon lock 与 adaptive zoom（裁切约 1.2× 同步缩小残差）。测试 229 passed |
| 2026-08-15 | **平滑层闭环验证（msgyro-smooth 参照）** | 实现 `crates/msgyro-smoothing/src/bin/msgyro-smooth.rs`（喂 JSON 四元数跑 Rust DefaultAlgo，本地盘构建绕网络盘 IO），同一 DJI 真值四元数流喂两侧: **Python 与 Rust 移植版逐位一致**（角度差中位 0.00006°/max 0.006°，浮点噪声级）。至此三层全部获得独立参照: 数据层（telemetry-parser 真值 parity）、平滑层（msgyro Rust parity）、渲染层（合成闭环锁定）。另补齐 `ComputeParams.camera_diagonal_fovs`（上游 fov_ratio=dfov/120 消费，此前恒 fallback 120°；DJI 实际 110°）。**结论**: 剩余 vs 官方导出的差距（sm=0.5 时 4.0 vs 2.74°/s；sm=0.8 时 3.34）归因于官方导出时的用户设置（smoothness/horizon lock 等）而非移植缺陷——管线各层数学均已与独立 Rust 参照对齐 |
| 2026-08-15 | **P4-1 镜头库落地 + GoPro 官方镜头渲染** | 官方库下载入包（12409 档案）、MINF 机型提取、`_try_auto_load_lens_profile` 实现（品牌守卫修复 "Unknown" 误匹配 Insta360 的回归，229 全过）。GoPro 全片: med\|ω\| 5.77→**5.08（1.14×）**、med-jitter 0.86×、音频 682 完整；输出 `verify_lens_gopro.mp4`。至此 P4-1 完成；GoPro 剩余提升空间在平滑参数与高频震动（CORI 30Hz 采样率上限） |
| 2026-08-15 | **P2-1 渲染采样优化** | cpu_undistort 三项改造: ① cv2.remap 替换 numpy 双线性 gather（采样 333ms→4ms，**81×**；uint8 直采省两次全帧 float 转换；边缘半像素/无效像素语义精确保持——identity/亚像素/RS 像素级测试全过）; ② mgrid 坐标网格按尺寸缓存（~60ms/帧→0）; ③ float32 全链 + IBIS 活跃标志从小矩阵栈预判（省 100MB 级逐像素布尔扫描）。端到端: GoPro 1280×1120 稳定 923→465ms/帧（**2.0×**），DJI 1920×1080 ~2.0s→1.24s/帧（~1.6×）。剩余成本剖析: 畸变数学 175ms（arctan 链，地板）+ RS 逐像素矩阵 gather 91ms + 掩码 40ms——进一步提速需分带/多线程（P2-5） |
| 2026-08-15 | **批量优化轮（P2 收尾 + P4-3/4-6 + guess + P3 进展）** | ① P2-4 fov 预排序缓存; P2-2 ComputeParams 渲染期复用（省每帧 ~25k 条目拷贝）; P2-5 三段流水线（decode‖stabilize‖encode 三线程）——端到端渲染 0.94→**1.48 fps（1.57×）**，音频完整。② 自适应 zoom 审计: 分发/算法与上游逐行一致，此前"变差"实为裁切放大残差的物理现象，无需改码。③ P4-3 keyframes 接入 frame_transform（VideoRotation/BackgroundMargin/Feather/LensCorrection/ZoomCenter 逐帧查询 + get_fov，含 keyframes 管线接线修复与 2 个动画回归测试）。④ P4-5 forward.rs: 上游 v1.6.3 无此算法，不适用。⑤ guess_imu_orientation 产品化（48 置换 × 重积分 × RS 代价，上游 rs_sync.rs 语义；真实素材 15s 选出 YxZ；含合成回归测试；CORI/DJI 文件自动跳过）。⑥ P3-1 VQF 诊断精化: 纯 Z 偏差 24.3°→29.5°，与 z-gyro bias ~0.017 rad/s 的 bias 合并差异一致（排除 mag 路径），xfail 理由更新。⑦ P3-3 GPU: 488 项插值系数表从上游提取接入（ coeffs=None 全 0 表根因修复）+ CPU fallback 签名修复（pixel_value_limit/output_stride 前轮已修）。⑧ P4-6 GUI 最小可用: QThread 导出/AutoSync worker（不再冻结 UI）+ AutoSync 接真实 manager.synchronize + 时间轴 seek 显示稳定帧预览。测试 232 passed（+3） |
| 2026-08-16 | **P3-1 VQF 对齐达成（消 xfail）** | 逐阶段 Rust 参照（vqf_debug harness dump bias/quat3d/acc_i/quat6d）定位两个真 bug: ① `filter_step` 收到的是切片副本——IIR 状态更新全部静默丢失，滤波器退化为无记忆（影响所有 VQF 输出）; ② 包装器把全零磁测降级为 6D（mag=None），Rust 恒走 9D（Some(zeros)）——零磁测仍改变 VQF 类内部状态。双修后 **VQF 与 Rust golden 机器精度一致（max_err 1.1e-15，原 0.338）**，xfail 移除——**5/6 积分器全部对齐**（仅剩 complementary 论文版重写）。真实素材复验: DJI（VQF 为默认积分器）帧级 med\|ω\| 压缩 2.97×→**3.27×**。测试 233 passed |
| 2026-08-16 | **P3-2 complementary 论文版重写达成（消最后一个 xfail）** | 按 Rust complementary.rs（Valenti et al. 论文）完整移植 V1+V2（IIR 加计滤波、重力自动标定、自适应增益/稳定期加速、settle 斜坡），wrapper 驱动 V2 且输入变换/零加计微调逐行对齐。**与 Rust golden 机器精度一致（max_err 2.2e-16，原 0.71）**，xfail 移除——**6/6 积分器全部对齐独立 Rust 参照**。自生成 golden 按新语义重生。测试 234 passed，xfail 清零 |
| 2026-08-16 | **P3-4 fov_iterative 接入完整畸变模型** | FOV 多边形计算改为上游 undistort_points_with_rolling_shutter 全管线: 归一化 → 畸变模型 undistort_points（向量化）→ 光折射 → 逐点 RS 行时间旋转 → new_k 投影。原针孔投影在鱼眼镜头上错算稳定多边形。首版 RS 分支漏 org_c⁻¹ 消去（绝对姿态重复应用致 fov 塌缩到 0.56 过裁），修正三因子合成后 fovs=0.90、渲染指标回到基线（med-jitter 0.82×）且用真实鱼眼几何、无黑边、音频完整。测试 234 passed |
| 2026-08-16 | **RS 分带实验（负结果记录）** | 尝试游程分带替代逐像素矩阵 gather（避开 80MB gather）: 正确但在纯 Python 更慢（760/386 vs 380ms/帧，逐调用 numpy 开销在行碎片化时占主导），回退并代码内注释记录。P2 性能项至此收敛：进一步提速需编译内核 |
| 2026-08-16 | **P3-3 GPU 真机验证完成（Intel UHD 630 / Vulkan）** | 用户确认本机有 UHD 630，安装 wgpu 0.32 后真机调试:
  ① 管线此前"全零输出"在真机上表现为**强度减半**（127 vs 255）——逐阶段隔离（单点/梯度/交替像素探针 + 隔离 shader 复算）定位第四个根因: **`safe_area_rect` 默认 (0,0,0,0)，而 shader 的 `draw_safe_area` 对区外像素 ×0.5**（预览调光功能），只有 (0,0) 精确、其余全半——与症状完全吻合; 修复: frame_transform 设全帧安全区。
  ② 顺带给 WGSL override 加 `@id(100..103)`（spec 常量绑定）。
  **结果**: identity 逐位一致（max diff 0）、鱼眼 ≤2 灰度、真实 GoPro RS 变换 ≤8/255；**GPU 6.5×/帧（78 vs 505ms）、端到端渲染 2.4×（3.52 vs 1.48 fps）**，音频完整，质量与 CPU 渲染等价。测试重写为真机断言（2 个新测试，xfail 移除）→ **236 passed**。GPU 保持 opt-in（--gpu） |

### 真实视频效果验证与 frame_transform 关键修复（2026-08-14）

**验证方法**（可复用工具）:
- `tests/verify_stabilization.py` — 用项目自身光流+本质矩阵估计帧间角速度，输出鲁棒抖动指标（中位数/截尾均值，因本质矩阵在低纹理帧上有高达 180° 的离群）
- `tests/extract_comparison.py` — 输入/输出同刻帧上下拼接 PNG，供目视检查
- `crates/msgyro-golden-gen/src/bin/ft_compare.rs` — Rust 独立 frame_transform 参照 harness（喂同一四元数/参数 JSON，输出矩阵对比）

**过程**:
1. 初版渲染（含 P0 全部修复）用 GoPro 真实素材度量：输出中位抖动为输入 4.8×（旧版输出 19×）——稳定不仅无效还在添加运动，且增量与陀螺自身幅值吻合。
2. 闭环合成测试（已知相机旋转渲染 + 同源四元数稳定，输出应锁住平滑路径）：增益 ≈1，修正矩阵近似恒等。
3. DJI（出厂四元数，绕过 GoPro 解析/积分链）同样变差 → 问题在共享下游（smoothing→frame_transform→render）。
4. Rust 独立参照逐元素对比：Python 无 RS 分支矩阵偏差 0.1-0.19；四元数查询两侧一致。
5. **根因**: `frame_transform.py` 无 RS 分支用 `quat = smoothed·org_c⁻¹`（org 相对增量），上游/移植版为三项 `quat = smoothed·org_c⁻¹·org_row`，无 RS 时 `org_row ≡ org_c` 退化为**绝对平滑姿态** `quat = smoothed`。增量语义使输出仍跟随原始抖动，稳定完全失效。RS 分支（三项）本来就正确；golden 旧用例恰好全为 RS 场景，故从未暴露。
6. 修复后 Python 与 Rust 参照矩阵逐元素一致（max diff < 1e-5，float32 精度级）；frame_transform.json golden 重生成（12 行，无 RS 用例）；全量 226 passed。

**注意**: 此 bug 说明"自生成自比对"golden 无法发现语义级移植错误；frame_transform 现已具备三方参照（上游源码、Rust 移植 harness、golden）。后续 Rust harness 的依赖 patch 见父仓库 `Cargo.toml` TEMP 标记（网络无法拉取 pinned git rev 时用本地 cargo git db 恢复的 vendor 副本）。

### 追加修复（同日第二轮，真实视频效果验证驱动）

修复后闭环复测仍不锁定，继续深挖又发现并修复 **2 个语义级移植缺陷**（均有上游源码依据）:

1. **smoothed 校正量转换缺失**（`manager.py` `recompute_smoothing`）: 上游 `gyro_source.rs` `recompute_smoothness()` 末尾把平滑结果转换为**校正四元数** `smoothed[t] = sm(t)⁻¹·org[t]`（上游 638-641 行注释 "rotation quaternion from smooth motion -> raw motion to counteract it"）。Python 之前直接存平滑姿态，frame_transform 的三项合成因此语义错位。已补上转换。
2. **fov_iterative 消费语义**（`zooming/fov_iterative.py`）: 修正为与 frame_transform 相同的校正量组合（`correction·org_c⁻¹·org_row`）。修复前把校正量当姿态用，真实素材算出 fov=2.7（输出网格映射出界 2×，渲染全黑）；修复后 fov≈0.75、采样全部在界内。

**渲染链正确性最终证明**（合成闭环，约定匹配后）: 上游渲染矩阵含 `flip = S·R·S`（S=diag(1,-1,-1)）相似变换，配套其 IMU 约定链（`coordinate_transform(-y,x,z)` + 初始 π/2·X），等效于"数据侧 pitch 反号"。在"pitch 反号数据 + 镜头内参匹配"的闭环下: roll 轴映射误差 **0.000 px**，pitch/yaw 残差 9px **完全**归因于闭环成像 f=448 vs 默认镜头 f=512（512/448−1=14.3% ✓）——渲染链（frame_transform→cpu_undistort）对正确约定的数据是**精确正确**的。

### 真实素材现状与遗留（诚实边界）

修复后 GoPro/DJI 重渲（含 autosync）**输出抖动仍高于输入**（GoPro 像素帧差 6.11→8.47）。逐层诊断结论:

- 渲染层: 已证正确（上述闭环）。
- **数据层: 陀螺-视觉相关性缺失**。对 GoPro 真实素材做 48 种轴映射置换 × ±1.2s 偏移扫描，视觉（本质矩阵）与陀螺（四元数增量）逐轴相关性最高仅 0.32（无峰无区分性）——时间对齐与轴约定均无法建立对应。autosync 在不相关信号上得到的偏移无意义。GoPro 嫌疑集中在 GPMF 时间戳模型（P0-5: 每 DEVC 1s 均匀假设、ORIN 读而不用、`has_accurate_timestamps=True` 恒置位）；DJI（出厂四元数+固定乘子链）同样不相关，需对照 telemetry-parser 逐行核对约定。
- **下一轮最高优先级**: P0-5 升级为 P0-blocker——修正 GPMF 时间轴（按 STMP/实际采样率）与 ORIN 轴序，之后用 `verify_stabilization.py` + 48 置换校准复验相关性，相关性 >0.8 后再谈真实素材的稳定效果。

### 本轮沉淀的可复用验证工具

- `tests/verify_stabilization.py` — 真实视频稳定质量度量（鲁棒角速度抖动 + 像素帧差双指标）
- `tests/extract_comparison.py` — 输入/输出同刻帧拼接 PNG 目视对比
- `crates/msgyro-golden-gen/src/bin/ft_compare.rs` — Rust 独立 frame_transform 参照（喂同一 JSON，逐元素比矩阵；本轮定位 2 个语义 bug 的关键工具）
- 48 置换轴校准方法（本轮脚本化于验证过程，后续应固化为 `tests/` 工具）

## 本轮实施数据（2026-08-14）

- **测试**: `226 passed, 6 skipped, 2 xfailed`（开工基线 201 passed；+25 个新测试，含 10 个 `slow` 标记的真实编解码端到端）
- **真实视频验证**（父目录 GX010045.MP4，1280×1120@29.97，435 可解码帧）:
  - 渲染输出 435/435 帧（修复前丢 13 帧）
  - 音频 682/682 帧 AAC stream copy（修复前无声）
  - 自动同步 28.2s 完成，offset -254.7ms
- **渲染性能**: 未优化（~0.69s/帧），P2 待做——本轮全部是正确性接线；cpu_undistort 的 RS 路径因逐像素矩阵 gather + 双趟映射略慢于旧版（中心矩阵单趟），P2 静态 map 预计算时一并解决


## 2026-08-16（晚）用户报告坏区间排查：自动同步 + 镜头自动匹配两个根因

**用户报告**: 成片第 8 秒起约 3 秒剧烈抖动未消除，后段仍有类似区间。

**窗口化抖动定位**（`tests/verify_windows.py`，逐秒窗口中位抖动）:
- DJI 8-22s 为剧烈运动段（|ω| 中位 33-66°/s，峰值 437°/s），成片在 11-14s/21s 明显差于官方导出
- GoPro 成片全局中位抖动 2.74 vs 输入 2.94 —— 稳定基本没生效

**根因一：自动同步给出错误偏移**。正式成片实际未跑 autosync（offset=0），而 `--autosync` 找到的 179ms 会把稳定彻底毁掉（medspeed 20-61°/s ≈ 未稳定）。逐帧轨迹重建代价函数发现：200 帧采样（every=3，133ms 基线）在快动段光流匹配失效，代价曲线全片平坦（46-55），"极小值"是噪声。**修复**：
- `synchronize()` 默认 `sample_count=1000`（逐帧，DJI 代价函数出现真实极小：27.99 vs 邻域 31+，自动同步 179→7.9ms，真值 0）
- `_rs_sync_offset` 加平坦警戒线：最优代价与 ±50/100ms 邻域中位差距 <3% 时拒绝返回（GoPro 正确拒绝）
- `find_offset_visual_features` 加弱峰置信度门槛（corr<0.5 拒绝）

**根因二：镜头库自动匹配从未生效**。`_try_auto_load_lens_profile` 的 brand guard 在 `load_all()` 之前遍历 profiles —— 进程首次运行时 DB 未加载、`known_brands` 为空集，任何品牌都被静默拒绝。GoPro 一直无畸变校正（lens "---"）。**修复**：把 `load_all()` 移到 brand guard 之前。GoPro 现在自动匹配 `HERO12 Black 4k 8:7 Wide`（库内唯一 8:7 HERO12 档案）。

**终配置验证**（同编码 H264 sweep 序列，跨编码度量有 ±0.7 全局/±15 单窗口压缩噪声不可比）:
- 双片均 offset=0（时间戳模型与上游逐位一致，任何非零偏移渲染更差：+10ms 快动段 9.4-20.6、-10ms 打地鼠、179ms 灾难）
- smoothness 0.5→1.0：DJI 平静段 2.5-6.5→1.7-2.2，14s 窗 10.7→4.8，17s 5.7→1.8
- 像素级验证：终片配方与 sweep G 配方逐位一致（max diff 0）

**终片**: `final2_dji_gpu.mp4`（H264@16Mbps+音频，全局中位抖动 3.34 vs 原成片 3.74）、`final2_gopro_gpu.mp4`（2.69 vs 原 2.74、输入 2.94；6-14s 窗口 1.3-2.0 对输入 1.5-7.6）

**已知限制**: GoPro 2-3s 窗口对任何偏移都难（-242: 4.5/4.7、0: 5.4/6.6、-120: 6.2/6.2，打地鼠模式），疑似机内 EIS 与 CORI 校正互扰，留待后续。DJI 4s 窗口同理（官方同窗口也差：7.3/6.0）。度量工具链沉淀：`tests/verify_windows.py`、`tests/diag_bad_windows.py`、`tests/sweep_bad_segment.py`、`tests/reverse_official.py`、`tests/phase_vs_official.py`、`tests/verify_sync_axis.py`。
