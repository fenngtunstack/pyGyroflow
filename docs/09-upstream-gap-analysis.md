# 上游 Gyroflow 对照差异清单（穷尽）

> 版本: 1.0 | 日期: 2026-09-16
> 对照基准: `opensource/gyroflow/` (v1.6.3 源码) + vendored telemetry-parser
> 范围: 全仓库逐模块对照，覆盖 5 个域

## 阅读约定

每条标注**证据等级**：

- **[实测]** — 已在本机用真实素材/合成数据复现，给出可重放的输出
- **[代码确证]** — 已定位双方 file:line 并直接比对，未跑运行时
- **[报告]** — 来自子代理逐行阅读，未独立抽验

状态标记：**缺口 R**（上游有、我们无或结果错）、**差异 D**（都在，实现不同）、**等价 E**（核对过、一致——列出以便读者知道查过）。

**关于「逐位一致」**：把上游 Rust 代码原样拷进临时 crate 跑参照时，同一份源码**重新编译后结果不保证逐位相同**——实测 `splines.rs` 有 11 个用例的个别值差 1 ULP（debug 与 release 也互相不一致，指向 LLVM 的乘加合并）。所以凡是标「逐位一致」的地方，指的是与该次构建的输出一致；跨构建应对到 1e-12 相对容差。

---

## 第一部分：实测复现的缺陷

### G-01 [实测] 编码器异常被后台线程吞掉，CLI 报成功退出码 0

**影响**：任何编码失败都表现为"处理完成 + 文件不存在"。`--codec` 三个选项中 ProRes 恒失败。

```
INFO  Rendering to: /tmp/prores_out.mov
Exception in thread pgf-encode: av.error.ArgumentError:
    avcodec_open2("prores_ks") returned 22
INFO  Processed 6 frames
INFO  Done: /tmp/prores_out.mov
>>> main() 正常返回，退出码 0
ls: /tmp/prores_out.mov: 没有那个文件或目录
```

**根因**：`pygyroflow/rendering/ffmpeg_processor.py:173` 把 `pix_fmt` 硬编码 `yuv420p`，`prores_ks` 只接受 `yuv422p10le`。编码器打开失败，而工作线程里的异常没有传播回主线程。

**上游**：`rendering/mod.rs` 的编码器错误经 `Result` 一路返回到调用方；`ffmpeg_processor.rs` 打开编码器失败直接返回 `Err`。

**修复**：① 按 codec 选择 pix_fmt（ProRes → `yuv422p10le`，HEVC/AVC → `yuv420p`）；② 工作线程异常收集起来，`render()` 结束时 raise。

---

### G-02 [实测] 遥测格式检测用全文子串匹配，66 个真机素材误判 3 个

**影响**：误判素材的稳定化静默失效（零遥测 → 无陀螺 → 只做镜头校正）。

`pygyroflow/telemetry/parser.py` 的 `_detect_dji`：
```python
data.find(b"djmd") >= 0 or (data.find(b"dvtm") >= 0 and data.find(b"DJI") >= 0)
```

470 MB 文件里搜 4 字节。命中位置全在 H.264 压缩码流内部：

```
@8889047  : ...w\xf5DJI\xe08o\xabL...
@39689725 : ...\xb5roDJIC\x12...
@201677380: ...b\tdvtmthb1dGtCt...
```

分派顺序 GoPro→DJI→Sony，命中即返回，走不到 Sony。

**同机位 A/B 对照（干净证据）**：

| 素材 | 检测结果 | IMU 样本数 |
|---|---|---|
| `issue-44-29-...-dist-compensation-off` | Sony ZV-E1 ✓ | 30030 |
| `issue-44-30-...-dist-compensation-on` | **DJI** ✗ | **0** |

误判的三个（`issue-44-12`、`-23`、`-30`）全部零遥测。

**上游**：telemetry-parser `lib.rs` 走结构性检查——遍历 mp4 track，找对应 codec tag 的 handler，不做全文搜索。

---

### G-03 [实测] IMU 变换与滤波整条链静默跳过，且滤波器本身会崩

三个缺陷叠加：

**(a) `clear()` 抹掉用户设置的 IMU 变换**
```python
# pygyroflow/gyro_source/source.py:97-103
def clear(self) -> None:
    self.quaternions.clear(); self.smoothed_quaternions.clear(); self.raw_imu.clear()
    self.imu_transforms = IMUTransforms()   # ← 无条件重置
```
`load_from_telemetry` 开头调用 `clear()`，任何 load 前设的 `gyro_bias`/`imu_orientation` 全被丢弃。

```
用户设 gyro_bias = 5 deg/s → load 前后四元数最大差 = 0.0
```

**上游** `gyro_source/mod.rs:517-527` 的 `clear()` 刻意保留 `imu_orientation` 与 `gyro_bias`。

**(b) 抹掉后走退化分支**：`has_any()` 变 False → `else: self.raw_imu.clear()`。

**(c) 空 `raw_imu` 静默回退到未变换数据**
```python
# pygyroflow/gyro_source/source.py:270-274
# raw_imu 为空时回退到 file_metadata.raw_imu（原始未处理值）
```
结果是用未做零偏修正的数据积分，表面无异常。

**(d) 即便绕过 (a)，滤波器直接崩**
```
>>> lowpass_filter_imu: AttributeError: 'TimeIMU' object has no attribute 'get'
>>> median_filter_imu:  AttributeError: 'TimeIMU' object has no attribute 'get'
```
`filtering/lowpass.py` 与 `filtering/median.py` 对每个样本调 `sample.get("gyro")`，而 `TimeIMU` 是 dataclass，没有 `.get()`。**`imu_lpf`/`imu_mf` 这两个功能从未工作过。**

---

### G-04 [实测] `--gpu` 忽略插值选项，默认输出比 CPU 差一档

```
GPU --interpolation bilinear(0) vs lanczos4(2)
→ 8 帧逐帧 max diff = 0，逐位相同
```

**根因**：`pygyroflow/stabilization/frame_transform.py:420` 写死
```python
kernel_params.interpolation = 2   # 注释写 "Bilinear"
```
这个 `2` 在 **CPU 约定**下是 Lanczos4（对），但 `gpu/backend.py:240-242` 把它当 pipeline 常量原样传给 shader，而 **shader 约定**里 2 = Bilinear。

两套索引约定撞车，用户的 `--interpolation` 从来没进过 shader。结合实测锐度差：Lanczos4 相对 Bilinear 的 Laplacian 方差 524.7 → 983.8（+87%），`--gpu` 默认落在低锐度那档且无法改变。

**上游** `stabilization/mod.rs:20-33` 只有一套约定：`Bilinear=2, Bicubic=4, Lanczos4=8, EWA 10-13`。

---

### G-05 [实测] `gyro_export` 的 stab 列与上游差一个 `org`

判别性测试（org 与 smoothed 绕不同轴）：

| | 值 |
|---|---|
| 我们写入 | `[1, 0, 0, 0]`（smoothed 原值） |
| 上游 / 正确值 | `[0.981856, 0.153439, -0.091158, 0.064071]` |
| 匹配 | **False** |

**上游** `gyro_export.rs:192`：`stab_quat = (quat_smooth / quat_org).inverse()` — 稳定后**相机运动**。
**我们**：写 `smoothed[ts]` — 是**校正量**。二者互为逆。喂进 Blender/AE 会得到反向的相机朝向。

分量顺序 [w,x,y,z] 两边一致。

---

### G-06 [实测] ~~STMap 的 redistort 图没有畸变模型~~ 已修（`bcc927b`，图本体）；可达性（render_queue 的 stmap job）仍属 C-12

`pygyroflow/stmap/exporter.py:225-236` 是逐像素 Python 双重循环，只把 `matrices[0]` 的 3×3 求逆。注释自认 "simplified approach"。

不含镜头畸变模型，不用逐 RS 行矩阵。**上游** 用 `FrameTransform::at_timestamp_for_points(..., true)` + `undistort_points` 逐像素取真实前向映射。任何非零畸变下输出都是错的。

可达性：全仓库只有 `compute_undistort_map` 有调用方（测试），`compute_distort_map` **无调用方**，`render_queue._process_stmap_job` 直接 `raise NotImplementedError`。

---

### G-07 [实测] 容器旋转元数据没读

**上游** `ffmpeg_processor.rs:609` 用 `av_display_rotation_get` 读 display matrix；`render_queue.rs:1336` 转成 `set_video_rotation((360 - info.rotation) % 360)`。

**我们** `manager._get_video_info` 只返回 width/height/fps/duration/frame_count/image_sequence。

竖屏素材渲染会歪。testvideos 里没有带旋转元数据的素材，故为代码级确证。

---

### G-08 [实测] 地平线锁应用了两次

```python
# pygyroflow/smoothing/registry.py:92-105
result = self.current().smooth(quats, duration_ms, compute_params)
if self.horizon_lock.lock_enabled or (...is_keyframed(LockHorizonAmount)):
    self.horizon_lock.lock(result, org_quats, grav, use_grav, compute_params)
return result
```

```python
# pygyroflow/manager.py:336-371
smoothed = self.smoothing.smooth(...)          # ← 内部已锁一次
if self.smoothing.horizon_lock.lock_enabled:
    ...
    self.smoothing.horizon_lock.lock(smoothed, ...)   # ← 又锁一次
```

`lock()` 是按百分比向锁定朝向 slerp，不是幂等操作。开地平线锁时角度被叠加两次。

**上游** `gyro_source/mod.rs:626-629` 只调用一次，且在 `alg.smooth()` **之前**。

---

### G-09 [实测] ~~三个 offset method 只剩一个实现~~（method 0 已移植 `46be830`、method 1 已移植 `d1036a5`；`estimate_rolling_shutter` 的 for_rs 模式仍缺，见 B-02）

```python
# pygyroflow/synchronization/find_offset/__init__.py:69
if method == 2:
    return find_offset_rs_sync(...)
# Methods 0 and 1 both use cross-correlation
return find_offset_visual_features(...)
```

注释白纸黑字。上游 method 0 与 method 1 是两套完全不同的算法：

- **method 0**（`find_offset/essential_matrix.rs:13-131`）本质矩阵 + TimeIMU 代价。两路信号各做 20 Hz 前后向低通；代价 `Σ(gx−ox)²·70 + (gy−oy)²·70 + (gz−oz)²·100`；`max_angle < 3.0` 的窗口跳过；粗搜 1 ms 步长 → 精搜 ±1 ms@0.01 ms；`|lowest−initial| < search_size·0.9` 才接受。
- **method 1**（`find_offset/visual_features.rs`）光流点对**畸变校正后**的残差最小化，取最短 90% 流线求和；含滚动快门模式。

附带：`initial_offset_ms` 在 method 0/1 路径上被丢弃（`__init__.py:75-79` 不转发）。

---

### G-10 [实测] ~~`get_visual_rotations` 的时间戳——注释与代码矛盾~~ 已修（`0420488`）

```python
# pygyroflow/synchronization/pose_estimator.py:274-275
# Midpoint timestamp between this frame and next
ts = fr.timestamp_us          # ← 就是帧时间戳，没有中点
```

**上游** `core/synchronization/mod.rs` 的 `recalculate_gyro_data` 做三件我们没做的事：① 把样本放在两帧时间戳**中点**（`ts += (next_ts - ts) / 2.0`，`iter.peek()` 取下一个 sync 结果）；② `final_pass` 时对姿态估计失败帧用**左右最近的成功帧线性插值**补上欧拉角（按时间戳比例加权），首尾缺口不补——外推是凭空造方向；③ `lpf > 0` 时对拼好的信号做零相位前后向低通（`lowpass_filter(freq, fps)`，注意上游把它存成「百分之一赫兹的整数」，精度 0.01 Hz）。中点这一条最要紧：offset 搜索量的就是这个时间平移，把半帧偏置加在每个样本上等于给被测量本身加了个恒差。

已按上游逐条补齐（`_midpoint_us` / `_interpolate_missing` / `_filtered`），并暴露 `PoseEstimator.lowpass_filter` 与 `AutosyncProcess.set_lpf`。注意上游 CLI 里没有任何东西驱动 lpf，它是 GUI 侧旋钮——仍然移植并保持可达，因为低对比度素材上光流信号噪声大时调用方没有别的入口。

---

## 第二部分：按域列出剩余差异

### A. 遥测与陀螺源

| # | 等级 | 项 | 证据 |
|---|---|---|---|
| A-01 | R | **25 种格式只支持 4 种**。上游 `telemetry-parser/lib.rs:161-187` 注册 GoPro/Sony/Canon/Nikon/DJI/Xtra/Insta360/GyroflowGcsv/GyroflowProtobuf/BlackBox/BlackmagicBraw/RedR3d/Runcam/WitMotion/PhoneApps/ArduPilot/Vuze/KanDao/QoocamEgo/Camm/EspLog/Cooke/SenseFlow/Freefly/Zcam。我们只有 GoPro/DJI/Sony/Insta360 | [实测] |
| A-02 | R | **GCSV/Protobuf 输入缺失**（`gyroflow.proto` + `gcsv.rs`，1307 行）。这是"相机无陀螺、用外部 IMU"场景的唯一入口 | [报告] |
| A-03 | R | **无 sidecar 回退**。上游 `lib.rs:107-124`：mp4 内无数据时回退同名 `.gcsv/.bbl/.bfl/.csv`。我们 `parser.py:47-51` 非 mp4/mov/insv 直接抛 | [报告] |
| A-04 | R | **整个文件读进内存**。`parser.py:66-68` `f.read()`；上游 `lib.rs:73-84` 只读头尾。多 GB 素材会爆 | [报告] |
| A-05 | R | **FileMetadata 19 字段中 9 个声明但从不写入**：`gravity_vectors`/`image_orientations`/`lens_positions`/`lens_params`/`per_frame_time_offsets`/`digital_zoom`/`camera_stab_data`/`mesh_correction`/`frame_readout_direction`。逐个 grep 确证（仅 `thin()` 会置 None） | [实测] |
| A-06 | R | **Sony 专项全缺**（`sony.rs` 621 行，我们只读 7 个 tag）：`init_lens_profile`（从 LensDistortion 0xe421 建档案，6 项 SVD + Newton 逆）、`stab_collect`+`stab_calc_splines`（IBIS 0xe40f/0xe450、OIS 0xe416 → CatmullRom 样条）、`get_time_offset`（0xe40c/0xe40d/0xe437/0xe435）、`get_mesh_correction`（0xe42f Mesh + 0xe423 FPD）。**注意**：`distortion_models/sony.py` 我们写了 353 行的着色器侧模型，但**无任何东西喂它系数** | [实测] |
| A-07 | R | **`splines.rs` 在 Python 侧不存在**（CatmullRom + BivariateSpline）。上游用于 Sony IBIS/OIS 与 mesh，不是死代码 | [报告] —— 已实现（`8bfa9e1`）：`gyro_source/splines.py`，CatmullRom 与 BivariateSpline 与上游 Rust 逐位一致（17 用例 / 108 个数值，见 `tests/golden/splines.json`） | [实测] |
| A-08 | D | **GoPro method-0 `QuaternionConverter` 公式不同**。`converter.py:103` 写 `n_quat * io_quat * org⁻¹`，上游 `mod.rs:47` 是 `n_quat * (org_quat * io_quat⁻¹)⁻¹`。实测短片段影响小（差异是恒定 20.5°±0.40° 安装角，逐帧相对旋转只差 0.043°），但公式错是事实 | [实测] |
| A-09 | D | **滤波边界行为不同**。上游零状态因果前后向；我们用 `sosfiltfilt`（奇数对称填充）与 `medfilt`（居中窗）。序列两端系统性不同 | [报告] |
| A-10 | D | **VQF 磁力计输入不同**。上游 `mod.rs:126` 硬编码 `let m = [0,0,0]`；我们在 `TimeIMU.magn` 有值时传真实磁力计。当前 Python 解析器从不填 magn，属潜伏 | [报告] |
| A-11 | R | **`get_checksum` 截断**。`source.py:448-462` 只哈希 detected_source/orientation/duration/lpf/两个计数；上游还哈希 rotation/bias/offsets/method/首末四元数。改这些参数不触发重算 | [报告] |
| A-12 | R | **`find_bias` 未做偏移修正**。上游 `mod.rs:933-935` 用 `offset_at_video_timestamp` 修正取样窗口 | [报告] |
| A-13 | D | **`adjust_offsets` 是朴素最小二乘**。上游 `mod.rs:700-776` 是 RANSAC 式成对斜率搜索 + 5ms 内点阈值；一个坏同步点会污染我们的线性拟合 | [报告] |

### B. 同步子系统

| # | 等级 | 项 | 证据 |
|---|---|---|---|
| B-01 | R | offset method 0 未移植（见 G-09） —— 已移植（`46be830`）：`essential_matrix.rs::find_offsets` 的角速度匹配算法（尽管文件名如此，无本质矩阵）：20 Hz 前后向低通、70/70/100 加权平方差、ceil 查找不插值、过半匹配规则、<3 °/s 跳过、±search_size 1 ms 粗搜 + ±2 ms 0.01 ms 细搜、90 % 接受窗（此处为活守卫）。AutosyncProcess method 0 分支接线，无结果回落互相关 | [实测] |
| B-02 | R | offset method 1 未移植；`estimate_rolling_shutter` 模式完全缺失（全树无 `for_rs`） —— method 1 已移植（`d1036a5`）；`estimate_rolling_shutter` 已移植（`be76284`）：候选值作为 frame_readout_time 换入私有副本、零偏移量同一成对距离、±(1000/fps) 粗搜 + 0.01 ms 细搜。**场景级验证受阻于一个未解发现**：成对距离对读出误差对称不敏感（行偏移同作用两帧、帧对差分一阶相消；跨行运动/各帧自身行的组合均试过，最小值钉 0）——疑移植逐点 RS 路径缺陷，待专项核查 | [实测] |
| B-03 | R | **rs-sync 缺 `− readout/2` 修正**。`autosync.py:265` `return -delay_ms`；上游 `rs_sync.rs:179` `offset = -delay - frame_readout_time/2`。符号约定本身一致 —— 已在 `recompute_smoothing`/autosync 补上（随 rs-sync 移植修正，`return -delay_ms - frame_readout_time_ms / 2.0`） | [实测] |
| B-04 | R | rs-sync 优化器退化。上游 = 3ms 粗网格 + LBFGS 逐帧平移向量 + Backtrack 延迟优化 + 鲁棒损失 `ρ=√log(1+r²)` + LMedS 平移方向估计；我们 = 3ms 粗网格 + 四次网格细分，无平移模型/无鲁棒损失/无梯度 —— 已实现（`8ff80fa` 移植 + `00b6478` 桥接 + 向量化 `f0f5994`）：rs-sync crate（@fc5cf51）逐文件移植为 `rs_sync_problem.py`（样条/四元数/Backtrack/LMedS/鲁棒损失与解析梯度链/pre_sync+sync+full_sync；每帧运动以 scipy L-BFGS-B 替 argmin LBFGS，同一拟牛顿族与同一梯度，以收敛到真值验证）；`RollingShutterSync.full_sync` 改为构造 SyncProblem 驱动之，旧网格细分退役；平坦地形守卫改用优化器自身代价（两尺度混用曾假阳性）。端到端：正弦场景恢复 −50 ms 真值、代价 ~1e-6；e2e 台架 +191.6 ms（真值 200）；梯度链对有限差分 <1e-5；向量化与标量逐位一致 | [实测] |
| B-05 | R | **rs-sync 不做镜头畸变校正**。`rs_sync.py:195` 注释自认畸变系数 "not yet used"；`camera_matrix=None` 时把原始像素坐标当归一化坐标（`:218-220`）。上游在归一化前过 `undistort_points_for_optical_flow` —— 已换（`76de345`）：add_track_from_frames 走 undistort_points_for_optical_flow | [实测] |
| B-06 | R | `calc_initial_fast` 缺失（上游用本质矩阵估中位数偏移、把 search_size 收窄到 3000ms 再跑 rs-sync） —— 已实现（`ed114ad`）：AutosyncProcess 的 calc_initial_fast 旋钮，`_rs_sync_offset` 前置 essential-matrix 预跑、窗口无条件 ±3000 ms | [报告] |
| B-07 | R | `initial_offset_inv` 缺失（±initial_offset 各跑一次取点多者） —— 已实现（`ed114ad`）：run() 的 ± 试验包裹统一分派 `_offset_dispatch`，存在者胜、其次代价低者（上游按点数，端口单区间故以存在性+代价） | [报告] |
| B-08 | R | **接受条件不同**。上游 `|offset−initial| < radius·0.9`；我们换成自造"平坦地形"启发式（`cost > 0.97·邻域中位数` 即拒绝，`autosync.py:243-261`） | [实测] |
| B-09 | D | 精搜窗口 ±2 ms vs 上游 ±1 ms（`visual_features.py:31`） —— 已对齐（`d1036a5`）：忠实的逐点搜索细搜 ±1 ms；互相关回退仍 ±2 ms | [实测] |
| B-10 | R | **AKAZE 三个常数不同**。我们 threshold `0.001`、Lowe ratio `0.7`、无上限（`akaze.py:34,78`）；上游 `0.0007`、`0.5`、`maximum_features=200`（`akaze.rs:26,13`）。ratio 0.7 放进大量歧义匹配 —— 已修：threshold 0.0007 / maximum 200 / ratio 0.5（akaze.py:30-32，docstring 记录） | [实测] |
| B-11 | D | **DIS 用错 preset**。我们 `PRESET_MEDIUM`（`opencv_dis.py:41`）；上游 `PRESET_FAST`（`opencv_dis.rs:59`） —— 已修：DISOPTICAL_FLOW_PRESET_FAST（opencv_dis.py:43） | [实测] |
| B-12 | R | **`per_frame_time_offsets` 全链路无消费者**。`file_metadata.py:59` 声明、`:90` 置空、全仓库无读取点。上游 `frame_transform.rs:216` 在查四元数前加此偏移 | [实测] |
| B-13 | R | `filter_of_lines`（30° 平均角过滤）+ `get_of_lines_for_timestamp`（按帧距取缓存 OF）缺失 —— 已补（`b96c4bd`）：30° 严格门 + 朴素圆均值（正交多数全灭的上游怪癖钉住）、2 ms 就近 + next_no 跳帧；num_frames>1 显式 NotImplementedError（多帧距缓存是 B-14 剩余） | [报告] |
| B-14 | R | `cache_optical_flow` / 多帧距 OF / `cleanup` / `processed_frames` / `get_ranges`（>100ms 断档切分）缺失。`FrameResult` 只有 d=1 的点对 —— `get_ranges` 已补（`2f916c4`，>100 ms 断档切分 + 上游孤立帧空列表怪癖钉住）；多帧距 OF 缓存族仍缺 | [报告] |
| B-15 | R | **rank<13 质量门缺失**。上游 `render_queue.rs:1485-1504` 按 `sync_data.rank` 跳过低运动段的偏移；我们完全不用 rank，改用 `_valid_sync_points`（>40ms 偏差剔除）与 `_drift_significant`（≥3 点、斜率>15ms、残差<10ms） | [报告] |
| B-16 | R | **`SyncParams` 结构缺失**：`custom_sync_pattern`/`auto_sync_points`/`every_nth_frame`/`calc_initial_fast`/`initial_offset_inv` 全无。且 `manager.py:673-674` 在 OptimSync 给 <2 点时直接放弃，上游 `render_queue.rs:1448-1462` 会退化成均匀分布 —— calc_initial_fast/initial_offset_inv 已补（`ed114ad`）；OptimSync 空点回退均匀分布与兼容设置 sync_settings 叠加已补（`126f7db`）；`custom_sync_pattern` 图案本体与 `auto_sync_points` 开关仍缺 | [报告] |
| B-17 | D | **逐点搜索窗口 120ms vs 上游 5000ms**（`manager.py:_SYNC_POINT_SEARCH_MS`） —— profile 的 search_size 覆盖已接（`8614034`，秒→ms 按 render_queue.rs:1470）；默认保持 120 ms 并记录原因（串行 Python 代价 ~40x，逐点各跑一次搜索） | [报告] |
| B-18 | D | 流式 API 缺失：无 `feed_frame`、无取消、无增量估计、无 OF 缓存生命周期。批处理 `run()` 功能上够用，但长视频无进度/无中断 | [报告] |
| B-19 | D | `guess_imu_orientation` 过程不同：我们用 6 个三帧窗口 + `_compute_cost`（`manager.py:852-964`）；上游 `rs_sync.rs:190-222` 是 48 个朝向的 `pre_sync` 代价求和 —— 48 个朝向的枚举已在（`_POSSIBLE_ORIENTATIONS` 恰 48 项）；差异仅在窗口选取（6 个三帧窗口 vs 全程）与代价函数（`_compute_cost` vs crate `pre_sync`）。B-04 落地后代价侧可换 crate 的 pre_sync，留作后续 | [报告] |
| B-20 | D | Almeida 的最终取逆方向与上游相反（`almeida.py:200-202` 返回相机旋转，`almeida.rs:34-36` 返回点旋转）。两者相对差一个逆——**所有 Almeida 推导出的欧拉角符号相反**。另 `_CameraK` 只有 K，无畸变/无逐帧内参 —— 已修（`034d033`）：delta 按上游 Camera::delta 独立重写并逐值比对，符号随之一致；_CameraK 持有 ComputeParams | [报告] |
| B-21 | D | 方法 0 配置未复刻：上游用 LMEDS/prob 0.999/threshold 1e-5/maxIters 4000/focal 1e5（`find_essential_mat.rs:34-42`）；我们用 RANSAC/1.0px/真实 K（`eight_point.py:20-22`）。且我们的 method 0 与 method 2 **是同一个函数**（`__init__.py:79-98`），上游不是 —— 已修（`3697a72`）：method 0 → find_essential_mat.py（LMEDS/1e-5/4000/1e5），与 method 2 分离 | [报告] |
| B-22 | D | 单应方法：上游 `find_homography_ext(RANSAC, 0.001, 2000, 0.999)` + 按 **max|t|²** 选分解（`find_homography.rs:38-52`）；我们 `cv2.findHomography(RANSAC, 5.0px)` + **取 `Rs[0]`** —— 已修（`3697a72`）：RANSAC 0.001/2000/0.999 + 单位阵 K 分解 + 取 |t|² 最小（上游 fold 原义） | [报告] |
| B-23 | R | **OptimSync 频谱公式不同**。`optimsync.py:129` 用 `np.abs(np.fft.rfft(chunk))`（模长）；上游 `optimsync.rs:98-101` 是 `zip(cm, cm.rev()).take(n/2).map(a+b).norm()`，即 bin k = `cm[k] + cm[n-1-k]`，对实信号 = `cm[k] + conj(cm[k+1])` —— **相邻两 bin 的带相位求和**。下游阈值（rank<50、450/650、0.1）按折返尺度标定 → 用模长等于换了尺度。**注：本条初稿写作「2·|Re(FFT)|」是错的**，已用 rustfft 参照程序推翻（参照向量 bin2：折返 11.58 vs 模长 0.56） —— 已修：相邻 bin 带相位求和已实现（optimsync.py:132-136 注释记录推导） | [实测] |
| B-24 | R | **OptimSync 缺低运动分支**。上游 `mf_max < 50.0` 时切 `(lf+mf)/penalty`（`optimsync.rs:134-148`）；我们恒走正常公式。慢速平移的能量全在 2Hz 以下，正常公式会把它罚没，结果是选不出任何同步点 | [实测] |
| B-25 | D | OptimSync 返回**已裁**的 rank（`:219`，就地清零于 `:177-189`）；上游返回未裁的 `rank_clone` | [报告] |
| B-26 | D | 多同步点窗口筛选用陀螺时间戳去筛**视频**帧（`manager.py:684`）。偏移量级相对 ±500ms 窗口可忽略，但两套时间线混用。**（子代理报的"键落在错误时间线"不成立——`points_ms` 来自 `OptimusSync(ts_ms,...)`，`ts_ms` 在 `manager.py:664` 明确取自 gyro_data）** | [实测] |
| B-27 | D | 上游同步期 `keyframes.clear()` + `lens_correction_amount = 1.0`（`autosync.rs:86-89`）；我们无强制校正也无关键帧抑制 —— 已修（`034d033`）：_build_sync_compute_params: keyframes 清空 + lens_correction_amount=1.0 | [报告] |
| B-28 | D | Python 侧自加的安全启发式（上游无）：DJI 先验 8ms/窗口 40ms、互相关峰 0.5 门、RS 平坦地形 0.97、≥5 视觉旋转、≥3 RS 轨迹、采样内存上限 | [报告] |

### C. 渲染 / 输出管线

| # | 等级 | 项 | 证据 |
|---|---|---|---|
| C-01 | R | **输出编解码矩阵**。上游 H.264/H.265/AV1/ProRes(6 profile)/DNxHD(8)/CineForm/**EXR 序列**/**PNG 序列**；我们 libx264/libx265/prores_ks(坏)/libaom-av1，另加两个上游没有的（vp9/mpeg4）。**无序列输出**——输入侧做了序列，输出侧仍只有视频 —— **序列输出已实现**（PNG/EXR：`_SEQUENCE_CODECS` + printf 模式校验 + image2 容器 + 音频跳过 + CLI 模式生成，`tests/test_rendering.py` 覆盖）；ProRes 编码已在 G-01 修复。DNxHD/CineForm/AV1 profile 细项仍缺 | [实测] |
| C-02 | R | **逐平面/位深/HDR 管线缺失**。上游按解码器原生格式逐平面处理（NV12/P010/YUV420P10/16、GBRPF32LE、RGB48BE…，`mod.rs:563-651`），按位深设 `pixel_value_limit`；我们 `ffmpeg_processor.py:216` 一律 `to_ndarray("rgb24")` 再编码回 yuv420p。**10/12/16-bit 与浮点输入在稳定化前就被量化到 8bit**

**C-02 勘察记录（2026-09-22，C-03 落地后）**。实现原生平面管线需要过四道关，前两道是环境硬阻塞：

1. **PyAV `to_ndarray` 覆盖缺口**：`yuv444p10le` 直接 `ValueError: not yet supported`（10-bit 旗舰格式！）；4:2:0 的 10/12-bit 同样不支持。绕过须走 `frame.planes[i]` 原始字节缓冲 + line-size/stride 处理 + RGB48BE 等大端格式的字节序重排——这层胶水没有上游对应物可对照验证。
2. **形状不一致**：`to_ndarray()` 对 `yuv444p` 给 `(3,H,W)`（平面在轴 0）、对 `yuv444p16le` 给 `(H,W,3)`（通道在尾轴）、`gbrpf32le` 给 `(H,W,3)` float。每个格式都要实测形状表（已探明：`yuv444p (3,48,64) u8`、`yuv444p16le (48,64,3) u16`、`gbrpf32le (48,64,3) f32`、`rgb48be (48,64,3) u16`、`gray (48,64) u8`）。
3. **色度子采样坐标**：4:2:0/4:2:2 的色度平面是半分辨率。上游 shader 在归一化 uv 空间工作（`wgpu_undistort.wgsl` 只用 plane_index 做 fix_range 的 is_y 判定），坐标缩放藏在 `zero_copy.get_plane_buffer` 的缓冲几何里——移植前必须把这条坐标链读穿，否则色度映射差半像素是赌出来的。
4. **编码器格式协商**：上游输入/输出帧格式由 ffmpeg 管线分别决定（平面表按 (in_frame, out_frame) 成对建），移植的输出流 pix_fmt 是按 codec 选的（`_PIX_FMT_MAP`）；「原生直通」要求两者一致时才可走，否则维持现有转换。

建议路径：先做「平面几何 == 帧几何」的格式（yuv444p 族、GBRP、打包 RGB/AYUV——C-03 的表已列全），4:2:0/4:2:2 等色度坐标链读穿后再进。 | [实测] |
| C-03 | R | **底层缺像素格式类型**。上游 `pixel_formats.rs` 12 种（含 NV12/P010/AYUV16/RGBAf16/BGRA8 通道交换 + Rec709 full→limited 重映射）；我们 `pixel_formats.py:15-55` 只做 dtype 探测。这是 C-02 的根因 | [报告] |
| C-04 | E | **trim ranges 渲染时不生效**。`params.trim_ranges` 只有平滑用；`manager.render` 与 `ffmpeg_processor` 完全忽略 → 永远整片渲染。上游 `mod.rs:194-200,278-280` 定位+时间戳重基+`pad_with_black`+`export_trims_separately` —— 已实现（`11294ef`） | [实测] |
| C-05 | E | **帧率控制 / 变速缺失**。上游回调可设 `repeat_times`/`out_timestamp_us`（`mod.rs:460-479`）+ `fps_scale` VFR；我们的回调只读时间戳，`video_speed`/`fps_scale` 渲染时被忽略 —— 已实现（`6a95077`），两个机制分开：`video_speed` 改帧数、`fps_scale` 只改查询时间戳 | [实测] |
| C-06 | E | **`.gyroflow` 工程文件读写完全没有**。官方工程里 `gyro_source.file_metadata`/`integrated_quaternions`/`smoothed_quaternions`/`adaptive_zoom_fovs`/`synced_imu_timestamps` 全是 base91+压缩 CBOR 大块（实测 753 帧的工程）。仅 `tests/decode_gyroflow_project.py` 有只读解码器。**这是 CLI 不能吃工程文件/preset 的原因** —— 读写层与 CLI 接线都已实现（`1077433`、`e7fb663`）；`file_metadata` 的读与写也已补齐（`ef8801d`），`--export-project` 的 2/3 模式也已实现（`9772a50`）——**此前的判断「还差 `raw_imu` 编码器」本身是错的**：上游从来不写那几个 bincode 字段，非 Simple 分支只写 `file_metadata` 一块，所以它一就位 2/3 就齐了。**另**：CBOR 编码器此前只有「按规范 + nalgebra serde 推导」的验证，已找到真机 `WithProcessedData` 导出（`DJI_20260507160359_0005_D.gyroflow`，25185 样本）并据此修掉一处浮点宽度错误（`b590c4b`）；`file_metadata` 这一唯一的大块此前无参照，现在也可以它为准 | [实测] |
| C-07 | E | **容器旋转元数据没读**（见 G-07）—— 已实现（`3aadd0d`） | [实测] |
| C-08 | R | 无 GPU 解码/编码（上游 `ffmpeg_hw.rs` 412 行 + 各平台 interop）；无 GPU 解码重试阶梯、无像素格式回退 | [报告] |
| C-09 | R | 渲染健壮性：上游写 `.tmp` 再改名、拷贝容器元数据/timecode、清残留 `%Nd` 文件、保持系统唤醒；我们直接写目标路径、无元数据 —— `.tmp` 原子发布 + 容器元数据拷贝已实现（`1304d50`，含扩展名→封装表与未知扩展名的直写回退）；timecode 拷贝、残留 `%Nd` 清理、防休眠仍缺 | [报告] |
| C-10 | R | **`settings.py` 是死代码**。106 行类，**全仓库零引用**。CLI 默认值硬编码在 argparse。上游 settings.json 约 25 个键驱动导出/同步默认值 | [报告] |
| C-11 | E | **CLI 表面积**。上游有 `--export_project`(4 模式)/`--export_metadata`(3)/`--export_stmap`(2)/`-p`/`-s`/`--preset`/`-t`/`-j`/`-d`/`--stdout_progress`/`--watch`/`--version`/`-f`。已补：多类型位置输入分流、`--preset`/`--export-project 1`/`-p`/`-s`/`-t`/`-f`/`--version`（`e7fb663`）。未补：`-j`/`-d`/`--watch`/`--open`/`-b`/`-r`/`--no-gpu-decoding`/`--export_metadata`；`--stdout_progress`（`6a81788`，JSON 行流）与 `--export-stmap`（`6a81788`，1=undistort/2=+redistort，接真实模型导出器）已补 | [报告] |
| C-12 | R | **RenderQueue 是骨架**。上游 1740 行（并行渲染、暂停/取消、队列持久化、preset 批量、渲染前 autosync、缩略图、when_done）；我们 227 行顺序执行，export 类 job 只 `json.dump` options，stmap job 抛 `NotImplementedError` | [报告] |
| C-13 | R | `compute_distort_map` 无畸变模型（见 G-06） —— 已修（`bcc927b`）：整网格喂 undistort_points_with_rolling_shutter（use_fovs=true、全额校正，逐行旋转由点族内部处理），未收敛点归零 | [实测] |
| C-14 | R | **manager setter 面**：上游约 45 个 `set_*`，我们 10 个。功能性缺失：`frame_readout_direction`/`additional_rotation`/`additional_translation`/`zooming_method`/`max_zoom`/`video_speed`/`digital_lens`/背景全套/IMU 变换全套 —— 24 个缺口已补（`b6ba695`：stab_enabled/video_speed/max_zoom/frame_offset/readout_direction/additional 全家/拉伸/光折射/背景全套/horizon_lock/zooming_method/of_method/debug overlay/digital_lens 双 setter 落镜头档案）；IMU 变换全套（imu_bias/rotation/orientation/lpf/mf/acc_rotation）仍缺 | [报告] |
| C-15 | R | **`util.rs` 辅助缺失**：`get_video_metadata`（扩展名白名单）、base91+zlib 编解码（**工程文件陀螺载荷用**）、`MapClosest::get_closest`（100ms 容差就近取）、`merge_json`（镜头库 sync_settings 合并）、`map_coord`。`util.py` 只有 `timestamp_at_frame`/`frame_at_timestamp` —— base91+zlib/MapClosest/map_coord 已在前批落地；`merge_json` 与扩展名白名单补齐（`e2b929f`）；`get_video_metadata` 本体（容器探测）在移植侧由 PyAV open 承担，不再单列 | [实测] |
| C-16 | R | **`LensProfile` 保存/命名/校验缺失**，且 `get_all_matching_profiles`（`lens/profile.py:430-523`）**把 `sync_settings` 覆盖整个丢掉**——其余 20 项覆盖都移植对了 —— sync_settings 叠加已补（`126f7db`：merge_json 键级合并 + custom_sync_pattern 先删后并）；保存/命名/校验仍缺 | [报告] |
| C-17 | D | 镜头库 `_insert`（`lens/database.py:374-381`）不算 crc32 校验和；无收藏/评分/去重；`search` 无"交换长宽比优先级"。别名表（gopro5-13/bmpcc/a7x/session5）逐条一致 | [报告] |

### D. stabilization 核心

| # | 等级 | 项 | 证据 |
|---|---|---|---|
| D-01 | R | **逐时间戳镜头数据路径整体缺失**。上游 `frame_transform.rs:82-155`（70 行）做四件事：按 `lens_positions` 插值变焦内参、按 `lens_params` 覆盖内参与畸变系数并重算 `radial_distortion_limit`、按 `digital_zoom` 缩放、非对称镜头 `invert_asym_lens`。我们 `_get_lens_data_at_timestamp`（`frame_transform.py:190-220`）**连时间戳参数都不用** → **变焦镜头全程用一组静态内参**。（`LensProfile.get_interpolated_profile_at` 数学正确但无逐帧调用点）—— 已实现（`3be3629`）：四件事全部落地（`lens_positions` 插值 / `lens_params` 覆盖内参与畸变系数 + 重算 `radial_distortion_limit` / `digital_zoom` / `invert_asym_lens`），配套移植 `MapClosest`（`util.ClosestMap`）；Sony 解析器逐包读 `0x8005` 焦距喂 `lens_positions`。**留一处有意不改**：`_build_compute_params` 把 `calib_width/height` 设成视频尺寸，标定分辨率缩放仍是 no-op | [实测] |
| D-02 | R | **`additional_rotation` 是死的**。`compute_params.py:107` 定义、`manager.py:1239` 拷贝、`gui/main_window.py:306` 唯一写入方（水平锁滑块 → additional_rotation[2]），但 `recompute_smoothing`（`manager.py:333-384`）**不乘这个旋转**。上游 `gyro_source/mod.rs:611-624` 在平滑/锁定**之前**对每个 org 四元数左乘。**GUI 上那个滑块对平滑结果无效** —— 已修：recompute_smoothing 步骤 1 逐时间戳（关键帧或静态值）乘 additional_rotation | [实测] |
| D-03 | R | **`optimal_fov` 未应用**。`frame_transform.py:302` 注释自认 `# (simplified: not handling lens.optimal_fov here)`；上游 `frame_transform.rs:185-191` 有 `fov *= adj` / `ui_fov /= adj` 分支 —— 已修：frame_transform.py:667-671 fov *= adj / ui_fov /= adj | [实测] |
| D-04 | R | **地平线锁顺序反了且应用两次**（见 G-08） | [实测] |
| D-05 | R | **max-zoom 反馈回路缺失**。上游 `lib.rs:548-605` 最多 5 轮 {夹紧 FOV 上限 → 按阈值 `[0.95,0.9,0.85,0.8]` 逐帧缩放 `smoothing_fov_limit_per_frame` → 重平滑 → 重缩放}。我们 `recompute_adaptive_zoom`（`manager.py:386-404`）只算一次；`ComputeParams` 无 `smoothing_fov_limit_per_frame` 字段，`default_algo.py:423`/`plain.py:141` 用 `getattr(...,{})` 读，**永远是空字典**。另无 `video_speed_affects_zooming_limit` —— 已修：manager._apply_max_zoom_limit：max_zoom_iters 轮反馈 + 阈值 (0.95,0.9,0.85,0.8)，smoothing_fov_limit_per_frame 已有生产者 | [实测] |
| D-06 | R | **`at_timestamp_for_points` / `undistort_points*` 家族整体缺失**。上游 `frame_transform.rs:344-430` + `cpu_undistort.rs:634-803`：逐点按各自行时刻取旋转、逐点 IBIS 位移、mesh 校正、数字镜头、GoPro 数字镜头 0.91/0.81 x 修正。我们只有 `zooming/fov_iterative.py:104-253` 一个简化版；`almeida.py:43-44` 注释自认点**没有**畸变校正。**自动同步与自适应缩放的采样点全程跑在畸变坐标上** —— **家族本身已实现（`c7cbf1f`，参照见 `6b7b84e`）**：`util.map_coord`、`splines.as_catmull_rom`、`ComputeParams` 的 mesh/stab/digital-lens 四字段与 manager 接线、`frame_transform.at_timestamp_for_points` + 三个辅助、`cpu_undistort` 的七个函数全部落地。`fov_iterative` 的调用方也已换过来，那份副本已删除（见下方「D-06 余下」行） | [实测] |
| D-07 | E | **`frame_readout_time` 未按传感器裁切缩放**。上游 `frame_transform.rs:22-36` 乘 `capture_area_size/sensor_size_px`；我们 `frame_transform.py:83-107` 只移植符号逻辑 —— 已实现（`3be3629`）：从 `lens_params` 的 `capture_area_size[1]/sensor_size_px[1]` 取到缩放 | [实测] |
| D-08 | E | **焦距平滑整文件缺失**。上游 `smoothing/focal_length.rs:8-146`（高斯 + 自适应两套）+ `lib.rs:442-513` 编排 + `frame_transform.rs:71-80` `focal_length_fov_compensation`。字段在（`stabilization_params.py:104-106`）但从不填充 —— 已实现（`cb606bb`）：两个滤波器 + 编排 + 补偿全部移植，`zooming.get_checksum` 顺带补齐两个字段。**唯一没做到的是真机端到端**：解析器只填 `lens_positions`，`lens_params` 的写入侧仍是缺口，没有现成素材带逐帧焦距，验证止步于「与上游 Rust 逐位一致 + 合成内参接线」 | [实测] |
| D-09 | R | **自适应缩放丢失全部关键帧支持**。`zooming/__init__.py:52-100` 构造工作副本时**不传 `keyframes`**（默认空 KeyframeManager）也不传 `sync_offsets_adjusted`；上游 `zooming/mod.rs:40` 克隆完整 params。连带 `zoom_dynamic.py:27-66` 只有静态窗口路径（无 `DataPerTimestamp`/`min_rolling_dynamic`/`convolve_dynamic`/逐时间戳 envelope alpha），`fov_iterative.py:308-312` 只用常量 kv —— 两半都已实现：工作副本 `dataclasses.replace` 全量克隆（`538c8af`，keyframes/sync_offsets/逐帧镜头数据随之到位）；`zoom_dynamic` 的关键帧窗口分支（`DataPerTimestamp`/`min_rolling_dynamic`/`convolve_dynamic`/逐样本 envelope alpha）已逐行移植（`cfc6ceb`） | [实测] |
| D-10 | R | **关键帧查询忽略陀螺同步偏移**。`keyframes/manager.py:302-311` `value_at_gyro_timestamp` 直接委托 `value_at_video_timestamp`，无偏移（docstring 自认）；无 `update_gyro`、无 `gyro_offsets`。上游 `keyframes.rs:79,205-208` —— 已实现（`8cf03b8`）：gyro_offsets 镜像 + value_at_gyro_timestamp 加偏移 + manager 三包装 | [报告] |
| D-11 | E | **`camera_diagonal_fovs` 塌缩为单值**。上游 `compute_params.rs:140-155` 变焦镜头逐帧一值；我们 `manager.py:1189-1219` 恒 `[单值]` —— 已实现（`3be3629`）：`ComputeParams.calculate_camera_fovs()`，仅当 `lens_params` 多于一项（标定真在动）才逐帧算，定焦保持单值以免做 `frame_count` 次恒等查找 | [实测] |
| D-12 | R | **`framebuffer_inverted` 时自适应缩放中心未翻转**。上游 `frame_transform.rs:310-312` `adaptive_zoom_center_y *= -1.0` —— 已修：frame_transform.py:848-849 打包 translation2d 前翻转，位置与上游一致 | [报告] |
| D-13 | R | **`input_rotation`/`output_rotation` 完全未处理**。`types/kernel_params.py:63-64` 有字段，**从不设置也从不读取**。上游 `cpu_undistort.rs:483-489,590-596` 按旋转量转 uv 与帧尺寸 —— 已实现（`5e1b3be`）：cpu_undistort 在映射后按上游旋转坐标并取旋转包围盒为采样边界（背景 clamp 保留未旋转边界的怪癖）；at_timestamp 增 input_rotation/output_rotation 可选参数打包（上游从渲染缓冲取，PyAV 交原始帧故默认 0）；output_rotation CPU 侧上游也无消费方，仅透传 | [报告] |
| D-14 | R | **速度斜坡缺失**。`stabilization_params.py:65` 有 `speed_ramped_timestamps` 字段，**无生产者无消费者**。上游 `stabilization_params.rs:230-283` 用于输出时间→源时间映射 —— **经复核改判**：上游 `stabilization_params.rs` 那对方法（`calculate_ramped_timestamps`/`get_source_timestamp_at_ramped_timestamp`）本身是死代码，全仓库零调用方；活的斜坡机器在 `rendering/mod.rs:441-475` 的编码回调里（skip/repeat/重写 out_timestamp_us），移植的 `FrameRateControl`（ffmpeg_processor.py，C-05 之 `6a95077`）就是它的忠实移植（含关键帧速度的可调用 speed）。死方法不移植，字段保持空置并记录 | [实测] |
| D-15 | R | **Sony 标识符哈希序列化不一致**：`camera/identifier.py:313-320` 用 `json.dumps(..., sort_keys=True)`（分隔符 `", "`/`": "`）；上游 `camera_identifier.rs:114-131` 用 `serde_json::json!({...}).to_string()`（紧凑）。实测 `'{"a": 1, "b": "x"}'` vs `'{"a":1,"b":"x"}'` → **CRC32 不同 → 镜头库里 Rust 侧生成的 Sony 标识符永远匹配不上** —— 已修：compact 分隔符 + Rust 参照程序验证（906280fe），见 _sony_distortion_hash docstring | [实测] |
| D-16 | R | `adjust_lens_profile` / `rescale_coeffs` 缺失。前者在加载时把 superview(4:3→×1.3333)/hyperview(8:7→×1.5556) 的标定尺寸与 `lens_model` 改对（`distortion_models/mod.rs:43-47`）；后者是 hugin 半径归一化系数重标定（`k[0] *= s²/(1−k0)³` 等） | [报告] |
| D-17 | R | 关键帧序列化格式不兼容：上游 easing 序列化为**字符串**（`"EaseInOut"`）、`id` 缺失时随机；我们序列化为 **int**（`kf.easing.value`）、`id` 必需（`KeyError`）。**`.gyroflow` 工程无法往返** —— 已实现（`02b9dd0`）：serialize 输出上游 serde 形状（包装 + 变体名字符串 + gyro_offsets + timestamp_scale），deserialize 双向兼容（旧扁平整数格式仍可读、id 缺失随机补、未知 easing 回退 NoEasing 不抛）；工程读写两向接线（`_apply_project_stabilization` 读 `stabilization.keyframes`、save_project 写入），偏移权威在陀螺源、加载后镜像回 keyframes | [报告] |
| D-18 | R | `zooming` 的 `get_checksum` 少两个焦距平滑字段（`zooming/__init__.py:127-161` vs `zooming/mod.rs:72-95`）——修了 D-08 才有意义 | [报告] |
| D-19 | R | `keyframes.clear()` 语义不同：上游 `*self = Self::new()`（含 custom_provider/timestamp_scale/gyro_offsets）；我们只清关键帧，**显式保留** provider 与 scale（`manager.py:413-417`） —— 已修：clear() 连 custom_provider/timestamp_scale/gyro_offsets 一起清 | [报告] |
| D-20 | D | **关键帧时间戳缩放应用两次**（潜伏）。`manager.py:299` `timestamp_us = round(ms*1000*scale)` → `value_at_timestamp` → `:240` `ts_ms = timestamp_us/1000*scale`。上游只应用一次。当前树内无 provider 注册，故潜伏 | [实测] |
| D-21 | D | **FOV 迭代次数 5 vs 上游 4**（`fov_iterative.py:372` `range(5)` vs `fov_iterative.rs:110` `1..5`） —— 已修：range(4) | [报告] |
| D-22 | D | `_get_frames_per_window` 多一个 `max(frames, 3)` 下限（`zoom_dynamic.py:75`），上游无 | [实测] |
| D-23 | D | 帧索引取整：我们用 `int(ts/1000*scaled_fps)` **截断**（`default_algo.py:434`、`plain.py:146`）；上游 `lib.rs:2060` 用 `.round()` —— 已修：default_algo.py:436 / plain.py:146 均 int(round(...)) | [实测] |
| D-24 | D | `camera_identifier` tag 扫描语义不同：上游 GoPro 只看首个含 Default 组的 sample 然后 break、Sony 只看首个、通用分支最多 2 个；我们对每个 tag 独立扫**全部** sample（`identifier.py:208-230`）。通常更鲁棒，偶尔得出不同标识符 | [报告] |
| D-25 | D | `FILL_WITH_BACKGROUND` 快路径 / `source_rect`/`output_rect` 映射缺失。两个 flag 我们从不设置，rect 恒为整帧（`frame_transform.py:472-473`）。当前等价，加子帧渲染会露 | [报告] |
| D-26 | D | 除法保护：我们在 `default_algo.py:461-464,572-621` 加了 `max(...,1e-9)`/`if max_distance>0` 守卫；上游直接除（得 inf/NaN） | [报告] |
| D-27 | D | slerp 近平行阈值：我们在 `dot > 0.9995` 回退 nlerp；上游用 nalgebra（epsilon≈`f64::EPSILON`）。近四元数上 <1e-4 偏差 | [报告] |
| D-28 | D | `zooming_debug_points`（UI 调试多边形）缺失——不影响输出值 | [报告] |
| D-29 | D | 无效 ZoomMethod：上游记日志回退 GaussianFilter；我们 `ZoomMethod(...)` 抛 `ValueError` | [报告] |
| D-30 | D | 校验和算法全局不同（上游 `DefaultHasher` over `to_bits()`，我们 Python `hash()` of tuple）。**输入字段集合逐算法核对一致**，用途相同 | [报告] |

### E. 已核对等价（列出以便读者确认查过）

| # | 项 | 核对方式 |
|---|---|---|
| E-01 | `KernelParams` 320 字节布局逐字段一致；COEFFS 表 488 项 | 程序化比对，**0 处 >1e-6** |
| E-02 | EWA 的 b/c 常数与 p/q 多项式（`ewa.py:38-41,287-294` vs `mod.rs:281-294`） | 逐位一致 |
| E-03 | `StabilizationParams` 45 字段全在、默认值逐项相同 | 逐项比对 |
| E-04 | `_rotation_to_angular_velocity` = `scaled_axis · scaled_fps/every_nth_frame` | 公式一致 |
| E-05 | DefaultAlgo 全部常量/参数 JSON/alpha 函数/双向 EMA/两趟流程 | 逐行一致（差异见 D-05/D-23/D-26） |
| E-06 | HorizonLock 结构/默认值/自动锁滚转率/重力分支/动态倾斜夹紧/插值 | 逐行一致（差异见 G-08/D-04） |
| E-07 | KeyframeType 27 个变体（含颜色/显示文本/格式化器）与 Easing 真值表 | 逐条一致（差异见 D-17/D-20） |
| E-08 | KeyframeManager CRUD / 最近时间戳吸附 ±1000µs / 插值核心 / next-prev | 一致 |
| E-09 | zoom_dynamic 静态高斯路径与静态 Envelope 路径 | 一致（差异见 D-22） |
| E-10 | FovIterative 几何（31×31 边界点、margin 2.0、`interpolate_points`、`nearest_edge` 折叠、最终 `fov = rect.0·2/output_w`、trim 填充） | 逐行一致（差异见 D-09/D-21） |
| E-11 | NoneSmoothing / PlainSmoothing / FixedSmoothing | 一致（差异见 D-23/D-30） |
| E-12 | Smoothing 注册表（算法列表、默认索引 1、clone 往返、校验和组合） | 一致 |
| E-13 | `set_fovs` / `min_fov` 公式 | 一致 |
| E-14 | trim 相关：`get_trimmed_quats` 范围推进/prev-next 播种/终端 (MAX,MAX)；`get_max_angles` 轴映射 | 一致（边界取整见 D-23 类） |
| E-15 | 10 个畸变模型中 9 个的数学模型与常数 | 一致（唯一缺 gopro6_superview） |
| E-16 | PyrLK 常数（200/0.01/10.0/bs3/win21/lvl3/30-0.01/1e-4） | 逐项一致 |
| E-17 | 相机标识符的 GoPro tag（EISA/EISE/VFOV/ZFOV/PRJT）、RED fps=0、hero12/13→11、runcam/caddx 默认值、镜头别名表 | 逐条一致 |
| E-18 | `calculate_fovs` 策略选择（空/静态 `<-0.9`/动态 `>0.0001`/禁用） | 一致 |
| E-19 | GoPro method-0 mounting 旋转（实测恒定 20.5°±0.40°，逐帧差 0.043°） | **实测证明非缺口** |

---

## 修复进展（2026-09-16）

第一、二部分里已落地的项，按提交顺序（P0 全部完成，P1 大部分，P2 部分）：

| 项 | 提交 | 验证 |
|---|---|---|
| G-01 编码异常上抛 + 按 codec 选 pix_fmt | `f2a71d0` | 三种 codec 端到端出文件；用 xyz12le 复现同一 EINVAL 并断言抛 VideoIOError |
| 同上的连带：解码线程正常结束不再丢尾部帧 | `b50cffa` | 1/5/6/7/30/61 帧逐一断言输出帧数 == 输入帧数 |
| G-02 遥测检测改结构性标记 + 首尾有界窗口 | `44eb067` | 77 个素材全量：Sony 29→32、GoPro 24→28、DJI 4→1 |
| G-03 IMU 变换/滤波三连 | `6925eb8` | bias 跨 load 存活、raw_imu 不再被清空、两个滤波器首次真正工作 |
| G-15 Sony 畸变哈希字节一致 | `12f2eb6` | rustfft 同款 Rust 参照程序，双方 CRC 均 `906280fe` |
| G-08 + D-02 + D-04 地平线锁单次施加、顺序对齐、additional_rotation 接线 | `8c1963b` | 锁恰好调用一次（spy）；5° 旋转使校正量移动 sin(2.5°) |
| G-04 GPU 插值核真正到达 shader（含 pipeline 缓存 key） | `cd705a0` | GPU 2 vs 8 锐度比 1.56x；三个核与 CPU 对应项差 ≤2 级 |
| G-05 gyro_export stab 语义 | `8c1963b` | 写出的值等于平滑朝向、不等于校正量 |
| B-10 AKAZE 常数、B-11 DIS preset | `b50cffa` | 常数逐项对照 akaze.rs / opencv_dis.rs |
| B-03 rs-sync `−readout/2` 与 90% 搜索半径门 | `8c1963b` | 5 项算术测试 |
| D-03 optimal_fov、D-12 framebuffer_inverted、B-12 per_frame_time_offsets | `8c1963b` | 逐项断言字段生效且不误伤相邻项 |
| B-23/B-24 OptimSync 折返公式 + 低运动分支 | `f2e05ce` | rustfft 参照逐值比对（<1e-5） |
| D-19/D-20 关键帧 clear 语义与时间缩放 | `f2e05ce` | provider 收到的时间只缩放一次 |
| D-09/D-21/D-22/D-23 缩放与平滑四小项 | `f2e05ce` | 关键帧透传、窗口下限、迭代次数、帧索引取整 |
| A-11 get_checksum 补齐字段集 | `f2e05ce` | 11 项参数逐一断言影响哈希 |
| D-05 max-zoom 反馈回路（此前 max_zoom 完全无效） | `bfb11d9` | max_zoom=110 时限制 0.3946 且 fovs 随之变化；130 不触发 |
| C-07 容器旋转元数据（tkhd 矩阵） | `3aadd0d` | 手工构造 v0/v1 tkhd + 与 `ffmpeg -display_rotation` 三方对齐 |
| C-01 PNG/EXR 图像序列输出 | `3aadd0d` | 30 帧序列进出帧数一致；EXR 用 `gbrpf32le` |
| C-06 `.gyroflow` 工程读写 + base91/zlib/bincode/CBOR 编解码 | `1077433` | 两个官方工程 JSON 层往返「缺失键 [] / 变化键 {}」；base91/bincode 与仓库内独立解码器逐字节一致；**CBOR 已升级为与真机导出逐字节对照（`b590c4b`）**——见下方 C-06 余下行，此前只有「按规范推导」 |
| C-04 渲染时裁剪区间 + 音频同步裁剪 + 逐区间导出 | `11294ef` | 8 组区间组合断言保留帧数与帧内容（帧身份写进画面象限）；输出 pts 相邻差值恒定（无空洞）；回调收到源时间线；带音轨素材裁后音频时长同步变短 |
| C-11/C-06 余下：CLI 吃工程与 preset、`--preset`/`--export-project 1`/`-p`/`-s`/`-t`/`-f`/`--version`、`python -m pygyroflow` | `e7fb663` | subprocess 跑真 CLI：真工程（自带 base91 bincode 陀螺块）→ 读工程 → 从块里取得陀螺 → 出片帧数正确；`--export-project 1` 无运动载荷且 fov 随 preset 变化；缺 `-f` 时拒绝覆盖 |
| G-10 视觉角速度信号：中点时间戳 + 失败帧欧拉角插值 + 可选低通 | `0420488` | 中点用 1 ms 等距夹具断言精确值、30fps 用规则断言；缺口按位置加权插值（中点=均值）且首尾不外推；低通使单帧尖峰幅度降到 60% 以下且时间戳不变 |
| C-05 渲染时变速与 `fps_scale` | `6a95077` | `video_speed` 2.0 出 6/12 帧且留下的正是奇数帧、0.5 出 22 帧、4.0 出 3 帧；`fps_scale=2` 帧数不变且查询时间戳被 spy 证实为 `ts/2`；变速时自动丢弃音轨 |
| D-01 逐时间戳镜头数据（`lens_positions`/`lens_params`/`digital_zoom`/`invert_asym_lens`）+ `MapClosest`、D-11 逐帧 `camera_diagonal_fovs`、D-07 读出时间按裁切缩放 | `3be3629` | 真素材：`a7s3-sony85mm` 450 包里 200 个带 `0x8005`，每个都是 85.0 mm（15015 条 IMU 行）；`rx100-7-ois-only` 无该标签、map 为空。`ClosestMap` 的等距返回 None、严格 `<` 上限、缺席侧 −99999 哨兵逐条断言；`stretch_lens` 门控与主点重置；`lens_params` 多于一项才逐帧算 FOV |
| C-06 余下（前半）：`file_metadata` 的 CBOR 编解码 + `WithProcessedData` 工程的载入路径 + readout 方向解析 | `ef8801d` | ciborium 逐字节对照全部 19 个字段（scratch crate 用真 ciborium 0.2.2 生成，`--check` 可一键复查 fixture 是否同步）；**真机工程 1059160 字节整块逐字节往返**（四元数须以原始分量喂编码器，`Quat64` 构造时会归一化，偏差上界 2.2e-16 已单独断言）；另用真工程验证载入后拿到 25185 个四元数、`lens_profile`、`additional_data` 与 readout 时间/方向 |
| C-06 余下（后半）：`--export-project` 2/3 模式 | `9772a50` | 真机工程：3 模式导出的 `synced_imu_timestamps` 与该工程原本导出**逐值相同**（25185 项），验证的是派生公式本身；2/3 产物都能被没见过原片的新 manager 载入并取回 25185 个四元数；CLI 侧 subprocess 跑真命令行验 2 与 3 |
| A-07 `splines.rs`（CatmullRom + BivariateSpline） | `8bfa9e1` | 上游 Rust 原样跑参照，17 用例 108 个数值逐位相等；顺带查出参照跨编译不可复现（1 ULP），生成脚本改用容差比较并把这条写进 fixture 的 `_provenance` |
| D-08 焦距平滑（两个滤波器 + strength 映射 + fov 补偿 + FOV 缓存键） | `cb606bb` | **与上游 Rust 逐位一致**：上游 `focal_length.rs` 原样拷进临时 crate、只新写 `main()`，22 个用例全 `max_rel = 0.0`；生成脚本复现的驱动脚本与实跑逐字节相同，fixture 明确标注非自生成。接线在恒等陀螺下按 `矩阵[0,0] = fov/相机fx` 的比值断言补偿量本身 |
| D-06（家族本身）`at_timestamp_for_points` + `undistort_points*` 七个函数 | `c7cbf1f` | 45 项测试。级联顺序用「把某一级的输出当下一级的输入、结果不变」断言（数字镜头、mesh），不重推畸变数学——那部分由 `test_distortion_models.py`/`test_distortion_cv2_parity.py` 覆盖。未收敛点的 `(-1000000,-1000000)` 哨兵用 `k1=-0.5`、归一化半径 1.0 触发（实测该点 10 次迭代不收敛），并断言只有失败的那个点进哨兵。变异检查：把 `_partial_correction` 换成恒等函数，`test_zero_correction_redistorts_the_point_back_where_it_started` 立刻失败 |
| **D-06 参照**：`undistort_points` 对上游 Rust 逐值比对 | `6b7b84e` | `cpu_undistort.rs:649-803` 逐字节切出、`distortion_models/` 与 `gyro_source/splines.rs` 整流原样复制，编译后跑 26 用例 304 个坐标值，全部落在 3.7e-7 相对误差内（f32 vs f64 的本底）。**这一步查出上面三条缺陷**，其中两条结构性测试按构造就测不到。参照工程不入库，`generate_undistort_points_reference.py --stage` 现场从只读的 `opensource/gyroflow/` 抽取，所以不存在第二份会漂的副本 |
| CPU 采样器消费 input_rotation（D-13） | `5e1b3be` | 7 项测试。90/180 度坐标映射按手算值比对（绕中心旋转，非角对称——首版测试在此翻过车）、零旋转恒等、fringe clamp 用旋转宽度、背景 clamp 天花板用未旋转宽度（1079 不被压到 1077，怪癖专测）、at_timestamp 打包两向、梯度帧端到端逐像素值 |
| rs-sync 优化器 crate 移植 + 桥接（B-04） | `8ff80fa` + `00b6478` | 25 项模块测试 + 桥接后 e2e。梯度链（运动/延迟）对有限差分 <1e-5；端到端正弦场景恢复 −50 ms、代价 ~1e-6、残差景观真值处单点凹陷（**恒定角速度对该残差退化**——θ(t₁)+θ(t₂) 平移不变，须非线性 pan，记入测试 docstring）；向量化与标量路径 max diff 0.0（手展开公式在 y 分量翻过车，旋转轴对齐时差异项恰好消失——教训在提交信息）；e2e 实测 +191.6/200 ms |
| manager setter 面 24 个缺口（C-14 主体） | `b6ba695` | 17 项测试：逐 setter 落点、分量 setter 组合、背景 float32、horizon >0.01 启用门、数码镜头三态（None 首写默认零、越界忽略）、失效 ID 单调与缩放校验和清零。digital_lens 按 upstream 落镜头档案非 params |
| filter_of_lines + get_of_lines_for_timestamp（B-13） | `b96c4bd` | 13 项测试：方向过滤（含正交多数全灭的朴素均值怪癖——上游语义）、缩放贯通、2 ms 容差两向、next_no 跳帧、filter 开关、多帧距显式未实现（不静默给错基线） |
| PoseEstimator.get_ranges（B-14 部分） | `2f916c4` | 6 项测试：连续一区间、断档二分、恰好 100 ms 不切（严格 >）、孤立帧产生空列表（上游 != 守卫怪癖，注释钉为上游语义）、空/单帧、乱序插入先排序 |
| CLI --export-stmap 与 --stdout-progress（C-11 两块） | `6a81788` | 5 项测试：模式 1/2 产物集合且不渲染、拒绝覆盖、进度 JSON 逐行（fraction 序列 + done 收尾）、无旗标回调为 None。progress_callback 贯通 render→process_frames，帧数估计 = 容器时长×fps（VFR 下钳 1.0，注释写明） |
| 逐点搜索窗的 profile 覆盖（B-17） | `8614034` | 2 项测试：1.5 s → 1500 ms 到达 proc.run；无覆盖回落 120 ms 默认（与上游 5000 的差及原因写进代码注释与表行） |
| OptimSync 均匀回退 + sync_settings 叠加（B-16/C-16 部分） | `126f7db` | 5 项测试：键级合并且原档案不动、设置冲突胜、无块整体采纳、pattern 替换不拼接（上游 393-404 的先删后并语义）、空点回退块中心 1000/9000 ms 且不再早退 |
| STMap distort 图走真实模型（C-13/G-06） | `bcc927b` | 6 项测试：畸变改变图（旧逆矩阵恒等场景给恒等网格——判别性断言）、恒等场景逐像素回自身、主点径向不动、卷帘逐行差异 >0.05 px、哨兵归零不泄漏、尺寸覆盖 |
| 视频输出原子发布 + 元数据拷贝（C-09 部分） | `1304d50` | 5 项测试：close 后改名发布且 .tmp 消失、flush 失败最终名不出现（PyAV 惰性建文件的断言边界写进注释）、未知扩展名警告直写、序列输出无改名路径、title 元数据往返 |
| estimate_rolling_shutter（for_rs 分支） | `be76284` | 4 项机制测试：已知最小值恢复（12.34 ms，粗搜+细搜落点 ±0.02）、半开区间边界（[-33,+32] @30fps，Rust 语义）、空输入、params 私有副本换字段且不改动调用方。场景级对称不敏感的发现见 B-02 行 |
| pose method 2 的 ARRSAC + eight-point 真身 | `5ad7bd1` | 13 项测试。精确场景 0.00000°、30% 外点仍 0.00000°（SPRT 的意义）；残差尺度校准专测（正确 ~1e-16 / 错解 ~2.0，阈值按此标定）；批量与标量残差逐位一致；列主序 reshape 的翻车记录（行主序当转置用 → 2° 偏差）。接线测试：单位射线断言、mild-lens 场景 0.07° |
| calc_initial_fast / initial_offset_inv（B-06/B-07） | `ed114ad` | ±试验与种子窗口各 3 项：符号序列 [50,−50]、\|initial\|>1 门、RS 种子经 full_sync 的 initial_delay 断言（+3000 ms 窗）。附带卫生修复：multi_sync 的 RS 桩从裸赋值改 monkeypatch（曾靠字母序不污染 e2e） |
| offset method 0 角速度匹配（B-01） | `46be830` | 10 项测试。独立合成场景（视觉滞后 40 ms → −40 ±2，双信号等长——首版测试 10 s 视觉对 3 s IMU，过半匹配规则如实判 inf 的教训）；静默信号跳过、窗口边活守卫拒绝（与 method 1 的空洞守卫成对照）、z 轴 100/70 权重可分辨、ceil 查找、过半规则、run 分发与半窗换算 |
| 速度斜坡改判为已覆盖（D-14） | — | 复核：上游活实现是 rendering/mod.rs 的编码回调斜坡，移植侧 FrameRateControl（C-05）已是其忠实移植（含关键帧速度）；stabilization_params 那对方法全仓库零调用，属上游死代码，不移植 |
| 动态缩放关键帧窗口分支（D-09 动态半边） | `cfc6ceb` | 8 项测试。分支切换对输出有实际影响（关键帧近零窗口 → 早期帧不平滑）；video_speed 缩小窗口的方向性（alpha 变大 → 邻点贴近自身值）；**上游怪癖钉住**：Gaussian 分支的帧数与核恒取静态参数，均匀关键帧下与静态逐位相同——修「怪」即偏离 Gyroflow。envelope 正向遍历 alpha 正序配对（`.iter().rev().zip(&alphas)`）专测防错位；静态路径不回归 |
| 关键帧序列化对齐上游（D-17） | `02b9dd0` | 9 项测试。上游形状写出（包装结构、变体名字符串、随机 id）、上游形状读入（缺 id、未知 easing 字符串）、旧扁平整数格式仍可读、timestamp_scale/gyro_offsets 往返、真实 save→load 文件往返（偏移经陀螺源权威路径镜像回）。变异检查：退回整数 easing 写出 → 变体名断言挂 |
| 数字镜头档案加载时修正标定尺寸（D-16） | `89da8a7` | 8 项测试。4:3 加宽（1440→1920）、8:7 加宽、非匹配宽高比只改名不缩放（上游 rename 在 if 外）、gopro6 同行为、默认 no-op、from_json 钩子两向。`rescale_coeffs` 经查上游无调用方（死代码），不移植并记录 |
| **D-06 余下**：`fov_iterative` 换用逐点家族 + 工作副本全量克隆 | `538c8af` | 18 项测试。① spy 断言多边形确实走 `undistort_points_with_rolling_shutter` 且 `use_fovs=False`，并把该函数桩成定值反推收缩循环的闭式结果（八角形半宽 100 → fov = 2·(100/0.5625)/1920），把循环本身钉住；② 逐项断言 IBIS 位移/mesh/数码镜头/光学模型/缩放中心各自**改变** FOV 且方向正确——这五项在旧实现里全无影响，其中 IBIS 用两个幅度验证单调性；③ 关键帧分支逐个参数断言 `_find_fov` 收到的三元组随帧变化，并单列一条「关键帧取值与静态值相同则结果不变」把"查了"和"变了"分开；④ 工作副本按 `is` 断言七个新字段同一对象，并断言估算用的是输入尺寸、fovs 已清空。变异检查：把 undistort 换成只做旋转+畸变的针孔版本 → ② 全挂；把 `nearest_edge` 的宽高比改回从当前矩形推导 → 对应断言挂 |
| HyperView 数码镜头：x 多项式的括号位置 + 去掉自造的收敛保护 | `e229f05` | 括号那条：`x == width/2` 那一列上游原样返回、移植漂 6.7 px，修后两个 HyperView 用例对上 2.4e-7；另加 `digital_lens_hyperview_converging` 用例（原用例现在全 NaN，只钉住失败路径），以及 `sony_distortion`/`insta360_distortion`（表达式最长的两个模型，括号错误的高发区，对上 2.5e-6 / 1.7e-7）。保护那条：整帧量化了「12 步无保护」与「20 步 + 重置」的差别（NaN 9.0% vs 0%，未收敛 52% vs 11%），并新建 `TestTheHyperviewInversion` 把「迭代次数是上游的、且与 WGSL 一致」写成断言 |
| Almeida 姿态估计接上透镜（D-06 余下）+ 鱼眼批处理主点 NaN | `034d033` | 独立重写上游 `Camera::delta` 逐值比对：带畸变/不带畸变各 5 个旋转，误差 < 1e-6。`params=None` 与改动前逐位一致（有断言）。批处理 vs 标量鱼眼逐点比对 300 点含主点，最大相对差 8.5e-10。变异检查：把 `_undistort` 换回「只归一化」，`TestAgainstUpstreamDelta` 5 项全挂 |
| 姿态估计 dispatch 重排 + method 0/3 忠实化 + 三估计器接逐点反畸变 | `3697a72` | 22 项测试（`test_pose_dispatch.py`）。端到端正确性用 **cv2.fisheye 做独立合成源**（同一畸变方程的另一个实现，共享 bug 不可能）：旋转 0.8°/−0.5°/0.3° + 平移的对应点经 dispatch 走 method 0，恢复角误差 < 0.5°（实测 0.0°）；同数据走针孔路径误差 >5 倍。常量逐个钉死：LMEDS/0.999/1e-5/4000、identity K、recoverPose@1e5、findHomography RANSAC 0.001/2000/0.999、分解用单位阵 K、**argmin \|t\|² 选择规则**（桩分解双序验证，钉住上游 fold 读反的坑）。两帧各自时间戳、method 2 像素味、未知 method 回落 Almeida + 日志、NaN 成对丢弃各有专测。过程发现并记录：纯旋转对上游 method 0 也判 0 内点（台架改两层深度 + 滑动）；旧 sync e2e 实际由 RS 路径承载、旧代码逐对旋转为垃圾（现显式钉 pose_method=2） |
| RS 同步轨迹过透镜 + 读出时长规则 | `76de345` | 12 项测试（`test_rs_sync_lens.py`）。反畸变正确性用「直接调 `undistort_points_for_optical_flow` 再手工归一化」做期望值逐值比对（1e-12）——期望值与被测代码只共享最底层已钉死的家族函数，不共享接线。专测钉住两个易错点：逐点时间戳必须用**原始** y（上游 `rs_sync.rs:132-133` 读 `a_p[i].1`，反畸变后的 y 属于另一行曝光）；NaN 点丢弃而非带进 cost。读出时长四条规则各一测（显式值保留 / 0 回落半帧间隔 / scaled_fps 生效 / global_shutter 覆盖**显式值**到 0.01 ms）。管道测试桩掉 `RollingShutterSync` 断言 `compute_params`/`points_dims` 从 `AutosyncProcess` 贯通。变异检查：把 `add_track_from_frames` 的反畸变分支删掉，射线比对与管道断言同挂 |
| offset method 1 换成上游逐点距离搜索 | `d1036a5` | 13 项测试（`test_visual_features_offset.py`）。端到端：scipy `Rotation` 独立生成陀螺流、`cv2.fisheye.distortPoints` 独立渲染素材（滞后 60 ms），种子窗口搜出 −60 ms ±5——符号与 `visual = gyro + offset` 一致，推导写在测试 docstring。quirks 逐个钉死：u64 截断（dx=3.5 → 12.25 记 12）、严格边界、最长 10% 丢弃（含 `(len*0.9)` 截断的单点零贡献）、offsets 快照清理不 mutating 调用方、上游死代码守卫的保留。**量出来的度量性质**（写进测试与 docstring）：恒定角速度使度量退化（帧间旋转均匀则任何 offset 都让映射点重合）；短素材上全局搜索会落旁瓣，生产调用方须用先验 offset 做种子。调试中顺带修掉 `undistort_points_with_rolling_shutter` 对 ndarray 抛 ambiguous-truth 的守卫（`b8ab83c`） |
| 关键帧查询补上陀螺同步偏移（D-10） | `8cf03b8` | 11 项测试。偏移搬运查询时间线（陀螺滞后 250 ms → 查询 1000 ms 读视频曲线 1250 ms 处）、单点常量偏移、update_gyro 整表替换、clear 连清（D-19 语义）、序列化往返、util 与 gyro_source 两条路径逐点一致（由「util 持实现 + gyro_source 委托」构造保证）、manager 三包装贯通到 ComputeParams 持有的同一 keyframes 对象 |
| 补缺模型 `gopro6_superview`（此前静默回落成鱼眼） | `b718511` | 上游 `gopro6_superview.rs` 逐字节副本编译后跑出的 46 个参照值全部对上；容差按实测取（undistort 1.05e-7 / distort 1.14e-6 相对，来源是上游 f32、此处 f64），另有一条 2e-6 的聚合断言防漂移。两个探针钉住「看起来像 bug」的上游行为：`(100,900)` 需要 22 步而上游 12 步封顶（用「多项式被调用几次 == 12」断言，并断言返回值出自 fixture）；`(200,200)` 的原像在画面外约 -57 px（同时断言 undistort 能回来）。另断言两个 superview 在 1440 px 处结果不同（rel 1e-4）——把两者混为一个是这条缺陷的失效模式 |

**实施中新发现的、原清单没有的缺陷**：

- `default_algo`/`plain` 用 `if frame in fov_limit_per_frame:` 读逐帧限制——对 list 是**值成员测试**，浮点限制值永远不匹配，限制实际从未生效（`bfb11d9`）。
- `gpu/backend.py` 的 pipeline 缓存 key 只哈希 shader 源码、不含 pipeline 常量，某个畸变模型首次建出的 kernel 会被整个进程复用（`cd705a0`）。
- 测试夹具 `tests/test_e2e.py` 的合成 GoPro mp4 的 `hdlr` 只有裸 `gpmd`，而真机是 `mhlr`/`meta` + Pascal 串 `GoPro MET`（`12f2eb6`）。
- `zooming.get_checksum` 少哈希两个字段。上游 `zooming/mod.rs:91-92` 把 `focal_length_smoothing_enabled` 和 `_strength` 都算进 FOV 缓存键；我们只哈希了 `distortion_coeffs`/尺寸/`max_zoom`/`trim_ranges`/`video_rotation`/`adaptive_zoom_window`。后果是**改了焦距平滑滑块不会让 FOV 缓存失效**——重渲染静默沿用旧变焦，看起来"滑块没反应"。已补齐（`cb606bb`）。
- **CBOR 浮点宽度写错**。`_cbor_f64` 一律写 8 字节，`ciborium` 却把能在半精度/单精度里精确表示的值写窄。此前只对着 CBOR 规范和 nalgebra 的 serde 推导，没有外部参照，所以没人发现——读侧完全不受影响（cbor2 任何宽度都认），只有拿真字节比才看得出来。在一个真的 `WithProcessedData` 工程上现形：25385 个时间戳差 934 字节。规则按文件反推为「取能精确容下该值的最窄形式，半精度优先于单精度」——反过来的优先级会错 9 个值。已修（`b590c4b`），四个载荷全部逐字节一致。
- **这条顺带说明此前 C-06 的 CBOR 验证不够**。原验证是「按规范和 nalgebra serde 推导后按位固化」，即自证；真文件证明推导在一个细节上（浮点宽度）是错的。现在有了外部参照。

**做 file_metadata 读取时新发现的两处**（都是拿真机工程跑出来的）：

- **`WithProcessedData` 工程读进来是空的**。`_load_project_motion` 只认 `gyro_source` 里的 bincode 块；真机 1.6.3 导出的那四个字段全是 `null`，陀螺在 `file_metadata`（CBOR）与 `integrated_quaternions`（CBOR）里。实测：一个带 25185 个样本的真工程载入后陀螺数 **0**。此前没发现，是因为手工验证用的参考工程四个字段恰好全是 `null`，"看起来是对的"。已按上游的分支顺序补齐（`ef8801d`）。
- **`stabilization.frame_readout_direction` 按整数解析，真机写的是枚举名字**。我们写 `ReadoutDirection(int(...))`，真机导出是 `"TopToBottom"`（serde 对 unit enum 的默认行为）→ 解析失败 → 静默回落到默认方向 → **滚动快门按错方向扣**。上游两条都吃（`as_i64` 再 `as_str`）。同处还漏了上游的隐含规则：`frame_readout_time` 为负表示 BottomToTop，且**保留符号**（我们载入时就 `abs()` 了，符号信息丢失）。已一并修（`ef8801d`）。

**做 C-06 时新发现的缺陷**：

- **工程文件的版本号不是装饰**。上游当前写 `"version": 4`；`import_gyroflow_data` 把 `project_version` 存进 `FileLoadOptions`，`gyro_source/mod.rs:365,447` 用它决定 RED 素材是否要做"旧版陀螺时间戳偏移"。我最初的实现固定写 2，会让真 Gyroflow 去剥一个我们从没加过的偏移。改为写 4，并补齐 v4 才有的字段（`stabilization.frame_offset`/`focal_length_smoothing_*`、`gyro_source.sample_index`/`detected_source`、`video_info.created_at`），否则读了版本号的下游会找不到它该有的字段。
- **两类 blob 的载荷编码不同**，此前 `util.py` 的注释写成笼统的"bincode/cbor"。按 `util.rs`：`gyro_source` 家族的 `quaternions`/`raw_imu`/`image_orientations`/`gravity_vectors` 走 `compress_to_base91`（bincode legacy），`WithProcessedData` 导出的 `integrated_quaternions`/`smoothed_quaternions`/`adaptive_zoom_fovs`/`synced_imu_timestamps*`/`focal_lengths` 走 `compress_to_base91_cbor`（CBOR）。用错编码器不会报错，只会读出垃圾——`gravity_vectors` 每项 32 字节而非 40，`raw_imu` 的行长还是变长的（`Option<[f64;3]>` 带 1 字节 tag）。
- `json.dump` 默认 `allow_nan=True` 会写出裸 `NaN`/`Infinity`，生成一个别的解析器（含 Gyroflow）读不了的文件；已改 `allow_nan=False`，让它在写的时候炸而不是交付一个坏文件。
- `StabilizationParams` 是 numpy 支撑的，`params.fov` 是 `float32`，**`json.dump` 直接拒收**——任何一次真实保存都会 `TypeError`。已在 JSON 边界统一做 `_jsonable` 归一（numpy 标量取 `.item()`，数组转 list），对应上游 serde 把 f32 写成 f64。

**做 C-06 接线（CLI）时又发现的两处，都是我自己那一版 `load_project` 的问题**：

- `load_project` 把 `params.background` 写成了 tuple，而该字段是 `np.ndarray`（`_build_compute_params` 对它调 `.copy()`）。于是"读工程 → `recompute_blocking()`"必炸 `AttributeError: 'tuple' object has no attribute 'copy'`。之前的验证只走到 load 就停了，没往下跑管线，所以没暴露。
- **工程自带的陀螺数据根本没被载入**。`load_project` 只套用了 IMU 变换，没解码 `quaternions`/`raw_imu`/`gravity_vectors`/`image_orientations` 并交给 gyro source。参考工程里这四个字段恰好全是 `null`（它们靠原片遥测），所以最初的手工验证看起来是对的——而 `WithGyroData` 工程存在的全部意义就是脱离原片。修的时候还要注意顺序：`load_from_telemetry` 内部会 `clear()`，把 IMU 变换清掉，所以变换必须在载入数据**之后**套用。

**做 D-06 时新发现的缺陷**：

- **姿态估计的 method 0 跑的是 method 2 的代码，而 `estimate_essential_matrix` 根本没人调**（**已修，`3697a72`**）。上游 `estimate_pose/mod.rs:27-35` 是 `0=PoseFindEssentialMat, 1=PoseAlmeida, 2=PoseEightPoint, 3=PoseFindHomography`，未知值**回落到 Almeida**。本移植 `estimate_pose/__init__.py:45-47` 把 0 映射成 `"essential_matrix"`，但 `estimate_rotation` 里 0 和 2 的**实现是同一个函数** `estimate_pose_eight_point`；`estimate_essential_matrix` 在包外只被 `__init__.py` 的 `__all__` 导出，`estimate_rotation` 从不调用它——即**默认 method（manager 三处都传 `pose_method=0`）跑的是 method 2 的代码**。修后：method 0 走新 `find_essential_mat.py`（上游逐常量：LMEDS、阈值 1e-5、4000 迭代、归一化坐标 + 单位阵 K、三角化 cheirality 的 recoverPose@1e5、内点 <10 判失败）；method 3 走 `estimate_pose_find_homography`（归一化 + 单位阵 K 分解、RANSAC 0.001/2000/0.999、按上游 fold 的**原义**取 |t|² **最小**者——guard 是 `m < dot` 保留原值，读反了会以为取最大）；未知值回落 Almeida + error 日志。**method 2 已换 ARRSAC 真身**（`5ad7bd1`）：arrsac crate v0.11 + rust-cv eight-point 逐文件移植（`estimate_pose/arrsac.py`）——单位射线 + 阈值阶梯 1e-10/1e-8/1e-6、SPRT 序贯拒绝、列主序本征矩阵、三角化 cheirality 残差（向量化，与标量逐位一致）；旧 findEssentialMat 替身保留为针孔回退。量出的边界：强镜头最小样本噪声 → 阈值阶梯全败 → 该帧无姿态——上游同款行为；真实光流噪声（~1e-4 射线误差）下三档阈值本就够不着，method 2 在真实素材上罕用（默认 0）
- **姿态估计的点没做畸变校正**（同上一批的根因；**已修，`3697a72`**，除 method 2 的估计器本身）。上游 `find_essential_mat.rs:25-26`、`eight_point.rs:26-27`、`find_homography.rs:25-26` 都先 `undistort_points_for_optical_flow(pts, ts, params, size)`；移植现于 dispatch 层对 method 0/2/3 统一先反畸变，**两帧各自按自己的时间戳取镜头数据**（`estimate_rotation` 新增 `next_timestamp_ms`，`pose_estimator` 两个调用点传入真实下一帧时间戳与真实帧尺寸——尺寸此前干脆不传，Almeida 靠主点猜）。params 为 None 时退回针孔：原始点 + 真 K（`findEssentialMat` 内部归一化，二者等价）。
- **e2e 合成场景与三角化 cheirality 的相性**（`3697a72` 过程中量出来的）。上游 method 0 的 `recoverPose` 用**三角化**判内点（`recover_pose_triangulated(..., 1e5)`）：纯旋转对（零平移）三角化退化，**0 内点直接失败**——真实手持素材总有平移所以能跑，但本仓库的纯旋转合成台架不行。更扎心的是量出来的历史：旧台架上旧代码的逐对旋转本来就是**垃圾**（±3000 deg/s，平面场景对本质矩阵是退化的，而 `estimate_pose_eight_point` 只查 findEssentialMat 的 mask、从不看 recoverPose 的 cheirality 内点数），旧 sync 测试能过**全靠 RS 搜索路径**（点轨迹 + 四元数，不经过姿态估计）扛着，视觉旋转信号从未真正参与过。现 guess 测试改两层深度 + 滑动相机（method 0 真实通路，恢复 <0.5°），sync 测试显式钉 `pose_method=2` 走 RS（与旧行为的实际承载路径一致），`manager.synchronize` 顺带补上 `pose_method` 旋钮（上游 SyncParams 本有）。

- **HyperView 的 x 多项式，y² 耦合项写在乘法外面**（`e229f05`）。上游 `uv.0 * (P(x²) + y2*-0.1086027)`，移植写成 `uv.0 * (P(x²)) + y2*-0.1086027`——差一个因子 x，所以两者在**竖直中线**上相同、离开就分岔，写成加项时看代码完全正常。是模块自己带的 WGSL 文本（照抄上游的）把它出卖了：同一文件里两处自相矛盾。实测 `x == width/2` 那一列上游返回原列不变，移植漂出 6.7 px（相对 7e-3）。前向 `undistort_point` 同样受影响。
- **HyperView 的逆迭代被移植自造了收敛保护**（`e229f05`）。上游 12 步、无保护；移植 20 步 + 发散重置，连 WGSL 也被改成 20（而 docstring 仍声称与 Gyroflow 的 WGSL 完全一致）。量了整帧（1920x1080，8 px 采样）：上游 NaN 9.0%、跑满 12 步未收敛 52%；移植 NaN 0%、11% 重置回「拉伸过但没求逆」的输入——那个重置的语义其实是**把数码镜头整个跳过**。两种都不好，选择改回上游：移植的目的是与 Gyroflow 一致，自造一个更"稳"的版本会让每次 HyperView 对比都对不上而原因看不出来。**上游这个逆迭代本身是坏的**（多数帧不收敛），记为上游性质而非我们的偏差。
- **Almeida 姿态估计的 `delta` 缺了整段透镜反演**（`034d033`，见下方 D-06 余下行）。原 docstring 的理由「上游把透镜校正是关了的」是错的：`init` 设 `compute_params.lens_correction_amount = 0`，但 `delta` 给 `undistort_points` 的 `lens_correction_amount` 参数传的是**字面量 1.0**，所以它设的字段不决定分支。而且即便混合真跑了，0 的语义是「保留原始观感」= 之后再施加一次前向畸变，不是跳过。
- **鱼眼批处理把正主点算成 NaN**（`034d033`）。`OpenCVFisheyeModel.undistort_points` 的收敛条件多了 `|theta_d| > EPS`，标量版对同一输入返回 (0,0)。只有主点那一个像素会踩到，所以一直没暴露；NaN 对两个消费方（图像路径的镜片校正混合、Almeida 求解器）都意味「这个点失败了」。

- **IBIS 位移分支的旋转顺序写反**（`cpu_undistort.py` 的 `undistort_points`）。上游是顺序赋值：先 `x = cos·x − sin·y + cx`，**再**用这个新的 x 算 `y = sin·x + cos·y + cy`；移植写成 `x, y = …, …`，Python 先求值右侧元组，于是 y 用的是**旧的** x。只有 `sin(角度) == 0`（纯平移）时两者才一致，所以看代码看不出来、看 `almeida` 那类对称测试也看不出来。实测：`shift_per_point` 用例相对误差 1.5e-2，参照容差 1e-5。已修。
- **`math.sqrt` 在负值上抛异常，而上游的 `f64::sqrt` 返回 NaN**。光折射校正把 `sin_theta_d` 除以系数，系数小于 1 时它会超过 1，根号下变负。上游在那里得到 NaN 并继续——消费方把 NaN 坐标当作"没有答案"（NaN 的任意比较都是假，点自动从边缘搜索里掉出去）——**渲染照常出片**；移植则抛 `ValueError`，整个渲染中断。用真工程里合法的系数（0.8）就能触发。已在两处改为返回 NaN 的辅助函数。这两条都是先有参照、后有发现：结构性测试对第一条天然免疫（断言两边共享同一个错误），对第二条也测不到（它压根没进过那条分支）。
- **HyperView 数码镜头的逆迭代，移植比上游多两处**（仍在，未改）：上游 `gopro_hyperview.rs:44` 是 `for _ in 0..12`，移植 `_MAX_ITER = 20`；上游**没有**发散保护，移植有（`|px| > 2.0` 时重置回未求逆的输入）。两处都让移植"更稳"，也都会改变输出——在这条用例上上游全返回 NaN，移植返回有限值，**而且那个有限值是"拉伸过但没求逆"的输入，不是原像**（把它喂回前向映射回不到原点）。这不是本次引入的，是有意的既有设计（`gopro_hyperview.py` 的模块 docstring 写了"guarded by clamping"）。**移除它等于改变所有 HyperView 片子的输出，不是修 bug**，所以留在这里等决策，测试改成把差异**显式断言**出来（`test_the_digital_lens_inversion_is_guarded_where_upstream_is_not`），而不是散在注释里。

- **`gopro6_superview` 这个畸变模型在本移植里根本不存在，却被两处代码当成存在**。上游有独立文件 `distortion_models/gopro6_superview.rs`（自带 `id() -> "gopro6_superview"`，多项式与 `gopro_superview` 不同，走 `1.0 - 0.48*|x|` 那一支）；本移植 `distortion_models/` 下没有它，`_MODEL_REGISTRY`（`distortion_models/__init__.py:57-67`）也没有这一项。而 `_MODEL_REGISTRY.get(name, OpenCVFisheyeModel)` 的兜底是**静默回落到 OpenCVFisheye**——一个完全不同的模型。可 `lens/profile.py` 自己认得这个名字：`_DISTORTION_MODEL_IDS` 里有 `"gopro6_superview": 7`（`profile.py:30`），`get_all_matching_profiles` 的标定尺寸重标定分支也专门把它和 `gopro_superview` 并列（`profile.py:558`）。所以带 `digital_lens: "gopro6_superview"` 的镜头档一路走到 `from_name` 就变成鱼眼模型，不报错、不出警告。这条是从 D-06 的数码镜头分支里牵出来的——`cpu_undistort.py` 的 0.91 修正按上游同时匹配 `"gopro_superview"` 与 `"gopro6_superview"`，但后者在这个移植里永远取不到。**已补（`b718511`）**：`gopro6_superview.py` 逐行移植 + 注册 + 以上游 Rust 原样跑出的参照 fixture。这条也说明「兜底不报错」这个设计在上游和我们这里是一样的，缺模型时**没有任何信号**——`_DISTORTION_MODEL_IDS` 里那个 7 是不是该有独立值，尚无从确证，未动。

**原清单中经复核被推翻的结论**：

- B-23 初稿写"上游 = `2·|Re(FFT)|`"，错。实为 `cm[k] + cm[n-1-k]`（相邻 bin 带相位求和），已用 rustfft 参照程序推翻。
- B-26 初稿说"多同步点键落在错误时间线（视频而非陀螺）"，不成立。已降级为"窗口筛选用错时间线、量级可忽略"。
- D-03 初稿把"FrameTransform 的 fov 与 ui_fov 接反"当成候选缺陷，复核后确认我们的接线与上游一致（`frame_transform.rs:319` 收渲染 fov、`:337` 收 UI fov）。

---

## 第三部分：优先级

**P0 — 半天内可修，影响可观测性或让一个功能从"从未工作"变可用**

| 项 | 动作 | 验证方式 |
|---|---|---|
| G-01 编码异常 | 按 codec 选 pix_fmt + 异常上抛 | 渲染 ProRes 看文件存在 |
| G-02 遥测误判 | 改结构性检测 | **手上 3 个受害素材**（issue-44-12/-23/-30） |
| G-03 IMU 三连 | `clear()` 保留字段 + 滤波改属性访问 | 设 bias 后四元数应变化 |
| G-15 Sony 哈希 | 改分隔符 | 30+ Sony 素材验自动识别 |
| G-08 双锁 | 只在一处施加 | 开地平线锁比对角度 |

**P1 — 小改动、明确正确性收益**

| 项 | 动作 |
|---|---|
| G-04 GPU 插值约定 | 分离 CPU/GPU 索引映射 |
| G-05 gyro_export | 写 `(sm/org)⁻¹` |
| B-11 AKAZE 常数 | 改 3 个数（0.0007/0.5/200） |
| B-11 DIS preset | FAST |
| D-02 `additional_rotation` | 平滑前左乘 |
| D-03 `optimal_fov` | 补 `fov *= adj` |
| B-03 rs-sync `−readout/2` | 加一项 |
| D-12 `framebuffer_inverted` | 补翻转 |
| B-12 `per_frame_time_offsets` | 接进 `frame_transform` |

**P2 — 工程量中到大的功能**

| 项 | 说明 |
|---|---|
| D-06 `at_timestamp_for_points`/`undistort_points*` 逐点家族 | ~~家族本身~~ 已实现（`c7cbf1f`） |
| **D-06 余下** 其余调用方换用逐点家族 | `zooming/fov_iterative.py` **已换**（`538c8af`）：`_undistort_points_simple` 整份删除，改调 `undistort_points_with_rolling_shutter`；工作副本改成全量克隆，逐帧镜头数据与逐帧 stab 数据随之到位。`synchronization/estimate_pose/almeida.py` **已换**（`034d033`）：`_CameraK` 现在持有 `ComputeParams` 与时间戳，`delta` 走镜头反演；`params` 可省（省则退回旧的针孔行为），manager 侧新增 `_build_sync_compute_params()` 喂给 `AutosyncProcess`，贯通 `PoseEstimator` → `estimate_rotation`。`estimate_pose/` 三个 OpenCV 估计器 **已换**（`3697a72`）：dispatch 层统一先 `undistort_points_for_optical_flow`，两帧各按自己的时间戳。`find_offset/rs_sync.py` **已换**（`76de345`）：`add_track_from_frames` 接受 `compute_params` 后走 `undistort_points_for_optical_flow`，读出时长规则（global_shutter 0.01 ms、scaled_fps 回落）也对齐 `FindOffsetsRssync::new`，`autosync._rs_sync_offset` 与 manager 朝向猜测两个调用方贯通。`find_offset/visual_features.py` **已换**（`d1036a5`）：上游 `visual_features.rs::find_offsets` 的逐点距离搜索（点对经 `undistort_points_with_rolling_shutter(pts, ts − offs)` 映射后平方距离和，1 ms 粗搜 + 0.01 ms 细搜）替换掉原来的 1-D 角速度互相关；旧实现保留为 `_correlation_fallback` 供无 `ComputeParams` 的调用方回退。上游 offset method 0（`essential_matrix::find_offsets`）仍不存在于移植，其信号层入口走互相关 shim。**D-06 的调用方接线到此全部完成**；**遗留一项**：offset method 0 的 essential_matrix 搜索**也已移植**（`46be830`）——同步侧的忠实性缺口至此清零 |
| D-08 焦距平滑 | ~~整文件移植~~ 已实现（`cb606bb`），但需要 `lens_params` 写入侧才能上真机 |
| D-05 max-zoom 反馈回路 | 需加字段 + 循环 |
| D-09 自适应缩放关键帧 | 传 params 即可恢复大半 |
| B-01/B-02 offset method 0/1 | 两套算法 |
| B-23 OptimSync 尺度 | **先量新尺度再重标定阈值** |
| C-02/C-03 位深/HDR 管线 | 需先补像素格式类型 |
| C-06 余下 | ~~已完成~~（`ef8801d`、`9772a50`）：读写两向、三种模式、真机逐字节对照；余下的只有 `raw_imu`/`gravity_vectors` 的 bincode **写**侧——上游自己也不写，可继续不做 |
| C-01 序列输出 | 与已完成的序列输入对称 |

---

## 附：验证素材索引

`/home/ft/workspace/testvideos/`（77 文件，18 GB，上游 issue 素材）

- `issue-44-12/-23/-30`：遥测误判的受害素材（G-02）
- `sony-ois-only.MP4`：Sony OIS 靶子，库内无档案、当前零畸变校正（A-06）
- `issue-44-*` 中 30+ 个 Sony 素材含 IBIS/变焦/comp-on-off 配对
- 官方 `.gyroflow` 工程 + 官方稳定化渲染成片（可做逐帧对照）

`/home/ft/workspace/PreReserach/msGyroFlow/DJI_20260507160359_0005_D.gyroflow`（3.4 MB）

真机 Gyroflow 1.6.3 导出的 `WithProcessedData` 工程，25185 个 IMU 样本。**这是仓库里唯一带真实载荷的工程文件**：`testvideos/` 下那两个 `extra-*` 的所有运动载荷都是 `null`（更老的版本写出，靠原片遥测），所以只有它能用来核对编码器写出的字节。已用它修掉 CBOR 浮点宽度错误（`b590c4b`）；后续 2/3 模式导出与 `file_metadata` 路径都应以它为准。
