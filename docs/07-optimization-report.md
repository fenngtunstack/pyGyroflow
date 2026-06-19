# PyGyroFlow 优化成果报告

> 日期: 2026-06-19 | 范围: 多轮审计、正确性修复、bit-exact 真值对照、性能向量化
> 行数测量约定: 本工作区部分文件 inode stat 损坏，`wc -l` 失真，本文行数均用 `git show HEAD:<file> | wc -l` 取得。

## 0 结论先行

这一轮工作让 pyGyroFlow 从"对外宣称 production-grade、bit-exact，实则测试坏、GPU 损坏、bit-exact 是两份手抄互校"的状态，变成**核心算法可验证、缺陷诚实标注**的状态。但**不能宣称"效果最优"**——核心算法（frame_transform + 4/6 积分器）可信，complementary/VQF 与真实算法有偏差，端到端真实视频从未验证过。

三个维度的真实边界：

| 维度 | 状态 |
|------|------|
| 数值正确性 | frame_transform + 4/6 积分器机器精度一致；complementary/VQF 偏差 0.71/0.34（xfail） |
| 完整性 | GPU 路径损坏（默认关闭）；CPU 渲染管线从未端到端验证过真实视频 |
| 性能 | default_algo 7.2×、卷帘快门 1.75×（算法层已优化，无端到端基准） |

**测试**: `201 passed, 5 skipped, 3 xfailed`（开工前为 `1 failed, 174 passed`）。

---

## 1 修复的正确性与诚信问题

### 1.1 测试套件曾长期失败，文档谎报全绿

开工前干净仓库实跑 `1 failed`（`test_frame_transform_matches_golden`，frame 15 相对误差 2.0），文档却写"175 通过 0 失败"。

**根因调查**：构造非交换旋转对，用三方独立来源（scipy 矩阵复合 / scipy `Rotation.__mul__` / pyGyroFlow `Quat64`）对照 `Quat64.__mul__`，确认四元数乘法约定**正确**（<3e-16）。合成顺序与上游 Gyroflow `frame_transform.rs:247-249` 完全一致。真正根因是 **golden 文件陈旧**——用旧版代码生成、从未跟随更新。

**修复**：重新生成 `frame_transform.json`；收紧测试容差 `1e-5 → 5e-6`（实测 max diff 3.1e-6）；新增 `tests/test_quaternion_convention.py`（13 个 case 固化三方约定，防未来回归）；清理 `quaternion.py` 里自相矛盾的 `__mul__` 注释（"Wait, let's think again"）。

### 1.2 GPU 渲染路径默认开启但产出垃圾（critical）

`gpu/backend.py` 的 uint8 路径：`_pack_to_u32` 把每个 uint8 像素值位重解释为 float32（255 → 3.57e-43 垃圾）；输出 buffer 恒 f32 与 u32 shader 路径冲突。**默认 `use_gpu=True`**，即默认出垃圾，且**无任何 GPU 测试**。

**止损**：`use_gpu` 默认改 False，显式开启时 `warnings.warn`；CLI 从 `--no-gpu`（默认开）改为 `--gpu`（opt-in）；给 `WgpuBackend` 加 `force_fallback` 参数支持 headless 测试。

**部分修复**：uint8 改走 f32 上传（值保持，非位重解释），删 `_pack_to_u32`。`tests/test_gpu_undistort.py` xfail(strict) 跟踪。

**环境阻塞**：本机只有 llvmpipe（纯软件 Vulkan，非合规），完整修复需真实 GPU 逐层调试，本地无法可靠验证。

### 1.3 bit-exact 曾是虚假宣称

原 README 写 "bit-exact algorithms"，但 `msgyro-golden-gen` 手写了 5 个积分器（complementary 自称 "simplified version"），与真实 `msgyro-imu-integration` 零依赖。两份手抄互相对齐 ≠ 与上游一致。测试实际用 `1e-6` 容差（松 8 数量级）。

### 1.4 静默降级 / 假成功

多处路径吞错误报成功：
- `render_queue.process_next`：job 失败仍 `return True` + `progress=1.0`，调用方无法检测 → 改为记录 `failures` 列表。
- `render_queue._process_stmap_job`：写 identity-map 占位假数据 → 改 `raise NotImplementedError`。
- `cli` 镜头加载失败只 warn 继续 → 改非零退出。
- `cli` 无陀螺仪数据静默继续（产出未防抖视频）→ 加显式警告。
- `gui._auto_sync` 导入不存在的 `AutoSync` 类（真实是 `AutosyncProcess`）并吞异常 → 改诚实弹窗"未实现"。

---

## 2 bit-exact 真值对照（核心成果）

让 `msgyro-golden-gen` 委托真实 `msgyro-imu-integration`（生产级 Gyroflow 算法移植），把对照从"两份手抄互校"变成"Python 与独立 Rust 实现互校"。

### 2.1 结果

```
=== Rust vs Python IMU Integration Comparison ===
  simple_gyro       : PASS  max_err=2.33e-15   ← 机器精度一致
  simple_gyro_accel : PASS  max_err=9.99e-16   ← 机器精度一致
  mahony            : PASS  max_err=5.83e-16   ← 机器精度一致
  madgwick          : PASS  max_err=9.99e-16   ← 机器精度一致
  complementary     : FAIL  max_err=7.14e-01   ← xfail(待对齐)
  vqf               : FAIL  max_err=3.38e-01   ← xfail(待对齐)
```

**4/6 积分器真正 bit-exact**。

### 2.2 complementary/VQF 偏差的根因（重要发现）

- **complementary**：Python 实现是早期 golden-gen 手写简化版的镜像（约 8 行），Rust 真实实现是论文 "Keeping a Good Attitude" 的 V1/V2 完整算法（约 600 行）。**算法级不同**，"对齐" = 用论文版重写 Python。
- **VQF**：Python 与 Rust 同源（Laidig VQF），但初始 heading 处理不同（约 24° 偏差）。同算法，需逐行对齐 heading 初始化。

历史版本曾声称"5 个积分器全部机器精度一致"——那是 Python 与手写简化 Rust 副本互校的结果（complementary 恰好两边都是同一份简化实现，故"一致"）。换成独立真实基准后该一致性不成立。

---

## 3 工程基础

### 3.1 CI 与 lint

新增 `.github/workflows/ci.yml`（matrix Python 3.11/3.12：ruff + mypy + pytest）；`pyproject.toml` 加 `[tool.ruff.lint]` 规则集（E,F,W,I,UP，原先只有 line-length）。ruff/mypy 首次运行 `continue-on-error` 容忍存量问题。

### 3.2 测试覆盖

- **VQF 单测**：`TestVQF` 4 个 case（空输入/单位四元数/确定性/旋转累积），填补 196 行最复杂积分器零测试的缺口。
- **smoke 测试**：`tests/test_smoke_untested_modules.py` 覆盖 11 个零测试模块（synchronization/telemetry/stmap/calibration/cli/camera）。**顺带新发现 stmap 导入损坏 bug**（见 3.3）。
- **GPU 回归测试**：`tests/test_gpu_undistort.py` xfail(strict)，`force_fallback_adapter` headless 运行。
- **四元数约定测试**：`tests/test_quaternion_convention.py` 黄金三角（nalgebra/scipy/Quat64）。

### 3.3 顺带修复的隐藏 bug

smoke 测试一跑就暴露 `pygyroflow.stmap` **整个模块不可导入**——`exporter.py` 导入的 `_rotate_and_distort` 在重构时改名 `_vectorized_rotate_distort`，标量版删了，stmap 没跟上。审计从没抓到（因为没人测过 stmap import）。

**修复**：改用向量化版，顺带消除 `compute_undistort_map` 的 O(h×w) Python 双重循环 → 整图 meshgrid 一次调用。

### 3.4 清理

- 删空壳 `telemetry_parser_bridge/`（仅 Cargo.toml 无 src）+ `telemetry/_native.py`（永远 ImportError 的桥接壳）+ 根目录 `parse_gopro.py`/`parse_gopro2.py`（孤儿脚本）。
- `pyproject.toml` 删未用的 `hypothesis` 依赖。
- `imu_integration/__init__.py` 修虚假 docstring（把已实现的 VQF/Complementary 标成"placeholder"）。

### 3.5 文档真实化

- **行数造假修正**：原文档声称 18,215 行，审计一度误报"虚高 7 倍"（实为 `wc` 在本工作区 inode 损坏下失真），最终用 `git show | wc -l` 确认 **18,925 行是真实值**。README/04-implementation/gzh-02/04/06/07 全部按实测重写。
- **bit-exact 措辞**：README 从"bit-exact algorithms"改为可验证的"4/6 积分器对独立 Rust 实现机器精度一致；complementary/VQF 标 xfail 待对齐"。
- 新增 `LICENSE`（GPL-3.0 全文）+ `pygyroflow/CLAUDE.md`（Python 包专属规范，含工作区 inode 损坏已知问题）。

### 3.6 父目录纳入版本控制

`msGyroFlow/` Rust workspace 原本无 git 仓库。`git init` + `.gitignore`（排除 19G target、vendor、opensource 上游克隆、pygyroflow 子仓库、大视频），首次 commit 纳入 142 个 Rust 源文件。

---

## 4 性能向量化

热路径优化全程保持 bit-exact（golden 容差 1e-10、rust golden 通过、三方约定测试通过）。

| 热路径 | baseline | 最终 | 提升 |
|--------|----------|------|------|
| default_algo smoothing（6000 样本） | 9771ms | 1351ms | **7.2×** |
| 卷帘快门 frame_transform（9 帧） | 5657ms | 3239ms | 1.75× |

### 4.1 三步递进

1. **卷帘快门预排序 keys**：`_quat_at_timestamp` 每行调一次、每次 `sorted(keys)` O(N log N)。改为每帧排一次传入。5657ms → 4532ms（~20%）。
2. **Quat64.slerp 改 numpy 公式**：profile 显示 default_algo 90% 时间在 slerp（24000 次），全是 scipy `Slerp` 对象构造开销。改闭式 numpy slerp（<6e-16 vs scipy）。default_algo 9771ms → 3868ms（2.5×）。
3. **default_algo 全 numpy 重写**：slerp 之后剩余开销仍是 scipy `Rotation` 对象构造（from_quat/as_quat/inv）。默认路径全程切到裸 `[x,y,z,w]` 数组，仅在返回边界构造 Quat64。3868ms → 1351ms（再 2.9×，累计 7.2×）。

per_axis 路径保留 Quat64（涉及 euler 分解，向量化复杂且 golden 不测）。

### 4.2 benchmark 基准脚本

`tests/benchmark_hotpaths.py`（手动跑，非 pytest 收集）记录四个热路径的 before/after，供后续优化对照。

---

## 5 遗留项与诚实边界

### 5.1 已知限制（06-test-report 跟踪）

| 项 | 状态 |
|----|------|
| complementary 对齐 | Python 是简化版镜像（8 行）vs Rust 论文 V1/V2（600 行），max_err=0.71，xfail |
| VQF 对齐 | 同源但 heading 初始化差异（~24°），max_err=0.34，xfail |
| GPU 完整修复 | `_pack_to_u32` 已修，compute 管线仍全 0；**环境阻塞**：本机无真实 GPU（仅 llvmppe） |
| 端到端验证 | 从未做真实视频逐帧 PSNR 对比 |

### 5.2 未做的事（按"其他不做"决策）

- VQF 向量化（独立工程）
- 上游 gyroflow-core 第三参照系 harness（边际收益递减，现有独立 Rust 实现已足够）
- numba/cython 等新依赖（性能优化只用纯 numpy/scipy）

### 5.3 "效果最优"无法保证的核心原因

1. **complementary 算法实现错误**（简化版而非论文版）——若用户选 complementary 积分，姿态估计是错的。
2. **端到端真实视频从未跑通**——CPU 渲染管线、telemetry 真实视频解析、cpu_undistort 的除零警告，都没在真实视频上验证过。
3. **GPU 路径损坏**——只能走 CPU，且 CPU 管线未端到端验证。

---

## 6 提交记录

### pygyroflow（14 个 commit，未 push）

```
81dfd96 perf(smoothing): rewrite default_algo default path on raw numpy quaternions
a38fbfd perf(quaternion): vectorize Quat64.slerp with closed-form numpy
ad80142 perf(frame_transform): pre-sort quaternion keys once per frame (RS path)
0586904 docs: note GPU fix is blocked on real-GPU env (host has only llvmppe)
3b13457 test(rust-golden): switch to msgyro-imu-integration reference; add VQF; xfail comp/vqf
0875075 fix(gpu): eliminate _pack_to_u32 bit-reinterpretation; uint8 now uploads as f32
88bfa85 fix(stmap): switch to vectorized _vectorized_rotate_distort; unbreak module
7c87655 ci: add smoke tests, GitHub Actions, ruff rules; correct line-count measurement
01499e0 docs: replace fabricated stats with measured values; add LICENSE and CLAUDE.md
3de8233 fix: surface silent failures across render queue, CLI, and GUI
8c2f3a9 refactor: remove empty telemetry_parser_bridge scaffold and orphan scripts
c05bd12 test(imu): add VQF coverage; fix false 'placeholder' docstring
810ff0a fix(types): verify Quat64 convention; regenerate stale frame_transform golden
0b75bda fix(gpu): default-disable broken GPU undistort path; add xfail regression test
```

### 父仓库 msGyroFlow（2 个 commit）

```
5b922ff refactor(golden-gen): delegate to real msgyro-imu-integration; add VQF
36fb401 Initial commit: msGyroFlow Rust workspace (msgyro-* crates)
```

## 7 验证方式

```bash
# 全量测试（应得 201 passed, 5 skipped, 3 xfailed）
cd pygyroflow && python -m pytest tests/ -q

# 性能基准（手动）
python tests/benchmark_hotpaths.py

# bit-exact 对照（需父目录 Rust workspace）
cd .. && cargo run -p msgyro-golden-gen -- pygyroflow/tests/golden/imu_data.json pygyroflow/tests/golden/
cd pygyroflow && python -m pytest tests/test_rust_golden.py -v
```
