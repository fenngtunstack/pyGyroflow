# pyGyroFlow (Python) 项目规范

> 本文件管辖 `pygyroflow/` Python 包。父目录 `msGyroFlow/CLAUDE.md` 管辖 Rust workspace，两者互不覆盖。

## 项目定位

pyGyroFlow 是 Gyroflow（Rust）核心防抖算法的 **Python 完整移植**（约 18,925 行、107 个源文件），面向算法验证、Jupyter 交互、嵌入式端侧推理场景。

诚实约束（这些是已知事实，不要在文档/对外宣称里夸大）：
- **数值一致性**：5 个 IMU 积分器与 Rust 移植版（`msgyro-imu-integration`）数值一致（max_err < 1e-14）；**VQF 暂无 Rust 对照**。golden 基准来自 Rust 移植版，**未与上游 Gyroflow 逐帧/逐像素验证**。
- **GPU 路径**：wgpu undistort 的 uint8/RGB 路径**已修复并验证**（cf41506）——identity 与非零鱼眼系数下与 CPU bilinear **逐位一致**（`test_gpu_undistort.py`，2026-09-15 复测 max diff=0）。默认仍 CPU（`use_gpu=False`），`--gpu` opt-in。加速比依赖 Vulkan 实现：本机 lavapipe 软件渲染实测 ~2.2x（1280x1120，465→207 ms/帧），代码注释中 ~6.5x 为真硬件（Intel UHD 630）数字，lavapipe 环境复现不了。
- **STMap**：此前"导入损坏"的记录已过时——模块当前可导入、`STMapExporter` 可构造、smoke 测试全部通过（原 xfail 已消除）。功能级（实际导出 stmap 文件）验证仍薄。
- **测试覆盖**：核心算法模块有测试；`synchronization`/`telemetry`/`cli`/`gui` 已加 smoke 测试，但功能测试仍薄。
- **镜头标定（calibration）**：`LensCalibrator` 有合成基准的功能测试（`tests/test_calibration.py`：渲染已知鱼眼畸变的棋盘→检测→标定→在未参与标定的位姿上比对重投影，实测最大 0.60 px）。**只支持 `opencv_fisheye`/`poly3`/`poly5`/`ptlens` 四种模型**——`opencv_standard` 被显式拒绝，因为渲染端的 12 参数布局是 Gyroflow 自己的前向/反向拆分，与 OpenCV 有理模型的系数顺序从第 5 个起就不一致，直接传会渲染错误。`digital_lens` 也被拒绝（上游会先把角点过一遍数码镜头再求解，这步没实现）。poly3/poly5/ptlens 的系数由鱼眼曲线最小二乘拟合而来，残差写在 profile 的 `distortion_model_fit_error` 里（poly3 在宽视场上可达 180%，profile 仍会写但会告警）。

## ⚠️ 工作区 inode 损坏（重要）

本工作区部分文件的 inode/stat 元数据损坏（`stat` 报告的文件大小与实际内容不符）。后果：
- `wc -l`、`stat`、`ruff`、`mypy` 等按 stat 大小读取的工具会读到越界二进制垃圾，**行数严重偏低、报 UTF-8 错误**。
- `git show HEAD:<file>`、`awk`、Python `open()` 按 EOF 读，**拿到正确内容**。
- 测行数/跑 lint 必须用 `git show HEAD:<file> | wc -l`，不要直接 `wc -l`。
- 在干净的 CI clone 里不会出现此问题（git checkout 出来的 inode 正常）。
- 这是工作区文件系统层面的脏状态，不在版本控制内，git 仓库内容本身是干净的。

**2026-09-15 升级：页缓存级内容腐化**。上述 stat 损坏在重 I/O 负载（全天 4K 渲染）后升级为跨文件、读法相关的内容不一致：
- 同一文件 `grep`（整块读）、`sed`（行读）、Python `read()` 可给出**三种不同内容**；连 Python 解释器加载模块都会编译到丢行的坏版本（症状：幽灵 NameError / ImportError）。
- dmesg 无 I/O 错误——是缓存页损坏，不是盘坏。此时**没有单一可靠的读法**，不要试图"找到对的读法继续打补丁"。
- 自救（按有效性排序）：① `git fetch origin` 重新拉对象 + `git checkout origin/main -- <file>` 还原——网络重传字节是唯一可信基线；② `os.posix_fadvise(fd,0,0,POSIX_FADV_DONTNEED)` 逐文件驱逐缓存页（部分有效）；③ 改文件后必须三读一致才算数（sed 逐行 + Python 整读 + `py_compile`），并删对应 `__pycache__`。
- 警惕误建的近似文件名（如 `test_gpu_undisort.py` vs `test_gpu_undistort.py`）——pytest 会一起收集，同名测试类互相污染状态，产生"全量挂、单跑过"的假象。

## 目录约定

```
pygyroflow/                 # Python 包源码（按分层组织）
  types/                    # Layer 0：四元数、kernel_params、枚举、错误
  imu_integration/          # Layer 1：6 个积分器（含 VQF）
  filtering/ lens/ gyro_source/ keyframes/ camera/  # Layer 1
  stabilization/ smoothing/ zooming/ stmap/          # Layer 2（防抖核心）
  synchronization/ gpu/ rendering/ telemetry/ calibration/  # Layer 3
  gui/ cli/                 # Layer 4
  manager.py                # 顶层编排器 StabilizationManager
tests/                      # pytest 测试
  golden/                   # 黄金数据（含 generate_references.py 生成器）
docs/                       # 0X-*.md 主文档 + gzh-*.md 公众号版（与 0X 同步）
```

分层依赖约束：Layer N 仅依赖 Layer 0..N-1，禁止反向/循环依赖。

## 开发规则

- 改完跑验证（见下方命令），不要只改不验。
- 新增公共 API 加 `#[cfg(test)]` 等价的 pytest 测试。
- **不修改** `opensource/gyroflow/`（父目录的上游只读参考）。
- 不为了让代码跑起来注释掉报错或加绕过标记，找根本原因。
- 文档里的统计数字（行数、测试数）必须实测：行数用 `git show HEAD:<file> | wc -l`（**不要直接 `wc -l`**，因工作区 inode 损坏会失真，见上），测试数用 `pytest --collect-only`。

## golden 数据机制（重要）

`tests/golden/` 的对照数据由 `tests/golden/generate_references.py` 生成。修改 `frame_transform.py`、积分器、平滑算法后，若语义变化要重新生成对应 golden 并同步测试容差。

- `rust_imu_*.json`：Rust 移植版（`msgyro-golden-gen`）产出，**不在本仓库**（在父 Rust workspace，未纳入版本控制）。本仓库只持有 JSON 快照。
- `frame_transform.json`：本仓库 Python 代码自生成（`generate_references.py`），曾经陈旧导致测试 FAIL，已修复。
- 单纯"自生成自比对"是循环验证，**不能**作为唯一正确性证据。算法正确性还要靠：① 与 Rust 移植版对照（5 个积分器）；② 四元数约定三方对照（`test_quaternion_convention.py`）；③ 上游 Gyroflow 源码逐行核对（未自动化）。

## 验证命令

```bash
# 单模块测试
python -m pytest tests/test_<module>.py -v
# 全量（含 xfail GPU 测试）
python -m pytest tests/ -q
# 重新生成 golden（语义变更后）
python tests/golden/generate_references.py
# 类型检查
mypy pygyroflow
# Lint
ruff check pygyroflow tests
# CLI
python -m pygyroflow input.mp4 -o output.mp4 --smoothness 0.5
```

## 命名规范

- 模块/文件：snake_case
- 公共类：PascalCase（如 `StabilizationManager`、`Quat64`）
- 积分器类：`<Method>Integrator`（如 `VQFIntegrator`、`SimpleGyroIntegrator`）
- Rust 移植版类名是 `AutosyncProcess`（不是 `AutoSync`）—— GUI 曾因误用 `AutoSync` 崩溃，勿重蹈。

## 依赖原则

- 核心算法依赖：numpy、scipy（数值）、wgpu（GPU）、av（视频 IO）、opencv-python、cbor2、PySide6、tqdm。
- Python >= 3.11（pyproject 要求）。
- 不引入 numba/cython 等额外编译依赖（性能优化用纯 numpy/scipy 向量化）。
- dev 依赖：pytest、pytest-cov、pytest-xdist、mypy、ruff。
