# pyGyroFlow (Python) 项目规范

> 本文件管辖 `pygyroflow/` Python 包。父目录 `msGyroFlow/CLAUDE.md` 管辖 Rust workspace，两者互不覆盖。

## 项目定位

pyGyroFlow 是 Gyroflow（Rust）核心防抖算法的 **Python 完整移植**（约 18,925 行、107 个源文件），面向算法验证、Jupyter 交互、嵌入式端侧推理场景。

诚实约束（这些是已知事实，不要在文档/对外宣称里夸大）：
- **数值一致性**：5 个 IMU 积分器与 Rust 移植版（`msgyro-imu-integration`）数值一致（max_err < 1e-14）；**VQF 暂无 Rust 对照**。golden 基准来自 Rust 移植版，**未与上游 Gyroflow 逐帧/逐像素验证**。
- **GPU 路径**：wgpu undistort 的 uint8/RGB 路径**已知损坏**（pack/unpack 契约错误），默认关闭（`use_gpu=False`），仅作实验性 opt-in。
- **STMap 导入损坏**：`stmap/exporter.py` 导入的 `_rotate_and_distort` 已被重构改名（`_vectorized_rotate_distort`），整个 `pygyroflow.stmap` 模块当前不可导入（smoke 测试 xfail 跟踪）。
- **测试覆盖**：核心算法模块有测试；`synchronization`/`telemetry`/`calibration`/`cli`/`gui` 已加 smoke 测试，但功能测试仍薄。

## ⚠️ 工作区 inode 损坏（重要）

本工作区部分文件的 inode/stat 元数据损坏（`stat` 报告的文件大小与实际内容不符）。后果：
- `wc -l`、`stat`、`ruff`、`mypy` 等按 stat 大小读取的工具会读到越界二进制垃圾，**行数严重偏低、报 UTF-8 错误**。
- `git show HEAD:<file>`、`awk`、Python `open()` 按 EOF 读，**拿到正确内容**。
- 测行数/跑 lint 必须用 `git show HEAD:<file> | wc -l`，不要直接 `wc -l`。
- 在干净的 CI clone 里不会出现此问题（git checkout 出来的 inode 正常）。
- 这是工作区文件系统层面的脏状态，不在版本控制内，git 仓库内容本身是干净的。

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
