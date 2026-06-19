# PyGyroFlow 代码实现报告

> 版本: 0.1.0 | 日期: 2026-05-28

## 1 项目规模

> 行数用 `git show HEAD:<file> | wc -l` 测量（本工作区部分文件 inode stat 元数据损坏，`wc -l`/`stat`/`ruff` 直接读取会失真，须以 git 内容为准；详见 pygyroflow/CLAUDE.md 已知问题）。

| 指标 | 数值 |
|------|------|
| Python 源文件 | 107 |
| Python 源代码行数 | ~18,925（`pygyroflow/` 包） |
| 测试文件 | 16 |
| 测试代码行数 | ~2,544 |
| 测试用例（pytest 收集） | 198 |
| Shader 文件 | 3 (WGSL + 2×SPIR-V) |
| Rust bridge crate | 0（telemetry_parser_bridge 空壳已移除） |

## 2 模块实现明细

> 各模块行数为 `git show | wc -l` 实测（含 `__init__.py`）。分层依据见 [02-概要设计](02-architecture.md)。

| 层 | 模块 | 行数 | 关键文件 |
|----|------|------|----------|
| Layer 0 类型 | `types/` | 542 | quaternion.py (Quat64)、kernel_params.py (ctypes 320B) |
| Layer 1 数据 | `imu_integration/` | 1,911 | vqf.py (1,072, VQF 离线)、simple/mahony/madgwick/complementary |
| | `filtering/` | 277 | lowpass.py、median.py |
| | `gyro_source/` | 744 | source.py、file_metadata.py、imu_transforms.py |
| | `lens/` | 1,033 | profile.py (LensProfile)、database.py (CBOR) |
| | `keyframes/` | 650 | manager.py、types.py |
| | `camera/` | 454 | identifier.py (品牌检测) |
| Layer 2 核心 | `stabilization/` | 2,836 | frame_transform.py、cpu_undistort.py、12 个畸变模型 |
| | `smoothing/` | 1,910 | default_algo.py、horizon.py、plain/fixed/none/trim |
| | `zooming/` | 695 | 自适应 FOV |
| | `stmap/` | 522 | exporter.py (ST-Map，导入损坏见已知限制) |
| Layer 3 集成 | `synchronization/` | 2,292 | rs_sync.py、pose_estimator.py、光流/位姿估计 |
| | `gpu/` | 555 | backend.py (WgpuBackend，uint8 路径已知损坏，默认关闭) |
| | `rendering/` | 665 | ffmpeg_processor.py、render_queue.py |
| | `telemetry/` | 933 | parser.py (纯 Python GPMF/DJI) |
| | `calibration/` | 611 | calibrator.py |
| Layer 4 应用 | `gui/` | 958 | PySide6 主窗口/时间线/设置面板 |
| | `cli/` | 153 | argparse CLI；manager.py (636) 在包顶层 |

## 3 与 Gyroflow Rust 代码的对应关系

> 列 Python 模块到上游 Gyroflow 源文件的对应。Rust 行数未列（如需，以 `opensource/gyroflow/` 实测为准）。

| Python 模块 | 上游 Rust 源文件 |
|---|---|
| manager.py | src/core/lib.rs (StabilizationManager) |
| frame_transform.py | src/core/stabilization/frame_transform.rs |
| gpu/backend.py | src/core/gpu/wgpu.rs |
| default_algo.py | src/core/smoothing/default_algo.rs |
| vqf.py | src/core/imu_integration/vqf.rs |
| gyro_source/source.py | src/core/gyro_source/mod.rs |
| lens/profile.py | src/core/lens_profile.rs |
| cpu_undistort.py | src/core/stabilization/cpu_undistort.rs |
| synchronization/autosync.py | src/core/synchronization/autosync.rs |

## 4 构建和安装

```bash
# 安装 Python 包 (开发模式)
pip install -e .

# 运行测试
pytest tests/ -q

# CLI 使用
python -m pygyroflow.cli.main input.mp4 -o output.mp4 --smoothness 0.5
```
