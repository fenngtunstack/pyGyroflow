# PyGyroFlow 验证方案

> 版本: 0.1.0 | 日期: 2026-05-28

## 1 验证策略

采用四级验证体系：

```
Level 4: 端到端验证 (视频输入 → 视频输出 → PSNR/SSIM 对比)
Level 3: 管线集成测试 (StabilizationManager 完整流程)
Level 2: Rust-Python 黄金对比 (同输入 → 数值一致性)
Level 1: 单元测试 (每个函数/类独立验证)
```

## 2 Level 1: 单元测试

### 2.1 测试矩阵

| 测试文件 | 测试数 | 覆盖模块 | 关键测试点 |
|----------|--------|----------|-----------|
| test_quaternions.py | 28 | Quat64 | 单位四元数、乘法、slerp、逆、欧拉角、旋转矩阵、scaled axis |
| test_imu_integration.py | 15 | 6种积分器 | 空输入、时间戳保留、输出维度、单位旋转 |
| test_smoothing.py | 8 | 4种平滑 | 时间戳保留、平滑度、固定方向 |
| test_distortion_models.py | 22 | 10种模型 | distort→undistort roundtrip、模型名称、基类 |
| test_frame_transform.py | 4 | FrameTransform | 基本变换、无抖动恒等 |
| test_filtering.py | 9 | 低通+中值 | 频率响应、通道滤波、IMU 包装 |
| test_keyframes.py | 15 | KeyframeManager | 插值、easing、序列化、provider |
| test_gyro_source.py | 15 | GyroSource | 加载、变换、积分调度 |
| test_lens_profile.py | 14 | LensProfile+DB | 序列化、搜索、FOV |
| test_stabilization_manager.py | 6 | StabilizationManager | 初始化、参数 |
| test_rendering.py | 3 | FfmpegProcessor | 导入、初始状态 |
| test_zooming.py | 5 | 自适应缩放 | 空输入、禁用、静态、动态 |
| **合计** | **144** | | |

### 2.2 四元数 double-cover 处理

四元数 q 和 -q 表示同一旋转。所有对比测试必须先处理 double-cover：

```python
if np.dot(q1, q2) < 0:
    q2 = -q2
np.testing.assert_allclose(q1, q2, atol=tol)
```

## 3 Level 2: Rust-Python 黄金对比

### 3.1 方法

用相同的合成 IMU 数据分别输入 Rust 和 Python 实现的积分器，对比输出四元数。

```
               imu_data.json (1000 样本, 5s, 200Hz)
                    │
          ┌─────────┴─────────┐
          v                   v
    Rust (nalgebra)      Python (scipy)
    ahrs crate            直接移植
          │                   │
          v                   v
    rust_imu_*.json     Python TimeQuat
          │                   │
          └─────────┬─────────┘
                    v
            数值对比 (tol=1e-6)
```

### 3.2 Rust 黄金数据生成器

位置: `msGyroFlow/crates/msgyro-golden-gen/`

使用和 Gyroflow 相同的外部依赖：
- `nalgebra 0.33` — 线性代数
- `ahrs 0.7` — Mahony/Madgwick 实现

```bash
cargo build --release -p msgyro-golden-gen
./target/release/generate-golden tests/golden/imu_data.json tests/golden/
```

### 3.3 对比结果

| 积分器 | max_err | mean_err | 状态 |
|--------|---------|----------|------|
| simple_gyro | 5.44e-15 | 2.74e-15 | **PASS** |
| simple_gyro_accel | 2.22e-15 | 1.31e-15 | **PASS** |
| mahony | 5.83e-16 | 2.69e-16 | **PASS** |
| madgwick | 9.99e-16 | 5.24e-16 | **PASS** |
| complementary | 4.33e-15 | 2.63e-15 | **PASS** |

全部在 **机器精度级别**（~1e-15），远优于 1e-6 容忍度。

### 3.4 时间戳容差

浮点 `timestamp_ms * 1000.0` 转 int 时 Python 和 Rust 可能有 ±1μs 差异。对比测试使用最近邻匹配（2μs 容差）。

### 3.5 算法移植中发现和修复的问题

| 模块 | 问题 | 修复 |
|------|------|------|
| simple_gyro_accel | 原始代码检查原始加速度模长 0.9-1.1，Rust 先 normalize 再检查 | 统一为先 normalize |
| mahony | nalgebra 内部 [x,y,z,w]，重力公式索引用错 | 修正为正确的 nalgebra 索引 |
| mahony | prev_time 单位混用（ms vs s） | 统一为 Rust 的混合单位方式 |
| madgwick | Jacobian 第 3 行 `2*y` 应为 `2*z` | 修正系数 |
| complementary | Python 用 body-frame correction，Rust 用 world-frame | 重写为 world-frame 方式 |
| complementary | 权重公式不同：Rust 用 `alpha*angle` | 修正为 `alpha × angle` |

## 4 Level 3: 管线集成测试

### 4.1 Python 自比黄金测试

用 Python 生成参考数据，重新运行后对比。

| 测试 | 数量 | 说明 |
|------|------|------|
| IMU 积分自比 | 5 | 5种积分器 × 1000 样本 |
| 平滑自比 | 3 | default_0.3, default_0.7, plain |
| 畸变自比 | 5 | 5种模型 × 5测试点 |
| 帧变换自比 | 1 | 5帧变换矩阵 |
| VQF 确定性 | 1 | 两次运行结果完全相同 |

### 4.2 StabilizationManager 集成测试

```python
mgr = StabilizationManager()
info = mgr.load_video("test.mp4")
mgr.recompute_blocking()
transform = mgr.get_frame_transform(0.0, 0)
assert transform.matrices.shape[1] == 14
```

## 5 Level 4: 端到端验证

### 5.1 视频级对比（待完善）

```
输入: test_video.mp4
    │
    ├── Gyroflow (Rust) ──→ reference_stable.mp4
    │
    └── PyGyroFlow (Python) ──→ python_stable.mp4
                                    │
                            逐帧 PSNR/SSIM 对比
                            容忍度: PSNR > 40dB
```

需要:
1. 安装 Gyroflow 桌面版
2. 用相同参数处理同一视频
3. 用 ffmpeg 提取帧做逐帧对比

### 5.2 GPU-CPU 对比

对同一帧，GPU 输出和 CPU undistort 输出差异 < 2 pixel value（ULP 差异）。

## 6 测试执行

```bash
# 全量测试
cd /home/ft/workspace/PreReserach/msGyroFlow/pygyroflow
python3 -m pytest tests/ -q --tb=short

# 仅 Rust-Python 对比
python3 -m pytest tests/test_rust_golden.py -v -s

# 仅 Python 黄金自比
python3 -m pytest tests/test_golden.py -v

# 重新生成黄金数据
PYTHONPATH=. python3 tests/golden/generate_references.py
```

## 7 测试覆盖率

```
180 tests collected
175 passed, 5 skipped, 0 failed

覆盖模块:
  types/            100%  (Quat64, KernelParams, Enums)
  imu_integration/  100%  (6种积分器 + converter)
  filtering/        100%  (低通 + 中值)
  smoothing/        100%  (5种算法)
  stabilization/    100%  (帧变换 + 畸变模型)
  zooming/          100%  (FOV + 动态)
  lens/             100%  (Profile + Database)
  keyframes/        100%  (Manager + Types)
  gyro_source/      100%  (Source + Metadata)
  manager/          100%  (StabilizationManager)
  rendering/        80%   (FfmpegProcessor 初始化)
  Rust对比          100%  (5种积分器 × Rust黄金数据)
```
