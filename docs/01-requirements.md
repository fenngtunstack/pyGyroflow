# PyGyroFlow 需求规格说明书

> 版本: 0.1.0 | 日期: 2026-05-28 | 状态: 已评审

## 1 项目背景

Gyroflow v1.6.3 是一个开源视频防抖工具，约 69K 行 Rust 代码，广泛用于运动相机的视频后期稳定化处理。项目目标是将 Gyroflow 的核心防抖算法完整移植到 Python，提供一个可编程、可扩展的视频防抖库。

### 1.1 为什么要做 Python 版

- **可集成性**: Python 生态中有大量 AI/ML 工具（PyTorch、TensorRT），Python 版可直接嵌入现有视频处理管线
- **可研究性**: 算法研究人员可以用 Jupyter Notebook 交互式调试防抖参数
- **可定制性**: Python 的动态特性允许运行时替换算法组件，无需重新编译

### 1.2 约束条件

- 算法输出必须和 Gyroflow Rust 版本**数值一致**（机器精度级别）
- 不修改上游 Gyroflow 源码（`opensource/gyroflow/` 为只读参考）
- Python 3.12+

## 2 功能需求

### 2.1 核心功能

| ID | 功能 | 优先级 | 说明 |
|----|------|--------|------|
| F01 | IMU 数据解析 | P0 | 从 GoPro/DJI/Sony 等相机视频中提取陀螺仪和加速度计数据 |
| F02 | IMU 积分 | P0 | 6 种积分算法：SimpleGyro, SimpleGyroAccel, Mahony, Madgwick, Complementary, VQF |
| F03 | 数据滤波 | P0 | Butterworth 低通 + 中值滤波 |
| F04 | 平滑算法 | P0 | 5 种：DefaultAlgo（速度自适应双向）、Plain、Fixed、None、HorizonLock |
| F05 | 畸变校正 | P0 | 10 种畸变模型：OpenCV Fisheye/Standard、Poly3/5、PTLens、Insta360、Sony、GoPro SuperView/HyperView、DigitalStretch |
| F06 | 帧变换 | P0 | 四元数→旋转矩阵、卷帘快门逐行校正、IBIS 参数 |
| F07 | 自适应缩放 | P0 | 31x31 边界采样 FOV 计算 + 动态高斯/包络平滑 |
| F08 | GPU 加速 | P1 | wgpu-py 加载原版 WGSL/SPIR-V shader，compute shader 8x8 workgroup |
| F09 | 视频渲染 | P0 | PyAV 解码/编码、音频直通 |
| F10 | 自动同步 | P1 | 光流特征匹配 + 本质矩阵/单应性矩阵位姿估计 + 时间偏移搜索 |
| F11 | 关键帧 | P1 | 27 种关键帧类型、12 种缓动函数、自定义 provider |
| F12 | 镜头标定 | P1 | 棋盘格自动标定、5 种畸变模型、RANSAC 随机采样 |
| F13 | ST-Map 导出 | P2 | EXR/NPZ/PNG16 格式坐标映射 |
| F14 | 相机识别 | P1 | GoPro/Sony/Insta360/DJI 品牌检测、镜头配置自动匹配 |

### 2.2 接口需求

| ID | 接口 | 说明 |
|----|------|------|
| I01 | Python API | `StabilizationManager` 统一入口，可编程控制完整管线 |
| I02 | CLI | `pygyroflow input.mp4 -o output.mp4 --smoothness 0.5` |
| I03 | GUI | PySide6 桌面应用，视频预览 + 参数调节 + 时间线 |

### 2.3 IMU 积分算法需求

| 算法 | 输入 | 输出 | 特点 |
|------|------|------|------|
| SimpleGyro | gyro | Quat64 | 纯陀螺仪积分，无漂移校正，零延迟 |
| SimpleGyroAccel | gyro + accel | Quat64 | 加速度计重力方向校正 |
| Mahony | gyro + accel | Quat64 | PI 控制器互补滤波，Kp=0.5, Ki=0.0 |
| Madgwick | gyro + accel | Quat64 | 梯度下降法，β=0.02 |
| Complementary | gyro + accel | Quat64 | 高低频互补，初始稳定期高增益 |
| VQF | gyro + accel (+ magn) | Quat64 | 前向-后向离线滤波，tau=40s，最精确 |

### 2.4 畸变模型需求

| 模型 | 数学基础 | 系数数量 |
|------|----------|----------|
| OpenCV Fisheye | Kannala-Brandt | 4 (k1-k4) |
| OpenCV Standard | Brown-Conrady | 14 |
| Poly3 | 三次多项式 | 1 |
| Poly5 | 五次多项式 | 2 |
| PTLens | 三次分段 | 3 (a,b,c) |
| Insta360 | 统一模型 + ξ | 5 |
| Sony | 6 系数径向 + 缩放 | 8 |
| GoPro SuperView | 迭代多项式 | 0 (内置) |
| GoPro HyperView | 七阶多项式 | 0 (内置) |
| Digital Stretch | 仿射缩放 | 2 |

## 3 非功能需求

| ID | 需求 | 目标值 | 验证方法 |
|----|------|--------|----------|
| N01 | 数值一致性 | Python vs Rust 误差 < 1e-6 | Rust 黄金数据对比测试 |
| N02 | 可用性 | 纯 CPU 模式可运行 | 无 GPU 环境下测试 |
| N03 | 代码覆盖 | 核心模块测试覆盖 > 80% | pytest --cov |
| N04 | 包大小 | wheel < 5MB（不含模型数据） | pip wheel 检查 |
| N05 | Python 版本 | 3.12+ | CI 矩阵测试 |

## 4 依赖关系

### 4.1 外部依赖

| 包 | 版本 | 用途 |
|----|------|------|
| numpy | >=2.0 | 矩阵/向量运算 |
| scipy | >=1.14 | 四元数 (Rotation)、信号滤波 (butter/filtfilt) |
| av | >=13.0 | FFmpeg 视频编解码 |
| opencv-python | >=4.10 | 棋盘格检测、光流 |
| wgpu | >=0.18 | GPU compute shader |
| PySide6 | >=6.7 | GUI |
| cbor2 | >=5.6 | 镜头配置文件解析 |
| tqdm | >=4.66 | 进度条 |

### 4.2 可选依赖

| 包 | 用途 |
|----|------|
| telemetry-parser (PyO3) | 高精度 IMU 数据解析 |
| torch | GPU 加速备选方案 |

## 5 数据管线

```
视频文件 (MP4/MOV)
  │
  ├─→ [PyAV 解码] ─→ 逐帧 ndarray
  │
  └─→ [Telemetry 解析] ─→ FileMetadata (IMU 样本)
        │
        v
[GyroSource.load_from_telemetry()]
  ├─ apply_transforms()     # 坐标系变换 (-y, x, z)、偏差、滤波
  └─ integrate(method)      # raw IMU → TimeQuat
        │
        v
[Smoothing.smooth()]        # DefaultAlgo 速度自适应双向平滑
  └─ 可选: HorizonLock 后处理
        │
        v
[calculate_fovs()]          # 自适应缩放
  ├─ FovIterative (31×31 边界采样)
  └─ zoom_dynamic (高斯/包络平滑)
        │
        v
[FrameTransform.at_timestamp()]  # 每帧变换
  ├─ 旋转 = smoothed * org⁻¹
  ├─ 卷帘快门逐行旋转
  └─ KernelParams 构建
        │
        v
[GPU/CPU undistort]         # 像素重映射
        │
        v
[PyAV 编码] ─→ 输出视频
```
