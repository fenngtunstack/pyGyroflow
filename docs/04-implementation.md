# PyGyroFlow 代码实现报告

> 版本: 0.1.0 | 日期: 2026-05-28

## 1 项目规模

| 指标 | 数值 |
|------|------|
| Python 源文件 | 108 |
| Python 源代码行数 | 18,215 |
| 测试文件 | 18 |
| 测试代码行数 | 2,294 |
| 测试用例 | 180 |
| Shader 文件 | 3 (WGSL + 2×SPIR-V) |
| Rust bridge crate | 0 (telemetry_parser_bridge 空壳已移除) |

## 2 模块实现明细

### 2.1 基础类型层 (Layer 0, 546 行)

| 文件 | 行数 | 说明 |
|------|------|------|
| quaternion.py | 196 | Quat64 封装 scipy Rotation，支持 [w,x,y,z] 接口 |
| kernel_params.py | 108 | ctypes.Structure 320B，`from_bytes`/`to_bytes`/`from_dict` |
| time_types.py | 40 | TimeQuat, TimeVec 类型别名, TimeIMU dataclass |
| enums.py | 96 | BackgroundMode, ReadoutDirection (含 is_horizontal/is_inverted), Interpolation, DistortionModelType |
| errors.py | 36 | GyroFlowError 异常层次 |

### 2.2 数据处理层 (Layer 1, 4,415 行)

| 文件 | 行数 | 说明 |
|------|------|------|
| **IMU 积分** (9 文件) | **1,911** | |
| vqf.py | 1,072 | 完整 VQF 离线算法：前向-后向滤波、偏差 EKF、Butterworth、静止检测 |
| simple_gyro.py | 78 | 纯陀螺仪积分，from_scaled_axis |
| simple_gyro_accel.py | 95 | 加速度计重力校正，前 15s 高权重 |
| mahony.py | 186 | PI 控制器，Kp=0.5, Ki=0.0，精确移植 ahrs crate |
| madgwick.py | 200 | 梯度下降法，β=0.02，精确移植 ahrs crate |
| complementary.py | 130 | 世界坐标系叉积修正，settle_time 高增益 |
| base.py | 30 | GyroIntegrator ABC |
| converter.py | 50 | QuaternionConverter 积分方法切换 |
| **滤波** (3 文件) | **277** | |
| lowpass.py | 140 | scipy.signal.butter + filtfilt，IMU 专用包装 |
| median.py | 137 | scipy.signal.medfilt，forward-backward 双向 |
| **镜头** (3 文件) | **1,033** | |
| profile.py | 680 | LensProfile 完整数据模型、JSON/CBOR 序列化、FOV 计算 |
| database.py | 347 | CBOR+Gzip 镜头数据库、搜索、下载 |
| **关键帧** (3 文件) | **650** | |
| types.py | 184 | 27 种 KeyframeType、12 种 Easing 函数、Keyframe dataclass |
| manager.py | 466 | bisect 插值、easing 计算、自定义 provider、序列化 |
| **GyroSource** (4 文件) | **744** | |
| source.py | 507 | GyroSource: raw_imu/quaternions/smoothed、IMU transform、积分调度 |
| file_metadata.py | 130 | FileMetadata: 检测源、IMU 方向、读出时间 |
| imu_transforms.py | 107 | IMUTransforms: 坐标变换、偏差、滤波 |

### 2.3 防抖核心层 (Layer 2, 7,522 行)

| 文件 | 行数 | 说明 |
|------|------|------|
| **平滑** (7 文件) | **1,910** | |
| default_algo.py | 595 | 速度自适应双向平滑，smoothness 0-1 参数化 |
| horizon.py | 471 | 地平线锁定后处理，up 向量偏差修正 |
| plain.py | 95 | 固定时间常数指数平滑 |
| fixed.py | 110 | 固定相机方向（适合三脚架拍摄） |
| none.py | 55 | 无平滑直通 |
| trim.py | 85 | 裁剪范围处理 |
| registry.py | 99 | Smoothing 管理器，算法注册和切换 |
| **帧变换** (stabilization/) | **2,917** | |
| frame_transform.py | 420 | at_timestamp() 核心算法，卷帘快门逐行校正 |
| cpu_undistort.py | 326 | CPU 双线性插值 undistort 回退实现 |
| compute_params.py | 119 | ComputeParams 53 字段 dataclass |
| pixel_formats.py | 62 | 像素格式工具 |
| **畸变模型** (12 文件) | **1,970** | |
| base.py | 80 | DistortionModelBase ABC: distort/undistort/WGSL |
| opencv_fisheye.py | 260 | Kannala-Brandt 模型，Newton-Raphson 逆向 |
| opencv_standard.py | 245 | Brown-Conrady 模型 |
| poly3.py / poly5.py | 180 | 多项式模型 |
| ptlens.py | 175 | PTLens 三次分段模型 |
| insta360.py | 210 | 统一模型 + ξ 参数 |
| sony.py | 240 | 6 系数径向 + 后缩放，radial_distortion_limit 二分搜索 |
| gopro_superview.py | 195 | 迭代多项式，发散保护 |
| gopro_hyperview.py | 210 | 七阶多项式，20 次迭代，发散保护 |
| digital_stretch.py | 75 | 仿射缩放 |
| **自适应缩放** (3 文件) | **695** | |
| fov_iterative.py | 366 | 31×31 边界采样、最多 5 次迭代 FOV 搜索 |
| zoom_dynamic.py | 329 | 高斯加权 / 包络跟随时间平滑 |

### 2.4 集成层 (Layer 3, 4,127 行)

| 文件 | 行数 | 说明 |
|------|------|------|
| **同步** (16 文件) | **2,292** | |
| autosync.py | 260 | 自动同步管线 |
| optimsync.py | 210 | 频域最优同步 |
| pose_estimator.py | 190 | 位姿估计协调器 |
| optical_flow/ | 580 | AKAZE、DIS、PyrLK 三种光流算法 |
| estimate_pose/ | 450 | 八点法、本质矩阵、单应性矩阵 |
| find_offset/ | 602 | 视觉特征 + 卷帘快门感知偏移搜索 |
| **GPU** (5 文件) | **539** | |
| backend.py | 378 | WgpuBackend: 惰性初始化、设备管理、管线创建 |
| shader_builder.py | 95 | WGSL 模板加载 + 畸变模型注入 |
| buffers.py | 66 | BufferDescription, BufferSource |
| **渲染** (5 文件) | **647** | |
| ffmpeg_processor.py | 280 | PyAV 解码/编码、帧回调 |
| render_queue.py | 210 | 批量渲染队列 |
| audio_resampler.py | 157 | PyAV 音频直通/重采样 |
| **Telemetry** (2 文件) | **~170** | |
| parser.py | 170 | GPMF/DJI 纯 Python 解析 |
| (bridge 已移除) | — | 原 telemetry_parser_bridge 为空壳，已删除 |

### 2.5 应用层 (Layer 4, 3,569 行)

| 模块 | 行数 | 说明 |
|------|------|------|
| manager.py | 606 | StabilizationManager 中央协调器 |
| camera/identifier.py | 449 | GoPro/Sony/Insta360/DJI 品牌检测 |
| calibration/calibrator.py | 606 | 棋盘格标定、RANSAC、5 种模型 |
| stmap/exporter.py | 517 | ST-Map 生成 + EXR/NPZ/PNG16 导出 |
| gui/ (7 文件) | 963 | PySide6 主窗口、视频控件、时间线、设置面板 |
| cli/main.py | 141 | argparse CLI |

## 3 与 Gyroflow Rust 代码的对应关系

| Python 模块 | Rust 源文件 | Rust 行数 |
|---|---|---|
| manager.py | src/core/lib.rs | 2,232 |
| frame_transform.py | src/core/stabilization/frame_transform.rs | 783 |
| gpu/backend.py | src/core/gpu/wgpu.rs | 714 |
| default_algo.py | src/core/smoothing/default_algo.rs | 782 |
| vqf.py | src/core/imu_integration/vqf.rs | 1,485 |
| gyro_source/source.py | src/core/gyro_source/mod.rs | 1,223 |
| lens/profile.py | src/core/lens_profile.rs | 665 |
| cpu_undistort.py | src/core/stabilization/cpu_undistort.rs | 1,116 |
| autosync.py | src/core/synchronization/autosync.rs | 665 |

## 4 构建和安装

```bash
# 安装 Python 包 (开发模式)
pip install -e .

# 运行测试
pytest tests/ -q

# CLI 使用
python -m pygyroflow.cli.main input.mp4 -o output.mp4 --smoothness 0.5
```
