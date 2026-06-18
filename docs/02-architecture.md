# PyGyroFlow 概要设计（HLD）

> 版本: 0.1.0 | 日期: 2026-05-28

## 1 设计目标

将 Gyroflow v1.6.3（~69K 行 Rust）的核心算法管线移植为 Python 库，同时满足：
- **数值等价**: 算法输出与 Rust 版本在机器精度级别一致
- **模块解耦**: 各算法组件可独立替换和测试
- **CPU/GPU 双路径**: 无 GPU 环境可降级到 CPU 运算

## 2 架构分层

> 详见 `docs/images/architecture-overview.drawio`（可用 draw.io 打开）

```
┌─────────────────────────── Layer 4: 应用层 ───────────────────────────┐
│  StabilizationManager (中央协调器)  │  CLI (argparse)  │  GUI (PySide6) │
└─────────────────────────────────────────────────────────────────────────┘
         ↑                    ↑                    ↑
┌─────────────────────────── Layer 3: 集成层 ──────────────────────────┐
│ 同步引擎        │ GPU 管线          │ 渲染引擎      │ Telemetry 解析  │
│ (光流+位姿+RS)  │ (wgpu+WGSL/SPIR-V)│ (PyAV+音频)   │ (GPMF/DJI+PyO3)│
└─────────────────────────────────────────────────────────────────────────┘
         ↑                    ↑                    ↑
┌─────────────────────────── Layer 2: 防抖核心 ────────────────────────┐
│ 平滑算法          │ 畸变模型          │ 帧变换            │ 自适应缩放     │
│ (5种+HorizonLock) │ (10种+WGSL shader)│ (RS逐行+KernelParams)│ (FOV+动态)   │
└─────────────────────────────────────────────────────────────────────────┘
         ↑                    ↑                    ↑
┌─────────────────────────── Layer 1: 数据处理 ────────────────────────┐
│ IMU 积分 (6种)   │ 滤波 (Butter+Med) │ 镜头 (Profile+DB+Cal) │ 关键帧 (27种) │
└─────────────────────────────────────────────────────────────────────────┘
         ↑                    ↑                    ↑
┌─────────────────────────── Layer 0: 基础类型 ────────────────────────┐
│ Quat64 (scipy)   │ KernelParams (ctypes 320B) │ TimeQuat/TimeIMU  │ Enums/Errors │
└─────────────────────────────────────────────────────────────────────────┘
```

### 2.1 依赖规则

- Layer 0 无内部依赖
- Layer N 仅依赖 Layer 0 ~ N-1
- 禁止循环依赖
- 外部依赖通过 workspace 统一版本

## 3 核心组件设计

### 3.1 StabilizationManager（中央协调器）

对应 Gyroflow `src/core/lib.rs` 的 `StabilizationManager` struct。

```python
class StabilizationManager:
    gyro: GyroSource           # IMU 数据源
    lens: LensProfile          # 镜头参数
    smoothing: Smoothing       # 平滑引擎
    params: StabilizationParams # 防抖参数
    keyframes: KeyframeManager  # 关键帧

    def load_video(path) -> dict           # 加载视频 + 解析 telemetry
    def recompute_smoothing()              # 重新计算平滑
    def recompute_adaptive_zoom()          # 重新计算 FOV
    def get_frame_transform(ts, frame)     # 获取单帧变换
    def render(input, output, options)     # 渲染输出
```

状态管理：无 Arc<RwLock>，依赖 Python GIL 保证线程安全。`ComputeParams` 作为快照对象，渲染时无需持有锁。

### 3.2 Quat64（四元数）

封装 `scipy.spatial.transform.Rotation`，匹配 nalgebra `UnitQuaternion<f64>`。

| 操作 | Python | Rust |
|------|--------|------|
| 内部格式 | scipy [x,y,z,w] | nalgebra [w,x,y,z] 内部 |
| 对外格式 | [w,x,y,z] | [w,x,y,z] |
| 乘法 | `Rotation.__mul__` | `UnitQuaternion * UnitQuaternion` |
| 插值 | `RotationSlerp` | `UnitQuaternion::slerp` |
| 矩阵 | `as_matrix()` 3x3 | `to_rotation_matrix()` |

### 3.3 KernelParams（GPU 参数）

`ctypes.Structure`，320 字节，16 字节对齐，精确匹配 WGSL `struct KernelParams` 布局。

```
偏移量    字段                           类型
0-47     i[0..11], mat_size 等          int32 × 12
48-63    focal_length                   vec4<f32>
64-79    optical_center                 vec2<f32> + padding
80-127   k1, k2, k3                     vec4<f32> × 3
128-159  radial_distortion_limit 等     f32 × 8
160-175  input_dimension                vec2<f32> + padding
176-191  background_color               vec4<f32>
192-207  output_dimension               vec4<i32>
208-223  flags                          vec4<i32>
224-239  digital_lens_params            vec4<f32>
240-255  viewport                       vec4<f32>
256-263  fov + padding                  f32 + i32 + i32 + f32
264-271  mesh_stretch + padding         f32 + i32
272-303  additional_rot/trans           vec4<f32> × 2
304-319  reserved                       vec4<f32>
```

### 3.4 GPU 管线

复用 Gyroflow 原版 shader 的两种路径：

| 路径 | 源文件 | 方式 |
|------|--------|------|
| WGSL | `wgpu_undistort.wgsl` (662行) | 模板替换：注入畸变模型函数、SCALAR→f32 |
| SPIR-V | `stabilize.spv` / `stabilize_u32.spv` | 直接加载字节码 |

Compute shader workgroup: 8×8。

## 4 Rust-Python 类型映射

| Rust (nalgebra) | Python |
|---|---|
| `UnitQuaternion<f64>` | `Quat64` (封装 scipy Rotation) |
| `Vector3<f64>` | `np.ndarray` shape (3,) |
| `Matrix3<f64>` | `np.ndarray` shape (3,3) |
| `BTreeMap<i64, Quat64>` | `dict[int, Quat64]` + bisect |
| `Option<Vector3>` | `np.ndarray | None` |

## 5 关键设计决策

| 决策 | 选择 | 理由 |
|------|------|------|
| 四元数库 | scipy.spatial.transform.Rotation | 成熟稳定，nalgebra 级精度 |
| GPU | wgpu-py + 原版 WGSL/SPIR-V | 效果保证一致，复用已验证 shader |
| 视频 I/O | PyAV | FFmpeg 全功能绑定 |
| GUI | PySide6 | Gyroflow 用 Qt，API 映射直接 |
| IMU 解析 | PyO3 bridge + Python fallback | 复用 telemetry-parser crate |
| 数据结构 | TimeQuat = dict[int, Quat64] + bisect | Python 中无需 BTreeMap，bisect 够用 |

## 6 包结构

```
pygyroflow/
├── types/                    # Layer 0: 6 文件, 546 行
├── filtering/                # Layer 1: 3 文件, 277 行
├── imu_integration/          # Layer 1: 9 文件, 1911 行
├── lens/                     # Layer 1: 3 文件, 1033 行
├── keyframes/                # Layer 1: 3 文件, 650 行
├── gyro_source/              # Layer 1: 4 文件, 744 行
├── smoothing/                # Layer 2: 7 文件, 1910 行
├── stabilization/            # Layer 2: 15 文件, 2917 行
│   └── distortion_models/    #           1970 行
├── zooming/                  # Layer 2: 3 文件, 695 行
├── synchronization/          # Layer 3: 16 文件, 2292 行
├── gpu/                      # Layer 3: 5 文件 + shaders, 539 行
├── rendering/                # Layer 3: 5 文件, 647 行
├── telemetry/                # Layer 3: 3 文件, 209 行
├── camera/                   # Layer 4: 2 文件, 454 行
├── calibration/              # Layer 4: 2 文件, 611 行
├── stmap/                    # Layer 4: 2 文件, 522 行
├── gui/                      # Layer 5: 7 文件, 963 行
├── cli/                      # Layer 5: 2 文件, 141 行
├── manager.py                # 协调器, 606 行
└── ...                       # 其他辅助模块
```
