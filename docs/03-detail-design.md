# PyGyroFlow 详细设计（LLD）

> 版本: 0.1.0 | 日期: 2026-05-28

## 1 四元数模块 (types/quaternion.py)

### 1.1 Quat64 类

```python
class Quat64:
    _rot: scipy.spatial.transform.Rotation  # 内部存储

    @classmethod
    def from_euler_angles(cls, roll, pitch, yaw) -> Quat64
    @classmethod
    def from_quaternion(cls, wxyz: array_like) -> Quat64
    @classmethod
    def from_scaled_axis(cls, axis: array_like) -> Quat64
    @classmethod
    def from_rotation_matrix(cls, mat: ndarray 3x3) -> Quat64

    def quaternion(self) -> ndarray          # 返回 [w, x, y, z]
    def euler_angles(self) -> ndarray       # 返回 [roll, pitch, yaw]
    def to_rotation_matrix(self) -> ndarray # 返回 3x3
    def inverse(self) -> Quat64
    def slerp(self, other: Quat64, t: float) -> Quat64
    def __mul__(self, other: Quat64) -> Quat64  # Hamilton 乘积
    def angle(self) -> float                # 旋转角度
```

**关键约束**: scipy 内部用 [x,y,z,w]，对外接口统一 [w,x,y,z]。

### 1.2 坐标变换

IMU 坐标系 → 相机坐标系: `(x, y, z) → (-y, x, z)`

陀螺仪单位转换: 度/秒 → 弧度/秒 (`× π/180`)

## 2 IMU 积分模块 (imu_integration/)

### 2.1 基类

```python
class GyroIntegrator(ABC):
    @abstractmethod
    def integrate(self, imu_data: list[TimeIMU], duration_ms: float) -> TimeQuat
```

### 2.2 SimpleGyroIntegrator

**算法**: 纯角速度积分，一阶旋转向量近似。

```
初始姿态: from_euler_angles(π/2, 0, 0)   # 相机朝向
每步:
  ω = coordinate_transform(gyro) × DEG2RAD
  dt = (timestamp_ms - prev_ms) / 1000
  Δq = from_scaled_axis(ω × dt)
  orientation = orientation × Δq
  输出: timestamp_us → orientation
```

### 2.3 VQFIntegrator (最复杂的积分器，1072 行)

**算法**: 离线前向-后向双向滤波，分离 3D 倾斜和航向估计。

```
参数: tau_acc=40.0, tau_mag=40.0 (大值，离线模式补偿延迟)

前向遍历:
  1. update_gyr(gyro): 陀螺仪积分 → gyr_quat
  2. update_acc(accel): Butterworth 低通 + 倾斜修正 → acc_quat
  3. update_bias_ekf(): 扩展卡尔曼滤波估计陀螺仪偏差
  4. 6D姿态: q6d = acc_quat × gyr_quat

后向遍历: 相同算法，时间反转

偏差合并:
  - 前向/后向 bias 通过协方差加权平均
  - 用合并后的 bias 重新前向积分

加速度计处理:
  - forward-backward Butterworth filtfilt (零相位延迟)
  - 静止检测 (gyro norm < threshold)
  - 磁场干扰拒绝
```

### 2.4 MahonyIntegrator

**算法**: PI 控制器互补滤波。

```
参数: Kp=0.5, Ki=0.0
初始: from_euler_angles(π/2, 0, 0)

每步:
  1. 估计重力方向: v = rotate(q, [0,0,1])
  2. 误差 = cross(v, accelerometer)
  3. 积分反馈: integral += error × Ki × dt
  4. 校正角速度: ω_corrected = ω + Kp×error + integral
  5. 一阶四元数积分: q += 0.5×dt×q×ω_corrected
  6. 归一化
```

**注意**: nalgebra 四元数内部存储为 [x,y,z,w]，重力向量公式索引需对应。

### 2.5 MadgwickIntegrator

**算法**: 梯度下降法。

```
参数: β=0.02
初始: from_euler_angles(π/2, 0, 0)

每步:
  1. 构造目标函数 F = gravity_error(q, accelerometer)
  2. 计算雅可比矩阵 J 的梯度 ∇F = Jᵀ × F
  3. 梯度下降修正: q_corrected = q - β × ∇F/|∇F| × dt
  4. 陀螺仪积分: q_gyro = q + 0.5×dt×q×ω
  5. 融合: q = lerp(q_gyro, q_corrected, β)
  6. 归一化
```

## 3 平滑模块 (smoothing/)

### 3.1 DefaultAlgo（核心算法，595 行）

速度自适应双向平滑，对应 Gyroflow `default_algo.rs`。

```
输入: org_quats (TimeQuat), duration_ms, ComputeParams
输出: smoothed_quats (TimeQuat)

步骤:
  1. 计算相邻四元数角速度序列
  2. 双向指数平滑角速度 (alpha自适应)
     - 低速 → 强平滑 (α→0.02, τ≈1s)
     - 高速 → 弱平滑 (α→0.4, τ≈0.1s)
  3. 第一遍: 双向自适应指数平滑四元数
  4. 计算平滑后与原始的距离序列
  5. 距离归一化, <0.5 设为 0
  6. 平滑距离序列
  7. 第二遍: alpha 按 速度×距离 缩放

smoothness 参数:
  - 0.0: 几乎不平滑
  - 0.5: 默认
  - 1.0: 最大平滑
```

### 3.2 HorizonLock（地平线锁定后处理器，471 行）

```
目标: 锁定输出视频的水平线

步骤:
  1. 对平滑后四元数序列，提取每帧的 up 向量
  2. 计算 up 向量与 [0,0,1] 的偏差角
  3. 用滑动窗口平滑偏差角
  4. 构造修正四元数，补偿偏差
  5. 应用到平滑后四元数
```

## 4 帧变换模块 (stabilization/frame_transform.py, 420 行)

### 4.1 FrameTransform.at_timestamp()

每帧变换的核心算法。

```
输入: ComputeParams, timestamp_ms, frame_index
输出: FrameTransform { matrices: ndarray(N,14), fov: float, kernel_params }

步骤:
  1. 查询 org_quat: 时间戳处的原始四元数
  2. 查询 smoothed_quat: 同时刻的平滑四元数
     (考虑 video_speed ramp 映射)
  3. R = smoothed_quat × org_quat⁻¹   # 旋转补偿
  4. 构建 new_k (输出内参矩阵):
     - 含 FOV 缩放、水平拉伸
     - 主点偏移
  5. 卷帘快门校正:
     对每行 y ∈ [0, height):
       ts_row = timestamp + readout_time × (y/height)
       quat_row = interp(org_quats, ts_row)
       smooth_row = interp(smoothed, ts_row)
       R_row = smooth_row × quat_row⁻¹
       matrix[i] = [R_row[0:3,0], R_row[0:3,1], R_row[1:3,2],
                    ibis_x, ibis_y]  # 14 floats
  6. 构建 KernelParams
```

## 5 畸变模型模块 (stabilization/distortion_models/)

### 5.1 架构

每个畸变模型同时实现 CPU 和 GPU（WGSL）两个版本。

```python
class DistortionModelBase(ABC):
    @abstractmethod
    def distort_point(self, x, y, z, params) -> tuple[float, float]

    @abstractmethod
    def undistort_point(self, dx, dy, params) -> tuple[float, float]

    @abstractmethod
    def get_wgsl_functions(self) -> str     # WGSL shader 代码

    def radial_distortion_limit(self, params) -> float
```

### 5.2 OpenCV Fisheye (Kannala-Brandt)

```
正向 (3D → 2D):
  θ = atan2(r, z)      # r = sqrt(x²+y²)
  θ_d = θ(1 + k₁θ² + k₂θ⁴ + k₃θ⁶ + k₄θ⁸)
  x' = (θ_d/r) × x × fx + cx
  y' = (θ_d/r) × y × fy + cy

逆向 (2D → 3D): Newton-Raphson 迭代
  θ_d = sqrt(((dx-cx)/fx)² + ((dy-cy)/fy)²)
  迭代求 θ: θ_d = θ(1 + k₁θ² + k₂θ⁴ + ...)
  x = sin(θ) × (dx-cx) / (r×fx)
  y = sin(θ) × (dy-cy) / (r×fy)
  z = cos(θ)
```

### 5.3 GoPro HyperView (七阶多项式)

```
正向: x' = Σᵢ aᵢ × x^(i+1)  (i=0..6, 7个系数)
逆向: Newton-Raphson 迭代, 最多 20 次
  发散保护: |x'| > 2.0 时重置到初始猜测
```

## 6 自适应缩放模块 (zooming/)

### 6.1 FovIterative (366 行)

```
对每帧:
  1. 在输出图像上均匀采样 31×31 个点
  2. 对每个采样点，逆变换到输入图像坐标
  3. 检查边界: 找超出输入图像范围的点
  4. 迭代调整 FOV 直到所有采样点在边界内
     (最多 max_zoom_iterations=5 次)
  5. 返回最小 FOV 值
```

### 6.2 ZoomDynamic

```
对 FOV 序列做时间平滑:
  方法 1: 高斯加权 (window=4s)
  方法 2: 包络跟随 (上包络 + 下包络)
```

## 7 GPU 管线 (gpu/)

### 7.1 Shader Builder

WGSL 模板处理:
```
输入: wgpu_undistort.wgsl (662 行)
替换:
  LENS_MODEL_FUNCTIONS;  →  对应畸变模型的 WGSL 函数
  SCALAR                 →  f32
  {texture_input}...{/texture_input}  →  移除 (使用 buffer_input)
  {buffer_input}...{/buffer_input}    →  保留
```

### 7.2 渲染流程

```
1. 创建 compute pipeline (shader module + bind group layout)
2. 每帧:
   a. 上传 KernelParams → uniform buffer (320B)
   b. 上传 matrices → storage buffer (N×14×4B)
   c. 上传输入帧 → storage buffer (W×H×4B)
   d. dispatch(W/8, H/8, 1)
   e. 读回输出帧
```

## 8 数据结构关键定义

### 8.1 TimeQuat 范围查询

```python
TimeQuat = dict[int, Quat64]  # timestamp_us → orientation

# 范围查询: 用 bisect 查找最近的 timestamp
import bisect
keys = sorted(quats.keys())
idx = bisect.bisect_right(keys, target_ts)
if idx > 0:
    q0, q1 = quats[keys[idx-1]], quats[keys[idx]]
    # slerp 插值
```

### 8.2 ComputeParams 快照

```python
@dataclass
class ComputeParams:
    # 视频: width, height, output_width, output_height, frame_count, fps
    # 陀螺仪: quaternions, smoothed_quaternions (TimeQuat)
    # FOV: fovs, minimal_fovs (list[float])
    # 镜头: camera_matrix (3x3), distortion_coeffs (12,), model_name
    # 卷帘快门: frame_readout_time, readout_direction
    # 缩放: adaptive_zoom_window, adaptive_zoom_method
    # 关键帧: keyframes (KeyframeManager)
    # 背景: background_mode, background_color
```

StabilizationManager 在每次计算前生成快照，渲染线程只读不写。
