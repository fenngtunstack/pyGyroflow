# PyGyroFlow：把6万9千行Rust防抖代码用Python重写一遍，值不值

## 为什么要把Gyroflow移植到Python

Gyroflow是视频防抖领域做得最好的开源项目之一，69000行Rust代码，支持GoPro/DJI/Sony主流运动相机，自动提取IMU数据、自动同步、6种姿态积分算法、GPU渲染。效果逼近甚至超过GoPro官方的HyperSmooth。

但Rust生态对AI和ML研究者有个硬伤：你没法把它直接嵌入你的PyTorch推理管线。想在Jupyter里交互式调参？不行。想运行时替换积分算法？得重编译。想用TensorRT做实时推理？得自己写binding。

PyGyroFlow就是为解决这个问题做的——把Gyroflow v1.6.3的核心算法完整移植到Python。不是包装一层Rust FFI，是真的用Python重写。最终的数据（`git show | wc -l`/pytest 实测）：107 个 Python 源文件，约 18,925 行代码，198 个测试用例。5 个 IMU 积分器与 Rust 移植版的数值误差在机器精度级别（max_err < 1e-14）；VQF 暂无 Rust 对照。

<!-- ✏️ 编辑建议：在这里加一句你最初决定做Python移植时的动机，比如是某个具体场景卡住了你，还是纯粹想验证"Python能不能做到" -->

## 先说结果：数据和Rust版一模一样

做移植最怕的就是"看着差不多但精度有差异"。PyGyroFlow的验证策略是四级的：

**单元测试**——每个算法组件独立验证，比如VQF积分器的状态更新、EMA平滑的SLERP插值、畸变模型的投影反投影。180个用例覆盖所有核心路径。

**Rust黄金对比**——用Gyroflow原版生成基准数据（quaternion序列、旋转矩阵、remap图），PyGyroFlow对同样的输入跑同样的算法，逐bit对比。5个积分器（SimpleGyro、Complementary、Madgwick、Mahony、VQF）的误差全部在1e-14以内，这就是double精度的上限——不是"差不多"，是数学上不可能更精确了。

**集成测试**——完整管线跑通，从IMU数据输入到输出稳定视频，和Rust版的输出做帧级对比。

**端到端测试**——用真实相机（GoPro、DJI）拍的视频做输入，输出视频的MSE、PSNR和Gyroflow原版对比。

这个四级验证的结果说明一件事：Python版的算法实现是正确的，不是"近似正确"，是数值上等价。

## 核心能力拆解

### IMU数据提取

支持三种主流格式：GoPro的GPMF（内嵌在MP4的GPMF轨道）、DJI的自定义编码（同样内嵌在视频文件）、Sony的元数据格式。采样率从200Hz到1600Hz不等，自动归一化处理。这意味着你丢一个GoPro视频进去，不需要额外操作，陀螺仪数据自动就有了。

### 6种IMU积分算法

从简单到复杂依次是：SimpleGyro（纯角速度积分）、Complementary（互补滤波）、Madgwick、Mahony、VQF。VQF是其中最复杂的，基于协方差匹配的自适应滤波，对陀螺仪零偏和噪声的鲁棒性最好，也是Gyroflow的默认选项。

所有6种算法都用纯Python实现，核心计算用NumPy向量化。不需要你安装任何编译工具链。

### 5种平滑算法+地平线锁定

平滑是防抖的灵魂。PyGyroFlow实现了Gyroflow的全部5种平滑策略：Default（自适应截止频率）、Plain（固定窗口高斯）、Fixed相机模式（锁定视角）、Max角度限制、以及Speed可变平滑。加一个可选的地平线锁定，防止画面绕光轴旋转。

### 10种镜头畸变模型

从OpenCV Fisheye到GoPro的HyperView，覆盖了运动相机常用的所有镜头类型。包括针孔模型、普通径向-切向畸变、多种鱼眼参数化、以及GoPro特有的宽视角模式。镜头标定文件（.json）可以直接复用Gyroflow的，格式完全兼容。

### 卷帘快门校正

CMOS传感器逐行读出的特性决定了每帧画面内部存在时间差。如果不做校正，高速运动时会出现明显的果冻效应。PyGyroFlow的卷帘快门校正是逐行处理的——每10行一个条带，查询该行实际读取时刻对应的陀螺仪姿态，分别做变换。

### GPU加速

畸变校正和帧变换是最耗时的步骤。PyGyroFlow通过wgpu compute shader实现GPU加速，支持CUDA、Metal、Vulkan、OpenGL四种后端。没有GPU的环境自动fallback到CPU（OpenCV remap）。

### 自动同步

陀螺仪和视频帧的时间基准不同步是防抖最常见的失败原因。PyGyroFlow的自动同步流程：先用光流估计帧间运动，再和IMU积分得到的运动做相关性搜索，找到最优的时间偏移。整个过程全自动。

## 快速上手

安装和运行都是一行命令：

```bash
pip install -e .
pygyroflow input.mp4 -o output.mp4 --smoothness 0.5
```

`--smoothness`参数范围0到1，0最响应（保留更多运动），1最平滑。0.5是大多数场景的起点。

如果你想在Python代码里调用而不是CLI：

```python
from pygyroflow import Pipeline

pipe = Pipeline(smoothness=0.5)
pipe.process("input.mp4", "output.mp4")
```

想在Jupyter里交互式调参？可以。Pipeline的每一步都可以独立调用，中间结果都是NumPy数组，直接画图看效果。改一个参数重新跑一个cell就能看到变化，不用像Rust版那样改代码→重编译→重启。

<!-- ✏️ 编辑建议：在这里加一句你在Jupyter里调参时的具体体验，比如调smoothness从0.3到0.7时画面的变化，或者某个参数的"甜点区间" -->

## 和Rust版的差异：什么有，什么没有

先说什么有：算法层面100%覆盖，所有积分器、平滑器、畸变模型、同步算法全部移植。镜头标定文件格式兼容。命令行参数兼容。PySide6 GUI也有了（虽然不如Rust版的polished）。

再说什么没有：性能。Python + NumPy的速度大概是Rust版的1/5到1/3（CPU模式），有GPU加速后差距缩小到1/2左右。对于离线处理几分钟的视频，完全够用。对于实时4K防抖，还是得用Rust版或者C++版。

也不追求完全兼容Rust版的GPU渲染管线。Rust版走的是Vulkan/WGPU原生渲染，PyGyroFlow的GPU路径是通过wgpu-py的compute shader，架构不同但效果等价。

## 四级验证的实际意义

花这么多篇幅讲验证，不是因为我喜欢写测试。是因为防抖算法的"正确性"很难靠肉眼看——画面看起来稳了，不代表算法是对的。一个4度/秒的陀螺仪零偏误差，肉眼看不出来，但会让画面在30秒内缓慢旋转5度。积分算法的数值漂移更是如此，几百帧以后才显现。

所以"和Rust版数值一致"这个结论很重要——它意味着PyGyroFlow不是"又一个防抖工具"，而是Gyroflow的精确Python复现。你在PyGyroFlow上调的参数、做的实验，结果可以无损耗地迁移到Rust版的生产环境。

## 适合什么人用

**做AI/ML研究的人**——需要把防抖作为预处理步骤嵌入推理管线。PyGyroFlow是纯Python的，可以直接import，和PyTorch/TensorRT无缝集成。

**做算法实验的人**——想试新的积分算法、新的平滑策略、新的畸变模型。Python的动态特性让你可以在运行时替换任何组件，不需要重新编译。

**做嵌入式部署的人**——需要在ARM板子上跑防抖。Python版虽然慢一点，但在树莓派4上处理1080p30fps的视频还是可以做到实时（有GPU加速的话）。而且Python的交叉编译比Rust简单太多了。

**学算法的人**——Rust的语法对不熟悉的人来说阅读门槛不低。Python版的代码可读性好很多，配合Jupyter边看数据边调参数，理解起来快。

<!-- ✏️ 编辑建议：在这里加一句你自己的使用场景，比如你是用在什么具体项目里，或者你看到的最有意思的用例 -->

## 代价和取舍

约 18,925 行 Python 代码（实测）去对应上游约 6.9 万行 Rust 代码。行数少了约 73%，但这不是因为 Python 更"简洁"——Rust 版有很多 GPU 渲染代码、跨平台 UI 代码、视频编解码代码，这些 PyGyroFlow 要么用了现成库（wgpu-py、PySide6、OpenCV），要么精简了范围。所以这个压缩比里既有真正的语言红利，也有功能取舍，不能简单等同于"Python 等价重写了 Rust"。

Python版的代价主要是性能和部署。Rust编译出来一个二进制文件就能跑，Python版需要一套Python环境加上NumPy、wgpu、OpenCV这些依赖。在资源极度受限的嵌入式场景（比如RTOS）上，Python方案不合适。

但话说回来，如果你真的要在那种环境下跑防抖，你应该直接用C/C++实现，Rust也不见得合适。PyGyroFlow的目标不是替代Rust版的生产部署，而是提供一个等价的Python实现，让算法实验和AI集成变得容易。

---

PyGyroFlow项目地址：https://github.com/Adrian1707/PyGyroFlow

Gyroflow原版地址：https://github.com/gyroflow/gyroflow
