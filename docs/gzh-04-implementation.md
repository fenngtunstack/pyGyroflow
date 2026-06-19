# 用Python重写Gyroflow：约19000行代码的五层解剖

Gyroflow 用 Rust 写的视频防抖引擎，源码十几万行。把它用 Python 重写一遍是什么体验？107 个源文件、约 18,925 行 Python 代码、198 个测试用例。这篇文章不聊算法原理，只看代码是怎么组织的——哪些模块最厚、哪些翻译最难、Python 和 Rust 之间的工程鸿沟在哪里。

> 注：行数用 `git show HEAD:<file> | wc -l` 测量。本工作区部分文件 inode stat 元数据损坏，`wc -l` 直接读取会严重失真（曾误测为 2,579 行），须以 git 内容为准。

## 为什么有人要拿Python重写Gyroflow

Gyroflow是Rust写的桌面应用，功能强大但对嵌入式开发者不友好。想在自己的板子上跑防抖、想裁剪一个子集做推理验证、想在Jupyter里可视化IMU积分过程——Rust的编译工具链和GUI框架都是障碍。

Python版本（PyGyroFlow）的目标不是替代原版，而是提供一套可交互、可裁剪、可嵌入的防抖算法库。牺牲性能换可读性和可组合性，这在算法验证阶段是合理的取舍。

198 个测试用例覆盖了核心算法路径，这在算法移植项目里算高的。移植项目的测试不是锦上添花——Rust的类型系统和所有权模型在编译期帮你挡住的bug，Python全靠测试在运行时补。

## 五层架构，代码量分布不均匀

整个项目按依赖关系分了五层，从底向上：

**Layer 0 基础类型（542 行）**——四元数、IMU数据、枚举、错误类型。这层最薄，但它是后续所有计算的基石。Quat64 占了核心，封装四元数乘法、归一化、SLERP 插值。Python 没有原生的四元数类型，所以包了一层 scipy.spatial.transform.Rotation，用双精度 float 承载，精度损失在可接受范围内。

**Layer 1 数据处理（约 4,800 行）**——IMU 积分、滤波、镜头模型、关键帧、GyroSource。其中 VQF（Versatile Quaternion-based Filter）独占 1,072 行，是整个 IMU 积分模块的最大单文件。VQF 是 Gyroflow 的默认姿态估计算法，论文发在 Sensors 期刊，实现涉及自适应增益、运动检测、磁力计融合。1,072 行 Python 翻译上游约 1,485 行 Rust，代码行数减少约 28%，主要因为 Python 省掉了 Rust 的所有权标注和生命周期声明。

镜头模型 1,033 行，覆盖了 Gyroflow 支持的所有畸变模型——从简单的针孔到复杂的 PolyFisheyeLUT。这部分代码密度高，数学推导密集，每个模型都要实现正向投影、反向投影、WGSL 着色器函数三组方法。

**Layer 2 防抖核心（约 6,000 行）**——整个项目最厚的一层。三个子模块：平滑算法 1,910 行、帧变换（stabilization/，含 12 个畸变模型）2,836 行、自适应缩放 695 行。

帧变换模块是代码量最大的单体，其中畸变模型占了 12 个文件。为什么畸变模型在这里又出现了一次？Layer 1 的镜头模型定义畸变参数和投影函数，Layer 2 的帧变换用这些函数做像素级的逆向映射。同一个数学模型在两个不同层级有不同的职责。

DefaultAlgo 是 Gyroflow 的默认平滑算法实现，HorizonLock 是地平线锁定功能。这两个算法在 Rust 原版里各自对应数百行的文件，Python 版通过列表推导和 numpy 向量化操作压缩了代码量。

**Layer 3 集成层（约 5,000 行）**——同步模块 2,292 行是这一层的重头戏。陀螺仪和视频的时间对齐是整个防抖流程里最容易出问题的环节，AutoSync 和 OptimSync 两个算法分别处理粗对齐和精对齐。光流相关代码占了相当比例——需要从视频帧间提取运动特征，然后和陀螺仪积分出的旋转序列做相关性匹配。

GPU 模块 555 行，渲染模块 665 行。Python 的 GPU 抽象层没有 Rust/wgpu 那样的零拷贝优势，但用 compute shader 做批量像素变换的思路是一致的。

**Layer 4 应用层（约 1,750 行）**——manager（636 行）是整个防抖管线的编排入口，把 IMU 数据、同步结果、平滑算法、帧变换串成一条完整的处理链。GUI 模块 958 行，CLI 153 行。说明这个 Python 版同时面向脚本化调用和 GUI 操作。

## 六组Rust到Python的翻译对照

挑代码量最大的几个模块做对照：

| Python模块 | Python行数 | Rust源文件 | Rust行数 | 压缩比 |
|---|---|---|---|---|
| manager.py | 636 | lib.rs | 2,232 | 72% |
| frame_transform.py | ~420 | frame_transform.rs | 783 | ~46% |
| gpu/backend.py | ~378 | wgpu.rs | 714 | ~47% |
| default_algo.py | 595 | default_algo.rs | 782 | 24% |
| vqf.py | 1,072 | vqf.rs | 1,485 | 28% |
| gyro_source/source.py | 507 | mod.rs | 1,223 | 59% |

manager.py 和 lib.rs 的差距最大。Rust 的 lib.rs 2,232 行里包含大量类型声明、trait 实现、错误处理样板代码。Python 用动态类型省掉了这些，但代价是运行时才发现类型错误。198 个测试用例的部分价值就在这里——用测试覆盖 Rust 编译器帮你覆盖的部分。

vqf.py 和 default_algo.py 的压缩比在 25%-28% 左右，这是数学密集型模块的典型比例。算法本身的数学逻辑无法压缩，压缩掉的只有 Rust 的语法噪音。

gpu/backend.py 对应的是 wgpu.rs，这里的翻译不是简单的语言转换。Rust 通过 wgpu 库直接操作 Vulkan/Metal/DX12 后端，Python 版用 wgpu-py 直接驱动同一套 WGSL compute shader（复用了 Gyroflow 的 stabilize.wgsl/SPIR-V），不再经过任何 Rust bridge。这是纯 Python 路径：Python 做算法编排，shader 做 GPU 像素重映射。

## GPU 与 Telemetry 路径的取舍

项目早期曾尝试用一个 PyO3 bridge crate（telemetry_parser_bridge）桥接 Rust 的二进制解析能力，但最终只留下空壳——bridge 既没有落地实现，纯 Python 的 GPMF/DJI 解析也已够用，于是把这个空壳移除，避免误导。GPU 路径同理：与其维护一层 Rust FFI 转发，不如直接复用上游成熟的 WGSL shader。这种取舍的代价是 Python 在二进制解析、逐像素重映射上比 Rust 慢，但换来了零交叉编译负担和可调试性——对一个研究型库来说更务实。

## 构建和测试的工程细节

安装一步：`pip install -e .`。没有 Rust 工具链依赖（bridge 已移除），这对嵌入式/算法验证场景是优点。

测试跑 `pytest tests/ -q`，16 个测试文件约 2,544 行测试代码，收集到 198 个用例。测试本身也是文档——通过测试用例可以理解每个模块的输入输出契约，比读源码更快。

## 从数字里能看出什么

约 18,925 行 Python 代码翻译 Gyroflow 的核心功能，代码分布呈明显的金字塔结构：Layer 0 的 542 行是塔尖，Layer 2 的防抖核心是塔基。防抖核心（平滑 + 帧变换 + 自适应缩放）集中了核心复杂度——视频防抖的难点就在姿态补偿和像素变换。

IMU 积分里 VQF 独占 1,072 行，是单个算法模块里最厚的。如果你要理解 Gyroflow 的算法核心，VQF 是第一个要啃的文件。

Python 和 Rust 的代码量比例大致在 0.25-0.75 之间（数学密集型模块压缩少、基础设施代码压缩多）。这个数据对评估"要不要用 Python 重写一个 Rust 项目"有参考价值——数学逻辑无法压缩，能省的是工程胶水代码。
