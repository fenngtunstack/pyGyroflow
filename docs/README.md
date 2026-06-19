# PyGyroFlow 项目文档

Gyroflow v1.6.3 Python 完整移植版——视频防抖库。

## 文档目录

| 文档 | 说明 |
|------|------|
| [01-需求规格说明书](01-requirements.md) | 功能需求、非功能需求、数据管线、依赖关系 |
| [02-概要设计](02-architecture.md) | 分层架构、核心组件、类型映射、设计决策 |
| [03-详细设计](03-detail-design.md) | 算法伪代码、数据结构、模块接口 |
| [04-代码实现报告](04-implementation.md) | 模块明细、代码统计、Rust 对应关系 |
| [05-验证方案](05-verification.md) | 四级验证策略、Rust-Python 黄金对比、测试矩阵 |
| [06-测试报告](06-test-report.md) | 测试结果、数值一致性数据、缺陷修复记录 |

## 架构图

- `images/architecture-overview.drawio` — 分层架构图（用 draw.io 打开）

## 快速开始

```bash
pip install -e .
pygyroflow input.mp4 -o output.mp4 --smoothness 0.5
```

## 项目统计

> 行数用 `git show HEAD:<file> | wc -l` 测量（本工作区部分文件 inode stat 损坏，`wc -l` 直读失真，以 git 内容为准）。

- 107 个 Python 源文件，约 18,925 行代码（`pygyroflow/` 包）
- 16 个测试文件，约 2,544 行测试代码，198 个测试用例
- Rust-Python 数值一致性：5 个 IMU 积分器 max_err < 1e-14（机器精度）；
  VQF 暂无 Rust 对照（见 [06-测试报告](06-test-report.md) 已知限制）
