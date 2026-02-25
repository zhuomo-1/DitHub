# DitHub 项目进度追踪

> 本文档是项目的真理来源（Source of Truth），记录所有开发进度、变更和关键决策。

---

## 项目概览

**DitHub** 是 NeurIPS 2025 论文的官方实现，用于增量开放词汇目标检测（Incremental Open-Vocabulary Object Detection）。基于 GroundingDINO，通过模块化的类特定适应模块管理版本控制式的增量学习。

**核心特性**：
- 基于 GroundingDINO 的增量学习框架
- 模块化的类特定适应模块
- 支持 COCO、ODinW-35/13 数据集
- 零样本评估能力

**项目结构**：
```
DitHub/
├── groundingdino/           # 核心模块
│   ├── config/              # 模型配置
│   ├── datasets/            # 数据集处理
│   ├── models/              # 模型定义
│   └── utils/               # 工具函数
├── tools/                   # 辅助脚本
├── train_dithub.sh          # 训练脚本
├── eval_dithub.sh           # 评估脚本
└── requirements.txt         # 依赖
```

---

## 待办事项 (TODO)

- [ ] 设置开发环境（Python >= 3.9, CUDA >= 11.8, GCC >= 11.4）
- [ ] 安装依赖并验证环境
- [ ] 下载数据集（COCO、ODinW-35）
- [ ] 下载预训练权重
- [ ] 验证训练流程
- [ ] 验证评估流程

---

## 开发日志

### [2026-02-25 --:--] 创建 CLAUDE.md 项目指南
- **Status**: Completed
- **Changes (变更详情)**:
    - 📂 `File/Folder`: 新增 `CLAUDE.md` 文件
    - ⚙️ `Function/API`: 无
    - 🔧 `Tech`: 为 Claude Code 提供项目上下文指南
- **Pitfalls & Solutions (踩坑与修复)**:
    - 无
- **Dev Notes (开发备忘/自由空间)**:
    - CLAUDE.md 包含：常用命令、架构概览、核心类说明、训练流程、数据集结构
    - 重点记录了 LoRA 注入机制（LinearPool）和 TaskMemory 单例管理
    - 未来维护时注意：修改 LoRA 相关代码需同步更新文档

### [2026-02-25 --:--] 项目初始化
- **Status**: Completed
- **Changes (变更详情)**:
    - 📂 `File/Folder`: 创建 `progress.md` 外部记忆文件
    - 🔧 `Tech`: 初始化进度锚点工作流
- **Pitfalls & Solutions (踩坑与修复)**:
    - 无
- **Dev Notes (开发备忘/自由空间)**:
    - 项目采用【进度锚点工作流】进行管理
    - 所有交流和文档使用中文
    - 每次任务完成后需更新 progress.md 并提交 Git