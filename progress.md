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

- [x] 设置开发环境（Python >= 3.9, CUDA >= 11.8, GCC >= 11.4）
- [x] 安装依赖并验证环境
- [x] 下载数据集（COCO、ODinW-35）
- [x] 下载预训练权重
- [x] 验证训练流程（单卡 + 多卡）
- [x] 验证评估流程
- [ ] 运行全量复现实验（ODinW-13）
- [ ] 分析结果并与论文对比

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

### [2026-02-27 16:xx] 环境配置与 CUDA Ops 编译
- **Status**: Completed
- **Changes (变更详情)**:
    - 📂 `File/Folder`: 新增 `setup.py` 用于编译 CUDA ops
    - 📂 `File/Folder`: 新增 `MultiScaleDeformableAttention.cpython-39-x86_64-linux-gnu.so`
    - ⚙️ `Function/API`: 修改 `main.py:308-310` 添加 detectron2 v0.6 兼容性处理
- **Pitfalls & Solutions (踩坑与修复)**:
    1. **HuggingFace 网络超时**
       - 问题: 训练启动时尝试从 huggingface.co 下载 BERT tokenizer，网络超时
       - 解决: 设置 `HF_HUB_OFFLINE=1` 环境变量使用本地缓存
    2. **detectron2 `lr_factor_func` 参数不兼容**
       - 问题: `get_default_optimizer_params()` 在 detectron2 v0.6 中不支持 `lr_factor_func` 参数
       - 解决: 在 `main.py` 中添加兼容性处理，移除该参数
    3. **CUDA Ops 编译缺少 `WITH_CUDA` 宏**
       - 问题: 编译成功但运行时报错 `RuntimeError: Not compiled with GPU support`
       - 原因: `ms_deform_attn.h` 中使用 `#ifdef WITH_CUDA` 条件编译，但编译时未定义该宏
       - 解决: 在 `setup.py` 的 `extra_compile_args` 中添加 `-DWITH_CUDA`
    4. **CUDA Ops 链接错误 `undefined symbol: get_cudart_version`**
       - 问题: 编译后导入模块时报链接错误
       - 原因: `vision.cpp` 引用了 `get_cudart_version()`，但 `cuda_version.cu` 未加入编译
       - 解决: 在 `setup.py` 的 `sources` 列表中添加 `cuda_version.cu`
    5. **PyTorch CUDA 库路径问题**
       - 问题: 导入编译好的模块时报 `libc10.so: cannot open shared object file`
       - 解决: 设置 `LD_LIBRARY_PATH` 包含 PyTorch 的 lib 目录
- **Dev Notes (开发备忘/自由空间)**:
    - 运行训练需要的环境变量:
      ```bash
      export LD_LIBRARY_PATH=/opt/data/private/conda_envs/dithub/lib/python3.9/site-packages/torch/lib:$LD_LIBRARY_PATH
      export HF_HUB_OFFLINE=1
      ```
    - conda 环境路径: `/opt/data/private/conda_envs/dithub`
    - 数据集路径: `/opt/data/private/why/DitHub/datasets/odinw13/` (软链接)
    - 预训练权重: `groundingdino_swint_ogc.pth` (693MB)

### [2026-02-27 17:xx] 多卡训练验证
- **Status**: Completed
- **Changes (变更详情)**:
    - 📂 `File/Folder`: 修改 `train_dithub.sh` 将 `--num-gpus` 从 1 改为 4
    - 📂 `File/Folder`: 修改所有 ODinW-13 配置文件 `total_batch_size` 从 2 改为 8
    - ⚙️ `Function/API`: 修复 `main.py` 三个 bug:
      1. `do_test()` 函数 - 处理评估结果可能没有 'bbox' 键的情况
      2. `main()` 函数 - 修复除以零错误（当没有评估结果时）
      3. `main()` 函数 - 修复输出路径重复拼接问题
- **Pitfalls & Solutions (踩坑与修复)**:
    1. **多卡训练 batch size 不兼容**
       - 问题: `AssertionError: Total batch size (2) must be divisible by the number of gpus (4)`
       - 解决: 修改配置文件将 `total_batch_size` 改为 8（能被 4 整除）
    2. **评估阶段 KeyError: 'bbox'**
       - 问题: 训练完成后评估阶段返回值可能没有 'bbox' 键
       - 解决: 在 `do_test()` 函数中添加 'bbox' 键检查
    3. **除零错误 ZeroDivisionError**
       - 问题: 当评估结果为空时计算平均 AP 导致除零错误
       - 解决: 添加条件判断 `len(avg_res) - coco_count > 0`
    4. **路径拼接错误 FileNotFoundError**
       - 问题: `json_path` 路径重复拼接导致路径不存在
       - 解决: 使用 `cfg.train.output_dir` 替代 `os.path.join(args.output_dir, cfg.train.output_dir)`
    5. **CUDA CUBLAS 错误**
       - 问题: GPU 显存未释放导致 `CUBLAS_STATUS_NOT_INITIALIZED`
       - 解决: 等待 GPU 显存释放后重新运行
- **Dev Notes (开发备忘/自由空间)**:
    - 多卡训练命令: `bash train_dithub.sh` (已配置 4 GPU)
    - GPU 显存使用: ~8.5GB/卡 (RTX 4090 24GB)
    - 训练输出目录: `./output/dithub_output/`
    - **快速测试结果 (20 iterations)**:
      - AP: 35.90 | AP50: 45.91 | AP75: 41.09
      - 耗时: ~1 分钟
    - **完整训练配置**:
      - 13 个数据集，4 epochs/数据集
      - 预计总时长: 1-2 小时

### [2026-02-27 19:xx] 全量复现实验
- **Status**: Partially Completed (训练完成，评估超时)
- **Changes (变更详情)**:
    - 📂 `File/Folder`: 修复后重新运行完整训练流程
- **Pitfalls & Solutions (踩坑与修复)**:
    1. **NCCL 超时错误**
       - 问题: 训练完成后评估阶段 NCCL 超时 (30分钟)，导致评估未完成
       - 状态: 待修复（需要调整评估策略）
- **Dev Notes (开发备忘/自由空间)**:
    - 训练脚本: `bash train_dithub.sh`
    - 日志文件: `train_full.log`
    - **训练结果**: 13 个数据集全部训练完成
    - **评估结果** (部分):
      - NorthAmericaMushrooms: 47.08%
      - ShellfishOpenImages: 44.39%
      - pistols: 73.09%
      - thermalDogsAndPeople: 77.42%
    - **下一步**: 需要修复 NCCL 超时问题，完成剩余数据集的评估

### [2026-03-01 02:xx] 分离训练与评估 - 修复 NCCL 超时
- **Status**: Completed
- **Changes (变更详情)**:
    - 📂 `File/Folder`: 修改 `main.py` 添加 `--train-only` 参数
    - 📂 `File/Folder`: 修改 `train_dithub.sh` 添加 `--train-only` 参数
    - ⚙️ `Function/API`: `main.py:366-368` - 条件添加 EvalHook
    - ⚙️ `Function/API`: `main.py:416-419` - 条件跳过评估循环
- **Pitfalls & Solutions (踩坑与修复)**:
    1. **NCCL 超时**
       - 问题: 训练完成后评估阶段多 GPU 通信超时 (30分钟)
       - 解决: 添加 `--train-only` 参数，训练评估分离执行
    2. **HuggingFace 网络超时**
       - 问题: 评估时尝试下载 BERT tokenizer
       - 解决: 设置 `HF_HUB_OFFLINE=1` 环境变量
- **Dev Notes (开发备忘/自由空间)**:
    - **训练命令**: `bash train_dithub.sh` (带 --train-only)
    - **评估命令**:
      ```bash
      HF_HUB_OFFLINE=1 python -u main.py \
        --config-file test/test_odinw13 \
        --model-config-file groundingdino/config/GroundingDINO_SwinT_OGC_dt_dithub.py \
        --model-checkpoint-path groundingdino_swint_ogc.pth \
        --output-dir ./output/dithub_output \
        --num-gpus 1 \
        --dithub \
        --eval-only
      ```
    - **训练结果**:
      - 13 个任务全部训练完成
      - 训练时间: 9小时29分
      - 迭代次数: 29998
      - 模型文件: `last_lora.pth` (297MB)
    - **评估结果** (使用 last_lora.pth):
      | 数据集 | AP |
      |--------|-----|
      | CottontailRabbits | 70.29 |
      | Egohands | 68.25 |
      | NorthAmericaMushrooms | 49.32 |
      | Packages | 66.09 |
      | PascalVoc | 71.46 |
      | Raccoon | 72.24 |
      | ShellfishOpenImages | 41.21 |
      | VehiclesOpenImages | 67.53 |
      | AerialMaritimeDrone | 34.15 |
      | Aquarium | 42.84 |
      | Pistols | 72.02 |
      | Pothole | 50.41 |
      | thermalDogsAndPeople | 72.50 |
      - **平均 AP**: 58.54
      - **COCO 零样本**: 45.83

---

## 关键路径索引

### 训练与评估
- 训练脚本: `./train_dithub.sh`
- 评估脚本: `./eval_dithub.sh`
- 训练日志: `./output/dithub_output/log.txt`
- 评估日志: `./output/dithub_output/eval_log.txt`
- 主模型文件: `./output/dithub_output/last_lora.pth` (本次训练)
- 合并模型文件: `./output/dithub_output/model_final.pth` (之前训练)

### 数据集
- ODinW-13 数据集: `./datasets/odinw13/`
- COCO 数据集: `./datasets/coco/`

### 输出目录
- 训练输出: `./output/dithub_output/odinw13/<dataset_name>_cet/`
- 评估结果: `./output/dithub_output/odinw13/<dataset_name>_cet/result.json`

### 预训练权重
- GroundingDINO: `./groundingdino_swint_ogc.pth`

### 环境变量
```bash
export LD_LIBRARY_PATH=/opt/data/private/conda_envs/dithub/lib/python3.9/site-packages/torch/lib:$LD_LIBRARY_PATH
export HF_HUB_OFFLINE=1
```