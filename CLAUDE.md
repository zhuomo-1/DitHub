# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## 项目概述

DitHub 是一个模块化框架，用于**增量式开放词汇目标检测** (NeurIPS 2025)。本仓库在 DitHub 基础上扩展了 **NCRN** (Neural-symbolic Concept Reasoning Network)，实现可解释、零遗忘的持续学习。

**架构**: GroundingDINO (冻结骨干) + LoRA 适配器 + NCRN_Head (替代 ContrastiveEmbed 的即插即用模块)

## 常用命令

### 训练

```bash
# 标准 DitHub 训练
bash train_dithub.sh

# NCRN 连续学习训练 (ODinW-13 全部 13 个任务)
bash run_ncrn_cl.sh ./output/ncrn_10shot_vX 3 0
# 参数: 输出目录, seed, GPU ID

# NCRN 单任务训练 (由 run_ncrn_cl.sh 调用)
python -u main_ncrn.py \
    --task-config test/test_odinw13_10shot/for_train/test_AerialMaritimeDrone_tiled.py \
    --task-index 0 --total-tasks 13 \
    --model-config-file groundingdino/config/GroundingDINO_SwinT_OGC_dt_ncrn.py \
    --model-checkpoint-path groundingdino_swint_ogc.pth \
    --output-dir ./output/ncrn_diag --seed 3 --dithub
```

### 评估

```bash
# 标准 DitHub 评估
bash eval_dithub.sh

# NCRN 评估
bash eval_ncrn.sh
# 或指定输出目录和 checkpoint
CHECKPOINT=./output/ncrn_10shot/ncrn_shared.pth OUTPUT_DIR=./output/ncrn_eval bash eval_ncrn.sh
```

### 测试

```bash
# NCRN 单元测试 (验证 4 个核心模块)
conda run -n dithub python tests/test_ncrn.py

# 带冒烟测试 (需要 COCO 128 数据集)
conda run -n dithub python tests/test_ncrn.py --smoke-test --data-dir /path/to/coco128
```

## 环境设置

```bash
# 必需的环境变量
export LD_LIBRARY_PATH=/opt/data/private/conda_envs/dithub/lib/python3.9/site-packages/torch/lib:$LD_LIBRARY_PATH
export HF_HUB_OFFLINE=1
export TOKENIZERS_PARALLELISM=false  # 避免 fork 警告

# Conda 环境: dithub (Python 3.9, PyTorch 2.4.0+cu118)
conda activate dithub
```

## 关键文件

| 文件 | 用途 |
|------|------|
| `main.py` | DitHub 训练/评估入口 |
| `main_ncrn.py` | NCRN 训练/评估入口 (独立进程模式) |
| `groundingdino/config/GroundingDINO_SwinT_OGC_dt_ncrn.py` | NCRN 超参数配置 |
| `groundingdino/models/GroundingDINO/ncrn/` | NCRN 核心模块 (4 个文件) |
| `progress.md` | 开发进度锚点 (修改后需更新) |
| `run_ncrn_cl.sh` | 编排脚本：顺序执行 13 个任务训练 |

## NCRN 架构

```
groundingdino/models/GroundingDINO/ncrn/
├── concept_dict.py   # OrthogonalConceptDict - SVD 零空间扩展
├── fuzzy_ops.py      # FuzzyLogicOperators - log-space T-norm/T-conorm
├── dnf_engine.py     # DifferentiableDNF - Binary STE, L1+conflict 正则化
└── ncrn_head.py      # NCRN_Head - 顶层 forward/loss/incremental_update
```

**数据流**: `Z (特征) -> OrthogonalConceptDict -> C (概念真值) -> DifferentiableDNF -> Y (类别概率)`

**核心逻辑**:
1. 双极性扩展: `L = [C, 1-C]`
2. 退火 STE: `W = sigmoid(Omega * beta)` (beta: 1→10)
3. AND (log-space geometric mean): P_rules
4. OR (probabilistic sum): Y_hat

## 配置参数

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `ncrn_feat_dim` | 256 | 输入特征维度 (= hidden_dim) |
| `ncrn_total_concepts` | 2048 | 字典总容量 M |
| `ncrn_init_active` | 64 | 初始激活概念数 K |
| `ncrn_num_rules` | 3 | 每类 DNF 规则数 R |
| `ncrn_lr` | 0.001 | NCRN 学习率 (Omega: 0.01) |
| `ncrn_lambda_l1` | 1e-3 | 稀疏正则化系数 |
| `ncrn_lambda_conflict` | 1e-2 | 互斥正则化系数 |
| `init_tau` | 0.07 | 温度参数 (当前过小，需调大) |

## 数据集

- **ODinW-13**: 13 个检测数据集用于增量学习 (通过 `python tools/download_odinw.py` 下载)
- **COCO**: 零样本评估
- 数据路径: `datasets/odinw13/` 和 `datasets/coco/` (指向 DitHub-ori 的软链接)

## 已知问题

1. **延迟导入**: `GroundingDINO/__init__.py` 使用延迟导入 - 必须显式 `import groundingdino.models.GroundingDINO.groundingdino_dt` 后才能构建模型
2. **CUDA ops**: `MultiScaleDeformableAttention.cpython-39-x86_64-linux-gnu.so` 必须编译或从已安装环境复制
3. **mmap 耗尽**: DataLoader 使用 `num_workers=0` 防止多任务训练时的 fork 相关 mmap 问题
4. **seed_all_rng**: 位于 `detectron2.utils.env`，不在 `groundingdino.util.misc`
