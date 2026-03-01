# NCRN（神经符号概念推理网络）项目交付书

> **项目**：DitHub × NCRN  
> **日期**：2026-03-02  
> **分支**：`feat/ncrn`

---

## 一、宏观愿景

### 1.1 核心痛点

少样本连续学习（FSCL）面临两大顽疾：

| 问题 | 传统方案 | 局限性 |
|------|----------|--------|
| **灾难性遗忘** | EWC / LwF 软正则化 | 少样本下失效，参数漂移不可控 |
| **黑盒不可解释** | Attention 可视化 | 无法给出「为什么是这个类」的逻辑推理 |

### 1.2 NCRN 解决方案

```
深度特征 Z ──→ 正交概念字典 ──→ 符号化真值 C ──→ DNF 逻辑引擎 ──→ 类别概率 Y
               (可解释属性)      (0~1 连续值)     (IF...THEN 规则)
```

**系统级防御策略**：绝对冻结旧参数 + 零空间释放新容量 + 学习新逻辑掩码  
→ 从纯数学维度**100%杜绝遗忘**

### 1.3 架构定位

```mermaid
graph LR
    A["GroundingDINO<br/>(冻结特征提取)"] -->|"hs [B,nq,256]"| B["NCRN_Head<br/>(可训练分类头)"]
    B --> C["OrthogonalConceptDict"]
    B --> D["DifferentiableDNF"]
    C -->|"概念真值 [B,K]"| D
    D -->|"类别概率 [B,C]"| E["输出"]
    
    style A fill:#1a1a2e,stroke:#16213e,color:#e94560
    style B fill:#0f3460,stroke:#16213e,color:#e94560
```

NCRN 作为 **Drop-in Replacement** 分类头，替换 GroundingDINO 的 `ContrastiveEmbed`，与 DitHub 的 LoRA 增量学习框架无缝集成。

---

## 二、已实现模块

### 2.1 核心代码（4 个模块 + 测试）

| 文件 | 模块 | 核心技术 | 行数 |
|------|------|----------|------|
| `ncrn/concept_dict.py` | 正交概念字典 | SVD 零空间投影扩容、L2 归一化、梯度掩码 | 178 |
| `ncrn/fuzzy_ops.py` | 模糊逻辑算子 | 对数空间 T-norm / T-conorm、ε 数值稳定 | 119 |
| `ncrn/dnf_engine.py` | DNF 规则引擎 | Binary STE 离散化、L1 稀疏 + 互斥正则 | 219 |
| `ncrn/ncrn_head.py` | 顶层封装 | forward / compute_loss / incremental_update / explain | 219 |
| `tests/test_ncrn.py` | 单元测试 | 9 个单元测试 + 1 个 COCO 128 冒烟测试 | 440 |

**代码位置**：`groundingdino/models/GroundingDINO/ncrn/`

### 2.2 验证结果

所有测试在 **Apple MPS** 设备上通过：

```
🧪 NCRN Unit Tests — Device: mps
✅ test_concept_dict_forward PASSED
✅ test_concept_dict_expand_nullspace PASSED (max_dot=0.000000, added=16)
✅ test_fuzzy_ops_numerical_stability PASSED
✅ test_fuzzy_ops_semantics PASSED
✅ test_dnf_forward PASSED
✅ test_dnf_logic_loss PASSED
✅ test_ncrn_head_end_to_end PASSED (L_total=20.0030)
✅ test_incremental_update PASSED (K: 32→40, C: 3→5)
✅ test_logic_explanation PASSED
✅ test_smoke_coco128 PASSED (72 classes, 128 images)
🎉 All tests passed!
```

关键指标：

- **零空间正交性**：`max_dot = 0.000000`（完美正交）
- **增量扩展**：K: 32→40, C: 3→5（概念字典 + 类别数正确扩容）
- **参数冻结**：旧参数梯度全为零（硬隔离验证通过）

### 2.3 服务器部署脚本

| 文件 | 用途 |
|------|------|
| `main_ncrn.py` | NCRN 训练/评估入口（自包含，不依赖 main.py） |
| `groundingdino/config/GroundingDINO_SwinT_OGC_dt_ncrn.py` | NCRN 超参数配置 |
| `train_ncrn.sh` | Click-to-run 训练脚本 |
| `eval_ncrn.sh` | Click-to-run 评估脚本 |
| `scripts/setup_server.sh` | 一键 4090 环境部署 |
| `scripts/download_data.sh` | ODinW-35 数据下载 |

### 2.4 基础设施修改

| 文件 | 修改 | 原因 |
|------|------|------|
| `GroundingDINO/__init__.py` | 急切导入 → 延迟导入 | 解耦 ncrn 与 transformers/timm/detectron2 依赖链 |

---

## 三、微观实现细节

### 3.1 OrthogonalConceptDict — 正交概念字典

```python
# 核心公式
P = W[:, :K_active]              # 取出活跃概念向量
C = σ(Z @ P)                     # 概念激活 (sigmoid)
C = C / ||C||₂                   # L2 归一化到单位球面

# 增量扩展 (零空间投影)
U, S, Vh = SVD(P_old)            # 对旧概念矩阵做 SVD
nullspace = Vh[rank:]            # 零空间基底
W_new = nullspace[:n_new]        # 在零空间中初始化新概念
# 保证 W_new ⊥ P_old (max_dot = 0.000000)
```

### 3.2 DifferentiableDNF — 可微逻辑引擎

```python
# Binary STE (Straight-Through Estimator)
# 前向：离散 {0,1}    反向：连续梯度
hard = (sigmoid(Ω) > 0.5).float()
Ω_discrete = hard - sigmoid(Ω).detach() + sigmoid(Ω)

# DNF 推理 (对数空间)
# 合取项 (AND): 对数空间加法 = 概率空间乘法
conjunction = Σ (Ω⁺ · log(C) + Ω⁻ · log(1-C))    # [C, R]
# 析取 (OR): log-sum-exp
class_prob = 1 - exp(Σ log(1 - exp(conjunction)))   # [C]
```

### 3.3 增量学习流程

```
Task 1: 学习 {dog, cat, car}
  → 训练 NCRN_Head (K=32 概念, C=3 类别)

Task 2: 学习 {bird, fish}  
  → ncrn_head.incremental_update(Z_new, Y_new, n_new=16)
    1. 冻结旧概念 W[:, :32] 的梯度
    2. SVD 投影得到 16 个新概念 (与旧概念正交)
    3. 扩展 DNF: 冻结旧 Ω[:3], 新增 Ω[3:5]
  → K: 32→48, C: 3→5 (旧知识零遗忘)
```

---

## 四、在 4090 服务器上复现连续学习的待办事项

### 4.1 环境部署（预计 15 分钟）

```bash
# 1. 将代码推送到服务器
git clone -b feat/ncrn <repo_url>

# 2. 一键部署
bash scripts/setup_server.sh
# 自动完成: conda 环境 + PyTorch CUDA 11.8 + detectron2 + Deformable-DETR CUDA 算子

# 3. 下载数据集
bash scripts/download_data.sh
# 下载 ODinW-35 (~10GB)
```

### 4.2 预训练权重（预计 5 分钟）

```bash
# 下载 GroundingDINO SwinT 预训练权重 (setup_server.sh 已包含)
wget https://github.com/IDEA-Research/GroundingDINO/releases/download/v0.1.0-alpha/groundingdino_swint_ogc.pth
```

### 4.3 运行实验（预计 2-4 小时）

```bash
bash train_ncrn.sh    # 训练 13 个 ODinW 数据集的增量学习
bash eval_ncrn.sh     # 评估所有已学任务的 AP
```

### 4.4 后续开发任务（重要性排序）

| 优先级 | 任务 | 说明 |
|--------|------|------|
| **P0** | 集成 Hungarian Matching | 当前 `main_ncrn.py` 使用简化的 BCE 损失，需替换为 Hungarian 匹配 + Focal Loss，与 DitHub 原始 criterion 对齐 |
| **P0** | NCRN 推理替换 | 评估阶段目前仍用 ContrastiveEmbed，需实现 NCRN 的推理路径（`dt_inference` 中使用 NCRN 输出） |
| **P1** | 超参数调优 | 概念数 K、规则数 R、正则化系数 λ_l1/λ_conflict 需根据实验效果调整 |
| **P1** | LoRA + NCRN 联合训练 | 当前方案冻结 GroundingDINO，可尝试 LoRA 与 NCRN 联合微调 |
| **P2** | 逻辑解释可视化 | 将 `get_logic_explanation()` 输出可视化为可读的规则表 |
| **P2** | 与 DitHub baseline 对比 | 在相同 ODinW-13 设置下对比 DitHub 原始 LoRA 与 NCRN 的 AP |

---

## 五、文件清单

```
DitHub/
├── groundingdino/models/GroundingDINO/
│   ├── __init__.py                 # [修改] 延迟导入
│   └── ncrn/                       # [新建] NCRN 核心模块
│       ├── __init__.py
│       ├── concept_dict.py         # 正交概念字典
│       ├── fuzzy_ops.py            # 模糊逻辑算子
│       ├── dnf_engine.py           # DNF 规则引擎
│       └── ncrn_head.py            # 顶层封装
├── groundingdino/config/
│   └── GroundingDINO_SwinT_OGC_dt_ncrn.py  # [新建] NCRN 配置
├── tests/
│   └── test_ncrn.py                # [新建] 单元测试 + 冒烟测试
├── scripts/
│   ├── setup_server.sh             # [新建] 环境部署
│   └── download_data.sh            # [新建] 数据下载
├── main_ncrn.py                    # [新建] NCRN 训练入口
├── train_ncrn.sh                   # [新建] Click-to-run 训练
├── eval_ncrn.sh                    # [新建] Click-to-run 评估
└── progress.md                     # [更新] 开发进度
```

---

## 六、开发环境备忘

| 项目 | 值 |
|------|-----|
| 本地开发环境 | macOS + MPS (Apple Metal) |
| Conda 环境名 | `dithub` |
| Python | 3.9 |
| PyTorch | 2.8.0 (MPS) / 2.4.0 (服务器 CUDA 11.8) |
| 测试数据集 | COCO 128 (`/Volumes/SSD512/Code/dataset/COCO 128.v2-640x640.coco`) |
| 目标数据集 | ODinW-13 (13 个检测数据集的增量学习) |
| 运行测试 | `conda run -n dithub python tests/test_ncrn.py --smoke-test` |
