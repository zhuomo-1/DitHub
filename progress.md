# DitHub 开发进度锚点

## 项目信息

- **项目名称**: DitHub — 增量式开放词汇目标检测
- **创建时间**: 2026-02-25
- **当前阶段**: 阶段十一 · V5 修复关键 bug (sparse init range), 3-task 验证完成, 准备架构重设计
- **Python环境**: `/opt/data/private/conda_envs/dithub/bin/python`
- **历史完整记录**: `progress_full_backup.md`

---

## 快速命令

```bash
# 3-task 快速验证 (Aerial + Cottontail + Egohands)
bash run_ncrn_3task.sh ./output/ncrn_v5 3 0

# 全量 13-task 连续学习
bash run_ncrn_cl.sh ./output/ncrn_10shot_vX 3 0
```

---

## 架构概览 (V5 当前)

```
GroundingDINO (全冻结, 含 LoRA 但不训练) -> hs [B,900,256] (文本条件化特征)
    -> NCRN_Head (唯一可训练部分)
        -> OrthogonalConceptDict: Z -> C [B,K] (随机正交基, sigmoid(cos/tau))
        -> DifferentiableDNF: C -> Y_hat [B,C] (硬 STE + 稀疏初始化)
```

**核心问题**: NCRN 完全绕过了 VLM 的文本-视觉对比分类能力, 概念字典随机初始化无语义

---

## 实验结果汇总

### Baseline (DitHub-ori, LoRA)
| 指标 | 值 |
|---|---|
| Average ODinW-13 AP | **58.54** |
| COCO 零样本 AP | 45.83 |

### NCRN 各版本

| 数据集 | V1 | V2 | V3 | V4 | **V5 (3-task)** |
|---|---|---|---|---|---|
| AerialMaritimeDrone | 3.68 | 3.68 | 0.00 | 1.82 | **0.26** |
| CottontailRabbits | 63.71 | 63.71 | 62.85 | 53.71 | **27.22** |
| Egohands | 11.26 | 11.26 | 6.68 | 6.76 | **52.30** |
| **Average (13-task)** | **23.12** | **23.28** | **15.90** | **14.82** | — |

### V5 关键改动与修复
1. **修复致命 bug**: `_sparse_init_omega` 在 4096 维中随机选 literal, 但实际只用前 128 维 (2×K_active=64). 选中的 +3 几乎从不落在有效范围, 导致之前所有稀疏初始化实验等同于全负初始化
2. **tau=0.5** (从 0.07 调大): C 分布合理 [0.28, 0.74]
3. **恢复硬 STE** (去掉退火): 简化架构
4. **去掉空规则=0 修复**: 避免 Y_hat 死锁在 0.0

### V5 训练动态 (首次正常!)
- iter=0: W active=2/128, P_rules=0.50, Y_hat=0.876 (理论预期)
- iter=50: C 展开 [0.28,0.75], Y_hat 降到 0.733, loss 4.0→1.39
- iter=100: Omega 出现 [0.6-0.9] [0.9-1.0] 区间, Y_hat=0.619
- iter=150: Y_hat=0.536, loss=0.21, 正样本 0.67 vs 负样本 0.54 (首次出现区分度!)

---

## 根因分析: 为什么 NCRN 远低于 Baseline (58.54 vs ~15)?

### 1. 概念字典无语义锚点
- OrthogonalConceptDict 用**随机正交向量**初始化 P_total
- 原始 ContrastiveEmbed 直接用文本 token embedding 做分类 (天然语义对齐)
- NCRN 必须从零学习概念, 200 iter 根本不够

### 2. LoRA 特征适配被禁用
- DitHub-ori: LoRA 可训练, 适配 encoder 特征空间 (lr=0.001, r=16)
- NCRN: LoRA 存在但冻结, 特征空间未针对新类适配

### 3. 文本-视觉协同被浪费
- hs 是文本条件化特征 (decoder 每层做 text cross-attention)
- 但 NCRN 只取 hs 的值, 完全不利用 encoded_text 做分类
- ContrastiveEmbed 的 hs @ text.T 能力被丢弃

---

## 下一步: Text-Grounded NCRN (V6 架构重设计)

### 核心思路: 不替代文本, 利用文本作为概念锚点

```
图像 → GroundingDINO (冻结 + LoRA 可训) → hs [B, nq, D]
                                            ↓
文本类名 → BERT → encoded_text ──┐          ↓
                                  ├──→ NCRN_Head
                                  │     ├── ConceptDict: P_init = encoded_text (文本初始化!)
                                  │     │   C = sigmoid(Z·P / tau)
                                  │     └── DNF: C → 规则推理 → Y_hat
                                  │
                                  └──→ ContrastiveEmbed: hs @ text.T → baseline_logits (辅助loss)
```

### 改动清单
1. **概念字典文本初始化**: P_total 从 encoded_text 的类别 token 初始化 (非随机)
2. **保留 ContrastiveEmbed 双头**: 辅助 loss 保持文本对齐
3. **解冻 LoRA**: 特征空间可适配
4. **NCRN 的规则推理保留可解释性优势**

---

## 关键配置参数 (V5 当前值)

| 参数 | 值 | 文件 |
|---|---|---|
| ncrn_num_rules | 3 | config |
| init_tau | 0.5 | concept_dict.py |
| Omega init | sparse +3/-3 (active_range=2*K_active) | dnf_engine.py |
| STE | 硬 STE (无退火) | dnf_engine.py |
| Omega lr | 10x base (0.01) | main_ncrn.py |
| 正则 | targeted sparsity (target=3) | dnf_engine.py |

## 输出目录

| 版本 | 目录 | 说明 |
|---|---|---|
| V1-V2 | `output/ncrn_10shot/` / `v2` | 初始 baseline |
| V3 | `output/ncrn_10shot_v3/` | 10x iter (过拟合) |
| V4 | `output/ncrn_10shot_v4/` | 架构修复 (tau 过激) |
| V5 | `output/ncrn_v5/` | 修复 sparse init bug, 3-task 验证 |
