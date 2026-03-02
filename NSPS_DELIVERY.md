# NSPS（神经符号产生式双系统）项目交付书

> **项目**：DitHub × NSPS v2  
> **日期**：2026-03-02  
> **分支**：`feat/ncrn-v2`  
> **前置依赖**：[NCRN_DELIVERY.md](./NCRN_DELIVERY.md)（v1 基础模块）

---

## 一、宏观定位

### 1.1 架构演进

```
v1 (NCRN_DELIVERY.md)          v2 (本文)
─────────────────────          ─────────────────────────────────────
单一共享 NCRN                   独立 NCRN 阵列 + 极化路由
增量扩展字典                    每任务物理隔离，零遗忘
单分类头                        System 1（路由）+ System 2（后件）
```

### 1.2 核心思路

```
                     System 1: 极化路由
                ┌─────────────────────────┐
                │  PolarizedAntecedentBase │
   Z [B,D] ────┤  cos(Z, K) / τ → Top-K  │
                └────┬──────┬──────┬───────┘
                     │      │      │  Top-K 候选任务
          ┌──────────┘      │      └──────────┐
          ▼                 ▼                 ▼
     System 2:         System 2:         System 2:
     NCRN_t0           NCRN_t1           NCRN_t2
     (冻结)            (冻结)            (冻结)
          │                 │                 │
          ▼                 ▼                 ▼
     p0 ≥ γ₀?         p1 ≥ γ₁?         p2 ≥ γ₂?
     → 采纳           → 拒识,           → 采纳
                        继续下一个
                              │
                         全拒识 → UNKNOWN (task_id=-1)
```

**关键设计**：

- 每个任务 NCRN **物理隔离**，训练后立即冻结，旧知识永不被修改
- 极化路由保证新任务原型与旧任务正交，防止路由坍塌
- 每任务专属动态 γ（训练集 5th percentile 置信度），比全局阈值更精准
- 硬负样本采样强制 NCRN 具备拒识能力

---

## 二、文件清单

```
DitHub/
├── groundingdino/models/GroundingDINO/ncrn/
│   ├── __init__.py                # [修改] 导出 v2 新模块
│   ├── concept_dict.py            # [v1 不动] 正交概念字典
│   ├── fuzzy_ops.py               # [v1 不动] 模糊逻辑算子
│   ├── dnf_engine.py              # [v1 不动] DNF 规则引擎
│   ├── ncrn_head.py               # [v1 不动] 顶层封装
│   ├── polarized_router.py        # [新建] System 1 极化路由器
│   ├── nsps_system.py             # [新建] NSPSSystem 编排层
│   └── hard_negative_sampler.py   # [新建] 硬负样本采样器
├── tests/
│   ├── test_ncrn.py               # [v1] 9 个单元测试
│   └── test_nsps.py               # [新建] 8 个 NSPS 单元测试
├── main_nsps.py                   # [新建] NSPS 服务器实验入口
└── NSPS_DELIVERY.md               # 本文件
```

---

## 三、核心模块说明

### 3.1 `polarized_router.py` — System 1 极化路由器

**类**：`PolarizedAntecedentBase`

| 方法 | 说明 |
|------|------|
| `register_task(Z_samples, task_id, class_names)` | 计算均值原型 → 零空间极化 → L2 归一化 → 追加入库 |
| `route(Z, top_k)` | cos 相似度 / τ → 返回 Top-K 任务索引+分数 |
| `_polarize(K_new, K_old)` | 零空间正交投影（Gram 矩阵法 + 退化保护） |
| `get_active_probes()` | 返回已注册的探针矩阵 K[T, D] |

**关键参数**：

```python
PolarizedAntecedentBase(
    feat_dim=256,    # 与 GroundingDINO hs 维度对齐
    max_tasks=100,
    tau=0.07,        # 路由温度（越小区分度越高）
    margin=0.3,      # 极化软排斥阈值
)
```

**极化原理**：

```
K_new^⊥ = K_new - K_old^T (K_old K_old^T + εI)^{-1} K_old K_new
```

> 奇异矩阵时退化为逐向量 Gram-Schmidt，过度投影（>90% 能量损失）时切换软排斥模式。

---

### 3.2 `nsps_system.py` — NSPSSystem 编排层

**类**：`NSPSSystem`

```python
NSPSSystem(
    feat_dim=256,
    concepts_per_task=64,    # 每个任务 NCRN 的概念字典大小
    rules_per_class=8,       # 每类 DNF 规则数
    lambda_l1=1e-3,          # 稀疏正则化（防维度浪费，L1 退火）
    lambda_conflict=1e-2,    # 互斥正则化
    top_k=3,                 # 路由候选数
    gamma=0.3,               # 全局置信度阈值（逐任务动态 γ 优先）
    neg_ratio=0.2,           # 硬负样本比例
    l1_anneal_epochs=5,      # L1 退火预热 epoch
)
```

**主要方法**：

| 方法 | 说明 |
|------|------|
| `add_task(Z, Y, class_names, num_epochs, lr)` | 完整编排：极化入库 → 训练 NCRN → 冻结 → 统计动态 γ |
| `inference(Z, top_k, gamma, return_details)` | 回退式逐样本推理（支持详细日志） |
| `inference_batched(Z, top_k, gamma)` | 同任务打包批量推理（高吞吐，行为与 inference 一致） |
| `save_checkpoint(save_dir)` | 序列化全系统（router + 所有 NCRN + meta + 特征缓存） |
| `load_checkpoint(save_dir, device)` | 反序列化，显式 `.to(device)` 解决 CPU/GPU 设备不匹配 |
| `get_global_class_mapping()` | 返回 `(task_id, local_cls) → class_name` 映射 |

**动态 γ 说明**：

- 每个任务训练完成后，在训练集上统计 `max_prob` 的第 5 百分位数作为该任务的 γ
- 推理时按 `task_meta[tid]["task_gamma"]` 逐任务调取，不使用全局默认值
- 显式传 `gamma` 参数时作为全局覆盖（如评估指定固定阈值）

---

### 3.3 `hard_negative_sampler.py` — 硬负样本采样器

**类**：`HardNegativeSampler`

从路由相似度最高的**其他任务**中采样负例，强制 NCRN 输出低概率（防过度自信）。

```python
# 每次 add_task 内部自动调用，无需手动干预
sampler.sample(Z_pos, Y_pos, stored_features, current_task_id, router_K)
# → Z_aug [N_pos + N_neg, D], Y_aug [N_pos + N_neg, C]
```

---

### 3.4 `main_nsps.py` — 服务器实验入口

```
核心函数
├── load_grounding_dino()         冻结 GroundingDINO，仅提取特征
├── extract_task_features()       hs[-1].mean(nq) → Z[N,D] + one-hot Y[N,C]
├── run_nsps_add_task()           单任务特征采集 → NSPSSystem.add_task
├── NSPSInferenceModel            detectron2 兼容推理包装
│   └── forward(batched_inputs)   NSPS 推理 → Instances 格式
├── do_train()                    顺序/乱序遍历 ODinW-13，断点续训
└── do_eval()                     全任务 COCO AP 评估，汇总 Mean AP
```

---

## 四、本地测试验证

在 Apple MPS（M 芯片）设备上，8 个单元测试全部通过：

```
🧪 NSPS Unit Tests — Device: mps

✅ test_polarized_router_register  (max_sim_t2=0.000000, 极化完美正交)
✅ test_polarized_router_route     (Top-K 路由格式正确)
✅ test_nsps_add_task              (final_loss=0.1792, gamma=0.6720)
✅ test_nsps_inference_fallback    (accepted=10/10)
✅ test_nsps_checkpoint            (保存/恢复推理结果一致)
✅ test_hard_negative_rejection    (OOD 样本路由测试通过)
✅ test_nsps_3task_continual       (3 tasks, 7 classes, accepted=30/30)
✅ test_feature_collapse           (sim_after=0.000004, 零向量保护有效)
🎉 All 8 tests passed!
```

关键指标：

- **极化正交性**：`max_sim_after ≈ 0.000000`（数学完美正交）
- **特征坍塌保护**：两个几乎相同任务极化后 cosine 相似度仍降至 `0.000004`
- **全系统冻结验证**：所有后件 NCRN 的 `requires_grad` 全为 `False`

---

## 五、服务器 4090 快速上手

### 5.1 环境准备（若未完成）

```bash
# 参考 NCRN_DELIVERY.md 中的 setup_server.sh
bash scripts/setup_server.sh
bash scripts/download_data.sh
```

### 5.2 拉取 v2 分支

```bash
cd /opt/data/private/why/DitHub
git fetch origin
git checkout feat/ncrn-v2
# 或首次使用：
git checkout -b feat/ncrn-v2 origin/feat/ncrn-v2
```

验证文件存在：

```bash
ls groundingdino/models/GroundingDINO/ncrn/
# 应看到: polarized_router.py  nsps_system.py  hard_negative_sampler.py  ...
ls main_nsps.py  # 应存在
```

### 5.3 运行单元测试（可选，约 2 分钟）

```bash
/opt/data/private/conda_envs/dithub/bin/python tests/test_nsps.py
```

> ⚠️ 注意：测试使用合成数据，无需 GPU 也可在 CPU 上跑。

### 5.4 完整连续学习实验

```bash
cd /opt/data/private/why/DitHub
export LD_LIBRARY_PATH=/opt/data/private/conda_envs/dithub/lib/python3.9/site-packages/torch/lib:$LD_LIBRARY_PATH
export HF_HUB_OFFLINE=1

nohup /opt/data/private/conda_envs/dithub/bin/python -u main_nsps.py \
    --config-file test/test_odinw13 \
    --model-config-file groundingdino/config/GroundingDINO_SwinT_OGC_dt_ncrn.py \
    --model-checkpoint-path groundingdino_swint_ogc.pth \
    --output-dir ./output/nsps_output \
    --seed 42 --num-gpus 1 --dithub \
    > output/nsps_output.log 2>&1 &

tail -f output/nsps_output.log
```

### 5.5 仅评估（训练已完成）

```bash
/opt/data/private/conda_envs/dithub/bin/python -u main_nsps.py \
    --config-file test/test_odinw13 \
    --model-config-file groundingdino/config/GroundingDINO_SwinT_OGC_dt_ncrn.py \
    --model-checkpoint-path groundingdino_swint_ogc.pth \
    --output-dir ./output/nsps_output \
    --eval-only
```

### 5.6 断点续训

训练中断后，直接重新运行 5.4 的命令。`main_nsps.py` 会自动检测 `output/nsps_output/nsps_checkpoint/` 并从上次完成的任务继续，跳过已完成的任务。

```bash
# 强制从头重新训练（忽略已有 checkpoint）
python -u main_nsps.py ... --overwrite
```

---

## 六、预期输出结构

```
output/nsps_output/
├── nsps_checkpoint/           # 全系统断点（每任务更新一次）
│   ├── router.pth             # PolarizedAntecedentBase 权重
│   ├── ncrn_0.pth             # Task 0 NCRN 权重（冻结）
│   ├── ncrn_1.pth             # Task 1 NCRN 权重（冻结）
│   ├── ...
│   ├── nsps_meta.pth          # 任务元信息（task_gamma / class_names 等）
│   └── feature_cache.pth      # 各任务特征缓存（用于硬负样本）
├── <task_output_dir>/         # 各任务评估输出（COCOEvaluator 产出）
└── nsps_output.log            # 完整训练日志
```

---

## 七、关键超参数（在 GroundingDINO_SwinT_OGC_dt_ncrn.py 中配置）

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `ncrn_feat_dim` | 256 | 特征维度（与 GroundingDINO hs 对齐） |
| `ncrn_init_active` | 64 | 每个任务 NCRN 的概念字典大小 K |
| `ncrn_num_rules` | 8 | 每类 DNF 规则数 R |
| `ncrn_lambda_l1` | 1e-3 | L1 稀疏正则（防字典维度浪费） |
| `ncrn_lambda_conflict` | 1e-2 | 互斥正则 |
| `nsps_top_k` | 3 | 路由候选数 |
| `nsps_gamma` | 0.3 | 全局置信度阈值（逐任务动态 γ 优先于此值） |
| `nsps_neg_ratio` | 0.2 | 硬负样本比例 |
| `nsps_epochs` | 50 | 每任务训练 epoch 数 |
| `nsps_lr` | 0.001 | NCRN 学习率 |
| `nsps_max_samples` | 500 | 每任务最大特征采集量 |

---

## 八、与 v1（NCRN）的关系

| 维度 | v1 NCRN (`main_ncrn.py`) | v2 NSPS (`main_nsps.py`) |
|------|--------------------------|--------------------------|
| 模型结构 | 单一 NCRN_Head，增量扩展 | NCRN 阵列，每任务独立实例 |
| 遗忘防止 | 零空间扩展 + 梯度掩码 | 物理冻结，数学零遗忘 |
| 路由 | 无（单分类头） | 极化前件路由 System 1 |
| 推理 | 单 NCRN 直接输出 | Top-K 路由 + 回退式判决 |
| 训练入口 | `run_ncrn_cl.sh` | `main_nsps.py` |
| checkpoint | `ncrn_shared.pth` | `nsps_checkpoint/` 目录 |

**v1 代码完全保留**，v2 是在其上的纯粹上层编排，两者可以独立运行、独立对比。

---

## 九、已知限制和后续工作

| 优先级 | 事项 | 说明 |
|--------|------|------|
| P0 | 服务器端实验验证 | 需在 4090 上运行 13 任务验证 COCO AP |
| P1 | `NSPSInferenceModel` box 对齐 | 当前直接复用 GroundingDINO bbox_embed，box 精度待验证 |
| P1 | 与 DitHub LoRA baseline 对比 | 相同 ODinW-13 设置下对比 Mean AP |
| P2 | 超参数调优 | concepts_per_task / nsps_epochs / neg_ratio 对性能的影响 |
| P2 | 逻辑解释可视化 | `get_logic_explanation()` 输出可视化为规则表 |

---

## 十、联系 & 踩坑备忘

| 问题 | 解决方案 |
|------|----------|
| `aten::mode` 在 MPS 上不支持 | `tensor.mode()` → `bincount + argmax`（已在测试中修复） |
| Checkpoint 恢复设备不匹配 | `load_state_dict` 后显式 `.to(device)`（已修复） |
| `gamma or self.gamma` 当 gamma=0.0 误判 | 改为 `gamma if gamma is not None else self.gamma`（已修复） |
| GroundingDINO 模块未注册 | 在入口文件显式 `import groundingdino.models.GroundingDINO.groundingdino_dt` |
| mmap 耗尽（服务器 max_map_count=65530） | DataLoader `num_workers=0`，参考 `NCRN_DELIVERY.md` §3 |
