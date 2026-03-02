# DitHub 开发进度锚点

## 项目信息

- **项目名称**: DitHub — 增量式开放词汇目标检测
- **创建时间**: 2026-02-25
- **当前阶段**: 阶段八 · V3 (10x iter) 验证过拟合，需要架构层面优化
- **Python环境**: `/opt/data/private/conda_envs/dithub/bin/python`

---

## 环境对齐指南

### 1. Conda 环境激活

```bash
# 方式一：直接使用绝对路径（推荐，无需 conda init）
/opt/data/private/conda_envs/dithub/bin/python <script.py>

# 方式二：source activate
source /opt/data/private/conda_envs/dithub/bin/activate
python <script.py>
```

### 2. 关键依赖版本

```
Python 3.9
PyTorch 2.0.1+cu118
detectron2
transformers
timm
pycocotools
opencv-python
matplotlib
```

### 3. 数据集路径映射

| 数据集名称 | GT标注路径 (相对于 datasets/odinw13/) | 预测结果路径 |
|-----------|--------------------------------------|-------------|
| AerialMaritimeDrone_tiled | `AerialMaritimeDrone/tiled/test/annotations_without_background.json` | `output/dithub_output/odinw13/aerialmaritimedrone_tiled_cet/` |
| Aquarium | `Aquarium/Aquarium Combined.v2-raw-1024.coco/test/annotations_without_background.json` | `output/dithub_output/odinw13/aquarium_cet/` |
| CottontailRabbits | `CottontailRabbits/test/annotations_without_background.json` | `output/dithub_output/odinw13/CottontailRabbits_cet/` |
| Egohands | `EgoHands/generic/test/annotations_without_background.json` | `output/dithub_output/odinw13/Egohands_generic_cet/` |
| NorthAmericaMushrooms | `NorthAmericaMushrooms/North American Mushrooms.v2-416x416augmented.coco/train/annotations_without_background.json` | `output/dithub_output/odinw13/NorthAmericaMushrooms_cet/` |
| Packages | `Packages/augmented-v1/test/annotations_without_background.json` | `output/dithub_output/odinw13/Packages_cet/` |
| PascalVoc | `PascalVOC/valid/annotations_without_background.json` | `output/dithub_output/odinw13/PascalVoc_cet/` |
| Raccoon | `Raccoon/Raccoon.v38-416x416-resize.coco/test/annotations_without_background.json` | `output/dithub_output/odinw13/Raccoon_cet/` |
| ShellfishOpenImages | `ShellfishOpenImages/416x416/test/annotations_without_background.json` | `output/dithub_output/odinw13/ShellfishOpenImages_cet/` |
| VehiclesOpenImages | `VehiclesOpenImages/416x416/test/annotations_without_background.json` | `output/dithub_output/odinw13/VehiclesOpenImages_cet/` |
| pistols | `pistols/export/test_annotations_without_background.json` | `output/dithub_output/odinw13/pistols_cet/` |
| pothole | `pothole/test/annotations_without_background.json` | `output/dithub_output/odinw13/pothole_cet/` |
| thermalDogsAndPeople | `thermalDogsAndPeople/test/annotations_without_background.json` | `output/dithub_output/odinw13/thermalDogsAndPeople_cet/` |

### 4. 预训练模型权重

| 文件 | 路径 | 用途 |
|------|------|------|
| GroundingDINO 基础模型 | `groundingdino_swint_ogc.pth` (693MB) | 训练起点 |
| DitHub LoRA 权重 | `output/dithub_output/last_lora.pth` (297MB) | 增量训练结果 |
| 合并后完整模型 | `output/dithub_output/model_final.pth` (811MB) | 评估使用 |

### 5. 训练与评估命令

```bash
# 训练（使用 train-only 避免 NCCL 超时）
bash train_dithub.sh

# 评估（使用 last_lora.pth）
HF_HUB_OFFLINE=1 /opt/data/private/conda_envs/dithub/bin/python -u main.py \
    --config-file test/test_odinw13 \
    --model-config-file groundingdino/config/GroundingDINO_SwinT_OGC_dt_dithub.py \
    --model-checkpoint-path groundingdino_swint_ogc.pth \
    --output-dir ./output/dithub_output \
    --num-gpus 1 \
    --dithub \
    --eval-only
```

### 6. 失败分析命令

```bash
# 分析所有数据集
/opt/data/private/conda_envs/dithub/bin/python scripts/analyze_failures.py --all

# 分析单个数据集
/opt/data/private/conda_envs/dithub/bin/python scripts/analyze_failures.py --dataset VehiclesOpenImages --top-k 20

# 生成全量GT vs Prediction对比图
/opt/data/private/conda_envs/dithub/bin/python scripts/visualize_full_comparison.py \
    --dataset VehiclesOpenImages \
    --output output/full_comparison/VehiclesOpenImages
```

---

## 变更日志

### [2026-03-02 11:17] V3 训练完成 — 10x iter 实验 (Average ODinW AP = 15.90, 严重过拟合)
- **Status**: Completed
- **实验设置**: 所有 13 个 10-shot 配置 `iter_per_epoch` 乘以 10（max_iter 从 40~2000 → 400~20000），其余超参不变
- **V3 结果 vs V2 对比**:
    | 数据集 | V2 AP | V3 AP | V2 iter | V3 iter | V2 loss | V3 loss |
    |--------|-------|-------|---------|---------|---------|---------|
    | AerialMaritimeDrone | 3.18 | 1.65 | 200 | 2000 | 0.19 | 0.41 |
    | CottontailRabbits | 63.77 | 63.00 | 200 | 2000 | 3.42 | 2.89 |
    | Egohands | 10.11 | 7.14 | 200 | 2000 | 1.50 | 1.44 |
    | NorthAmericaMushrooms | 24.53 | 19.77 | 200 | 2000 | 4.08 | 4.81 |
    | Packages | 44.29 | 43.22 | 200 | 2000 | 2.74 | 2.02 |
    | PascalVoc | 2.61 | 1.36 | 2000 | 20000 | 4.42 | 8.77 |
    | Raccoon | 42.90 | 24.35 | 200 | 2000 | 9.35 | 6.38 |
    | ShellfishOpenImages | 10.73 | 8.20 | 400 | 4000 | 5.69 | 1.58 |
    | VehiclesOpenImages | 7.20 | 3.74 | 40 | 400 | 21.47 | 12.93 |
    | Aquarium | 3.47 | 1.77 | 200 | 2000 | 4.07 | 4.38 |
    | Pistols | 48.72 | 7.63 | 200 | 2000 | 23.38 | 14.79 |
    | Pothole | 10.20 | 4.63 | 200 | 2000 | 12.64 | 9.55 |
    | ThermalDogsAndPeople | 30.99 | 20.27 | 40 | 400 | 37.98 | 12.37 |
    | **Average ODinW** | **23.28** | **15.90** | | | | |
    | COCO zero-shot | 0.15 | 0.08 | | | | |
- **关键发现**:
    1. **严重过拟合**: Average AP 从 23.28 → 15.90，下降 31.7%；几乎所有数据集 AP 都下降
    2. **Pistols 崩溃**: 48.72 → 7.63（-84%），虽然 loss 从 23.38 降到 14.79，但泛化能力严重退化
    3. **CottontailRabbits 稳定**: 63.77 → 63.00（仅-1.2%），单类别数据集抗过拟合能力强
    4. **PascalVoc loss 反升**: 4.42 → 8.77，20000 iter 下 loss 震荡加剧而非收敛
    5. **后续任务 loss 仍不低**: Task 12 final_loss=12.37（vs V2 的 37.98），有下降但仍在两位数
- **结论**: 10-shot 下增加 iter 不能解决问题，NCRN 在少样本场景下容易过拟合训练集。baseline 的 iter 设定是合理的。
- **V3 结果已保存**: `output/ncrn_10shot_v3/`
- **配置已恢复**: 所有 13 个 10-shot 配置文件已恢复到 baseline 原始值
- **训练时间**: 8150s (136min), 其中 PascalVoc 占 3757s (63min)

### [2026-03-02 08:30] V2 训练完成 — loss 缩放修复验证 (Average ODinW AP = 23.28)
- **Status**: Completed
- **V2 结果 (Average ODinW AP = 23.28)**:
    | 数据集 | V1 AP | V2 AP | V1 loss | V2 loss | 变化 |
    |--------|-------|-------|---------|---------|------|
    | AerialMaritimeDrone | 3.68 | 3.18 | 329.69 | 0.19 | AP -0.50, loss 1736x↓ |
    | CottontailRabbits | 63.71 | 63.77 | 5988.63 | 3.42 | AP +0.06, loss 1750x↓ |
    | Egohands | 11.26 | 10.11 | 2671.91 | 1.50 | AP -1.15, loss 1781x↓ |
    | NorthAmericaMushrooms | 25.07 | 24.53 | 7330.95 | 4.08 | AP -0.54, loss 1797x↓ |
    | Packages | 44.73 | 44.29 | 4942.36 | 2.74 | AP -0.44, loss 1804x↓ |
    | PascalVoc | 2.24 | 2.61 | 7933.22 | 4.42 | AP +0.37, loss 1795x↓ |
    | Raccoon | 43.44 | 42.90 | 17419.13 | 9.35 | AP -0.54, loss 1863x↓ |
    | ShellfishOpenImages | 10.71 | 10.73 | 10539.65 | 5.69 | AP +0.02, loss 1852x↓ |
    | VehiclesOpenImages | 9.28 | 7.20 | 38791.49 | 21.47 | AP -2.08, loss 1806x↓ |
    | Aquarium | 3.65 | 3.47 | 7394.13 | 4.07 | AP -0.18, loss 1816x↓ |
    | Pistols | 49.73 | 48.72 | 43283.84 | 23.38 | AP -1.01, loss 1851x↓ |
    | Pothole | 11.04 | 10.20 | 23240.78 | 12.64 | AP -0.84, loss 1839x↓ |
    | ThermalDogsAndPeople | 32.98 | 30.99 | 68826.73 | 37.98 | AP -1.99, loss 1812x↓ |
    | **Average ODinW** | **23.96** | **23.28** | | | **AP -0.68** |
    | COCO zero-shot | 0.08 | 0.15 | | | AP +0.07 |
- **关键分析**:
    1. **loss 缩放修复验证成功**: 所有数据集 loss 均下降约 1800 倍（=移除 ×900 乘子 + cls_weight 2.0→1.0），梯度不再被 clip_grad_norm 裁死
    2. **AP 基本持平**: V2 平均 AP=23.28 vs V1=23.96，差异仅 0.68，说明 V1 中虽然 loss 值异常高但梯度裁剪后实际学习仍在进行
    3. **PascalVoc loss 仍有震荡**: 15→12→3.5→4.0→7.5→...→4.4（在 3~10 间跳动），20 类 10-shot 收敛困难
    4. **VehiclesOpenImages/ThermalDogsAndPeople**: 40 iter 是 baseline 原始设定（已确认与 DitHub-ori 对齐），非 bug
    5. **后续任务 loss 递增趋势**: Task 0 (0.19) → Task 6 (9.35) → Task 12 (37.98)，随 K_active 增大，模型优化难度增加
- **结论**: loss 缩放修复本身对 AP 影响不大（V1 中 clip_grad_norm 起到了"意外自适应"效果），真正的瓶颈在 NCRN 架构层面：
    - **瓶颈 1: 几何平均导致 Y_hat 初始值偏高** — sigmoid(dot_product) 映射到 [0.5, 1]，几何平均后更接近 1，focal loss 对高置信度样本的梯度极小
    - **瓶颈 2: STE 离散化导致优化landscape不光滑** — Omega 以 N(0,0.01) 初始化，sigmoid ≈ 0.5，STE 阈值恰在边界，训练初期震荡严重
    - **瓶颈 3: 后续任务累积的冻结参数占比增大** — K_active 从 64 增到 256，但只有最新 16 维可训练
- **V2 结果已保存**: `output/ncrn_10shot_v2/`
- **下一步优化路线 (V3)**:
    - P2: 概念激活值映射优化（当前 (cos+1)/2 映射使值域集中在 [0.5, 1]，考虑改为 sigmoid(scale * cos) 或 ReLU+clamp）
    - P2: lr scheduler（cosine decay）
    - P3: Omega 初始化策略优化（负偏移使 sigmoid(Ω) < 0.5，STE 后大部分文字默认关闭）
    - P3: 增加 num_rules 或 init_active

### [2026-03-02 07:30] V1 性能分析 + P0/P1 修复（类名重复 & loss 缩放）
- **Status**: Completed
- **V1 基线结果 (Average ODinW AP = 23.96)**:
    | 数据集 | 新类别 | K_active | max_iter | 初始loss | 最终loss | AP |
    |--------|--------|---------|----------|---------|---------|-----|
    | AerialMaritimeDrone | 5(首) | 64 | 200 | 24681 | 330 | 3.68 |
    | CottontailRabbits | 1 | 80 | 200 | 22889 | 5989 | **63.71** |
    | Egohands | 1 | 96 | 200 | 4550 | 2672 | 11.26 |
    | NorthAmericaMushrooms | 2 | 112 | 200 | 22018 | 7331 | 25.07 |
    | Packages | 1 | 128 | 200 | 10777 | 4942 | **44.73** |
    | PascalVoc | 20 | 144 | 2000 | 26913 | 7933 | 2.24 |
    | Raccoon | 1 | 160 | 200 | 22747 | 17419 | **43.44** |
    | ShellfishOpenImages | 3 | 176 | 400 | 23452 | 10540 | 10.71 |
    | VehiclesOpenImages | 5 | 192 | **40** | 38791 | 38791 | 9.28 |
    | Aquarium | 7 | 208 | 200 | 24113 | 7394 | 3.65 |
    | Pistols | 1 | 224 | 200 | 54416 | 43284 | **49.73** |
    | Pothole | 1 | 240 | 200 | 40253 | 23241 | 11.04 |
    | ThermalDogsAndPeople | 2 | 256 | **40** | 68827 | 68827 | **32.98** |
- **发现的关键问题**:
    1. **P0 类名重复 Bug**: `all_class_names` 中 boat/car/dog/person 等出现多次（Task0 和 PascalVoc 都有 boat/car），`set_eval_task` 只匹配首次出现的全局 ID，导致后续任务的同名类评估错误
    2. **P0 iter 数不足**: VehiclesOpenImages/ThermalDogsAndPeople 只有 40 iter，完全没训练
    3. **P1 loss 缩放异常**: `loss_cls` 被乘以 `Y_logits.shape[1]`（=900 queries）再乘 `cls_weight=2.0`，总计放大 1800 倍。梯度 norm 数百级别被 `clip_grad_norm(0.1)` 裁剪 1800 倍，导致实际学习率极低
    4. **P2 loss 震荡**: PascalVoc 2000 iter 的 loss 在 1833~18378 间跳动，未收敛
- **V1 结果已保存**: `output/ncrn_10shot_v1/`
- **Dev Notes**:
    - 高 AP 数据集共性：单类别 + 视觉特征独特（CottontailRabbits/Pistols/Raccoon/Packages）
    - 低 AP 数据集共性：多类别 + iter 不足 + loss 震荡（PascalVoc/AerialMaritimeDrone/Aquarium）
    - 后续优化路线：P0 修复 → P1 loss 缩放 → P2 lr scheduler → P3 Omega 稀疏初始化
- **Changes (变更详情) — 修复部分**:
    - 📂 `main_ncrn.py`:
        - ⚙️ `all_class_names` → `class_registry: list[(task_idx, [names])]`：每个任务的类名独立存储，避免跨任务重名冲突（boat/car/dog/person 等在多个任务中出现）
        - ⚙️ `save_ncrn_shared` / `load_ncrn_shared`：checkpoint 结构从 `all_class_names` 改为 `class_registry`，保持向后兼容
        - ⚙️ `NCRNInferenceModel.set_eval_task`：完全重写。不再按名字首次匹配，改为先找与评估类名重叠最多的任务 block，再在该 block 内按偏移量精确映射全局索引
        - ⚙️ `NCRNTrainer.__init__`: `cls_weight` 从 2.0 → 1.0
        - ⚙️ `NCRNTrainer.train_step`: 移除 `sigmoid_focal_loss(...) * Y_logits.shape[1]` 中的 `* Y_logits.shape[1]`（移除 ×900 乘子）
        - ⚙️ `clip_grad_norm_` 的 `max_norm` 从 0.1 → 1.0
- **Pitfalls & Solutions (踩坑与修复)**:
    - ❌ *Issue*: V1 中 `all_class_names=['boat','car',...,'boat','car','dog',...,'person',...]`，`set_eval_task("boat")` 总是匹配到 Task 0 的 boat（全局 ID 0），导致 PascalVoc 的 boat 预测使用了错误的分类器
    - ✅ *Fix*: 引入 `class_registry` 以任务为单位注册类名，评估时先找最匹配任务 block，再在 block 内查找
    - ❌ *Issue*: `sigmoid_focal_loss` 已除以 `num_boxes` 归一化，DETR 论文乘 `nq` 是因为 query 维度参与了 softmax，但 NCRN 用的是独立 sigmoid，不需要 nq 缩放。乘以 900 导致 loss 值数千~数万，`clip_grad_norm(0.1)` 几乎将梯度裁为零
    - ✅ *Fix*: 移除 `* Y_logits.shape[1]`，将 `cls_weight` 从 2.0 降为 1.0，`max_norm` 从 0.1 放宽到 1.0

### [2026-03-02 07:00] 数值下溢修复 + 评估类别映射修复 ✅
- **Status**: Completed
- **Changes (变更详情)**:
    - 📂 `fuzzy_ops.py` — `log_product_t_norm` 从乘积语义改为几何平均语义
        - ⚙️ 修复前: `result = exp(sum(W * log(L)))` — 当 K_active > 64 时，约 80 个 log(<1) 项累加导致 exp(极大负数)=0，Y_hat 全为零
        - ⚙️ 修复后: `result = exp(sum(W * log(L)) / W.sum().clamp(min=1))` — 除以有效文字数，转化为几何平均，防止连乘下溢
        - ⚙️ 数学含义: 从 "所有因子的乘积" 变为 "所有因子的几何平均"，两者都是合理的 AND 操作
    - 📂 `main_ncrn.py` — 评估阶段修复
        - ⚙️ `NCRNInferenceModel` 新增 `set_eval_task(task_categories)` 方法，评估每个数据集时只取该数据集对应的全局类别列，映射回本地类别 ID
        - ⚙️ `do_eval_all` 中为每个评估数据集调用 `set_eval_task`
        - ⚙️ 修复 `TaskMemory().task_mapping is None` 导致 `apply_lora` 报错（评估模式下初始化为空 dict）
- **Pitfalls & Solutions (踩坑与修复)**:
    - ❌ *Issue*: Task 0 (首个任务, K_active=64) 的 loss 能正常下降 (8.06→1.15)，但 Task 1+ (K_active=80+) 的 loss 卡在 8.0620 完全不变
    - ✅ *Root Cause*: `log_product_t_norm` 中 `exp(sum(W*log(L)))` 在 2*K_active 维上求和。当 K_active≥80 时，约一半的 W=1（因 sigmoid(0.01*randn)≈0.5），~80 个 log(0.5)≈-0.69 累加后 sum≈-55，exp(-55)=0。Task 0 虽然初始也为 0，但所有 5 类可训练，梯度能累积逃出；Task 1+ 只有 1 类可训练（旧类冻结），梯度太微弱
    - ✅ *Fix*: 将 sum 除以有效文字数 `num_active = W.sum(dim=-1).clamp(min=1)`，转化为几何平均
    - ❌ *Issue*: 评估时 NCRN 输出全局类别 ID (0~50+)，但 COCO Evaluator 期望每个数据集的本地 ID
    - ✅ *Fix*: `set_eval_task` 根据数据集类别名在全局类别列表中查找索引，只取对应列
    - ❌ *Issue*: 评估模式 `TaskMemory().task_mapping` 为 None → `apply_lora` 报 AttributeError
    - ✅ *Fix*: 评估前初始化 `TaskMemory().task_mapping = {}`
- **Dev Notes (开发备忘/自由空间)**:
    - 10-shot 全 13 任务训练耗时 937s (15.6min)，评估约 11min
    - Average ODinW AP = 23.96，部分数据集表现较好 (CottontailRabbits 63.71, Pistols 49.73, Packages 44.73)
    - COCO zero-shot AP 仅 0.08（预期如此：COCO 80 类大部分不在 ODinW13 训练集中）
    - 后续优化方向: loss 值仍偏高（几千级别），可能需要调 lr/weight_decay/max_iter；几何平均改变了 AND 的严格程度，可能需要配合调整 Omega 初始化
    - debug log 证据: Task 1 修复前 Y_hat.max=3.1e-6 (下溢), 修复后 Y_hat 在 [0.88, 0.99]，梯度 norm 从 1.28e-5 提升到 182.0 (提升 4-5 个数量级)

### [2026-03-02 05:35] DataLoader 死锁修复 + 数值下溢实验(已回滚) 🔄
- **Status**: In Progress
- **Changes (变更详情)**:
    - 📂 `File/Folder`: 修改 `main_ncrn.py` — 增量更新使用独立 DataLoader 实例
        - ⚙️ 修复前：增量更新和训练复用同一个 `train_loader`，`del temp_iter` 后重新 `iter(train_loader)` 导致 detectron2 `ToIterableDataset` 死锁
        - ⚙️ 修复后：增量更新创建独立 `sample_loader = instantiate(cfg.dataloader.train)`，用完即 `del`；训练用新建的 `train_loader`
    - 📂 `File/Folder`: `fuzzy_ops.py` — 尝试全程 log-space 计算（已回滚）
    - 📂 `File/Folder`: `dnf_engine.py` — 尝试 Omega 负偏移初始化（已回滚）
    - 📂 `File/Folder`: 新增 `run_ncrn_cl.sh` — 正式 10-shot 连续学习编排脚本，支持 GPU 参数
    - 📂 `File/Folder`: 新增 `run_parallel_debug.sh` — 4 GPU 并行诊断脚本
    - 📂 `File/Folder`: 新增 `debug_task2.py` — Task 2 精确定位脚本（诊断用）
- **Pitfalls & Solutions (踩坑与修复)**:
    1. **detectron2 ToIterableDataset 重复迭代死锁**
       - ❌ *Issue*: Task 1+ 训练时，`iter(train_loader)` 卡在 `next(data_iter)` 上，GPU 0% CPU 100%，无输出无报错
       - ❌ *根因*: 增量更新阶段对 `train_loader` 创建了 `temp_iter` 并消费了部分数据，`del temp_iter` 后再次 `iter(train_loader)` 触发 detectron2 `ToIterableDataset.__iter__` 中的某种状态异常/死锁
       - ✅ *Fix*: 增量更新使用独立的 DataLoader 实例 `sample_loader`，用完即销毁；训练阶段创建全新的 `train_loader`
       - ✅ *验证*: Task 0 + Task 1 在 81 秒内顺序完成，不再卡住
    2. **fuzzy_ops 数值下溢实验（已回滚）**
       - ❌ *Issue*: `log_product_t_norm` 中 `exp(sum(W·log(L)))` 在 K_active≥64 时下溢为 0（诊断报告 `docs/NCRN_Training_Halt_Debug_Report.md` 详细记录了此问题）
       - 🔧 *尝试*: 全程 log-space 计算 + 几何平均归一化 + logsumexp→sigmoid 聚合 + Omega 负偏移初始化
       - ❌ *结果*: 连续学习流程不再卡死，但模型不收敛（loss 从 4646→491 但不继续下降），原因是修改后 Y_hat 初始值~0.8 过高且方差过小
       - ✅ *决定*: 回滚 `fuzzy_ops.py` 和 `dnf_engine.py` 到原始逻辑，数值下溢问题留待后续用更细致的方案解决
    3. **实际卡住的根因不是数值下溢而是 DataLoader 死锁**
       - 之前误判为数值问题，实际上 Task 0 在原始逻辑下也能正常训练完成（loss 从 4.03 收敛到 0.3-0.6）
       - 真正的阻塞点是 `next(data_iter)` 而非 `train_step` 内部
- **Dev Notes (开发备忘/自由空间)**:
    - **连续学习已验证通过**: 4 个任务顺序执行成功（Task 0-3），每个任务 ~40s，共享 checkpoint 正确传递
    - **数值下溢是独立问题**: 虽然 Y_hat 在大 K_active 时可能下溢为 0，但：
        - Task 0 (K=64) 训练时 loss 正常收敛
        - 下溢主要影响后续任务的 NCRN 输出质量，不影响训练流程本身
        - 需要更细致的方案（如 top-k 稀疏 STE、分段 softmax 聚合等），留待后续优化
    - **正式训练命令**:
      ```bash
      cd /opt/data/private/why/DitHub
      nohup bash run_ncrn_cl.sh ./output/ncrn_10shot 3 0 > output/ncrn_10shot_run.log 2>&1 &
      ```
    - **下一步**: 解决数值下溢问题后重新跑正式的 13 任务连续学习

---

### [2026-03-02 05:00] 切换至 10-shot + 并行诊断测试通过 ✅
- **Status**: Completed
- **Changes (变更详情)**:
    - 📂 `File/Folder`: 修改 `run_ncrn_cl.sh` — CONFIG_DIR 从 `test/test_odinw13` 改为 `test/test_odinw13_10shot`
    - 📂 `File/Folder`: 修改 `run_parallel_debug.sh` — 同上
    - 📂 `File/Folder`: 修改 `train_ncrn.sh` — config-file 改为 10shot
    - 📂 `File/Folder`: 修改 `eval_ncrn.sh` — config-file 改为 10shot
    - 📂 `File/Folder`: 新增 `run_parallel_debug.sh` — 4 GPU 并行诊断脚本（每个数据集独立 task_index=0）
    - ⚙️ `Function/API`: 10-shot 配置关键差异：`iter_per_epoch=20`（vs 全量 300），数据集路径 `odinw_10shot/`，每任务 max_iter=200
- **Pitfalls & Solutions (踩坑与修复)**:
    - ❌ *Issue*: 之前一直用全量数据集（odinw13）跑测试，iter 数量 2000-30000 太慢
    - ✅ *Fix*: 切换到 10-shot（odinw13_10shot），每任务只需 200 iter，单任务 ~30s 完成
- **Dev Notes (开发备忘/自由空间)**:
    - **并行诊断结果**: 4/4 数据集在 4 张 GPU 上独立并行全部成功，60 秒内完成
    - AerialMaritimeDrone: loss 4.03→0.75，GPU 2.0GB，checkpoint 2.7M
    - CottontailRabbits: loss 4.03→0.20，GPU 1.9GB，checkpoint 2.2M
    - Egohands: loss 4.03→0.08，GPU 2.4GB，checkpoint 2.2M
    - NorthAmericaMushrooms: loss 4.03→0.35，GPU 1.9GB，checkpoint 2.3M
    - **确认**: 独立进程架构下无 mmap 问题，无卡死，无静默退出
    - **10-shot 任务 iter 分布**: 大部分 200 iter，PascalVoc 最大 2000 iter，ShellfishOpenImages 400 iter

---

### [2026-03-02 04:30] NCRN 单任务进程架构重构 + 冒烟测试修复 ✅
- **Status**: Completed
- **Changes (变更详情)**:
    - 📂 `File/Folder`: 完全重写 `main_ncrn.py` — 从"单进程多任务循环"改为"单任务独立进程"模式
        - 新增 `do_single_task_train(args)` — 单任务训练入口，接收 `--task-config` 和 `--task-index`
        - 新增 `do_eval_all(args)` — 全任务评估入口，使用 `--eval-only`
        - 新增 `save_ncrn_shared()` / `load_ncrn_shared()` — 磁盘共享 checkpoint 保存/加载
        - 新增 `create_fresh_ncrn()` — 首任务 NCRN_Head 创建工厂
        - 移除 `detectron2.engine.launch()` 调用 — 单 GPU 不再 fork 子进程
        - 移除旧的多任务 for 循环 (`_main_inner`)
        - 新增 `--task-config`, `--task-index`, `--total-tasks` CLI 参数
    - 📂 `File/Folder`: 新增 `run_ncrn_cl.sh` — 连续学习编排脚本
        - 自动枚举 `test/test_odinw13/for_train/*.py` 中的所有任务配置
        - 顺序调用 `python main_ncrn.py --task-config ... --task-index N`
        - 每个任务是独立 Python 进程，进程退出后释放全部 mmap/内存
        - 任务失败时立即停止并报错（`set -e` + exit code 检查）
        - 训练完成后自动启动评估
    - ⚙️ `Function/API`:
        - 共享 checkpoint 格式 `ncrn_shared.pth`：`{ncrn_head, K_active, num_classes, task_count, all_class_names}`
        - 增量学习状态传递：`frozen_mask`/`frozen_class_mask` (register_buffer) 在 state_dict 中自动序列化
        - `K_active` 和 `num_classes` 作为 Python 属性需额外保存，恢复时通过构造函数 `init_active=ckpt['K_active']` + `num_classes=ckpt['num_classes']` 传入
    - 🔧 `Tech`:
        - `num_workers=0` — 避免 DataLoader fork 导致 mmap 耗尽
        - `torch.multiprocessing.set_sharing_strategy('file_descriptor')` — 减少 mmap 使用
        - `gc.collect()` — 模型加载后和任务间显式回收 CPU 内存
        - `traceback` 模块 — train_step 异常捕获 + 详细日志
        - GPU 显存日志 — 每 50 iter 打印 `torch.cuda.max_memory_allocated()`
- **Pitfalls & Solutions (踩坑与修复)**:
    1. **max_map_count=65530 导致 mmap 失败**
       - ❌ GroundingDINO (BERT+SwinT+Transformer) 占用大量 mmap 映射；DataLoader worker fork 后继承所有映射，超出 65530 限制
       - ❌ 错误信息：`RuntimeError: unable to mmap 786432 bytes from file: Cannot allocate memory (12)`
       - ❌ 容器环境 `/proc/sys/vm/max_map_count` 只读，无法修改
       - ✅ 方案一：`num_workers=0` 消除 fork（但仍因进程内 mmap 累积在 Task 2 时被 OS 杀死）
       - ✅ 最终方案：每个任务独立进程，进程退出后 OS 自动回收所有 mmap 映射
    2. **detectron2.engine.launch() 在 num_gpus=1 时仍 fork 子进程**
       - ❌ launch() 使用 `torch.multiprocessing.start_processes`，即使单 GPU 也会 fork，导致 mmap 翻倍
       - ✅ 完全移除 launch()，直接调用 main()
    3. **进程被 OOM killer 静默杀死，无 Python traceback**
       - ❌ 进程在 Task 2 开始训练时被杀，日志无错误信息
       - ✅ 通过 `dmesg` 和 `/proc/<pid>/maps` 数量（56780 个 fd）定位到 mmap 耗尽
       - ✅ 新架构中每个进程只处理一个任务，内存占用可控
    4. **NCRN_Head 的 K_active/num_classes 不在 state_dict 中**
       - ❌ 这两个是 Python 属性（非 Parameter/Buffer），`load_state_dict` 不会恢复
       - ✅ checkpoint 中显式保存，恢复时通过构造函数参数 `init_active`/`num_classes` 传入
- **Dev Notes (开发备忘/自由空间)**:
    - **新的运行方式**:
      ```bash
      cd /opt/data/private/why/DitHub
      nohup bash run_ncrn_cl.sh ./output/ncrn_smoke 3 > output/ncrn_cl.log 2>&1 &
      ```
    - **Task 0 验证通过**: AerialMaritimeDrone 3000 iter，loss 从 4.03 收敛到 0.3-0.6，GPU 峰值 2018MB，进程正常退出
    - **Task 1 验证通过**: CottontailRabbits 增量更新成功（K_active: 64→80, classes: 5→6），从磁盘共享 checkpoint 正确恢复
    - **关键架构决策**: 放弃"单进程多任务循环"，改为"每任务独立进程 + 磁盘 checkpoint"。虽然每个任务都要重新加载 GroundingDINO（~15s），但完全避免了 mmap 累积问题
    - **连续学习信息保证**: `frozen_mask`/`frozen_class_mask` 作为 register_buffer 在 state_dict 中传递，确保旧概念/旧类别冻结状态跨进程保持
    - **预计总时间**: 13 任务训练 ~1.5h（每任务重加载 GroundingDINO 增加 ~3min 总开销）

---

### [2026-03-01 18:00] NCRN 端到端训练+评估集成（P0 任务）🔄
- **Status**: In Progress（冒烟测试运行中）
- **Changes (变更详情)**:
    - 📂 `File/Folder`: 完全重写 `main_ncrn.py` — NCRN 训练/评估入口
        - 新增 `NCRNHungarianMatcher` 类 — 基于 Focal Cost + L1 + GIoU 的匈牙利匹配器
        - 新增 `NCRNInferenceModel` 类 — 封装 GroundingDINO + NCRN_Head 的推理模型
        - 重写 `NCRNTrainer.train_step()` — 集成 Hungarian Matching + Focal Loss，替代简化 BCE
        - 重写 `extract_features()` — 共享的特征提取函数（训练+推理复用）
        - 新增 `do_test_ncrn()` — NCRN 专用评估函数
        - 修复类别数初始化：从首个任务配置文件读取实际类别数，而非硬编码 10
        - 修复增量更新：使用 decoder 输出 hs 的均值作为新任务特征（而非 backbone 最后层）
        - 修复模型复用：GroundingDINO 只加载一次并跨任务复用，避免内存泄漏
    - 📂 `File/Folder`: 修复 `train_ncrn.sh` / `eval_ncrn.sh` — OUTPUT_DIR 默认路径添加 `./` 前缀
    - 📂 `File/Folder`: 创建 `datasets/` 软链接 — odinw13 + coco 指向 DitHub-ori
    - 📂 `File/Folder`: 复制 `MultiScaleDeformableAttention.cpython-39-x86_64-linux-gnu.so` — CUDA 算子
    - 📂 `File/Folder`: 创建 `groundingdino_swint_ogc.pth` 软链接
    - ⚙️ `Function/API`:
        - `NCRNHungarianMatcher.forward()` — 输入 NCRN 概率 [0,1]，内部转 focal cost
        - `NCRNInferenceModel.forward()` — 接管 GroundingDINO 推理，用 NCRN 替代 ContrastiveEmbed
        - `extract_features()` — 冻结 GroundingDINO 前向，返回 hs/reference/pred_boxes/targets
        - `sigmoid_focal_loss()` — 从 DitHub criterion 移植的 focal loss
    - 🔧 `Tech`:
        - 添加 `import sys`
        - 添加 `import groundingdino.models.GroundingDINO.groundingdino_dt` 触发模块注册
        - 替换 `seed_all_rng` 的导入源：`detectron2.utils.env` → 正确位置
        - 设置 `TOKENIZERS_PARALLELISM=false` 避免 fork 警告
- **Pitfalls & Solutions (踩坑与修复)**:
    1. **`seed_all_rng` 导入错误**
       - ❌ 原代码从 `groundingdino.util.misc` 导入，但该函数实际在 `detectron2.utils.env`
       - ✅ 修正导入路径
    2. **`dtgroundingdino` 模块未注册**
       - ❌ `GroundingDINO/__init__.py` 改为延迟导入后，`@MODULE_BUILD_FUNCS.registe_with_name` 装饰器不再在模块加载时执行
       - ✅ 在 `main_ncrn.py` 中显式 `import groundingdino.models.GroundingDINO.groundingdino_dt` 触发注册
    3. **CUDA Ops 缺失**
       - ❌ `MultiScaleDeformableAttention` .so 文件不在 DitHub 目录
       - ✅ 从 DitHub-ori 复制已编译的 .so 文件
    4. **内存泄漏 (Cannot allocate memory)**
       - ❌ 每个任务循环中重新 `load_grounding_dino()` 导致模型累积占用 GPU 内存
       - ✅ 将 GroundingDINO 加载移到循环外，所有任务共享同一模型实例
    5. **TOKENIZERS_PARALLELISM 警告**
       - ❌ HuggingFace tokenizer 在 fork 后的子进程中产生死锁警告
       - ✅ 设置 `os.environ["TOKENIZERS_PARALLELISM"] = "false"`
- **Dev Notes (开发备忘/自由空间)**:
    - **冒烟测试命令**:
      ```bash
      cd /opt/data/private/why/DitHub
      export LD_LIBRARY_PATH=/opt/data/private/conda_envs/dithub/lib/python3.9/site-packages/torch/lib:$LD_LIBRARY_PATH
      export HF_HUB_OFFLINE=1
      /opt/data/private/conda_envs/dithub/bin/python -u main_ncrn.py \
          --config-file test/test_odinw13 \
          --model-config-file groundingdino/config/GroundingDINO_SwinT_OGC_dt_ncrn.py \
          --model-checkpoint-path groundingdino_swint_ogc.pth \
          --output-dir ./output/ncrn_smoke \
          --seed 3 --num-gpus 1 --dithub
      ```
    - **训练速度**: ~7s/50iter (Task 1 AerialMaritimeDrone, 3000 iter, ~7 min/task)
    - **Loss 趋势**: Task 1 初始 loss_cls=4.03 → 收敛到 0.3-0.6 区间
    - **增量更新验证**: Task 2 开始时 K_active: 64→80, total_classes: 5→6，零空间投影正常
    - **架构决策**: NCRN 只替换分类头(ContrastiveEmbed)，box 回归头(bbox_embed)保持冻结复用
    - **当前状态**: 全 13 任务冒烟测试正在运行中，预计 ~1.5h 完成（含评估）
    - **下一步**: 冒烟测试通过后，需观察评估阶段 NCRN 推理是否能产出有意义的 AP 值

---

### [2026-03-01] 失败案例分析与可视化 ✅

- **变更内容**:
  - 创建 `scripts/analyze_failures.py` — 多维度错误分析系统
    - 四种错误类型分类: Localization / Classification / Background / Missed
    - 混淆矩阵生成
    - 统计报告 (JSON + Markdown)
    - 对比可视化 (GT绿色 vs Pred红色)
  - 创建 `scripts/visualize_full_comparison.py` — 全量GT vs Prediction对比图生成

- **输出结果**:
  - `output/failure_analysis/summary_report.md` — 13数据集汇总报告
  - `output/failure_analysis/<dataset>/` — 各数据集详细分析
    - `statistics.json` — 统计数据
    - `confusion_matrix.png` — 混淆矩阵
    - `localization_errors/` — 定位错误可视化
    - `classification_errors/` — 分类错误可视化
    - `background_errors/` — 背景误报可视化
    - `missed_detections/` — 漏检可视化
  - `output/full_comparison/VehiclesOpenImages/` — 126张全量对比图

- **关键发现**:
  - 整体准确率: 92.4% (11456/12397)
  - 最大问题: False Positive (20525个)，主要集中在 PascalVOC (16803)
  - NorthAmericaMushrooms 分类错误最多 (292)，符合预期（蘑菇种类混淆）
  - AerialMaritimeDrone 定位困难 (小目标 + 航拍视角)

- **踩坑记录**:
  1. **COCO API 迭代**: `coco_dt = coco_gt.loadRes()` 返回 COCO 对象，需用 `coco_dt.anns.items()` 迭代，不能直接 `for ann in coco_dt`
  2. **数据集路径不一致**: ODinW-13 各数据集目录结构差异大，需从 `groundingdino/config/configs/common/data/odinw/*.py` 提取正确路径
  3. **pistols 数据集特殊命名**: 标注文件名为 `test_annotations_without_background.json`，与其他数据集不同

### [2026-03-02] 服务器部署 click-to-run 脚本 ✅

- **变更内容**:
  - 创建 `main_ncrn.py` — 自包含 NCRN 训练/评估入口（不依赖 main.py 导入）
  - 创建 `groundingdino/config/GroundingDINO_SwinT_OGC_dt_ncrn.py` — NCRN 超参数配置
  - 创建 `train_ncrn.sh` / `eval_ncrn.sh` — click-to-run 训练/评估脚本
  - 创建 `scripts/setup_server.sh` — 一键 4090 环境部署（conda + PyTorch CUDA 11.8 + detectron2 + Deformable-DETR 算子编译）
  - 创建 `scripts/download_data.sh` — ODinW-35 数据下载
  - 修改 `GroundingDINO/__init__.py` — 延迟导入解耦

- **踩坑记录**:
  1. **main.py 无 parse_args**: 参数解析内联在 `__main__` 块，无法直接 `from main import parse_args`，改为在 main_ncrn.py 中自包含
  2. **main.py 的 DetectionLoraCheckpointer**: 实际来自 `groundingdino.util.lora_utils`，不是 main.py 定义
  3. **GroundingDINO **init**.py 导入链**: 急切导入触发 transformers→timm→detectron2 依赖链，改为延迟导入
  4. **timm 安装**: timm 0.9.16/1.0.8 在 Python 3.9 上 `version.py` 缺失，需手动创建

- **开发备忘**:
  - 服务器使用: `bash scripts/setup_server.sh && bash scripts/download_data.sh && bash train_ncrn.sh`
  - NCRN 作为并行分支，不替换 ContrastiveEmbed
  - 评估暂复用 GroundingDINO 原始 inference 流程

- **变更内容**:
  - 创建 `groundingdino/models/GroundingDINO/ncrn/` 包，实现四个核心模块:
    - `concept_dict.py` — 正交概念字典（SVD 零空间投影扩容）
    - `fuzzy_ops.py` — 对数空间模糊逻辑算子（T-norm/T-conorm）
    - `dnf_engine.py` — 可微 DNF 规则引擎（Binary STE + 逻辑正则化）
    - `ncrn_head.py` — 顶层封装（forward/compute_loss/incremental_update）
  - 创建 `tests/test_ncrn.py` — 9 个单元测试 + 1 个 COCO 128 冒烟测试
  - 修改 `GroundingDINO/__init__.py` — 延迟导入解耦 ncrn 与重量级依赖

- **踩坑记录**:
  1. **conda 权限**: macOS SIP 限制导致 `/Users/why/.conda` 和 miniforge3 目录无写权限，需用户 `sudo chmod -R u+w` 手动修复
  2. **timm 安装**: timm 0.9.16/1.0.8 在 Python 3.9 上安装后 `version.py` 缺失，需手动创建
  3. **导入链**: `GroundingDINO/__init__.py` 急切导入整条依赖链（transformers→timm→detectron2），改为延迟导入 `build_dt_groundingdino` 解决
  4. **MPS SVD**: `torch.linalg.svd` 在 MPS 上不支持，自动回退 CPU 执行（UserWarning 可忽略，不影响正确性）
  5. **MPS 噪声**: MPS 运行时产生大量 "Error creating directory" 日志，是 macOS 权限问题，不影响计算

- **开发备忘**:
  - 环境: `conda activate dithub` (Python 3.9 + PyTorch 2.8.0 + MPS)
  - 测试: `conda run -n dithub python tests/test_ncrn.py --smoke-test`
  - 所有新代码兼容 CUDA/MPS/CPU，设备选择 `cuda > mps > cpu`
  - 零空间投影正交性验证: max_dot = 0.000000（完美）
  - COCO 128 数据集: `/Volumes/SSD512/Code/dataset/COCO 128.v2-640x640.coco`

### [2026-03-02] 初始化项目进度文件

- **变更内容**: 创建 `progress.md`，开始 NCRN 工程实施
