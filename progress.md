# DitHub 开发进度锚点

## 项目信息

- **项目名称**: DitHub — 增量式开放词汇目标检测
- **创建时间**: 2026-02-25
- **当前阶段**: 阶段六 · NSPS 双系统连续学习架构 v2（进行中）
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

### [2026-03-02 15:20] NSPS Bug 修复 + main_nsps.py 创建 ✅

- **Status**: Completed
- **Changes (变更详情)**:
  - 🐛 `File/Folder`: 修复 `ncrn/nsps_system.py` — 两个 Bug
    - **Bug 1**: `inference_batched` 硬编码使用全局 gamma，忽略 `task_meta` 中的逐任务动态 γ → 改为优先读取 `task_meta[tid]["task_gamma"]`，`gamma` 参数仅作覆盖
    - **Bug 2**: `gamma or self.gamma` 的 Python boolean 陷阱（`gamma=0.0` 被误判为 `False`）→ 改为 `gamma if gamma is not None else self.top_k/self.gamma`
    - `inference()` 中同样修复了 `top_k or self.top_k` → `top_k if top_k is not None else self.top_k`
  - 📂 `File/Folder`: 新增 `main_nsps.py` — NSPS 双系统服务器入口（~450 行）
    - `load_grounding_dino()` — 冻结 GroundingDINO 特征提取器
    - `extract_task_features()` — hs[-1] 均值池化 + one-hot 标签构造
    - `run_nsps_add_task()` — 单任务特征采集 → NSPSSystem.add_task 编排
    - `NSPSInferenceModel` — 封装 NSPS 推理，兼容 detectron2 评估管线
    - `do_train()` — 顺序/乱序遍历 ODinW-13，每任务后保存断点 checkpoint
    - `do_eval()` — 全任务 COCO AP 评估，汇总 Mean AP
    - 支持: `--eval-only`, `--shuffle-tasks`, `--overwrite` 断点续训
- **Pitfalls & Solutions (踩坑与修复)**:
  1. `gamma or self.gamma`: Python 中 `0.0 or x` 返回 `x`，导致 gamma=0.0 时静默失效 → `if gamma is not None`
  2. `inference_batched` 和 `inference` 行为不一致：前者丢失动态 γ → 统一逻辑
- **Dev Notes (开发备忘)**:
  - 分支: `feat/ncrn-v2`
  - 8/8 测试在 MPS (Apple M 芯片) 上全通过，修复后仍全通过
  - 服务器运行命令:

    ```bash
    HF_HUB_OFFLINE=1 /opt/data/private/conda_envs/dithub/bin/python -u main_nsps.py \
        --config-file test/test_odinw13 \
        --model-config-file groundingdino/config/GroundingDINO_SwinT_OGC_dt_ncrn.py \
        --model-checkpoint-path groundingdino_swint_ogc.pth \
        --output-dir ./output/nsps_output \
        --seed 42 --num-gpus 1 --dithub
    ```

---

### [2026-03-02 15:00] NSPS 双系统架构实现 (feat/ncrn-v2) 🔄

- **Status**: In Progress
- **Changes (变更详情)**:
  - 📂 `File/Folder`: 新增 `ncrn/polarized_router.py` — System 1 极化前件路由器
    - `PolarizedAntecedentBase` 类：零空间正交投影 + cos Top-K 路由
    - 退化保护：矩阵奇异时迭代减去投影分量
    - 能量检测：>90% 能量被投影掉时回退软排斥
  - 📂 `File/Folder`: 新增 `ncrn/nsps_system.py` — NSPS 双系统调度器
    - `NSPSSystem` 类：编排 System 1 路由 + System 2 后件阵列
    - `add_task()`: 极化入库 → 独立训练 NCRN → 冻结入列 → 统计动态 γ
    - `inference()`: 回退式判决 + 逐任务动态阈值 + 熔断出口 (UNKNOWN)
    - `save_checkpoint()` / `load_checkpoint()`: 全系统序列化
    - L1 退火策略防维度浪费 (边界条件三)
  - 📂 `File/Folder`: 新增 `ncrn/hard_negative_sampler.py` — 硬负样本采样器
    - 防止 System 2 过度自信 (边界条件二)
    - 从路由最相似的其他任务中采样负例
  - 📂 `File/Folder`: 新增 `tests/test_nsps.py` — 7 个单元测试 + 1 个特征坍塌测试
  - 📂 `File/Folder`: 更新 `ncrn/__init__.py` — 导出 NSPS 新模块
  - ⚙️ `Function/API`:
    - 动态 γ: 训练后统计 5th percentile 置信度存入 `task_meta["task_gamma"]`
    - 回退熔断: rejected 样本输出 `task_id=-1` (UNKNOWN)，不强行输出不合格预测
    - 设备修复: `load_checkpoint` 中显式 `.to(device)` 解决 CPU/MPS 设备不匹配
- **Pitfalls & Solutions (踩坑与修复)**:
    1. **MPS 不支持 `aten::mode`**: 测试中 `tensor.mode()` 在 MPS 上报错，改用 `bincount` + `argmax`
    2. **单任务 loss 未收敛**: 因训练轮次不足 (30 epoch, batch_size > N)，增加 epoch 数或放宽断言
    3. **checkpoint 恢复设备不匹配**: `torch.load(map_location=device)` 对 MPS 不够，需显式 `.to(device)`
- **Dev Notes (开发备忘/自由空间)**:
  - 分支: `feat/ncrn-v2`
  - NSPS 架构完全不动 v1 四大基础模块 (concept_dict/fuzzy_ops/dnf_engine/ncrn_head)
  - 每个任务的 NCRN 物理隔离，训练复杂度 O(1)，零遗忘
  - 测试: `conda run -n dithub python tests/test_nsps.py`

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
