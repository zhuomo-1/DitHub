# DitHub 开发进度锚点

## 项目信息

- **项目名称**: DitHub — NCRN 神经符号概念推理网络
- **创建时间**: 2026-03-02
- **当前阶段**: 阶段三 · 服务器部署 click-to-run（已完成）

---

## 变更日志

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
