# -*- coding: utf-8 -*-
"""
NSPS 双系统连续学习入口 (服务器版)
====================================
在 GroundingDINO 冻结特征之上，编排 NSPSSystem 进行 ODinW-13
的多任务增量学习。每个任务独立训练一个 NCRN 实例，极化入库后冻结。

架构:
    GroundingDINO (冻结) → hs [B, nq, D] → 均值池化 → Z [B, D]
                                                           ↓
                                              NSPSSystem.add_task(Z, Y)
                                                           ↓
                                              NSPSSystem.inference(Z)

用法:
    # 完整连续学习实验（顺序训练 + 末尾评估）
    python -u main_nsps.py \\
        --config-file test/test_odinw13 \\
        --model-config-file groundingdino/config/GroundingDINO_SwinT_OGC_dt_ncrn.py \\
        --model-checkpoint-path groundingdino_swint_ogc.pth \\
        --output-dir ./output/nsps_output \\
        --seed 42 --num-gpus 1 --dithub

    # 仅评估（加载已有 nsps_checkpoint/）
    python -u main_nsps.py ... --eval-only

    # 打乱任务顺序
    python -u main_nsps.py ... --shuffle-tasks
"""

import argparse
import gc
import glob
import logging
import os
import random
import sys
import traceback
from pathlib import Path

import torch
import torch.nn.functional as F

# detectron2
from detectron2.config import LazyConfig, instantiate
from detectron2.engine.defaults import setup_logger, _try_get_key, CfgNode
from detectron2.utils.file_io import PathManager
from detectron2.evaluation import COCOEvaluator, print_csv_format
from detectron2.modeling import detector_postprocess
from detectron2.structures import Boxes, Instances

# DitHub 核心
from groundingdino.models import build_model
from groundingdino.util.slconfig import SLConfig
from groundingdino.util.utils import clean_state_dict
from groundingdino.util.misc import NestedTensor, nested_tensor_from_tensor_list, inverse_sigmoid
from detectron2.utils.env import seed_all_rng
from groundingdino.util.task_memory import TaskMemory
from groundingdino.util.lora_utils import get_lora_modules, apply_lora
from groundingdino.util.box_ops import box_cxcywh_to_xyxy
from groundingdino.config.configs.common.data.odinw.mapping_classes import (
    ODINW_13_FILE_MAPPING,
    ODINW_OVERLAPPED_FILE_MAPPING,
)
from groundingdino.models.GroundingDINO.bertwarper import (
    generate_masks_with_special_tokens_and_transfer_map,
)

import groundingdino.models.GroundingDINO.groundingdino_dt  # noqa: F401 — 触发模块注册

# NSPS 核心（v2 双系统）
from groundingdino.models.GroundingDINO.ncrn import NSPSSystem

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), os.path.pardir)))
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

logger = logging.getLogger("nsps")

NSPS_CKPT_DIR = "nsps_checkpoint"


# ============================================================
# 设备工具
# ============================================================

def auto_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


# ============================================================
# GroundingDINO 加载（冻结）
# ============================================================

def load_grounding_dino(model_config_path: str, model_checkpoint_path: str,
                        output_dir: str, device: torch.device) -> torch.nn.Module:
    """加载并冻结 GroundingDINO，仅用于特征提取。"""
    cfg_model = SLConfig.fromfile(model_config_path)
    cfg_model.device = str(device)
    model = build_model(cfg_model)
    checkpoint = torch.load(model_checkpoint_path, map_location="cpu")
    model.load_state_dict(clean_state_dict(checkpoint["model"]), strict=False)

    if getattr(cfg_model, 'dithub', False):
        model.load_custom_attention()
        lora_modules = get_lora_modules(model, cfg_model.lora_out_min)
        apply_lora(modules=lora_modules, r=cfg_model.lora_r,
                   lora_alpha=cfg_model.lora_alpha, lora_dropout=cfg_model.lora_dropout)

    for param in model.parameters():
        param.requires_grad = False
    model.eval()
    model.to(device)
    logger.info(f"GroundingDINO 已加载并冻结 ({model_config_path})")
    return model


# ============================================================
# 特征提取（从 GroundingDINO decoder 最后一层提取 hs 均值）
# ============================================================

@torch.no_grad()
def extract_task_features(gd_model, task_loader, device: torch.device,
                          max_samples: int = 500):
    """
    从 DataLoader 中提取全部样本的特征和标签，返回 Z [N, D], Y [N, C]。

    特征: hs[-1].mean(dim=1) → [B, D]（Decoder 最后一层 query 均值池化）
    标签: one-hot [B, C]，C = 当前任务类别数
    """
    gd_model.eval()
    Z_list, Y_list = [], []
    n_collected = 0
    num_classes = None

    for batch in task_loader:
        if n_collected >= max_samples:
            break
        try:
            images = gd_model.preprocess_image(batch)
            samples = nested_tensor_from_tensor_list(images)
            captions = [x["captions"] for x in batch]
            names_list = [x["captions"][:-1].split(".") for x in batch]

            tokenized = gd_model.tokenizer(
                captions, padding="longest", return_tensors="pt"
            ).to(device)
            (
                text_self_attention_masks,
                position_ids,
                _cate_to_token_mask_list,
            ) = generate_masks_with_special_tokens_and_transfer_map(
                tokenized, gd_model.specical_tokens, gd_model.tokenizer
            )

            if text_self_attention_masks.shape[1] > gd_model.max_text_len:
                L = gd_model.max_text_len
                text_self_attention_masks = text_self_attention_masks[:, :L, :L]
                position_ids = position_ids[:, :L]
                tokenized["input_ids"] = tokenized["input_ids"][:, :L]
                tokenized["attention_mask"] = tokenized["attention_mask"][:, :L]
                tokenized["token_type_ids"] = tokenized["token_type_ids"][:, :L]

            if gd_model.sub_sentence_present:
                enc_input = {k: v for k, v in tokenized.items() if k != "attention_mask"}
                enc_input["attention_mask"] = text_self_attention_masks
                enc_input["position_ids"] = position_ids
            else:
                enc_input = tokenized

            bert_out = gd_model.bert(**enc_input)
            encoded_text = gd_model.feat_map(bert_out["last_hidden_state"])
            text_token_mask = tokenized.attention_mask.bool()
            if encoded_text.shape[1] > gd_model.max_text_len:
                L = gd_model.max_text_len
                encoded_text = encoded_text[:, :L, :]
                text_token_mask = text_token_mask[:, :L]
                position_ids = position_ids[:, :L]
                text_self_attention_masks = text_self_attention_masks[:, :L, :L]

            text_dict = {
                "encoded_text": encoded_text,
                "text_token_mask": text_token_mask,
                "position_ids": position_ids,
                "text_self_attention_masks": text_self_attention_masks,
            }

            features, poss = gd_model.backbone(samples)
            srcs, masks = [], []
            for l, feat in enumerate(features):
                src, mask = feat.decompose()
                srcs.append(gd_model.input_proj[l](src))
                masks.append(mask)
            if gd_model.num_feature_levels > len(srcs):
                _len_srcs = len(srcs)
                for l in range(_len_srcs, gd_model.num_feature_levels):
                    src = gd_model.input_proj[l](features[-1].tensors if l == _len_srcs else srcs[-1])
                    m = samples.mask
                    mask = F.interpolate(m[None].float(), size=src.shape[-2:]).to(torch.bool)[0]
                    pos_l = gd_model.backbone[1](NestedTensor(src, mask)).to(src.dtype)
                    srcs.append(src); masks.append(mask); poss.append(pos_l)

            hs, _, _, _, _ = gd_model.transformer(
                srcs, masks, None, poss, None, None, text_dict
            )
            Z_batch = hs[-1].mean(dim=1)  # [B, D] — query 均值池化

            # 构造 one-hot 标签（从 instances 取 gt_classes）
            B = Z_batch.shape[0]
            batch_labels = []
            has_gt = False
            for x in batch:
                if "instances" in x and len(x["instances"]) > 0:
                    cls_ids = x["instances"].gt_classes  # [n]
                    if num_classes is None:
                        # 从第一批确定类别数
                        nc = len(names_list[0])
                        num_classes = nc
                    y_b = torch.zeros(num_classes, device=device)
                    for c in cls_ids.tolist():
                        if 0 <= c < num_classes:
                            y_b[c] = 1.0
                    batch_labels.append(y_b)
                    has_gt = True
                else:
                    if num_classes is not None:
                        batch_labels.append(torch.zeros(num_classes, device=device))
                    else:
                        batch_labels.append(None)

            if not has_gt or any(l is None for l in batch_labels):
                continue

            Y_batch = torch.stack(batch_labels, dim=0)  # [B, C]
            Z_list.append(Z_batch)
            Y_list.append(Y_batch)
            n_collected += B

        except Exception as e:
            logger.warning(f"特征提取跳过一批: {e}")
            continue

    if not Z_list:
        raise RuntimeError("特征提取失败：没有收集到任何样本。请检查数据配置。")

    Z = torch.cat(Z_list, dim=0)[:max_samples]
    Y = torch.cat(Y_list, dim=0)[:max_samples]
    logger.info(f"特征提取完成: Z={Z.shape}, Y={Y.shape}, classes={Y.shape[1]}")
    return Z, Y


# ============================================================
# NSPS 推理包装（复用 detectron2 评估管线）
# ============================================================

class NSPSInferenceModel(torch.nn.Module):
    """将 NSPSSystem 包装为 detectron2 兼容的推理模型。"""

    def __init__(self, gd_model, nsps_system: NSPSSystem, global_class_names: list):
        super().__init__()
        self._gd = gd_model
        self.nsps = nsps_system
        self.global_class_names = global_class_names  # 全局类别列表（用于评估对齐）

    @property
    def device(self):
        return next(self._gd.parameters()).device

    @torch.no_grad()
    def forward(self, batched_inputs):
        images = self._gd.preprocess_image(batched_inputs)
        samples = nested_tensor_from_tensor_list(images)
        image_sizes = images.image_sizes

        captions = [x["captions"] for x in batched_inputs]
        names_list = [x["captions"][:-1].split(".") for x in batched_inputs]
        device = self.device

        tokenized = self._gd.tokenizer(
            captions, padding="longest", return_tensors="pt"
        ).to(device)
        (text_self_attention_masks, position_ids, cate_to_token_mask_list) = (
            generate_masks_with_special_tokens_and_transfer_map(
                tokenized, self._gd.specical_tokens, self._gd.tokenizer
            )
        )

        if text_self_attention_masks.shape[1] > self._gd.max_text_len:
            L = self._gd.max_text_len
            text_self_attention_masks = text_self_attention_masks[:, :L, :L]
            position_ids = position_ids[:, :L]
            tokenized["input_ids"] = tokenized["input_ids"][:, :L]
            tokenized["attention_mask"] = tokenized["attention_mask"][:, :L]
            tokenized["token_type_ids"] = tokenized["token_type_ids"][:, :L]

        if self._gd.sub_sentence_present:
            enc_input = {k: v for k, v in tokenized.items() if k != "attention_mask"}
            enc_input["attention_mask"] = text_self_attention_masks
            enc_input["position_ids"] = position_ids
        else:
            enc_input = tokenized

        bert_out = self._gd.bert(**enc_input)
        encoded_text = self._gd.feat_map(bert_out["last_hidden_state"])
        text_token_mask = tokenized.attention_mask.bool()

        if encoded_text.shape[1] > self._gd.max_text_len:
            L = self._gd.max_text_len
            encoded_text = encoded_text[:, :L, :]
            text_token_mask = text_token_mask[:, :L]
            position_ids = position_ids[:, :L]
            text_self_attention_masks = text_self_attention_masks[:, :L, :L]

        text_dict = {
            "encoded_text": encoded_text,
            "text_token_mask": text_token_mask,
            "position_ids": position_ids,
            "text_self_attention_masks": text_self_attention_masks,
        }

        features, poss = self._gd.backbone(samples)
        srcs, masks = [], []
        for l, feat in enumerate(features):
            src, mask = feat.decompose()
            srcs.append(self._gd.input_proj[l](src))
            masks.append(mask)
        if self._gd.num_feature_levels > len(srcs):
            _len_srcs = len(srcs)
            for l in range(_len_srcs, self._gd.num_feature_levels):
                src = self._gd.input_proj[l](features[-1].tensors if l == _len_srcs else srcs[-1])
                m = samples.mask
                mask = F.interpolate(m[None].float(), size=src.shape[-2:]).to(torch.bool)[0]
                pos_l = self._gd.backbone[1](NestedTensor(src, mask)).to(src.dtype)
                srcs.append(src); masks.append(mask); poss.append(pos_l)

        hs, reference, _, _, _ = self._gd.transformer(
            srcs, masks, None, poss, None, None, text_dict
        )

        hs_last = hs[-1]   # [B, nq, D]
        B, nq, D = hs_last.shape

        # NSPS 推理：逐 query 路由 → 回退式判决
        Z_all = hs_last.reshape(B * nq, D)  # [B*nq, D]
        result_dict = self.nsps.inference_batched(Z_all)

        # 将 NSPS 输出转回 detectron2 结果格式
        # 为每个 query 构造 pred_logits: 采纳的用置信度，拒识的用 0
        task_ids = result_dict["task_ids"]    # [B*nq]
        class_ids = result_dict["class_ids"]  # [B*nq]
        probs = result_dict["probs"]          # [B*nq]
        rejected = result_dict["rejected"]     # [B*nq]

        # 获取全局类别总数
        global_cls_map = self.nsps.get_global_class_mapping()
        total_classes = max((k[1] + 1 for k in global_cls_map), default=1)
        # 将 (task_id, local_cls) → global_cls 偏移映射
        task_offsets = {}
        offset = 0
        for meta in self.nsps.task_meta:
            task_offsets[meta["task_id"]] = offset
            offset += meta["num_classes"]

        pred_logits = torch.zeros(B * nq, max(total_classes, 1), device=device)
        for i in range(B * nq):
            if not rejected[i]:
                tid = task_ids[i].item()
                lid = class_ids[i].item()
                g_cls = task_offsets.get(tid, 0) + lid
                if g_cls < pred_logits.shape[1]:
                    pred_logits[i, g_cls] = probs[i]

        pred_logits = pred_logits.reshape(B, nq, -1)

        # 使用 GroundingDINO 的 box 输出
        last_ref = reference[-2]
        delta_unsig = self._gd.bbox_embed[-1](hs_last)
        pred_boxes = (delta_unsig + inverse_sigmoid(last_ref)).sigmoid()  # [B, nq, 4]

        # 标准 detectron2 后处理
        processed = []
        select_n = getattr(self._gd, 'select_box_nums_for_evaluation', 300)
        for i, (img_logits, img_boxes, img_size) in enumerate(
            zip(pred_logits, pred_boxes, image_sizes)
        ):
            # Top-K 选择
            scores_flat = img_logits.reshape(-1)
            topk_n = min(select_n, scores_flat.shape[0])
            _, topk_idx = scores_flat.topk(topk_n)
            q_idx = topk_idx // img_logits.shape[1]
            c_idx = topk_idx % img_logits.shape[1]
            scores_sel = scores_flat[topk_idx]
            boxes_sel = img_boxes[q_idx]

            inst = Instances(img_size)
            inst.pred_boxes = Boxes(box_cxcywh_to_xyxy(boxes_sel))
            inst.pred_boxes.scale(scale_x=img_size[1], scale_y=img_size[0])
            inst.scores = scores_sel
            inst.pred_classes = c_idx

            inp = batched_inputs[i]
            h = inp.get("height", img_size[0])
            w = inp.get("width", img_size[1])
            processed.append({"instances": detector_postprocess(inst, h, w)})

        return processed


# ============================================================
# 单任务特征采集 + NSPSSystem.add_task
# ============================================================

def run_nsps_add_task(
    nsps: NSPSSystem,
    gd_model,
    task_cfg,
    class_names: list,
    device: torch.device,
    nsps_epochs: int,
    nsps_lr: float,
    max_samples: int,
):
    """
    采集特征 → 调用 NSPSSystem.add_task 完成一个任务的完整注册。

    Returns:
        dict: add_task 返回的信息
    """
    task_cfg.dataloader.train.num_workers = 0
    train_loader = instantiate(task_cfg.dataloader.train)

    logger.info(f"  采集训练特征（最多 {max_samples} 样本）...")
    try:
        Z, Y = extract_task_features(gd_model, train_loader, device, max_samples=max_samples)
    except RuntimeError as e:
        logger.error(f"  ❌ 特征采集失败: {e}")
        raise

    del train_loader
    gc.collect()
    if device.type == "cuda":
        torch.cuda.empty_cache()

    logger.info(f"  注册到 NSPSSystem (epochs={nsps_epochs}, lr={nsps_lr})...")
    with torch.enable_grad():
        info = nsps.add_task(
            Z, Y,
            class_names=class_names,
            num_epochs=nsps_epochs,
            lr=nsps_lr,
            verbose=True,
        )

    logger.info(
        f"  ✅ Task {info['task_id']} 注册完成: "
        f"loss={info['train_stats']['final_loss']:.4f}, "
        f"gamma={info['task_gamma']:.4f}, "
        f"params={info['ncrn_params']}"
    )
    return info


# ============================================================
# 主训练流程
# ============================================================

def do_train(args):
    """顺序训练 NSPSSystem 所有任务。"""
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    seed_all_rng(args.seed)

    device = auto_device()
    logger.info(f"设备: {device}")

    # 加载 GroundingDINO（冻结）
    gd_model = load_grounding_dino(
        args.model_config_file,
        args.model_checkpoint_path,
        args.output_dir,
        device,
    )

    # 任务配置列表
    train_config_dir = os.path.join(args.config_file, "for_train")
    task_configs_paths = sorted(glob.glob(os.path.join(train_config_dir, "*.py")))
    if not task_configs_paths:
        raise RuntimeError(f"在 {train_config_dir} 找不到任务配置文件 (*.py)")

    if args.shuffle_tasks:
        random.shuffle(task_configs_paths)
        logger.info("任务顺序已打乱")

    logger.info(f"共 {len(task_configs_paths)} 个任务: {[Path(p).stem for p in task_configs_paths]}")

    # 检查是否从已有 checkpoint 恢复
    nsps_ckpt_dir = os.path.join(args.output_dir, NSPS_CKPT_DIR)
    ncrn_cfg = SLConfig.fromfile(args.model_config_file)

    nsps = NSPSSystem(
        feat_dim=getattr(ncrn_cfg, 'ncrn_feat_dim', 256),
        concepts_per_task=getattr(ncrn_cfg, 'ncrn_init_active', 64),
        rules_per_class=getattr(ncrn_cfg, 'ncrn_num_rules', 8),
        lambda_l1=getattr(ncrn_cfg, 'ncrn_lambda_l1', 1e-3),
        lambda_conflict=getattr(ncrn_cfg, 'ncrn_lambda_conflict', 1e-2),
        top_k=getattr(ncrn_cfg, 'nsps_top_k', 3),
        gamma=getattr(ncrn_cfg, 'nsps_gamma', 0.3),
        neg_ratio=getattr(ncrn_cfg, 'nsps_neg_ratio', 0.2),
    ).to(device)

    start_task = 0
    if os.path.exists(nsps_ckpt_dir) and not args.overwrite:
        try:
            nsps.load_checkpoint(nsps_ckpt_dir, device=device)
            start_task = nsps.num_tasks
            logger.info(f"从 checkpoint 恢复：已完成 {start_task} 个任务，继续训练")
        except Exception as e:
            logger.warning(f"无法恢复 checkpoint ({e})，从头开始")
            start_task = 0

    task_mapping = (
        ODINW_OVERLAPPED_FILE_MAPPING
        if 'odinwo' in args.config_file
        else ODINW_13_FILE_MAPPING
    )
    TaskMemory().task_mapping = task_mapping

    nsps_epochs = getattr(ncrn_cfg, 'nsps_epochs', 50)
    nsps_lr = getattr(ncrn_cfg, 'nsps_lr', 0.001)
    max_samples = getattr(ncrn_cfg, 'nsps_max_samples', 500)

    all_task_infos = []

    for task_idx, cfg_path in enumerate(task_configs_paths):
        if task_idx < start_task:
            logger.info(f"[Task {task_idx}] 已完成（跳过）")
            continue

        logger.info(f"\n{'='*60}")
        logger.info(f"[Task {task_idx}/{len(task_configs_paths)}] {Path(cfg_path).stem}")
        logger.info(f"{'='*60}")

        cfg = LazyConfig.load(cfg_path)
        cfg = LazyConfig.apply_overrides(cfg, args.opts)
        cfg.train.output_dir = os.path.join(args.output_dir, cfg.train.output_dir)
        PathManager.mkdirs(cfg.train.output_dir)

        class_names = cfg.dataloader.train.mapper.categories_names

        try:
            info = run_nsps_add_task(
                nsps, gd_model, cfg, class_names, device,
                nsps_epochs=nsps_epochs,
                nsps_lr=nsps_lr,
                max_samples=max_samples,
            )
            all_task_infos.append(info)
        except Exception:
            logger.error(f"[Task {task_idx}] 失败:\n{traceback.format_exc()}")
            raise

        # 每个任务后保存 checkpoint（断点续训）
        nsps.save_checkpoint(nsps_ckpt_dir)
        logger.info(f"[Task {task_idx}] Checkpoint 已保存到 {nsps_ckpt_dir}")

        gc.collect()
        if device.type == "cuda":
            torch.cuda.empty_cache()

    logger.info(f"\n{'='*60}")
    logger.info(f"✅ 所有 {nsps.num_tasks} 个任务训练完成！")
    logger.info(f"全局类别映射: {len(nsps.get_global_class_mapping())} 个类别")
    logger.info(f"{'='*60}\n")

    return nsps, gd_model


# ============================================================
# 评估流程
# ============================================================

def do_eval(args, nsps: NSPSSystem, gd_model):
    """对所有任务进行 COCO AP 评估。"""
    device = auto_device()
    ncrn_cfg = SLConfig.fromfile(args.model_config_file)

    # 如果是 eval-only，从 checkpoint 重新加载 NSPS
    if nsps is None:
        nsps = NSPSSystem(
            feat_dim=getattr(ncrn_cfg, 'ncrn_feat_dim', 256),
        ).to(device)
        nsps_ckpt_dir = os.path.join(args.output_dir, NSPS_CKPT_DIR)
        nsps.load_checkpoint(nsps_ckpt_dir, device=device)
        logger.info(f"从 {nsps_ckpt_dir} 加载 NSPS ({nsps.num_tasks} tasks)")

    if gd_model is None:
        gd_model = load_grounding_dino(
            args.model_config_file,
            args.model_checkpoint_path,
            args.output_dir,
            device,
        )

    # 构造全局类别列表（用于评估对齐）
    global_class_names = []
    for meta in nsps.task_meta:
        global_class_names.extend(meta.get("class_names", []))

    inference_model = NSPSInferenceModel(gd_model, nsps, global_class_names).to(device)
    inference_model.eval()

    eval_config_dir = args.config_file
    eval_paths = sorted(glob.glob(os.path.join(eval_config_dir, "for_train", "*.py")))
    coco_cfg = os.path.join(eval_config_dir, "test_zero_shot_coco.py")
    if os.path.exists(coco_cfg):
        eval_paths.append(coco_cfg)

    all_results = {}

    for cfg_path in eval_paths:
        dataset_name = Path(cfg_path).stem
        logger.info(f"\n[评估] {dataset_name}")

        try:
            cfg = LazyConfig.load(cfg_path)
            cfg = LazyConfig.apply_overrides(cfg, args.opts)
            cfg.train.output_dir = os.path.join(args.output_dir, cfg.train.output_dir)
            PathManager.mkdirs(cfg.train.output_dir)

            cfg.dataloader.test.num_workers = 0

            from groundingdino.util.inference import inference_on_dataset
            evaluator = COCOEvaluator(
                dataset_name=cfg.dataloader.test.dataset.names,
                output_dir=cfg.train.output_dir,
            )
            ret = inference_on_dataset(
                inference_model,
                instantiate(cfg.dataloader.test),
                cfg.dataloader.test.dataset.names,
                evaluator,
                output_dir=cfg.train.output_dir,
            )
            print_csv_format(ret)
            all_results[dataset_name] = ret.get('bbox', {})

            gc.collect()
            if device.type == "cuda":
                torch.cuda.empty_cache()

        except Exception:
            logger.error(f"[评估] {dataset_name} 失败:\n{traceback.format_exc()}")
            all_results[dataset_name] = {}

    # 汇总 AP
    ap_list = [r.get("AP", 0.0) for r in all_results.values() if isinstance(r, dict) and "AP" in r]
    if ap_list:
        mean_ap = sum(ap_list) / len(ap_list)
        logger.info(f"\n{'='*60}")
        logger.info(f"NSPS 评估汇总: Mean AP = {mean_ap:.2f} ({len(ap_list)} 数据集)")
        logger.info(f"{'='*60}")
        for ds, r in all_results.items():
            ap = r.get("AP", "N/A") if isinstance(r, dict) else "N/A"
            logger.info(f"  {ds}: AP={ap}")

    return all_results


# ============================================================
# CLI 参数解析
# ============================================================

def parse_args():
    parser = argparse.ArgumentParser(description="NSPS 双系统连续学习实验入口")
    parser.add_argument("--config-file", required=True,
                        help="ODinW-13 任务配置目录 (含 for_train/*.py)")
    parser.add_argument("--model-config-file", required=True,
                        help="GroundingDINO + NCRN 超参数配置文件")
    parser.add_argument("--model-checkpoint-path", required=True,
                        help="GroundingDINO 预训练权重路径")
    parser.add_argument("--output-dir", default="./output/nsps_output",
                        help="输出目录")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--eval-only", action="store_true",
                        help="仅评估（从 output-dir/nsps_checkpoint/ 加载）")
    parser.add_argument("--shuffle-tasks", action="store_true",
                        help="随机打乱任务顺序")
    parser.add_argument("--overwrite", action="store_true",
                        help="忽略已有 checkpoint，从头训练")
    parser.add_argument("--num-gpus", type=int, default=1,
                        help="GPU 数量（当前仅支持 1）")
    parser.add_argument("--dithub", action="store_true",
                        help="启用 DitHub LoRA 模式")
    parser.add_argument("opts", default=None, nargs=argparse.REMAINDER,
                        help="LazyConfig 覆盖选项")
    return parser.parse_args()


# ============================================================
# Main
# ============================================================

def main():
    args = parse_args()
    PathManager.mkdirs(args.output_dir)
    setup_logger(args.output_dir, distributed_rank=0, name="fvcore")
    global logger
    logger = setup_logger(args.output_dir, distributed_rank=0, name="nsps")

    logger.info(f"NSPS 双系统连续学习实验")
    logger.info(f"  config-file: {args.config_file}")
    logger.info(f"  output-dir:  {args.output_dir}")
    logger.info(f"  eval-only:   {args.eval_only}")
    logger.info(f"  shuffle:     {args.shuffle_tasks}")

    if args.eval_only:
        do_eval(args, nsps=None, gd_model=None)
    else:
        nsps, gd_model = do_train(args)
        do_eval(args, nsps=nsps, gd_model=gd_model)


if __name__ == "__main__":
    main()
