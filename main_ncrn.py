# -*- coding: utf-8 -*-
"""
NCRN 训练/评估入口 (单任务模式)
================================
每个任务作为独立进程运行，通过磁盘上的共享 checkpoint 传递增量学习状态。
由 run_ncrn_cl.sh 编排脚本按顺序调用。

用法:
    # 单任务训练 (由 run_ncrn_cl.sh 调用)
    python -u main_ncrn.py --task-config test/test_odinw13/for_train/test_XXX.py \
        --task-index 0 --total-tasks 13 \
        --model-config-file groundingdino/config/GroundingDINO_SwinT_OGC_dt_ncrn.py \
        --model-checkpoint-path groundingdino_swint_ogc.pth \
        --output-dir ./output/ncrn_smoke --seed 3 --dithub

    # 评估 (所有任务训练完成后)
    python -u main_ncrn.py --eval-only --config-file test/test_odinw13 \
        --model-config-file ... --output-dir ./output/ncrn_smoke
"""

import argparse
import datetime
import gc
import glob
import json
import logging
import os
import random
import sys
import time
import traceback
from pathlib import Path


import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from scipy.optimize import linear_sum_assignment

# detectron2
from detectron2.config import LazyConfig, instantiate
from detectron2.engine import default_argument_parser
from detectron2.engine.defaults import setup_logger, collect_env_info, _try_get_key, CfgNode
from detectron2.utils import comm
from detectron2.utils.file_io import PathManager
from detectron2.evaluation import COCOEvaluator, print_csv_format
from detectron2.modeling import detector_postprocess
from detectron2.structures import Boxes, Instances

# DitHub 核心组件
from groundingdino.models import build_model
from groundingdino.util.slconfig import SLConfig
from groundingdino.util.utils import clean_state_dict
from groundingdino.util.misc import (
    NestedTensor, inverse_sigmoid, nested_tensor_from_tensor_list,
)
from detectron2.utils.env import seed_all_rng
from groundingdino.util.task_memory import TaskMemory
from groundingdino.util.lora_utils import get_lora_modules, apply_lora
from groundingdino.util import ema, get_tokenlizer
from groundingdino.util.box_ops import box_cxcywh_to_xyxy, box_xyxy_to_cxcywh, generalized_box_iou
from groundingdino.config.configs.common.data.odinw.mapping_classes import (
    ODINW_13_FILE_MAPPING,
    ODINW_OVERLAPPED_FILE_MAPPING,
)
from groundingdino.models.GroundingDINO.bertwarper import (
    generate_masks_with_special_tokens_and_transfer_map,
)

import groundingdino.models.GroundingDINO.groundingdino_dt  # noqa: F401

from groundingdino.models.GroundingDINO.ncrn import NCRN_Head

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), os.path.pardir)))

logger = logging.getLogger("detectron2")

SHARED_CKPT_NAME = "ncrn_shared.pth"


# ============================================================
# 损失函数
# ============================================================

def sigmoid_focal_loss(inputs, targets, num_boxes, alpha=0.25, gamma=2.0):
    prob = inputs.sigmoid()
    ce_loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction="none")
    p_t = prob * targets + (1 - prob) * (1 - targets)
    loss = ce_loss * ((1 - p_t) ** gamma)
    if alpha >= 0:
        alpha_t = alpha * targets + (1 - alpha) * (1 - targets)
        loss = alpha_t * loss
    return loss.mean(1).sum() / num_boxes


# ============================================================
# Hungarian Matcher
# ============================================================

class NCRNHungarianMatcher(nn.Module):
    def __init__(self, cost_class=2.0, cost_bbox=5.0, cost_giou=2.0,
                 alpha=0.25, gamma=2.0):
        super().__init__()
        self.cost_class = cost_class
        self.cost_bbox = cost_bbox
        self.cost_giou = cost_giou
        self.alpha = alpha
        self.gamma = gamma

    @torch.no_grad()
    def forward(self, outputs, targets):
        bs, num_queries = outputs["pred_logits"].shape[:2]
        out_prob = outputs["pred_logits"].flatten(0, 1).clamp(1e-7, 1 - 1e-7)
        out_bbox = outputs["pred_boxes"].flatten(0, 1)
        tgt_ids = torch.cat([v["labels"] for v in targets])
        tgt_bbox = torch.cat([v["boxes"] for v in targets])

        if tgt_ids.numel() == 0:
            return [(torch.tensor([], dtype=torch.int64),
                     torch.tensor([], dtype=torch.int64)) for _ in range(bs)]

        alpha, gamma = self.alpha, self.gamma
        neg_cost = (1 - alpha) * (out_prob ** gamma) * (-(1 - out_prob + 1e-8).log())
        pos_cost = alpha * ((1 - out_prob) ** gamma) * (-(out_prob + 1e-8).log())
        cost_class = pos_cost[:, tgt_ids] - neg_cost[:, tgt_ids]
        cost_bbox = torch.cdist(out_bbox, tgt_bbox, p=1)
        cost_giou = -generalized_box_iou(
            box_cxcywh_to_xyxy(out_bbox), box_cxcywh_to_xyxy(tgt_bbox)
        )
        C = (self.cost_bbox * cost_bbox
             + self.cost_class * cost_class
             + self.cost_giou * cost_giou)
        C = C.view(bs, num_queries, -1).cpu()
        sizes = [len(v["boxes"]) for v in targets]
        indices = [linear_sum_assignment(c[i])
                   for i, c in enumerate(C.split(sizes, -1))]
        return [(torch.as_tensor(i, dtype=torch.int64),
                 torch.as_tensor(j, dtype=torch.int64)) for i, j in indices]


# ============================================================
# Utility
# ============================================================

def setup_logging(output_dir):
    if output_dir:
        PathManager.mkdirs(output_dir)
    setup_logger(output_dir, distributed_rank=0, name="fvcore")
    _logger = setup_logger(output_dir, distributed_rank=0)
    return _logger


def default_setup(cfg, args):
    output_dir = _try_get_key(cfg, "OUTPUT_DIR", "output_dir", "train.output_dir")
    if output_dir:
        PathManager.mkdirs(output_dir)
        path = os.path.join(output_dir, "config.yaml")
        if isinstance(cfg, CfgNode):
            with PathManager.open(path, "w") as f:
                f.write(cfg.dump())
        else:
            LazyConfig.save(cfg, path)
    if not (hasattr(args, "eval_only") and args.eval_only):
        torch.backends.cudnn.benchmark = _try_get_key(
            cfg, "CUDNN_BENCHMARK", "train.cudnn_benchmark", default=False
        )


def load_grounding_dino(model_config_path, model_checkpoint_path, output_dir,
                        eval_mode=False):
    cfg_model = SLConfig.fromfile(model_config_path)
    cfg_model.device = "cuda"
    model = build_model(cfg_model)
    checkpoint = torch.load(model_checkpoint_path, map_location="cpu")
    model.load_state_dict(clean_state_dict(checkpoint["model"]), strict=False)

    if hasattr(cfg_model, 'dithub') and cfg_model.dithub:
        model.load_custom_attention()
        lora_modules = get_lora_modules(model, cfg_model.lora_out_min)
        apply_lora(modules=lora_modules, r=cfg_model.lora_r,
                   lora_alpha=cfg_model.lora_alpha, lora_dropout=cfg_model.lora_dropout)
        if eval_mode:
            if 'model_final.pth' in model_checkpoint_path:
                ckpt = torch.load(model_checkpoint_path)
                model.load_state_dict(ckpt['model'])
            else:
                last_lora = os.path.join(output_dir, 'last_lora.pth')
                if os.path.exists(last_lora):
                    lora_ckpt = torch.load(last_lora)
                    model.load_state_dict(clean_state_dict(lora_ckpt['model']), strict=False)

    for param in model.parameters():
        param.requires_grad = False
    model.eval()
    return model


# ============================================================
# Shared NCRN checkpoint save/load
# ============================================================

def save_ncrn_shared(ncrn_head, class_registry, output_dir):
    """Save NCRN shared checkpoint with all state needed for continual learning.
    
    class_registry: list of (task_idx, [class_names]) — tracks which global class
    indices belong to which task, resolving duplicate class names across tasks.
    """
    ckpt_path = os.path.join(output_dir, SHARED_CKPT_NAME)
    os.makedirs(output_dir, exist_ok=True)
    torch.save({
        'ncrn_head': ncrn_head.state_dict(),
        'K_active': ncrn_head.K_active,
        'num_classes': ncrn_head.num_classes,
        'task_count': ncrn_head._task_count,
        'class_registry': class_registry,
    }, ckpt_path)
    all_names = [n for _, names in class_registry for n in names]
    logger.info(f"Shared NCRN checkpoint saved: {ckpt_path}")
    logger.info(f"  K_active={ncrn_head.K_active}, num_classes={ncrn_head.num_classes}, "
                f"task_count={ncrn_head._task_count}, classes={all_names}")


def load_ncrn_shared(output_dir, ncrn_cfg):
    """Load NCRN from shared checkpoint, restoring all continual learning state."""
    ckpt_path = os.path.join(output_dir, SHARED_CKPT_NAME)
    if not os.path.exists(ckpt_path):
        return None, []

    ckpt = torch.load(ckpt_path, map_location="cuda")
    logger.info(f"Loading shared NCRN checkpoint: {ckpt_path}")
    class_registry = ckpt.get('class_registry', [])
    if not class_registry and 'all_class_names' in ckpt:
        class_registry = [(0, ckpt['all_class_names'])]
    all_names = [n for _, names in class_registry for n in names]
    logger.info(f"  K_active={ckpt['K_active']}, num_classes={ckpt['num_classes']}, "
                f"task_count={ckpt['task_count']}, classes={all_names}")

    ncrn_head = NCRN_Head(
        feat_dim=getattr(ncrn_cfg, 'ncrn_feat_dim', 256),
        num_classes=ckpt['num_classes'],
        total_concepts=getattr(ncrn_cfg, 'ncrn_total_concepts', 2048),
        init_active=ckpt['K_active'],
        num_rules=getattr(ncrn_cfg, 'ncrn_num_rules', 8),
        lambda_l1=getattr(ncrn_cfg, 'ncrn_lambda_l1', 1e-3),
        lambda_conflict=getattr(ncrn_cfg, 'ncrn_lambda_conflict', 1e-2),
    ).to("cuda")

    ncrn_head.load_state_dict(ckpt['ncrn_head'])
    ncrn_head._task_count = ckpt['task_count']

    return ncrn_head, class_registry


def create_fresh_ncrn(ncrn_cfg, num_classes):
    """Create a fresh NCRN_Head for the first task."""
    ncrn_head = NCRN_Head(
        feat_dim=getattr(ncrn_cfg, 'ncrn_feat_dim', 256),
        num_classes=num_classes,
        total_concepts=getattr(ncrn_cfg, 'ncrn_total_concepts', 2048),
        init_active=getattr(ncrn_cfg, 'ncrn_init_active', 64),
        num_rules=getattr(ncrn_cfg, 'ncrn_num_rules', 8),
        lambda_l1=getattr(ncrn_cfg, 'ncrn_lambda_l1', 1e-3),
        lambda_conflict=getattr(ncrn_cfg, 'ncrn_lambda_conflict', 1e-2),
    ).to("cuda")
    return ncrn_head


# ============================================================
# Feature extraction
# ============================================================

def extract_features(model, batched_inputs, device):
    with torch.no_grad():
        images = model.preprocess_image(batched_inputs)
        samples = nested_tensor_from_tensor_list(images)

        captions = [x["captions"] for x in batched_inputs]
        names_list = [x["captions"][:-1].split(".") for x in batched_inputs]

        memory_classes = []
        for batch_elem, names_elem in zip(batched_inputs, names_list):
            if 'instances' not in batch_elem:
                names_elem = [f'class_{n[0].lower() + n[1:]}' for n in names_elem]
                memory_classes = names_elem
                break
            elem_classes = batch_elem['instances'].gt_classes.tolist()
            if not elem_classes:
                elem_classes = [0]
            curr_dataset = Path(batch_elem['file_name']).parts[2].lower()
            curr_class_int = random.choice(elem_classes)
            curr_class_str = TaskMemory().task_mapping[curr_dataset][curr_class_int]
            curr_class_str = curr_class_str[0].lower() + curr_class_str[1:]
            memory_classes.append(f'class_{curr_class_str}')
        TaskMemory().set_classes(memory_classes)

        if model.use_add_names and not model.training:
            non_overlap_classes = [c for c in model.learned_classes if c not in names_list[0]]
            for idx, (caption, names) in enumerate(zip(captions, names_list)):
                names_list[idx] = names + non_overlap_classes
                captions[idx] = caption + ".".join(non_overlap_classes)
                if not captions[idx].endswith("."):
                    captions[idx] = captions[idx] + "."

        tokenized = model.tokenizer(captions, padding="longest",
                                    return_tensors="pt").to(samples.device)
        (
            text_self_attention_masks,
            position_ids,
            cate_to_token_mask_list,
        ) = generate_masks_with_special_tokens_and_transfer_map(
            tokenized, model.specical_tokens, model.tokenizer
        )

        if text_self_attention_masks.shape[1] > model.max_text_len:
            text_self_attention_masks = text_self_attention_masks[
                :, :model.max_text_len, :model.max_text_len]
            position_ids = position_ids[:, :model.max_text_len]
            tokenized["input_ids"] = tokenized["input_ids"][:, :model.max_text_len]
            tokenized["attention_mask"] = tokenized["attention_mask"][:, :model.max_text_len]
            tokenized["token_type_ids"] = tokenized["token_type_ids"][:, :model.max_text_len]

        if model.sub_sentence_present:
            tokenized_for_encoder = {k: v for k, v in tokenized.items() if k != "attention_mask"}
            tokenized_for_encoder["attention_mask"] = text_self_attention_masks
            tokenized_for_encoder["position_ids"] = position_ids
        else:
            tokenized_for_encoder = tokenized

        bert_output = model.bert(**tokenized_for_encoder)
        encoded_text = model.feat_map(bert_output["last_hidden_state"])
        text_token_mask = tokenized.attention_mask.bool()

        if encoded_text.shape[1] > model.max_text_len:
            encoded_text = encoded_text[:, :model.max_text_len, :]
            text_token_mask = text_token_mask[:, :model.max_text_len]
            position_ids = position_ids[:, :model.max_text_len]
            text_self_attention_masks = text_self_attention_masks[
                :, :model.max_text_len, :model.max_text_len]

        text_dict = {
            "encoded_text": encoded_text,
            "text_token_mask": text_token_mask,
            "position_ids": position_ids,
            "text_self_attention_masks": text_self_attention_masks,
        }

        features, poss = model.backbone(samples)
        srcs, masks = [], []
        for l, feat in enumerate(features):
            src, mask = feat.decompose()
            srcs.append(model.input_proj[l](src))
            masks.append(mask)
        if model.num_feature_levels > len(srcs):
            _len_srcs = len(srcs)
            for l in range(_len_srcs, model.num_feature_levels):
                if l == _len_srcs:
                    src = model.input_proj[l](features[-1].tensors)
                else:
                    src = model.input_proj[l](srcs[-1])
                m = samples.mask
                mask = F.interpolate(m[None].float(), size=src.shape[-2:]).to(torch.bool)[0]
                pos_l = model.backbone[1](NestedTensor(src, mask)).to(src.dtype)
                srcs.append(src)
                masks.append(mask)
                poss.append(pos_l)

        input_query_bbox = input_query_label = attn_mask = dn_meta = None
        hs, reference, hs_enc, ref_enc, _ = model.transformer(
            srcs, masks, input_query_bbox, poss, input_query_label, attn_mask, text_dict
        )

        last_ref = reference[-2]
        last_hs = hs[-1]
        last_bbox_embed = model.bbox_embed[-1]
        delta_unsig = last_bbox_embed(last_hs)
        pred_boxes = (delta_unsig + inverse_sigmoid(last_ref)).sigmoid()

        targets = None
        if any('instances' in x for x in batched_inputs):
            try:
                gt_instances = [x["instances"].to(device) for x in batched_inputs if 'instances' in x]
                if gt_instances:
                    targets = model.prepare_targets(gt_instances, cate_to_token_mask_list, names_list)
            except Exception:
                targets = None

    return hs, reference, pred_boxes, cate_to_token_mask_list, names_list, targets, images.image_sizes


# ============================================================
# NCRNTrainer
# ============================================================

class NCRNTrainer:
    def __init__(self, grounding_dino, ncrn_head, dataloader, optimizer, cfg,
                 matcher, train_box_loss=False):
        self.grounding_dino = grounding_dino
        self.ncrn_head = ncrn_head
        self.dataloader = dataloader
        self.optimizer = optimizer
        self.cfg = cfg
        self.device = torch.device(cfg.train.device)
        self.matcher = matcher
        self.train_box_loss = train_box_loss
        self.iter = 0
        self.max_iter = cfg.train.max_iter
        self.cls_weight = 1.0
        self.box_weight = 5.0
        self.giou_weight = 2.0

    def train_step(self, batched_inputs):
        hs, reference, pred_boxes, cate_to_token_mask_list, names_list, targets, _ = \
            extract_features(self.grounding_dino, batched_inputs, self.device)

        if targets is None or len(targets) == 0:
            return {}

        hs_last = hs[-1]
        B, nq, D = hs_last.shape
        num_classes = self.ncrn_head.num_classes

        Z = hs_last.reshape(B * nq, D)

        do_diag = (self.iter % 50 == 0)
        if do_diag:
            Y_hat_flat, intermediates = self.ncrn_head(Z, return_intermediates=True)
        else:
            Y_hat_flat = self.ncrn_head(Z)
            intermediates = None

        Y_hat = Y_hat_flat.reshape(B, nq, -1)

        outputs = {"pred_logits": Y_hat, "pred_boxes": pred_boxes}
        indices = self.matcher(outputs, targets)

        num_boxes = max(sum(len(t["labels"]) for t in targets), 1)

        Y_logits = torch.log(Y_hat.clamp(1e-7, 1 - 1e-7) / (1 - Y_hat.clamp(1e-7, 1 - 1e-7)))

        idx = self._get_src_permutation_idx(indices)
        target_classes_o = torch.cat([t["labels"][J] for t, (_, J) in zip(targets, indices)])
        target_classes = torch.full(
            Y_logits.shape[:2], num_classes, dtype=torch.int64, device=Y_logits.device)
        target_classes[idx] = target_classes_o

        target_onehot = torch.zeros(
            [Y_logits.shape[0], Y_logits.shape[1], Y_logits.shape[2] + 1],
            dtype=Y_logits.dtype, device=Y_logits.device)
        target_onehot.scatter_(2, target_classes.unsqueeze(-1), 1)
        target_onehot = target_onehot[:, :, :-1]

        loss_cls = sigmoid_focal_loss(Y_logits, target_onehot, num_boxes)

        loss_bbox = torch.tensor(0.0, device=self.device)
        loss_giou = torch.tensor(0.0, device=self.device)

        logic_loss = self.ncrn_head.dnf.compute_logic_loss(
            K_active=self.ncrn_head.K_active,
            lambda_l1=self.ncrn_head.lambda_l1,
            lambda_conflict=self.ncrn_head.lambda_conflict,
        )

        L_total = (self.cls_weight * loss_cls
                   + self.box_weight * loss_bbox
                   + self.giou_weight * loss_giou
                   + logic_loss["L_total"])

        self.optimizer.zero_grad()
        L_total.backward()
        self.ncrn_head.apply_gradient_mask()

        grad_norm_before = torch.nn.utils.clip_grad_norm_(
            self.ncrn_head.parameters(), max_norm=1.0)
        self.optimizer.step()

        if do_diag:
            self._log_diagnostics(
                Z, Y_hat, Y_logits, intermediates,
                indices, targets, num_boxes, B, nq,
                grad_norm_before,
            )

        return {
            "loss_cls": loss_cls.item(),
            "loss_bbox": loss_bbox.item(),
            "loss_giou": loss_giou.item(),
            "loss_logic": logic_loss["L_total"].item(),
            "loss_total": L_total.item(),
        }

    @torch.no_grad()
    def _log_diagnostics(self, Z, Y_hat, Y_logits, intermediates,
                         indices, targets, num_boxes, B, nq,
                         grad_norm_before):
        """Print comprehensive layer-by-layer diagnostics."""
        C = intermediates["C"]
        L = intermediates["L"]
        W = intermediates["W"]
        P_rules = intermediates["P_rules"]
        omega_active = intermediates["omega_active"]
        sig_omega = torch.sigmoid(omega_active)

        lines = [f"\n{'='*60} DIAGNOSTICS iter={self.iter} {'='*60}"]

        # --- Layer 1: Input features Z ---
        lines.append(f"[L1 Z] shape={list(Z.shape)} "
                      f"min={Z.min():.4f} max={Z.max():.4f} "
                      f"mean={Z.mean():.4f} std={Z.std():.4f}")

        # --- Layer 2: Concept activations C ---
        q5, q25, q50, q75, q95 = torch.quantile(
            C.float(), torch.tensor([0.05, 0.25, 0.5, 0.75, 0.95], device=C.device))
        concept_means = C.mean(dim=0)
        lines.append(f"[L2 C] shape={list(C.shape)} "
                      f"min={C.min():.4f} max={C.max():.4f} "
                      f"mean={C.mean():.4f} std={C.std():.4f}")
        lines.append(f"  quantiles: 5%={q5:.4f} 25%={q25:.4f} 50%={q50:.4f} "
                      f"75%={q75:.4f} 95%={q95:.4f}")
        lines.append(f"  per-concept-mean std={concept_means.std():.4f} "
                      f"(low=no differentiation)")

        # --- Layer 3: STE mask W / sigmoid(Omega) ---
        active_ratio = W.sum().item() / W.numel()
        per_rule_active = W.sum(dim=-1).mean().item()
        total_lits = W.shape[-1]

        bins = [(0, 0.1), (0.1, 0.4), (0.4, 0.6), (0.6, 0.9), (0.9, 1.0)]
        sig_flat = sig_omega.flatten()
        bin_str = " ".join(
            f"[{lo}-{hi}]:{((sig_flat>=lo)&(sig_flat<hi)).sum().item()}"
            for lo, hi in bins)

        lines.append(f"[L3 W] shape={list(W.shape)} "
                      f"active_ratio={active_ratio:.4f} "
                      f"per_rule_active={per_rule_active:.1f}/{total_lits}")
        lines.append(f"  sig(Omega) mean={sig_omega.mean():.4f} "
                      f"distribution: {bin_str}")

        # --- Layer 4: Rule outputs P_rules ---
        lines.append(f"[L4 P_rules] shape={list(P_rules.shape)} "
                      f"min={P_rules.min():.4f} max={P_rules.max():.4f} "
                      f"mean={P_rules.mean():.4f} std={P_rules.std():.4f}")

        # --- Layer 5: Final output Y_hat ---
        Y_flat = Y_hat.reshape(-1, Y_hat.shape[-1])
        lines.append(f"[L5 Y_hat] shape={list(Y_hat.shape)} "
                      f"min={Y_hat.min():.4f} max={Y_hat.max():.4f} "
                      f"mean={Y_hat.mean():.4f} std={Y_hat.std():.4f}")

        idx = self._get_src_permutation_idx(indices)
        if idx[0].numel() > 0:
            target_cls = torch.cat([t["labels"][J] for t, (_, J) in zip(targets, indices)])
            matched_yhat = Y_hat[idx[0], idx[1]]  # [num_matched, C]
            pos_probs = matched_yhat[torch.arange(len(target_cls)), target_cls]
            neg_mask = torch.ones_like(matched_yhat, dtype=torch.bool)
            neg_mask[torch.arange(len(target_cls)), target_cls] = False
            neg_probs = matched_yhat[neg_mask]

            lines.append(f"  matched_queries={idx[0].numel()}/{B*nq} "
                          f"pos_prob: mean={pos_probs.mean():.4f} "
                          f"min={pos_probs.min():.4f} max={pos_probs.max():.4f}")
            lines.append(f"  neg_prob (matched queries, wrong classes): "
                          f"mean={neg_probs.mean():.4f} "
                          f"min={neg_probs.min():.4f} max={neg_probs.max():.4f}")

            unmatched_mask = torch.ones(B, nq, dtype=torch.bool, device=Y_hat.device)
            unmatched_mask[idx[0], idx[1]] = False
            unmatched_yhat = Y_hat[unmatched_mask]
            if unmatched_yhat.numel() > 0:
                lines.append(f"  unmatched_queries Y_hat: "
                              f"mean={unmatched_yhat.mean():.4f} "
                              f"min={unmatched_yhat.min():.4f} "
                              f"max={unmatched_yhat.max():.4f}")

        # --- Layer 6: Loss & gradients ---
        lines.append(f"[L6 logits] "
                      f"min={Y_logits.min():.2f} max={Y_logits.max():.2f} "
                      f"mean={Y_logits.mean():.2f} std={Y_logits.std():.2f}")
        lines.append(f"  num_boxes={num_boxes} total_queries={B*nq} "
                      f"ratio=1:{B*nq//max(num_boxes,1)}")

        omega_grad = self.ncrn_head.dnf.Omega.grad
        ptotal_grad = self.ncrn_head.concept_dict.P_total.grad
        gnb = grad_norm_before.item() if torch.is_tensor(grad_norm_before) else float(grad_norm_before)
        ogn = omega_grad.norm().item() if omega_grad is not None else 0.0
        pgn = ptotal_grad.norm().item() if ptotal_grad is not None else 0.0
        lines.append(f"  grad_norm_before_clip={gnb:.4f} "
                      f"Omega.grad.norm={ogn:.6f} "
                      f"P_total.grad.norm={pgn:.6f}")

        lines.append("=" * 130)
        logger.info("\n".join(lines))

    @staticmethod
    def _get_src_permutation_idx(indices):
        batch_idx = torch.cat([torch.full_like(src, i) for i, (src, _) in enumerate(indices)])
        src_idx = torch.cat([src for (src, _) in indices])
        return batch_idx, src_idx

    def train(self):
        logger.info(f"Starting NCRN training for {self.max_iter} iterations")
        data_iter = iter(self.dataloader)
        self.ncrn_head.train()
        skip_count = 0

        for self.iter in range(self.max_iter):
            try:
                data = next(data_iter)
            except StopIteration:
                data_iter = iter(self.dataloader)
                data = next(data_iter)

            try:
                loss_dict = self.train_step(data)
            except Exception as e:
                skip_count += 1
                if skip_count <= 5:
                    logger.warning(f"[iter {self.iter}] train_step error (skip {skip_count}): {e}")
                    logger.warning(traceback.format_exc())
                continue

            if self.iter % 50 == 0 and loss_dict:
                loss_str = " | ".join(f"{k}: {v:.4f}" for k, v in loss_dict.items())
                mem_mb = torch.cuda.max_memory_allocated() / (1024 ** 2)
                logger.info(f"[iter {self.iter}/{self.max_iter}] {loss_str} | gpu_mem: {mem_mb:.0f}MB")

        if skip_count > 0:
            logger.warning(f"Training finished with {skip_count} skipped iterations")
        logger.info("NCRN training finished")


# ============================================================
# NCRN Inference wrapper
# ============================================================

class NCRNInferenceModel(nn.Module):
    def __init__(self, grounding_dino, ncrn_head, class_registry=None):
        super().__init__()
        self.gd = grounding_dino
        self.ncrn_head = ncrn_head
        self.ncrn_head.eval()
        self.class_registry = class_registry or []
        self._task_class_indices = None

    def set_eval_task(self, task_categories):
        """Set which task's classes to evaluate; maps global → local IDs.
        
        Uses class_registry (list of (task_idx, [names])) to resolve
        duplicate class names across tasks by finding the best matching
        task block.
        """
        if not task_categories or not self.class_registry:
            self._task_class_indices = None
            return

        task_cats_lower = [c.lower() for c in task_categories]
        best_block = None
        best_overlap = 0
        global_offset = 0
        for _, block_names in self.class_registry:
            block_lower = [n.lower() for n in block_names]
            overlap = len(set(task_cats_lower) & set(block_lower))
            if overlap > best_overlap:
                best_overlap = overlap
                best_block = (global_offset, block_lower)
            global_offset += len(block_names)

        if best_block is None:
            self._task_class_indices = None
            return

        offset, block_lower = best_block
        indices = []
        for cat_l in task_cats_lower:
            if cat_l in block_lower:
                indices.append(offset + block_lower.index(cat_l))
            else:
                for gi_off, (_, bn) in enumerate(self.class_registry):
                    bn_lower = [n.lower() for n in bn]
                    if cat_l in bn_lower:
                        prev_offset = sum(len(names) for _, names in self.class_registry[:gi_off])
                        indices.append(prev_offset + bn_lower.index(cat_l))
                        break
        self._task_class_indices = indices if indices else None
        logger.info(f"Eval task class mapping: {list(zip(task_categories, self._task_class_indices or []))}")

    @property
    def device(self):
        return next(self.gd.parameters()).device

    def forward(self, batched_inputs):
        hs, reference, pred_boxes, cate_to_token_mask_list, names_list, _, image_sizes = \
            extract_features(self.gd, batched_inputs, self.device)
        hs_last = hs[-1]
        B, nq, D = hs_last.shape

        with torch.no_grad():
            Z = hs_last.reshape(B * nq, D)
            Y_hat = self.ncrn_head(Z).reshape(B, nq, -1)

        if self._task_class_indices is not None:
            idx = torch.tensor(self._task_class_indices, device=Y_hat.device)
            Y_hat = Y_hat[:, :, idx]

        pred_logits = torch.log(
            Y_hat.clamp(1e-7, 1 - 1e-7) / (1 - Y_hat.clamp(1e-7, 1 - 1e-7))
        )
        results = self._dt_inference(pred_logits, pred_boxes, image_sizes)
        processed_results = []
        for results_per_image, input_per_image, image_size in zip(
            results, batched_inputs, image_sizes
        ):
            height = input_per_image.get("height", image_size[0])
            width = input_per_image.get("width", image_size[1])
            r = detector_postprocess(results_per_image, height, width)
            processed_results.append({"instances": r})
        return processed_results

    def _dt_inference(self, box_cls, box_pred, image_sizes):
        select_box_nums = getattr(self.gd, 'select_box_nums_for_evaluation', 300)
        results = []
        prob = box_cls.sigmoid()
        topk_values, topk_indexes = torch.topk(
            prob.view(box_cls.shape[0], -1), select_box_nums, dim=1)
        scores = topk_values
        topk_boxes = torch.div(topk_indexes, box_cls.shape[2], rounding_mode="floor")
        labels = topk_indexes % box_cls.shape[2]
        boxes = torch.gather(box_pred, 1, topk_boxes.unsqueeze(-1).repeat(1, 1, 4))

        for i, (scores_per_image, labels_per_image, box_pred_per_image, image_size) in enumerate(
            zip(scores, labels, boxes, image_sizes)
        ):
            result = Instances(image_size)
            result.pred_boxes = Boxes(box_cxcywh_to_xyxy(box_pred_per_image))
            result.pred_boxes.scale(scale_x=image_size[1], scale_y=image_size[0])
            result.scores = scores_per_image
            result.pred_classes = labels_per_image
            results.append(result)
        return results


# ============================================================
# Evaluation
# ============================================================

def do_test_ncrn(cfg, inference_model, output_dir=None):
    from detectron2.config import LazyCall as L
    evaluator = instantiate(L(COCOEvaluator)(
        dataset_name=cfg.dataloader.test.dataset.names,
        output_dir=cfg.train.output_dir,
    ))
    cfg.dataloader.test.num_workers = 0
    from groundingdino.util.inference import inference_on_dataset
    ret = inference_on_dataset(
        inference_model, instantiate(cfg.dataloader.test),
        cfg.dataloader.test.dataset.names, evaluator, output_dir=output_dir
    )
    print_csv_format(ret)
    if 'bbox' not in ret:
        logger.warning(f"No 'bbox' key in eval result for {cfg.dataloader.test.dataset.names}")
        return {}
    returns = {'bbox': ret['bbox']}
    output_path = Path(cfg.train.output_dir).parent.parent / f'{cfg.dataloader.test.dataset.names}.out'
    with open(output_path, 'a') as f:
        f.write(f'{Path(cfg.train.output_dir).stem} {returns["bbox"]["AP"]}\n')
    return returns


# ============================================================
# Single-task training entry
# ============================================================

def do_single_task_train(args):
    """Train NCRN on a single task. Called by run_ncrn_cl.sh for each task."""
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    seed_all_rng(args.seed)
    setup_logging(args.output_dir)

    logger.info(f"{'='*60}")
    logger.info(f"NCRN Single-Task Training: task_index={args.task_index}")
    logger.info(f"Task config: {args.task_config}")
    logger.info(f"{'='*60}")

    ncrn_cfg = SLConfig.fromfile(args.model_config_file)

    # Determine task mapping
    config_dirs = os.path.dirname(os.path.dirname(args.task_config))
    TaskMemory().task_mapping = (
        ODINW_OVERLAPPED_FILE_MAPPING if 'odinwo' in config_dirs
        else ODINW_13_FILE_MAPPING
    )

    # Load task config
    cfg = LazyConfig.load(args.task_config)
    cfg = LazyConfig.apply_overrides(cfg, args.opts)
    cfg.train.output_dir = os.path.join(args.output_dir, cfg.train.output_dir)
    default_setup(cfg, args)

    categories_names = cfg.dataloader.train.mapper.categories_names
    num_classes_task = len(categories_names)
    logger.info(f"Task classes ({num_classes_task}): {categories_names}")

    # Load or create NCRN head
    task_idx = args.task_index
    if task_idx == 0:
        logger.info(f"First task: creating fresh NCRN_Head with {num_classes_task} classes")
        ncrn_head = create_fresh_ncrn(ncrn_cfg, num_classes_task)
        class_registry = []
    else:
        logger.info(f"Task {task_idx}: loading shared NCRN checkpoint...")
        ncrn_head, class_registry = load_ncrn_shared(args.output_dir, ncrn_cfg)
        if ncrn_head is None:
            raise RuntimeError(f"No shared checkpoint found at {args.output_dir}/{SHARED_CKPT_NAME} "
                             f"but task_index={task_idx} > 0!")

    logger.info(f"NCRN_Head state: K_active={ncrn_head.K_active}, "
                f"num_classes={ncrn_head.num_classes}, task_count={ncrn_head._task_count}")

    # Load frozen GroundingDINO
    gd_model = load_grounding_dino(
        args.model_config_file,
        args.model_checkpoint_path,
        args.output_dir,
    )
    gd_model.to("cuda")
    gc.collect()

    # DataLoader (num_workers=0 to avoid mmap issues with max_map_count=65530)
    cfg.dataloader.train.num_workers = 0

    # Incremental update for task_idx > 0
    # Use a separate DataLoader instance for sampling features,
    # then create the training DataLoader fresh (avoids re-iter deadlock
    # in detectron2's ToIterableDataset).
    if task_idx > 0:
        logger.info("Performing incremental update for new task...")
        sample_loader = instantiate(cfg.dataloader.train)
        sample_features = []
        sample_count = 0
        sample_iter = iter(sample_loader)
        for _ in range(20):
            try:
                data = next(sample_iter)
            except StopIteration:
                break
            hs_tmp, _, _, _, _, _, _ = extract_features(gd_model, data, torch.device("cuda"))
            if hs_tmp is not None:
                feat = hs_tmp[-1].mean(dim=1)
                sample_features.append(feat)
                sample_count += feat.shape[0]
            if sample_count >= 50:
                break
        del sample_iter, sample_loader
        torch.cuda.empty_cache()

        if sample_features:
            Z_new = torch.cat(sample_features, dim=0)[:50]
            Y_new = torch.zeros(Z_new.shape[0], num_classes_task, device=Z_new.device)
            info = ncrn_head.incremental_update(
                Z_new, Y_new,
                num_new_concepts=getattr(ncrn_cfg, 'ncrn_new_concepts', 16)
            )
            logger.info(f"Incremental update result: {info}")
        else:
            logger.warning("No features collected for incremental update!")

    train_loader = instantiate(cfg.dataloader.train)

    # Track class names per task (avoids duplicate name collisions)
    class_registry.append((task_idx, list(categories_names)))

    # Matcher
    matcher = NCRNHungarianMatcher(cost_class=2.0, cost_bbox=5.0, cost_giou=2.0).to("cuda")

    # Optimizer
    base_lr = getattr(ncrn_cfg, 'ncrn_lr', 0.001)
    omega_id = id(ncrn_head.dnf.Omega)
    omega_params = [ncrn_head.dnf.Omega]
    other_params = [p for p in ncrn_head.parameters()
                    if p.requires_grad and id(p) != omega_id]
    optimizer = torch.optim.AdamW([
        {"params": omega_params, "lr": base_lr * 10.0},
        {"params": other_params, "lr": base_lr},
    ], weight_decay=getattr(ncrn_cfg, 'ncrn_weight_decay', 0.01))

    TaskMemory().current_task = cfg.dataloader.train.dataset.names.split('_')[0].lower()

    # Train
    trainer = NCRNTrainer(
        grounding_dino=gd_model,
        ncrn_head=ncrn_head,
        dataloader=train_loader,
        optimizer=optimizer,
        cfg=cfg,
        matcher=matcher,
    )

    try:
        trainer.train()
    except Exception:
        logger.error(f"Training failed:\n{traceback.format_exc()}")
        raise

    peak_mem = torch.cuda.max_memory_allocated() / (1024 ** 2)
    logger.info(f"Task {task_idx} training finished. GPU peak memory: {peak_mem:.0f} MB")

    TaskMemory().end_task()

    # Save per-task checkpoint
    task_ckpt_path = os.path.join(cfg.train.output_dir, "ncrn_head.pth")
    os.makedirs(cfg.train.output_dir, exist_ok=True)
    torch.save({
        'ncrn_head': ncrn_head.state_dict(),
        'K_active': ncrn_head.K_active,
        'num_classes': ncrn_head.num_classes,
        'task_count': ncrn_head._task_count,
        'class_registry': class_registry,
    }, task_ckpt_path)
    logger.info(f"Task checkpoint saved: {task_ckpt_path}")

    # Update shared checkpoint for next task
    save_ncrn_shared(ncrn_head, class_registry, args.output_dir)

    all_names = [n for _, names in class_registry for n in names]
    logger.info(f"Task {task_idx} complete (K_active={ncrn_head.K_active}, "
                f"num_classes={ncrn_head.num_classes}, classes={all_names}). "
                f"Process will exit — OS reclaims all GPU/CPU/mmap resources.")


# ============================================================
# Evaluation entry (all tasks done)
# ============================================================

def do_eval_all(args):
    """Evaluate after all tasks are trained."""
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    seed_all_rng(args.seed)
    setup_logging(args.output_dir)

    logger.info("=" * 60)
    logger.info("EVALUATION PHASE")
    logger.info("=" * 60)

    ncrn_cfg = SLConfig.fromfile(args.model_config_file)

    # Load shared NCRN checkpoint
    ncrn_head, class_registry = load_ncrn_shared(args.output_dir, ncrn_cfg)
    if ncrn_head is None:
        # Try ncrn_final.pth fallback
        final_path = os.path.join(args.output_dir, "ncrn_final.pth")
        if os.path.exists(final_path):
            logger.info(f"Loading from {final_path}")
            ckpt = torch.load(final_path, map_location="cuda")
            ncrn_head = create_fresh_ncrn(ncrn_cfg, ckpt['num_classes'])
            ncrn_head.concept_dict.K_active = ckpt['K_active']
            ncrn_head.load_state_dict(ckpt['ncrn_head'])
            ncrn_head._task_count = ckpt['task_count']
            class_registry = ckpt.get('class_registry', [])
        else:
            raise RuntimeError("No NCRN checkpoint found for evaluation!")

    if TaskMemory().task_mapping is None:
        TaskMemory().task_mapping = {}

    gd_model = load_grounding_dino(
        args.model_config_file,
        args.model_checkpoint_path,
        args.output_dir,
        eval_mode=True,
    )
    gd_model.to("cuda")
    inference_model = NCRNInferenceModel(gd_model, ncrn_head, class_registry).to("cuda")
    inference_model.eval()

    config_dirs_base = args.config_file
    coco_config_file = os.path.join(config_dirs_base, "test_zero_shot_coco.py")
    eval_files = sorted(glob.glob(os.path.join(config_dirs_base, "for_train", "*.py")))
    if os.path.exists(coco_config_file):
        eval_files.append(coco_config_file)

    json_paths = {}
    for eval_config_file in eval_files:
        gc.collect()
        torch.cuda.empty_cache()
        cfg = LazyConfig.load(eval_config_file)
        cfg = LazyConfig.apply_overrides(cfg, args.opts)
        cfg.train.output_dir = os.path.join(args.output_dir, cfg.train.output_dir)
        default_setup(cfg, args)

        task_categories = cfg.dataloader.train.mapper.categories_names
        inference_model.set_eval_task(task_categories)
        logger.info(f"Evaluating {Path(eval_config_file).stem}: "
                     f"local classes={task_categories}")

        json_path = os.path.join(cfg.train.output_dir, "result.json")
        json_paths[eval_config_file] = json_path

        try:
            res = do_test_ncrn(cfg, inference_model, args.output_dir)
            os.makedirs(os.path.dirname(json_path), exist_ok=True)
            with open(json_path, "w") as jf:
                json.dump(res, jf)
        except Exception as e:
            logger.error(f"Evaluation failed for {eval_config_file}: {e}")

    avg_res = {}
    for k, v_path in json_paths.items():
        if os.path.exists(v_path):
            with open(v_path, "r") as jf:
                res = json.load(jf)
                if 'bbox' in res:
                    avg_res[k] = res['bbox']['AP']

    logger.info("\nAP results:")
    for k, v in avg_res.items():
        logger.info(f"  {Path(k).stem}: {v:.2f}")

    coco_count = 0
    sum_ = 0
    for k, v in avg_res.items():
        if k != coco_config_file:
            sum_ += v
        else:
            coco_count += 1
    odinw_count = len(avg_res) - coco_count
    if odinw_count > 0:
        logger.info(f"\nAverage ODinW AP: {sum_ / odinw_count:.4f}")
    if coco_config_file in avg_res:
        logger.info(f"COCO zero-shot AP: {avg_res[coco_config_file]:.4f}")


# ============================================================
# CLI
# ============================================================

if __name__ == "__main__":
    import socket
    print(socket.gethostname())

    parser = argparse.ArgumentParser(description="NCRN Training/Evaluation (single-task mode)")
    parser.add_argument("--task-config", type=str, default=None,
                        help="Path to a single task config file (for training)")
    parser.add_argument("--task-index", type=int, default=0,
                        help="0-based index of the current task in the CL sequence")
    parser.add_argument("--total-tasks", type=int, default=13)
    parser.add_argument("--config-file", type=str, default=None,
                        help="Config dir for evaluation (e.g. test/test_odinw13)")
    parser.add_argument("--model-config-file", "-c", type=str, required=True)
    parser.add_argument("--model-checkpoint-path", "-p", type=str, required=True)
    parser.add_argument("--seed", type=int, default=3)
    parser.add_argument("--output-dir", type=str, default="output/ncrn")
    parser.add_argument("--eval-only", action="store_true")
    parser.add_argument("--dithub", action="store_true", default=False)
    parser.add_argument("--shuffle-tasks", action="store_true")
    parser.add_argument('--lora-r', type=int, default=8)
    parser.add_argument('--lora-alpha', type=float, default=8)
    parser.add_argument('--lora-dropout', type=float, default=0.0)
    parser.add_argument('--lora-out-min', type=int, default=128)
    parser.add_argument('--opts', nargs='*', default=[])

    args = parser.parse_args()

    if args.eval_only:
        if not args.config_file:
            parser.error("--config-file required for --eval-only mode")
        do_eval_all(args)
    else:
        if not args.task_config:
            parser.error("--task-config required for training mode")
        do_single_task_train(args)
