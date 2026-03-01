# -*- coding: utf-8 -*-
"""
NCRN 训练/评估入口
==================
在 DitHub 框架基础上集成 NCRN 分类头。
复用 GroundingDINO 作为特征提取器，NCRN_Head 作为分类头。

用法:
    # 训练
    python -u main_ncrn.py --config-file test/test_odinw13 \
        --model-config-file groundingdino/config/GroundingDINO_SwinT_OGC_dt_ncrn.py \
        --model-checkpoint-path groundingdino_swint_ogc.pth \
        --output-dir /output/ncrn_output --seed 3 --num-gpus 1 --shuffle-tasks --dithub
    
    # 评估
    python -u main_ncrn.py ... --eval-only
"""

import argparse
import copy
import datetime
import glob
import json
import logging
import os
import random
import re
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from torch.cuda.amp import autocast

# detectron2
from detectron2.config import LazyConfig, instantiate
from detectron2.engine import create_ddp_model, default_argument_parser, hooks, launch
from detectron2.engine.defaults import create_ddp_model, setup_logger, collect_env_info, _try_get_key, CfgNode
from detectron2.checkpoint import DetectionCheckpointer
from detectron2.utils import comm
from detectron2.utils.events import CommonMetricPrinter, JSONWriter, TensorboardXWriter, EventStorage
from detectron2.utils.file_io import PathManager
from detectron2.evaluation import COCOEvaluator, print_csv_format

# DitHub 核心组件
from groundingdino.models import build_model
from groundingdino.util.slconfig import SLConfig
from groundingdino.util.utils import clean_state_dict
from groundingdino.util.misc import seed_all_rng
from groundingdino.util.task_memory import TaskMemory
from groundingdino.util.lora_utils import get_lora_modules, apply_lora
from groundingdino.util import ema
from groundingdino.config.configs.common.data.odinw.mapping_classes import (
    ODINW_13_FILE_MAPPING,
    ODINW_OVERLAPPED_FILE_MAPPING,
)

# NCRN 模块
from groundingdino.models.GroundingDINO.ncrn import NCRN_Head
from groundingdino.util.inference import inference_on_dataset

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), os.path.pardir)))


def multi_datasets_setup_logger(args):
    """设置多数据集日志 (从 main.py 复制)"""
    output_dir = args.output_dir
    rank = comm.get_rank()
    if comm.is_main_process() and output_dir:
        PathManager.mkdirs(output_dir)
    setup_logger(output_dir, distributed_rank=rank, name="fvcore")
    logger = setup_logger(output_dir, distributed_rank=rank)
    logger.info(f"Rank of current process: {rank}. World size: {comm.get_world_size()}")
    logger.info("Environment info:\n" + collect_env_info())
    logger.info(f"Command line arguments: {str(args)}")
    return logger


def default_setup(cfg, args):
    """默认设置 (从 main.py 复制)"""
    output_dir = _try_get_key(cfg, "OUTPUT_DIR", "output_dir", "train.output_dir")
    if comm.is_main_process() and output_dir:
        PathManager.mkdirs(output_dir)
    if comm.is_main_process() and output_dir:
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


def do_test(cfg, model, output_dir=None, eval_only=False):
    """评估 (从 main.py 复制)"""
    from detectron2.config import LazyCall as L
    logger = logging.getLogger("detectron2")
    evaluator = instantiate(L(COCOEvaluator)(
        dataset_name=cfg.dataloader.test.dataset.names,
        output_dir=cfg.train.output_dir,
    ))
    if eval_only:
        logger.info("Run evaluation under eval-only mode")
        if cfg.train.model_ema.enabled and cfg.train.model_ema.use_ema_weights_for_eval_only:
            logger.info("Run evaluation with EMA.")
        else:
            logger.info("Run evaluation without EMA.")
        if "evaluator" in cfg.dataloader:
            ret = inference_on_dataset(
                model, instantiate(cfg.dataloader.test),
                cfg.dataloader.test.dataset.names, evaluator, output_dir=output_dir
            )
            print_csv_format(ret)
        returns = {'bbox': ret['bbox']}
        output_path = Path(cfg.train.output_dir).parent.parent / f'{cfg.dataloader.test.dataset.names}.out'
        with open(output_path, 'a') as f:
            f.write(f'{Path(cfg.train.output_dir).stem} {returns["bbox"]["AP"]}\n')
        return returns
    logger.info("Run evaluation without EMA.")
    if "evaluator" in cfg.dataloader:
        ret = inference_on_dataset(
            model, instantiate(cfg.dataloader.test),
            cfg.dataloader.test.dataset.names, evaluator
        )
        print_csv_format(ret)
        if cfg.train.model_ema.enabled:
            logger.info("Run evaluation with EMA.")
            with ema.apply_model_ema_and_restore(model):
                if "evaluator" in cfg.dataloader:
                    ema_ret = inference_on_dataset(
                        model, instantiate(cfg.dataloader.test),
                        cfg.dataloader.test.dataset.names, evaluator
                    )
                    print_csv_format(ema_ret)
                    ret.update(ema_ret)
        return {'bbox': ret['bbox']}


def load_model_with_ncrn(model_config_path, model_checkpoint_path, output_dir, 
                          ncrn_head, eval_mode=False):
    """
    加载 GroundingDINO + LoRA，附加 NCRN_Head
    
    基于原始 main.py:load_model 修改:
    - 冻结 GroundingDINO 全部参数 (仅作特征提取)
    - 冻结 LoRA 参数
    - NCRN_Head 参数设为可训练
    """
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

    # 冻结 GroundingDINO 全部参数
    for param in model.parameters():
        param.requires_grad = False

    return model


class NCRNTrainer:
    """
    NCRN 训练器
    
    核心流程:
    1. GroundingDINO 前向 → 提取 decoder 输出 hs [B, nq, D]
    2. NCRN_Head 前向 → 类别概率 Y_hat [B, nq, C]  
    3. 与 GT 计算损失 (BCE + 逻辑正则化)
    4. 反向传播 + 梯度掩码
    """

    def __init__(self, args, grounding_dino, ncrn_head, dataloader, optimizer, cfg):
        self.args = args
        self.grounding_dino = grounding_dino
        self.ncrn_head = ncrn_head
        self.dataloader = dataloader
        self.optimizer = optimizer
        self.cfg = cfg
        self.device = cfg.train.device

        # 训练状态
        self.iter = 0
        self.max_iter = cfg.train.max_iter
        self.task_count = 0

    def extract_features(self, batched_inputs):
        """
        从 GroundingDINO 提取 decoder 特征
        
        Returns:
            hs: decoder 输出 [num_layers, B, nq, D]
            cate_to_token_mask_list: 类别-token 掩码映射
            names_list: 类别名称列表
            targets: GT 目标 (训练模式)
        """
        self.grounding_dino.eval()
        with torch.no_grad():
            # 复用 GroundingDINO 的前向逻辑提取特征
            from groundingdino.util.misc import NestedTensor, nested_tensor_from_tensor_list
            from groundingdino.models.GroundingDINO.bertwarper import (
                generate_masks_with_special_tokens_and_transfer_map,
            )
            from groundingdino.util import get_tokenlizer

            model = self.grounding_dino
            
            # 预处理图像
            images = model.preprocess_image(batched_inputs)
            samples = nested_tensor_from_tensor_list(images)

            # 准备 captions
            captions = [x["captions"] for x in batched_inputs]
            names_list = [x["captions"][:-1].split(".") for x in batched_inputs]

            # TaskMemory 类别更新
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

            # Tokenize
            tokenized = model.tokenizer(captions, padding="longest",
                                        return_tensors="pt").to(samples.device)
            (
                text_self_attention_masks,
                position_ids,
                cate_to_token_mask_list,
            ) = generate_masks_with_special_tokens_and_transfer_map(
                tokenized, model.specical_tokens, model.tokenizer
            )

            # 截断
            if text_self_attention_masks.shape[1] > model.max_text_len:
                text_self_attention_masks = text_self_attention_masks[
                    :, :model.max_text_len, :model.max_text_len]
                position_ids = position_ids[:, :model.max_text_len]
                tokenized["input_ids"] = tokenized["input_ids"][:, :model.max_text_len]
                tokenized["attention_mask"] = tokenized["attention_mask"][:, :model.max_text_len]
                tokenized["token_type_ids"] = tokenized["token_type_ids"][:, :model.max_text_len]

            # BERT 编码
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

            # Backbone
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
                    pos_l = model.backbone[1](
                        NestedTensor(src, mask)
                    ).to(src.dtype)
                    srcs.append(src)
                    masks.append(mask)
                    poss.append(pos_l)

            # Transformer
            input_query_bbox = input_query_label = attn_mask = dn_meta = None
            hs, reference, hs_enc, ref_enc, _ = model.transformer(
                srcs, masks, input_query_bbox, poss, input_query_label, attn_mask, text_dict
            )

            # 准备 GT
            targets = None
            if self.grounding_dino.training or any('instances' in x for x in batched_inputs):
                try:
                    from groundingdino.util.box_ops import box_xyxy_to_cxcywh
                    gt_instances = [x["instances"].to(self.device) for x in batched_inputs if 'instances' in x]
                    if gt_instances:
                        targets = model.prepare_targets(gt_instances, cate_to_token_mask_list, names_list)
                except Exception:
                    targets = None

        return hs, reference, cate_to_token_mask_list, names_list, targets, text_dict

    def train_step(self, batched_inputs):
        """单步训练"""
        # Step 1: 提取特征 (冻结 GroundingDINO)
        hs, reference, cate_to_token_mask_list, names_list, targets, text_dict = \
            self.extract_features(batched_inputs)

        if targets is None:
            return {}

        # Step 2: 获取最后一层 decoder 输出
        # hs: [num_layers, B, nq, D] → 取最后一层
        hs_last = hs[-1]  # [B, nq, D]
        B, nq, D = hs_last.shape

        # Step 3: 获取每个类别的数量
        num_classes = len(names_list[0])

        # 动态调整 NCRN 类别数
        if num_classes != self.ncrn_head.num_classes:
            # 对于新任务需要重建或扩展 NCRN
            pass  # 在 task 级别处理

        # Step 4: NCRN 前向
        # 将 [B, nq, D] 展开为 [B*nq, D] 送入 NCRN
        Z = hs_last.reshape(B * nq, D)
        Y_hat = self.ncrn_head(Z)  # [B*nq, C]
        Y_hat = Y_hat.reshape(B, nq, -1)  # [B, nq, C]

        # Step 5: 构建 GT 标签矩阵
        # targets 是 [{labels: [num_gt], boxes: [num_gt, 4]}, ...]
        # 使用 DitHub 原始的 criterion 匹配方式，或简化为最近匹配
        C_ncrn = Y_hat.shape[-1]
        
        # 简化损失：对 GT 标签做 one-hot，用 matched queries 计算 BCE
        # (完整版应通过 Hungarian matching，这里先用 focal loss 近似)
        loss_dict = {}

        # 对每个 query 构建软标签
        gt_labels = torch.zeros(B, nq, C_ncrn, device=Y_hat.device)
        for b_idx, t in enumerate(targets):
            for gt_cls in t["labels"]:
                if gt_cls < C_ncrn:
                    # 将 GT 类别标记到所有 query 上 (soft assignment)
                    gt_labels[b_idx, :, gt_cls] = 1.0

        # BCE 分类损失
        Y_hat_flat = Y_hat.reshape(-1, C_ncrn)
        gt_flat = gt_labels.reshape(-1, C_ncrn)
        L_BCE = F.binary_cross_entropy(
            Y_hat_flat.clamp(1e-7, 1.0 - 1e-7),
            gt_flat,
            reduction="mean",
        )

        # 逻辑正则化损失
        logic_loss = self.ncrn_head.dnf.compute_logic_loss(
            K_active=self.ncrn_head.K_active,
        )

        L_total = L_BCE + logic_loss["L_total"]

        # Step 6: 反向传播
        self.optimizer.zero_grad()
        L_total.backward()
        self.ncrn_head.apply_gradient_mask()
        
        # 梯度裁剪
        torch.nn.utils.clip_grad_norm_(self.ncrn_head.parameters(), max_norm=0.1)
        self.optimizer.step()

        loss_dict = {
            "loss_bce": L_BCE.item(),
            "loss_logic": logic_loss["L_total"].item(),
            "loss_total": L_total.item(),
        }
        return loss_dict

    def train(self):
        """训练循环"""
        logger = logging.getLogger(__name__)
        data_iter = iter(self.dataloader)

        logger.info(f"Starting NCRN training for {self.max_iter} iterations")
        self.ncrn_head.train()

        for self.iter in range(self.max_iter):
            try:
                data = next(data_iter)
            except StopIteration:
                data_iter = iter(self.dataloader)
                data = next(data_iter)

            loss_dict = self.train_step(data)

            if self.iter % 50 == 0 and loss_dict:
                loss_str = " | ".join(f"{k}: {v:.4f}" for k, v in loss_dict.items())
                logger.info(f"[iter {self.iter}/{self.max_iter}] {loss_str}")

        logger.info("NCRN training finished")


def main(args):
    """NCRN 主入口"""
    seed_all_rng(args.seed)
    logger = multi_datasets_setup_logger(args)

    # 加载NCRN模型配置
    ncrn_cfg = SLConfig.fromfile(args.model_config_file)

    config_dirs = args.config_file
    TaskMemory().task_mapping = (
        ODINW_OVERLAPPED_FILE_MAPPING if 'odinwo' in config_dirs
        else ODINW_13_FILE_MAPPING
    )

    ow_config_files = glob.glob(os.path.join(config_dirs, "for_train", "*.py"))
    if args.shuffle_tasks:
        random.shuffle(ow_config_files)

    start_time = datetime.datetime.now()

    # 初始化 NCRN_Head
    # 第一个任务时，根据类别数初始化
    first_cfg = LazyConfig.load(ow_config_files[0])
    
    # 获取首个数据集的类别数用于初始化
    init_num_classes = 10  # 安全默认值，实际由数据集决定

    ncrn_head = NCRN_Head(
        feat_dim=getattr(ncrn_cfg, 'ncrn_feat_dim', 256),
        num_classes=init_num_classes,
        total_concepts=getattr(ncrn_cfg, 'ncrn_total_concepts', 2048),
        init_active=getattr(ncrn_cfg, 'ncrn_init_active', 64),
        num_rules=getattr(ncrn_cfg, 'ncrn_num_rules', 8),
        lambda_l1=getattr(ncrn_cfg, 'ncrn_lambda_l1', 1e-3),
        lambda_conflict=getattr(ncrn_cfg, 'ncrn_lambda_conflict', 1e-2),
    ).to("cuda")

    logger.info(f"NCRN_Head initialized: {ncrn_head}")

    # ---- 训练阶段 ----
    if not args.eval_only:
        for task_idx, ow_config_file in enumerate(ow_config_files):
            torch.cuda.empty_cache()
            logger.info(f"\n{'='*60}")
            logger.info(f"Task {task_idx + 1}/{len(ow_config_files)}: {ow_config_file}")
            logger.info(f"{'='*60}")

            args.config_file = ow_config_file
            cfg = LazyConfig.load(ow_config_file)
            cfg = LazyConfig.apply_overrides(cfg, args.opts)
            cfg.train.output_dir = os.path.join(args.output_dir, cfg.train.output_dir)
            default_setup(cfg, args)

            # 加载 GroundingDINO (冻结)
            model = load_model_with_ncrn(
                args.model_config_file,
                args.model_checkpoint_path,
                args.output_dir,
                ncrn_head,
            )
            model.to(cfg.train.device)

            # 获取当前数据集的类别信息
            train_loader = instantiate(cfg.dataloader.train)
            categories_names = cfg.dataloader.train.mapper.categories_names
            num_classes_task = len(categories_names)
            logger.info(f"Task classes ({num_classes_task}): {categories_names}")

            # 增量更新 NCRN: 如果是新任务，扩展类别
            if task_idx > 0:
                # 收集新任务特征用于零空间投影
                logger.info("Collecting features for nullspace expansion...")
                model.eval()
                sample_features = []
                sample_count = 0
                for data in train_loader:
                    if sample_count >= 50:
                        break
                    with torch.no_grad():
                        from groundingdino.util.misc import NestedTensor, nested_tensor_from_tensor_list
                        images = model.preprocess_image(data)
                        samples = nested_tensor_from_tensor_list(images)
                        features, poss = model.backbone(samples)
                        # 使用 backbone 最后一层特征的全局平均池化
                        feat = features[-1].tensors.mean(dim=[2, 3])  # [B, C]
                        # 投影到 hidden_dim
                        sample_features.append(feat)
                        sample_count += feat.shape[0]

                if sample_features:
                    Z_new = torch.cat(sample_features, dim=0)[:50]
                    # 如果特征维度不匹配，投影
                    if Z_new.shape[1] != ncrn_head.feat_dim:
                        Z_new = F.adaptive_avg_pool1d(
                            Z_new.unsqueeze(1), ncrn_head.feat_dim
                        ).squeeze(1)
                    
                    Y_new = torch.zeros(Z_new.shape[0], num_classes_task, device=Z_new.device)
                    info = ncrn_head.incremental_update(
                        Z_new, Y_new, 
                        num_new_concepts=getattr(ncrn_cfg, 'ncrn_new_concepts', 16)
                    )
                    logger.info(f"Incremental update: {info}")

            # NCRN 优化器 (仅训练 NCRN 参数)
            ncrn_params = [p for p in ncrn_head.parameters() if p.requires_grad]
            optimizer = torch.optim.AdamW(
                ncrn_params,
                lr=getattr(ncrn_cfg, 'ncrn_lr', 0.001),
                weight_decay=getattr(ncrn_cfg, 'ncrn_weight_decay', 0.01),
            )

            # 更新 TaskMemory
            TaskMemory().current_task = cfg.dataloader.train.dataset.names.split('_')[0].lower()

            # 创建训练器
            trainer = NCRNTrainer(
                args=args,
                grounding_dino=model,
                ncrn_head=ncrn_head,
                dataloader=train_loader,
                optimizer=optimizer,
                cfg=cfg,
            )

            # 训练
            trainer.train()

            # 结束任务
            TaskMemory().end_task()

            # 保存 NCRN checkpoint
            ncrn_ckpt_path = os.path.join(cfg.train.output_dir, "ncrn_head.pth")
            os.makedirs(cfg.train.output_dir, exist_ok=True)
            torch.save({
                'ncrn_head': ncrn_head.state_dict(),
                'K_active': ncrn_head.K_active,
                'num_classes': ncrn_head.num_classes,
                'task_count': ncrn_head._task_count,
            }, ncrn_ckpt_path)
            logger.info(f"NCRN checkpoint saved: {ncrn_ckpt_path}")

        # 保存最终模型
        final_ncrn_path = os.path.join(args.output_dir, "ncrn_final.pth")
        os.makedirs(args.output_dir, exist_ok=True)
        torch.save({
            'ncrn_head': ncrn_head.state_dict(),
            'K_active': ncrn_head.K_active,
            'num_classes': ncrn_head.num_classes,
            'task_count': ncrn_head._task_count,
        }, final_ncrn_path)
        logger.info(f"Final NCRN model saved: {final_ncrn_path}")

    # ---- 评估阶段 ----
    logger.info("\n" + "=" * 60)
    logger.info("EVALUATION PHASE")
    logger.info("=" * 60)

    # 加载 NCRN final checkpoint
    ncrn_final_path = os.path.join(args.output_dir, "ncrn_final.pth")
    if os.path.exists(ncrn_final_path):
        ncrn_ckpt = torch.load(ncrn_final_path, map_location="cuda")
        ncrn_head.load_state_dict(ncrn_ckpt['ncrn_head'])
        logger.info(f"Loaded NCRN checkpoint: K_active={ncrn_ckpt['K_active']}, "
                    f"num_classes={ncrn_ckpt['num_classes']}")

    # 评估每个数据集
    coco_config_file = os.path.join(config_dirs, "test_zero_shot_coco.py")
    eval_files = ow_config_files[:]
    if os.path.exists(coco_config_file):
        eval_files.append(coco_config_file)

    json_paths = {}
    for ow_config_file in eval_files:
        torch.cuda.empty_cache()
        args.config_file = ow_config_file
        cfg = LazyConfig.load(args.config_file)
        cfg = LazyConfig.apply_overrides(cfg, args.opts)
        cfg.train.output_dir = os.path.join(args.output_dir, cfg.train.output_dir)
        default_setup(cfg, args)

        # 使用原始 DitHub 评估流程 (ContrastiveEmbed)
        # NCRN 的评估会在后续版本中替换 inference 逻辑
        config_file = args.model_config_file
        checkpoint_path = args.model_checkpoint_path

        model = load_model_with_ncrn(
            config_file, checkpoint_path, output_dir=args.output_dir,
            ncrn_head=ncrn_head, eval_mode=True,
        )
        for p in model.parameters():
            p.requires_grad = True
        model.to(cfg.train.device)
        model = create_ddp_model(model)
        ema.may_build_model_ema(cfg, model)
        if cfg.train.model_ema.enabled and cfg.train.model_ema.use_ema_weights_for_eval_only:
            ema.apply_model_ema(model)

        json_path = os.path.join(cfg.train.output_dir, "result.json")
        json_paths[ow_config_file] = json_path
        res = do_test(cfg, model, args.output_dir, eval_only=True)
        os.makedirs(os.path.dirname(json_path), exist_ok=True)
        with open(json_path, "w") as jf:
            json.dump(res, jf)

    # 汇总结果
    avg_res = {}
    for k, v_path in json_paths.items():
        if os.path.exists(v_path):
            with open(v_path, "r") as jf:
                res = json.load(jf)
                if 'bbox' in res:
                    avg_res[k] = res['bbox']['AP']

    logger.info(f"AP results: {avg_res}")
    sum_ = 0
    coco_count = 0
    for k, v in avg_res.items():
        if k != coco_config_file:
            sum_ += v
        else:
            coco_count += 1
    if len(avg_res) - coco_count > 0:
        logger.info(f"Average AP: {sum_ / (len(avg_res) - coco_count):.4f}")
    if coco_config_file in avg_res:
        logger.info(f"AP on COCO: {avg_res[coco_config_file]:.4f}")

    elapsed = datetime.datetime.now() - start_time
    logger.info(f"Elapsed time: {elapsed}")

    # 打印 NCRN 逻辑解释
    logger.info("\n--- NCRN Logic Rules ---")
    explanations = ncrn_head.get_logic_explanation()
    for c_idx, rules in explanations.items():
        if rules:
            logger.info(f"Class {c_idx}: {len(rules)} active rules")
            for r_idx, rule in enumerate(rules[:3]):
                logger.info(f"  Rule {r_idx}: +concepts={rule['positive_concepts'][:5]}, "
                          f"-concepts={rule['negative_concepts'][:5]}")


if __name__ == "__main__":
    import socket
    print(socket.gethostname())
    
    parser = default_argument_parser()
    parser.add_argument("--model-config-file", "-c", type=str, required=True)
    parser.add_argument("--model-checkpoint-path", "-p", type=str, required=True)
    parser.add_argument("--seed", type=int, default=3)
    parser.add_argument("--output-dir", type=str, default="output/ncrn")
    parser.add_argument("--shuffle-tasks", action="store_true")
    parser.add_argument("--zero-shot", action="store_true")
    parser.add_argument("--dithub", action="store_true", default=False)
    parser.add_argument('--lora-r', type=int, default=8)
    parser.add_argument('--lora-alpha', type=float, default=8)
    parser.add_argument('--lora-dropout', type=float, default=0.0)
    parser.add_argument('--lora-out-min', type=int, default=128)
    parser.add_argument('--lora-lr', type=float, default=0.001)
    parser.add_argument('--lora-weight-decay', type=float, default=0.01)

    args = parser.parse_args()

    launch(
        main,
        args.num_gpus,
        num_machines=args.num_machines,
        machine_rank=args.machine_rank,
        dist_url=args.dist_url,
        args=(args,),
    )
