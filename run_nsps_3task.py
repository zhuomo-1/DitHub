#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
NSPS 3-task 快速验证脚本
========================
只跑前 3 个任务的训练 + 逐任务评估，用于快速调试。
"""

import gc, glob, logging, os, sys, random, traceback
from pathlib import Path

import torch
import torch.nn.functional as F

os.environ["TOKENIZERS_PARALLELISM"] = "false"
sys.path.insert(0, os.path.dirname(__file__))

from detectron2.config import LazyConfig, instantiate
from detectron2.engine.defaults import setup_logger
from detectron2.utils.file_io import PathManager
from detectron2.evaluation import COCOEvaluator, print_csv_format
from detectron2.utils.env import seed_all_rng

from groundingdino.util.slconfig import SLConfig
from groundingdino.util.task_memory import TaskMemory
from groundingdino.config.configs.common.data.odinw.mapping_classes import ODINW_13_FILE_MAPPING

from main_nsps import (
    load_grounding_dino, extract_task_features, run_nsps_add_task,
    NSPSInferenceModel,
)
from groundingdino.models.GroundingDINO.ncrn import NSPSSystem
from groundingdino.util.inference import inference_on_dataset

logger = logging.getLogger("nsps_3task")

OUTPUT_DIR = "./output/nsps_3task"
CKPT_DIR = os.path.join(OUTPUT_DIR, "nsps_checkpoint")
CONFIG_FILE = "test/test_odinw13"
MODEL_CFG = "groundingdino/config/GroundingDINO_SwinT_OGC_dt_ncrn.py"
MODEL_CKPT = "groundingdino_swint_ogc.pth"
SEED = 42
NUM_TASKS = 3


def main():
    PathManager.mkdirs(OUTPUT_DIR)
    setup_logger(OUTPUT_DIR, distributed_rank=0, name="fvcore")
    global logger
    logger = setup_logger(OUTPUT_DIR, distributed_rank=0, name="nsps_3task")

    seed_all_rng(SEED)
    device = torch.device("cuda")

    TaskMemory().task_mapping = ODINW_13_FILE_MAPPING
    gd_model = load_grounding_dino(MODEL_CFG, MODEL_CKPT, OUTPUT_DIR, device)

    ncrn_cfg = SLConfig.fromfile(MODEL_CFG)
    nsps = NSPSSystem(
        feat_dim=getattr(ncrn_cfg, 'ncrn_feat_dim', 256),
        concepts_per_task=getattr(ncrn_cfg, 'ncrn_init_active', 64),
        rules_per_class=getattr(ncrn_cfg, 'ncrn_num_rules', 3),
        lambda_l1=getattr(ncrn_cfg, 'ncrn_lambda_l1', 1e-3),
        lambda_conflict=getattr(ncrn_cfg, 'ncrn_lambda_conflict', 1e-2),
    ).to(device)

    train_dir = os.path.join(CONFIG_FILE, "for_train")
    task_paths = sorted(glob.glob(os.path.join(train_dir, "*.py")))[:NUM_TASKS]
    logger.info(f"=== 3-Task 快速验证 ===")
    logger.info(f"Tasks: {[Path(p).stem for p in task_paths]}")

    # ========== 训练 ==========
    for task_idx, cfg_path in enumerate(task_paths):
        name = Path(cfg_path).stem
        logger.info(f"\n{'='*50}")
        logger.info(f"[Train] Task {task_idx}: {name}")
        logger.info(f"{'='*50}")

        cfg = LazyConfig.load(cfg_path)
        cfg.train.output_dir = os.path.join(OUTPUT_DIR, "odinw13", name.replace("test_", "") + "_cet")
        PathManager.mkdirs(cfg.train.output_dir)

        class_names = list(cfg.dataloader.train.mapper.categories_names)
        info = run_nsps_add_task(
            nsps, gd_model, cfg, class_names, device,
            nsps_epochs=100, nsps_lr=0.003, max_samples=500,
        )
        nsps.save_checkpoint(CKPT_DIR)
        logger.info(f"  => loss={info['train_stats']['final_loss']:.4f}, gamma={info['task_gamma']:.4f}")

        gc.collect(); torch.cuda.empty_cache()

    # ========== 训练后诊断 ==========
    logger.info(f"\n{'='*50}")
    logger.info(f"训练后诊断")
    logger.info(f"{'='*50}")

    for tid in range(nsps.num_tasks):
        Z = nsps._feature_cache[tid]
        indices, scores = nsps.router.route(Z, top_k=3)
        top1_acc = (indices[:, 0] == tid).float().mean().item()

        ncrn = nsps.consequents[tid]
        ncrn.eval()
        with torch.no_grad():
            Y_hat = ncrn(Z)
        max_probs = Y_hat.max(dim=1).values
        pred_cls = Y_hat.argmax(dim=1)
        unique = pred_cls.unique().tolist()

        meta = nsps.task_meta[tid]
        logger.info(
            f"  Task {tid} ({meta['num_classes']}cls, {meta['class_names']}): "
            f"route_top1={top1_acc:.0%}, "
            f"prob=[{max_probs.min():.3f},{max_probs.mean():.3f},{max_probs.max():.3f}], "
            f"unique_pred={unique}"
        )

    # ========== 逐任务评估 ==========
    global_class_names = []
    for meta in nsps.task_meta:
        global_class_names.extend(meta.get("class_names", []))

    inference_model = NSPSInferenceModel(gd_model, nsps, global_class_names).to(device)
    inference_model.eval()

    train_task_stems = [Path(p).stem for p in task_paths]
    all_results = {}

    for task_idx, cfg_path in enumerate(task_paths):
        name = Path(cfg_path).stem
        logger.info(f"\n{'='*50}")
        logger.info(f"[Eval] Task {task_idx}: {name}")
        logger.info(f"{'='*50}")

        inference_model.set_eval_task(task_idx)

        cfg = LazyConfig.load(cfg_path)
        cfg.train.output_dir = os.path.join(OUTPUT_DIR, "odinw13", name.replace("test_", "") + "_cet")
        PathManager.mkdirs(cfg.train.output_dir)
        cfg.dataloader.test.num_workers = 0

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
        ap = ret.get('bbox', {}).get('AP', 0.0)
        ap50 = ret.get('bbox', {}).get('AP50', 0.0)
        all_results[name] = ret.get('bbox', {})
        logger.info(f"  => AP={ap:.2f}, AP50={ap50:.2f}")

        gc.collect(); torch.cuda.empty_cache()

    # ========== 汇总 ==========
    logger.info(f"\n{'='*50}")
    logger.info(f"汇总")
    logger.info(f"{'='*50}")
    ap_list = []
    for name, r in all_results.items():
        ap = r.get("AP", 0.0)
        ap50 = r.get("AP50", 0.0)
        ap_list.append(ap)
        logger.info(f"  {name}: AP={ap:.2f}, AP50={ap50:.2f}")
    if ap_list:
        logger.info(f"  Mean AP = {sum(ap_list)/len(ap_list):.2f}")


if __name__ == "__main__":
    main()
