# -*- coding: utf-8 -*-
"""
NSPS 双系统调度器 (Neural-Symbolic Production System)
=====================================================
编排 System 1（极化路由）和 System 2（NCRN 后件阵列），
实现模块化连续学习。

核心流程:
    训练: add_task() → 提取原型 → 极化入库 → 独立训练 NCRN → 冻结入列
    推理: inference() → 路由 Top-K → 后件校验 → 回退判决

设计原则:
    - 每个任务的 NCRN 完全物理隔离，训练复杂度 O(1)
    - 零遗忘：旧后件永远不被修改
    - 可解释：每个判决附带 DNF 规则解释
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Dict, Optional, Tuple
import os
import logging

from .ncrn_head import NCRN_Head
from .polarized_router import PolarizedAntecedentBase
from .hard_negative_sampler import HardNegativeSampler

logger = logging.getLogger(__name__)


def _auto_device() -> torch.device:
    """设备选择: cuda > mps > cpu"""
    if torch.cuda.is_available():
        return torch.device("cuda")
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


class NSPSSystem(nn.Module):
    """
    神经符号产生式双系统

    System 1: PolarizedAntecedentBase — 路由（"我是谁？"）
    System 2: List[NCRN_Head] — 后件阵列（"我为什么是它？"）

    参数:
        feat_dim (int): ROI 特征维度 D
        concepts_per_task (int): 每个任务 NCRN 的字典大小 M
        rules_per_class (int): 每类 DNF 规则数 R
        lambda_l1 (float): 稀疏正则化系数
        lambda_conflict (float): 互斥正则化系数
        top_k (int): 路由 Top-K
        gamma (float): 逻辑置信度阈值（回退触发线）
        tau (float): 路由温度
        neg_ratio (float): 硬负样本比例
        l1_anneal_epochs (int): L1 正则化退火 epoch 数
    """

    def __init__(
        self,
        feat_dim: int = 256,
        concepts_per_task: int = 64,
        rules_per_class: int = 8,
        lambda_l1: float = 1e-3,
        lambda_conflict: float = 1e-2,
        top_k: int = 3,
        gamma: float = 0.3,
        tau: float = 0.07,
        neg_ratio: float = 0.2,
        l1_anneal_epochs: int = 5,
    ):
        super().__init__()
        self.feat_dim = feat_dim
        self.concepts_per_task = concepts_per_task
        self.rules_per_class = rules_per_class
        self.lambda_l1 = lambda_l1
        self.lambda_conflict = lambda_conflict
        self.top_k = top_k
        self.gamma = gamma
        self.l1_anneal_epochs = l1_anneal_epochs

        # System 1: 极化前件路由器
        self.router = PolarizedAntecedentBase(
            feat_dim=feat_dim, tau=tau,
        )

        # System 2: 后件阵列
        self.consequents = nn.ModuleList()

        # 硬负样本采样器
        self.neg_sampler = HardNegativeSampler(neg_ratio=neg_ratio)

        # 任务元信息
        self.task_meta: List[dict] = []

        # 各任务的特征缓存（用于硬负样本采样）
        self._feature_cache: Dict[int, torch.Tensor] = {}

    @property
    def num_tasks(self) -> int:
        return self.router.T

    def create_task_ncrn(self, num_classes: int) -> NCRN_Head:
        """
        为新任务创建全新的 NCRN 实例

        Args:
            num_classes: 该任务的类别数

        Returns:
            全新的、未训练的 NCRN_Head
        """
        ncrn = NCRN_Head(
            feat_dim=self.feat_dim,
            num_classes=num_classes,
            total_concepts=self.concepts_per_task,
            init_active=self.concepts_per_task,  # 全量使用
            num_rules=self.rules_per_class,
            lambda_l1=self.lambda_l1,
            lambda_conflict=self.lambda_conflict,
        )
        return ncrn

    def train_single_task(
        self,
        ncrn: NCRN_Head,
        Z_train: torch.Tensor,
        Y_train: torch.Tensor,
        task_id: int,
        num_epochs: int = 50,
        lr: float = 0.001,
        weight_decay: float = 0.01,
        batch_size: int = 64,
        verbose: bool = True,
    ) -> dict:
        """
        独立训练单个任务的 NCRN

        完全隔离：仅使用该任务的数据 + 硬负样本

        Args:
            ncrn: 待训练的 NCRN_Head
            Z_train: 训练特征 [N, D]
            Y_train: 训练标签 [N, C]
            task_id: 当前任务 ID
            num_epochs / lr / weight_decay / batch_size: 训练超参数
            verbose: 是否打印日志

        Returns:
            dict: 训练统计 {final_loss, epochs, ...}
        """
        device = Z_train.device
        ncrn = ncrn.to(device)
        ncrn.train()

        optimizer = torch.optim.AdamW(
            ncrn.parameters(), lr=lr, weight_decay=weight_decay,
        )

        N = Z_train.shape[0]
        stats = {"losses": [], "final_loss": float("inf")}

        for epoch in range(num_epochs):
            # L1 退火: 前 l1_anneal_epochs 内从 0 线性增长到 lambda_l1
            if epoch < self.l1_anneal_epochs:
                current_l1 = self.lambda_l1 * (epoch / max(1, self.l1_anneal_epochs))
            else:
                current_l1 = self.lambda_l1

            # 硬负样本增强
            Z_aug, Y_aug = self.neg_sampler.sample(
                Z_train, Y_train,
                stored_features=self._feature_cache,
                current_task_id=task_id,
                router_K=self.router.get_active_probes() if self.num_tasks > 0 else None,
            )

            # Mini-batch 训练
            perm = torch.randperm(Z_aug.shape[0], device=device)
            epoch_loss = 0.0
            n_batches = 0

            for i in range(0, Z_aug.shape[0], batch_size):
                idx = perm[i : i + batch_size]
                Z_batch = Z_aug[idx]
                Y_batch = Y_aug[idx]

                # 前向
                Y_hat = ncrn(Z_batch)  # [B, C]

                # BCE 分类损失
                L_BCE = F.binary_cross_entropy(
                    Y_hat.clamp(1e-7, 1.0 - 1e-7),
                    Y_batch.float(),
                    reduction="mean",
                )

                # 逻辑正则化（带退火）
                logic_loss = ncrn.dnf.compute_logic_loss(
                    K_active=ncrn.K_active,
                    lambda_l1=current_l1,
                    lambda_conflict=self.lambda_conflict,
                )

                L_total = L_BCE + logic_loss["L_total"]

                optimizer.zero_grad()
                L_total.backward()
                torch.nn.utils.clip_grad_norm_(ncrn.parameters(), max_norm=0.1)
                optimizer.step()

                epoch_loss += L_total.item()
                n_batches += 1

            avg_loss = epoch_loss / max(n_batches, 1)
            stats["losses"].append(avg_loss)

            if verbose and (epoch % 10 == 0 or epoch == num_epochs - 1):
                logger.info(
                    f"  [Task {task_id}] Epoch {epoch}/{num_epochs}: "
                    f"loss={avg_loss:.4f} (l1_coeff={current_l1:.5f})"
                )

        stats["final_loss"] = stats["losses"][-1] if stats["losses"] else float("inf")
        stats["epochs"] = num_epochs
        return stats

    @torch.no_grad()
    def add_task(
        self,
        Z_samples: torch.Tensor,
        Y_samples: torch.Tensor,
        class_names: Optional[List[str]] = None,
        num_epochs: int = 50,
        lr: float = 0.001,
        **train_kwargs,
    ) -> dict:
        """
        添加新任务 — 完整编排

        流程:
        1. 提取原型 → 极化入库 (System 1)
        2. 实例化新 NCRN → 独立训练 → 冻结 (System 2)
        3. 缓存特征 (用于未来任务的硬负样本)

        Args:
            Z_samples: 新任务样本特征 [N, D]
            Y_samples: 新任务标签 [N, C]
            class_names: 类别名列表
            num_epochs / lr: 训练超参数

        Returns:
            dict: {task_id, router_info, train_stats, ncrn_params}
        """
        task_id = self.num_tasks
        info = {"task_id": task_id}
        num_classes = Y_samples.shape[1]

        # Step 1: System 1 — 极化入库
        router_info = self.router.register_task(
            Z_samples, task_id=task_id, class_names=class_names,
        )
        info["router"] = router_info
        logger.info(
            f"[NSPS] Task {task_id} registered: "
            f"sim_before={router_info['max_sim_before']:.4f} → "
            f"sim_after={router_info['max_sim_after']:.4f}"
        )

        # Step 2: System 2 — 创建并训练 NCRN
        ncrn = self.create_task_ncrn(num_classes)
        ncrn = ncrn.to(Z_samples.device)

        # 解除 @torch.no_grad() 进行训练
        with torch.enable_grad():
            train_stats = self.train_single_task(
                ncrn, Z_samples, Y_samples,
                task_id=task_id,
                num_epochs=num_epochs,
                lr=lr,
                **train_kwargs,
            )
        info["train_stats"] = train_stats

        # Step 3: 冻结并入列
        ncrn.eval()
        for p in ncrn.parameters():
            p.requires_grad_(False)
        self.consequents.append(ncrn)

        # Step 4: 统计动态 γ — 用训练集自身的置信度下限
        with torch.no_grad():
            Y_hat_val = ncrn(Z_samples)  # [N, C]
            max_probs_val = Y_hat_val.max(dim=1).values  # [N]
            # 取 5th percentile 作为该任务的置信度下限
            sorted_probs = max_probs_val.sort().values
            p5_idx = max(0, int(len(sorted_probs) * 0.05))
            task_gamma = sorted_probs[p5_idx].item()
            # 限制在 [0.1, 0.9] 范围内
            task_gamma = max(0.1, min(0.9, task_gamma))

        # 缓存特征
        self._feature_cache[task_id] = Z_samples.detach().clone()

        # 元信息
        meta = {
            "task_id": task_id,
            "num_classes": num_classes,
            "class_names": class_names or [],
            "num_samples": Z_samples.shape[0],
            "task_gamma": task_gamma,  # 动态置信度阈值
        }
        self.task_meta.append(meta)

        info["ncrn_params"] = sum(p.numel() for p in ncrn.parameters())
        info["task_gamma"] = task_gamma
        logger.info(
            f"[NSPS] Task {task_id} trained: "
            f"loss={train_stats['final_loss']:.4f}, "
            f"params={info['ncrn_params']}, "
            f"gamma={task_gamma:.4f}"
        )

        return info

    def inference(
        self,
        Z: torch.Tensor,
        top_k: Optional[int] = None,
        gamma: Optional[float] = None,
        return_details: bool = False,
    ) -> dict:
        """
        推理 — 回退式软路由

        流程:
        1. System 1 路由得到 Top-K 候选任务
        2. 按相似度降序，逐个送入后件 NCRN
        3. 若 max(prob) >= γ → 采纳；否则拒识回退，尝试下一个

        Args:
            Z: 查询特征 [B, D]
            top_k: Top-K 候选数（默认 self.top_k）
            gamma: 全局置信度阈值覆盖（默认使用逐任务动态 γ）
            return_details: 是否返回详细信息

        Returns:
            dict: {
                'task_ids': 最终选中的任务 ID [B]（-1 = UNKNOWN/Background）
                'class_ids': 每个样本在选中任务内的类别 ID [B]
                'probs': 置信度 [B]
                'rejected': 是否全部拒识（回退熔断）[B]
                'details': (可选) 详细回退信息
            }
        """
        top_k = top_k if top_k is not None else self.top_k
        B = Z.shape[0]
        device = Z.device

        # Step 1: 路由
        indices, scores = self.router.route(Z, top_k=top_k)  # [B, K]

        # 结果容器
        final_task_ids = torch.full((B,), -1, dtype=torch.long, device=device)
        final_class_ids = torch.full((B,), -1, dtype=torch.long, device=device)
        final_probs = torch.zeros(B, device=device)
        rejected = torch.ones(B, dtype=torch.bool, device=device)
        details = [] if return_details else None

        # Step 2 & 3: 逐候选回退式校验
        for k_idx in range(indices.shape[1]):
            pending_mask = rejected  # [B]
            if not pending_mask.any():
                break

            candidate_task_ids = indices[:, k_idx]  # [B]
            candidate_scores = scores[:, k_idx]  # [B]

            for b in range(B):
                if not pending_mask[b]:
                    continue

                tid = candidate_task_ids[b].item()
                if tid < 0 or tid >= len(self.consequents):
                    continue

                # 动态 γ: 优先用逐任务阈值，否则用全局默认
                if gamma is not None:
                    thresh = gamma
                elif tid < len(self.task_meta) and "task_gamma" in self.task_meta[tid]:
                    thresh = self.task_meta[tid]["task_gamma"]
                else:
                    thresh = self.gamma

                ncrn = self.consequents[tid]
                z_single = Z[b : b + 1]  # [1, D]

                with torch.no_grad():
                    y_hat = ncrn(z_single)  # [1, C]

                max_prob, max_cls = y_hat.max(dim=1)

                if return_details:
                    details.append({
                        "batch_idx": b,
                        "k_idx": k_idx,
                        "task_id": tid,
                        "max_prob": max_prob.item(),
                        "max_cls": max_cls.item(),
                        "route_score": candidate_scores[b].item(),
                        "threshold": thresh,
                        "accepted": max_prob.item() >= thresh,
                    })

                if max_prob.item() >= thresh:
                    final_task_ids[b] = tid
                    final_class_ids[b] = max_cls.item()
                    final_probs[b] = max_prob.item()
                    rejected[b] = False

        # 回退熔断: rejected=True 的样本标记为 UNKNOWN (task_id=-1, class_id=-1)
        # 这是明确的出口，不会强行输出不合格的预测

        result = {
            "task_ids": final_task_ids,
            "class_ids": final_class_ids,
            "probs": final_probs,
            "rejected": rejected,
        }
        if return_details:
            result["details"] = details

        return result

    def inference_batched(
        self,
        Z: torch.Tensor,
        top_k: Optional[int] = None,
        gamma: Optional[float] = None,
    ) -> dict:
        """
        高效批量推理 — 将同一候选任务的样本打包处理

        与 inference() 行为完全一致，但将同一候选任务的样本打包前向，
        避免逐样本 Python 循环，提升吞吐。

        Args:
            Z: 查询特征 [B, D]
            top_k: Top-K 候选数（默认 self.top_k）
            gamma: 全局置信度阈值覆盖（默认使用逐任务动态 γ）

        Returns:
            同 inference()
        """
        top_k = top_k if top_k is not None else self.top_k
        # gamma=None 时走逐任务动态 γ；显式传入时作为全局覆盖
        global_gamma_override = gamma  # None 表示不覆盖
        B = Z.shape[0]
        device = Z.device

        indices, scores = self.router.route(Z, top_k=top_k)

        final_task_ids = torch.full((B,), -1, dtype=torch.long, device=device)
        final_class_ids = torch.full((B,), -1, dtype=torch.long, device=device)
        final_probs = torch.zeros(B, device=device)
        rejected = torch.ones(B, dtype=torch.bool, device=device)

        for k_idx in range(indices.shape[1]):
            pending = rejected.clone()
            if not pending.any():
                break

            candidate_tids = indices[:, k_idx]  # [B]

            # 按 task_id 分组批量处理
            for tid in range(len(self.consequents)):
                mask = pending & (candidate_tids == tid)
                if not mask.any():
                    continue

                # 动态 γ：优先逐任务阈值，显式覆盖时使用全局值
                if global_gamma_override is not None:
                    thresh = global_gamma_override
                elif tid < len(self.task_meta) and "task_gamma" in self.task_meta[tid]:
                    thresh = self.task_meta[tid]["task_gamma"]
                else:
                    thresh = self.gamma

                z_batch = Z[mask]  # [n, D]
                ncrn = self.consequents[tid]

                with torch.no_grad():
                    y_hat = ncrn(z_batch)  # [n, C]

                max_probs, max_cls = y_hat.max(dim=1)
                accept = max_probs >= thresh

                # 写回
                batch_indices = mask.nonzero(as_tuple=True)[0]
                for i, bi in enumerate(batch_indices):
                    if accept[i]:
                        final_task_ids[bi] = tid
                        final_class_ids[bi] = max_cls[i]
                        final_probs[bi] = max_probs[i]
                        rejected[bi] = False

        return {
            "task_ids": final_task_ids,
            "class_ids": final_class_ids,
            "probs": final_probs,
            "rejected": rejected,
        }

    def get_global_class_mapping(self) -> Dict[Tuple[int, int], str]:
        """
        获取全局类别映射: (task_id, local_class_id) → class_name
        """
        mapping = {}
        for meta in self.task_meta:
            tid = meta["task_id"]
            for c_idx, name in enumerate(meta.get("class_names", [])):
                mapping[(tid, c_idx)] = name
        return mapping

    def save_checkpoint(self, save_dir: str):
        """保存全系统状态"""
        os.makedirs(save_dir, exist_ok=True)

        # 1. 保存路由器
        torch.save(self.router.state_dict(), os.path.join(save_dir, "router.pth"))

        # 2. 保存每个后件
        for i, ncrn in enumerate(self.consequents):
            torch.save(ncrn.state_dict(), os.path.join(save_dir, f"ncrn_{i}.pth"))

        # 3. 保存元信息
        torch.save({
            "task_meta": self.task_meta,
            "num_tasks": self.num_tasks,
            "config": {
                "feat_dim": self.feat_dim,
                "concepts_per_task": self.concepts_per_task,
                "rules_per_class": self.rules_per_class,
                "lambda_l1": self.lambda_l1,
                "lambda_conflict": self.lambda_conflict,
                "top_k": self.top_k,
                "gamma": self.gamma,
            },
        }, os.path.join(save_dir, "nsps_meta.pth"))

        # 4. 保存特征缓存
        torch.save(self._feature_cache, os.path.join(save_dir, "feature_cache.pth"))

        logger.info(f"[NSPS] Checkpoint saved to {save_dir} ({self.num_tasks} tasks)")

    def load_checkpoint(self, save_dir: str, device: Optional[torch.device] = None):
        """恢复全系统状态"""
        device = device or _auto_device()

        # 1. 元信息
        meta_path = os.path.join(save_dir, "nsps_meta.pth")
        meta = torch.load(meta_path, map_location="cpu")
        self.task_meta = meta["task_meta"]
        num_tasks = meta["num_tasks"]

        # 2. 路由器
        router_path = os.path.join(save_dir, "router.pth")
        self.router.load_state_dict(torch.load(router_path, map_location=device))

        # 3. 后件
        self.consequents = nn.ModuleList()
        for i in range(num_tasks):
            ncrn_path = os.path.join(save_dir, f"ncrn_{i}.pth")
            task_meta = self.task_meta[i]
            ncrn = self.create_task_ncrn(task_meta["num_classes"])
            ncrn.load_state_dict(torch.load(ncrn_path, map_location="cpu"))
            ncrn = ncrn.to(device)  # 显式移动到目标设备
            ncrn.eval()
            for p in ncrn.parameters():
                p.requires_grad_(False)
            self.consequents.append(ncrn)

        # 4. 特征缓存 — 移动到目标设备
        cache_path = os.path.join(save_dir, "feature_cache.pth")
        if os.path.exists(cache_path):
            raw_cache = torch.load(cache_path, map_location="cpu")
            self._feature_cache = {
                k: v.to(device) if isinstance(v, torch.Tensor) else v
                for k, v in raw_cache.items()
            }

        # 移动路由器到目标设备
        self.router = self.router.to(device)

        logger.info(f"[NSPS] Checkpoint loaded from {save_dir} ({num_tasks} tasks)")

    def extra_repr(self) -> str:
        return (
            f"tasks={self.num_tasks}, "
            f"concepts/task={self.concepts_per_task}, "
            f"rules/class={self.rules_per_class}, "
            f"top_k={self.top_k}, gamma={self.gamma}"
        )
