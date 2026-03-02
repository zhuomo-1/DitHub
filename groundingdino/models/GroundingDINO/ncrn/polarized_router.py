# -*- coding: utf-8 -*-
"""
任务路由器 (System 1)
=====================
管理任务原型库。由于 GroundingDINO 冻结特征高度相似（任务间 cos_sim 0.93~0.99），
直接使用原始均值作为探针，配合低温度 softmax 路由。

关键设计决策:
    当特征本身高度相似时，零空间极化会把探针推离真实数据流形，
    导致路由完全失效。正确做法是保留真实均值，依赖 tau 放大微小差异。
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Optional, Dict, Tuple
import logging

logger = logging.getLogger(__name__)


class PolarizedAntecedentBase(nn.Module):
    """
    System 1: 任务路由器

    维护一个 [T, D] 的探针矩阵 K，每个任务对应一行。
    新任务注册时，直接使用 L2 归一化的均值原型入库（不做极化）。
    路由时用 cosine 相似度 / τ 做 Top-K 选择。

    Args:
        feat_dim: 特征维度 D
        max_tasks: 最大任务容量
        tau: 路由温度（越小区分度越高，对高相似度场景需要很低）
        margin: 保留用于兼容（未使用）
    """

    def __init__(
        self,
        feat_dim: int = 256,
        max_tasks: int = 100,
        tau: float = 0.01,
        margin: float = 0.3,
    ):
        super().__init__()
        self.feat_dim = feat_dim
        self.max_tasks = max_tasks
        self.tau = tau
        self.margin = margin

        self.register_buffer(
            "K", torch.zeros(max_tasks, feat_dim)
        )
        self.register_buffer(
            "T_count", torch.tensor(0, dtype=torch.long)
        )

        self._class_names: Dict[int, List[str]] = {}

    @property
    def T(self) -> int:
        return self.T_count.item()

    def get_active_probes(self) -> Optional[torch.Tensor]:
        """返回已注册的探针矩阵 K[T, D]，无任务时返回 None"""
        if self.T == 0:
            return None
        return self.K[: self.T].clone()

    def register_task(
        self,
        Z_samples: torch.Tensor,
        task_id: int,
        class_names: Optional[List[str]] = None,
    ) -> dict:
        """
        注册新任务原型

        流程: 计算均值原型 → L2 归一化 → 直接入库

        Args:
            Z_samples: 该任务的样本特征 [N, D]
            task_id: 任务 ID（应等于当前 self.T）
            class_names: 类别名列表

        Returns:
            dict: {max_sim_before, max_sim_after, polarized, energy_ratio}
        """
        assert task_id == self.T, f"Expected task_id={self.T}, got {task_id}"
        assert self.T < self.max_tasks, f"Router full ({self.max_tasks} tasks)"

        device = Z_samples.device
        info = {}

        K_new = Z_samples.mean(dim=0)  # [D]
        K_new = F.normalize(K_new, dim=0)

        if self.T > 0:
            K_old = self.K[: self.T].to(device)  # [T, D]
            sim_before = F.cosine_similarity(
                K_new.unsqueeze(0), K_old, dim=1
            )
            info["max_sim_before"] = sim_before.abs().max().item()
            info["max_sim_after"] = info["max_sim_before"]
            info["energy_ratio"] = 1.0
            info["polarized"] = "raw_mean"
        else:
            info["max_sim_before"] = 0.0
            info["max_sim_after"] = 0.0
            info["energy_ratio"] = 1.0
            info["polarized"] = "first_task"

        K_final = K_new

        self.K[self.T] = K_final.detach().to(self.K.device)
        self.T_count += 1

        if class_names is not None:
            self._class_names[task_id] = class_names

        return info

    def route(
        self, Z: torch.Tensor, top_k: int = 3
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        cosine 相似度 / τ → Top-K 路由

        Args:
            Z: 查询特征 [B, D]
            top_k: 返回的候选任务数

        Returns:
            indices: [B, K] 任务索引
            scores: [B, K] 路由分数（softmax 后）
        """
        assert self.T > 0, "No tasks registered yet"
        K_active = self.K[: self.T].to(Z.device)  # [T, D]

        Z_norm = F.normalize(Z, dim=1)  # [B, D]
        K_norm = F.normalize(K_active, dim=1)  # [T, D]

        sim = Z_norm @ K_norm.T  # [B, T]
        sim_scaled = sim / self.tau

        actual_k = min(top_k, self.T)
        top_scores, top_indices = sim_scaled.topk(actual_k, dim=1)  # [B, K]

        top_scores = F.softmax(top_scores, dim=1)

        if actual_k < top_k:
            B = Z.shape[0]
            pad_size = top_k - actual_k
            pad_idx = torch.full(
                (B, pad_size), -1, dtype=torch.long, device=Z.device
            )
            pad_scores = torch.zeros(B, pad_size, device=Z.device)
            top_indices = torch.cat([top_indices, pad_idx], dim=1)
            top_scores = torch.cat([top_scores, pad_scores], dim=1)

        return top_indices, top_scores

    def extra_repr(self) -> str:
        return (
            f"feat_dim={self.feat_dim}, T={self.T}/{self.max_tasks}, "
            f"tau={self.tau}, margin={self.margin}"
        )
