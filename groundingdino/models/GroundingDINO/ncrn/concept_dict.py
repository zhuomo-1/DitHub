# -*- coding: utf-8 -*-
"""
概念字典 (Concept Dictionary)
============================
使用可学习线性投影 + sigmoid 将高维特征映射到概念空间。

设计决策:
    GroundingDINO 冻结特征在不同任务间高度相似（cos_sim 0.93~0.99），
    简单的 (1+cos)/2 映射无法产生足够的区分力。
    改为 Linear(D, K) + sigmoid，让模型自行学习有区分度的概念投影。
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class OrthogonalConceptDict(nn.Module):
    """
    概念字典模块 — 可学习线性投影

    参数:
        feat_dim (int): 输入特征维度 D
        total_concepts (int): 字典总容量 M (保留用于增量扩容)
        init_active (int): 初始激活概念数 K_active
    """

    def __init__(
        self,
        feat_dim: int = 256,
        total_concepts: int = 2048,
        init_active: int = 64,
    ):
        super().__init__()
        self.feat_dim = feat_dim
        self.total_concepts = total_concepts
        self.K_active = init_active

        self.proj = nn.Linear(feat_dim, total_concepts, bias=True)
        nn.init.xavier_uniform_(self.proj.weight)
        nn.init.zeros_(self.proj.bias)

        self.register_buffer(
            "frozen_mask",
            torch.zeros(total_concepts, dtype=torch.bool),
        )

    @property
    def P_active(self) -> torch.Tensor:
        """兼容旧接口"""
        return self.proj.weight[:self.K_active]

    def forward(self, Z: torch.Tensor) -> torch.Tensor:
        """
        Args:
            Z: ROI 特征 [B, D]

        Returns:
            C: 概念激活值 [B, K_active]，值域 (0, 1)
        """
        logits = self.proj(Z)  # [B, total_concepts]
        C = torch.sigmoid(logits[:, :self.K_active])  # [B, K_active]
        return C

    @torch.no_grad()
    def expand_via_nullspace(
        self,
        Z_new: torch.Tensor,
        num_new_concepts: int = 16,
    ) -> int:
        remaining = self.total_concepts - self.K_active
        num_new = min(num_new_concepts, remaining)
        if num_new <= 0:
            return 0

        old_K = self.K_active
        self.frozen_mask[:old_K] = True
        self.K_active += num_new
        return num_new

    def get_frozen_params_mask(self) -> torch.Tensor:
        mask = self.frozen_mask[:self.K_active].unsqueeze(1).float()
        return mask

    def apply_gradient_mask(self):
        if self.proj.weight.grad is not None:
            frozen = self.frozen_mask.unsqueeze(1).expand_as(self.proj.weight)
            self.proj.weight.grad.data[frozen] = 0.0

    def extra_repr(self) -> str:
        return (
            f"feat_dim={self.feat_dim}, "
            f"total_concepts={self.total_concepts}, "
            f"K_active={self.K_active}"
        )
