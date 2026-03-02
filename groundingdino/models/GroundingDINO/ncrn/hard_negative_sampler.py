# -*- coding: utf-8 -*-
"""
硬负样本采样器
==============
从路由相似度最高的其他任务中采样负例，混入训练集，
强制 NCRN 输出低概率，防止 System 2 过度自信。

在每次 add_task() 内部自动调用，无需手动干预。
"""

import torch
import torch.nn.functional as F
from typing import Dict, Optional
import logging

logger = logging.getLogger(__name__)


class HardNegativeSampler:
    """
    硬负样本采样器

    从已有任务的特征缓存中，按路由相似度排序，
    从最相似的其他任务中采样负例。负例标签全部设为全零向量（背景）。

    Args:
        neg_ratio: 负样本占正样本的比例
    """

    def __init__(self, neg_ratio: float = 0.2):
        self.neg_ratio = neg_ratio

    def sample(
        self,
        Z_pos: torch.Tensor,
        Y_pos: torch.Tensor,
        stored_features: Dict[int, torch.Tensor],
        current_task_id: int,
        router_K: Optional[torch.Tensor] = None,
    ) -> tuple:
        """
        采样硬负样本并与正样本拼接

        Args:
            Z_pos: 正样本特征 [N_pos, D]
            Y_pos: 正样本标签 [N_pos, C]
            stored_features: {task_id: Z_tensor} 各任务的特征缓存
            current_task_id: 当前任务 ID（排除自身）
            router_K: 路由器探针矩阵 [T, D]，用于确定最相似任务

        Returns:
            Z_aug: [N_pos + N_neg, D]
            Y_aug: [N_pos + N_neg, C]
        """
        N_pos = Z_pos.shape[0]
        N_neg_target = max(1, int(N_pos * self.neg_ratio))
        C = Y_pos.shape[1]
        device = Z_pos.device

        other_task_ids = [
            tid for tid in stored_features.keys() if tid != current_task_id
        ]

        if not other_task_ids:
            return Z_pos, Y_pos

        if router_K is not None and router_K.shape[0] > 0:
            neg_features = self._sample_by_router_similarity(
                Z_pos, stored_features, other_task_ids,
                router_K, current_task_id, N_neg_target, device,
            )
        else:
            neg_features = self._sample_uniform(
                stored_features, other_task_ids, N_neg_target, device,
            )

        if neg_features is None or neg_features.shape[0] == 0:
            return Z_pos, Y_pos

        N_neg = neg_features.shape[0]
        neg_labels = torch.zeros(N_neg, C, device=device)

        Z_aug = torch.cat([Z_pos, neg_features], dim=0)
        Y_aug = torch.cat([Y_pos, neg_labels], dim=0)

        return Z_aug, Y_aug

    def _sample_by_router_similarity(
        self,
        Z_pos: torch.Tensor,
        stored_features: Dict[int, torch.Tensor],
        other_task_ids: list,
        router_K: torch.Tensor,
        current_task_id: int,
        N_neg_target: int,
        device: torch.device,
    ) -> Optional[torch.Tensor]:
        """按路由相似度从最相似的其他任务中采样"""
        if current_task_id >= router_K.shape[0]:
            return self._sample_uniform(
                stored_features, other_task_ids, N_neg_target, device,
            )

        current_proto = router_K[current_task_id]  # [D]
        sims = []
        for tid in other_task_ids:
            if tid < router_K.shape[0]:
                proto = router_K[tid]
                sim = F.cosine_similarity(
                    current_proto.unsqueeze(0).to(device),
                    proto.unsqueeze(0).to(device),
                ).item()
                sims.append((tid, abs(sim)))
            else:
                sims.append((tid, 0.0))

        sims.sort(key=lambda x: x[1], reverse=True)

        neg_chunks = []
        remaining = N_neg_target
        for tid, _ in sims:
            if remaining <= 0:
                break
            feats = stored_features[tid].to(device)
            n_take = min(remaining, feats.shape[0])
            perm = torch.randperm(feats.shape[0], device=device)[:n_take]
            neg_chunks.append(feats[perm])
            remaining -= n_take

        if not neg_chunks:
            return None
        return torch.cat(neg_chunks, dim=0)

    def _sample_uniform(
        self,
        stored_features: Dict[int, torch.Tensor],
        other_task_ids: list,
        N_neg_target: int,
        device: torch.device,
    ) -> Optional[torch.Tensor]:
        """均匀采样（无路由信息时的回退策略）"""
        all_feats = []
        for tid in other_task_ids:
            all_feats.append(stored_features[tid].to(device))

        if not all_feats:
            return None

        all_neg = torch.cat(all_feats, dim=0)
        n_take = min(N_neg_target, all_neg.shape[0])
        perm = torch.randperm(all_neg.shape[0], device=device)[:n_take]
        return all_neg[perm]
