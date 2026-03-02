# -*- coding: utf-8 -*-
"""
正交概念字典 (Orthogonal Concept Dictionary)
============================================
基于 GPM (Gradient Projection Memory) 的 SVD 零空间投影方案，
实现超参数化掩码空间策略的连续谓词字典。

核心思想:
- 预分配大字典 P_total ∈ R^{M×D}，正交初始化
- 维护 K_active 标量，仅截取前 K_active 行参与前向
- 扩容时基于 SVD 零空间投影，保证新旧谓词正交
- 严格 L2 归一化，确保点积值域在 [0,1]

参考: GPM (third_party_code/GPM/main_cifar100.py:update_GPM)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


def _get_device(preferred: str = "auto") -> torch.device:
    """设备选择: cuda > mps > cpu"""
    if preferred == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return torch.device("mps")
        else:
            return torch.device("cpu")
    return torch.device(preferred)


class OrthogonalConceptDict(nn.Module):
    """
    正交概念字典模块
    
    参数:
        feat_dim (int): 输入特征维度 D
        total_concepts (int): 字典总容量 M (默认 2048)
        init_active (int): 初始激活概念数 K_active (默认 64)
    
    前向:
        输入: Z ∈ R^{B×D} (ROI 特征)
        输出: C ∈ R^{B×K_active} (概念激活值，值域 [0,1])
    """

    def __init__(
        self,
        feat_dim: int = 256,
        total_concepts: int = 2048,
        init_active: int = 64,
        init_tau: float = 0.5,
    ):
        super().__init__()
        self.feat_dim = feat_dim           # D
        self.total_concepts = total_concepts  # M
        self.K_active = init_active        # 当前激活数

        # Learnable temperature for cosine similarity → concept activation
        self.log_tau = nn.Parameter(torch.tensor(float(init_tau)).log())

        # 预分配字典参数 P_total ∈ R^{M×D}，正交初始化
        P_total = torch.empty(total_concepts, feat_dim)
        num_blocks = (total_concepts + feat_dim - 1) // feat_dim
        for i in range(num_blocks):
            start = i * feat_dim
            end = min((i + 1) * feat_dim, total_concepts)
            block = torch.empty(end - start, feat_dim)
            nn.init.orthogonal_(block)
            P_total[start:end] = block

        P_total = F.normalize(P_total, p=2, dim=1)
        self.P_total = nn.Parameter(P_total)

        self.register_buffer(
            "frozen_mask",
            torch.zeros(total_concepts, dtype=torch.bool),
        )

    @property
    def P_active(self) -> torch.Tensor:
        """返回当前激活的字典子集 P_active ∈ R^{K_active×D}"""
        return self.P_total[:self.K_active]

    def forward(self, Z: torch.Tensor) -> torch.Tensor:
        """
        前向投影 (Grounding)
        
        Args:
            Z: ROI 特征 [B, D]
        
        Returns:
            C: 概念激活值 [B, K_active]，值域 (0, 1)
        """
        Z_norm = F.normalize(Z, p=2, dim=-1)                  # [B, D]
        P_norm = F.normalize(self.P_active, p=2, dim=-1)      # [K_active, D]

        cos_sim = torch.mm(Z_norm, P_norm.t())                # [B, K_active]
        tau = self.log_tau.exp().clamp(min=0.01)
        C = torch.sigmoid(cos_sim / tau)                      # [B, K_active]
        return C

    @torch.no_grad()
    def expand_via_nullspace(
        self, 
        Z_new: torch.Tensor, 
        num_new_concepts: int = 16,
    ) -> int:
        """
        基于 SVD 的零空间投影扩容
        
        核心公式 (复用 GPM Eq-8):
            Z_new⊥ = Z_new - U_old · U_old^T · Z_new
        
        Args:
            Z_new: 新任务样本特征 [N, D]
            num_new_concepts: 新增概念数量
        
        Returns:
            实际新增的概念数量
        """
        device = self.P_total.device
        Z_new = Z_new.to(device)

        # 检查容量
        remaining = self.total_concepts - self.K_active
        num_new = min(num_new_concepts, remaining)
        if num_new <= 0:
            return 0

        # Step 1: 获取旧字典的正交基 U_old
        # 使用 torch.linalg.svd 代替手工 Gram-Schmidt (数值更稳定)
        P_old = self.P_active.detach().float()  # [K_active, D]
        U, S, Vh = torch.linalg.svd(P_old, full_matrices=False)
        # U_old = Vh^T 的前 rank 列 (即旧字典在 D 维空间的正交基)
        # 保留有效奇异值对应的基
        rank = (S > 1e-6).sum().item()
        U_old = Vh[:rank].t()  # [D, rank]

        # Step 2: 零空间投影
        # Z_new⊥ = Z_new - U_old · U_old^T · Z_new
        Z_float = Z_new.float()  # [N, D]
        proj = Z_float @ U_old @ U_old.t()  # [N, D]
        Z_orth = Z_float - proj               # [N, D]

        # Step 3: 对正交化后的特征做 SVD，提取主成分
        U_new, S_new, Vh_new = torch.linalg.svd(Z_orth.t(), full_matrices=False)
        # U_new[:, :num_new] 就是新概念的正交基
        actual_new = min(num_new, U_new.shape[1])
        new_concepts = U_new[:, :actual_new].t()  # [actual_new, D]

        # L2 归一化
        new_concepts = F.normalize(new_concepts, p=2, dim=1)

        # Step 4: 写入字典的新维度
        old_K = self.K_active
        self.P_total.data[old_K:old_K + actual_new] = new_concepts.to(
            self.P_total.dtype
        )

        # Step 5: 冻结旧概念，更新 K_active
        self.frozen_mask[:old_K] = True
        self.K_active += actual_new

        return actual_new

    def get_frozen_params_mask(self) -> torch.Tensor:
        """返回 P_total 的逐行冻结掩码 [M, 1]，用于梯度 hook"""
        mask = self.frozen_mask[:self.K_active].unsqueeze(1).float()  # [K, 1]
        return mask

    def apply_gradient_mask(self):
        """在反向传播后调用，将冻结行的梯度清零"""
        if self.P_total.grad is not None:
            frozen = self.frozen_mask.unsqueeze(1).expand_as(self.P_total)
            self.P_total.grad.data[frozen] = 0.0

    def extra_repr(self) -> str:
        return (
            f"feat_dim={self.feat_dim}, "
            f"total_concepts={self.total_concepts}, "
            f"K_active={self.K_active}"
        )
