# -*- coding: utf-8 -*-
"""
可微 DNF 规则引擎 (Differentiable DNF Engine)
=============================================
将纯张量运算转化为标准 PyTorch nn.Module 的堆叠，
无缝融合神经与符号推理。

核心机制:
1. 双极性映射: L = cat([C, 1-C])，前半代表属性存在，后半代表不存在
2. Binary STE:  W = (σ(Ω)>0.5).float() - σ(Ω).detach() + σ(Ω)
                前向是稀疏布尔掩码，反向是连续 Sigmoid 梯度
3. 逻辑聚合:   用 FuzzyLogicOperators 的对数空间算子完成合取与析取

参考:
- pytorch_explain ConceptReasoningLayer 的 sign_attn + filter_attn 设计
- LTNtorch 的 stable 模式数值护栏
"""

import torch
import torch.nn as nn

from .fuzzy_ops import FuzzyLogicOperators


class BinarySTE(torch.autograd.Function):
    """
    Binary Straight-Through Estimator
    
    前向: 离散化为 0/1 (threshold=0.5)
    反向: 梯度直通 (identity)
    """

    @staticmethod
    def forward(ctx, input: torch.Tensor) -> torch.Tensor:
        return (input > 0.5).float()

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor) -> torch.Tensor:
        return grad_output  # straight-through


class DifferentiableDNF(nn.Module):
    """
    可微析取范式 (DNF) 规则引擎
    
    每个类别有 R 条规则，每条规则对 2*K_active 个文字（含正/负）
    进行加权合取，然后 R 条规则进行析取组合。
    
    参数:
        max_concepts (int): 字典总容量 M (与 OrthogonalConceptDict 对齐)
        num_classes (int): 初始类别数 C
        num_rules (int): 每类规则数 R (默认 8)
        init_scale (float): Omega 初始化标准差
    """

    def __init__(
        self,
        max_concepts: int = 2048,
        num_classes: int = 10,
        num_rules: int = 8,
        init_scale: float = 0.01,
        init_active_per_rule: int = 2,
        init_k_active: int = 64,
    ):
        super().__init__()
        self.max_concepts = max_concepts  # M
        self.num_classes = num_classes    # C_total
        self.num_rules = num_rules       # R
        self.init_scale = init_scale
        self.init_active_per_rule = init_active_per_rule
        self.init_k_active = init_k_active

        self.Omega = nn.Parameter(
            self._sparse_init_omega(num_classes, num_rules, 2 * max_concepts,
                                    init_active_per_rule,
                                    active_range=2 * init_k_active)
        )

        self.register_buffer(
            "frozen_class_mask",
            torch.zeros(num_classes, dtype=torch.bool),
        )

    @staticmethod
    def _sparse_init_omega(C, R, num_lits, active_per_rule=2, active_range=None):
        """Sparse deterministic init: each rule binds a few random literals at +3
        (within the active_range dims), all others at -3."""
        if active_range is None:
            active_range = num_lits
        omega = torch.full((C, R, num_lits), -3.0)
        for c in range(C):
            for r in range(R):
                idx = torch.randperm(active_range)[:active_per_rule]
                omega[c, r, idx] = 3.0
        return omega

    def _bipolar_expand(self, C: torch.Tensor) -> torch.Tensor:
        """
        双极性映射
        C [B, K] → L [B, 2K]
        前 K 维: 属性存在 (C)
        后 K 维: 属性不存在 (1-C)
        """
        return torch.cat([C, 1.0 - C], dim=-1)  # [B, 2K]

    def _get_w(self, omega: torch.Tensor) -> torch.Tensor:
        """Binary STE: hard 0/1 forward, sigmoid gradient backward."""
        sig = torch.sigmoid(omega)
        hard = (sig > 0.5).float()
        return hard - sig.detach() + sig

    def set_annealing_progress(self, progress: float):
        """Set training progress in [0, 1] for STE annealing.
        beta ramps from 1.0 to 10.0 over training."""
        self._current_beta = 1.0 + 9.0 * min(max(progress, 0.0), 1.0)

    def forward(
        self,
        C: torch.Tensor,
        K_active: int,
        return_intermediates: bool = False,
    ):
        B = C.shape[0]

        L = self._bipolar_expand(C)

        omega_active = self.Omega[:, :, :2 * K_active]
        W = self._get_w(omega_active)

        L_expanded = L.unsqueeze(1).unsqueeze(1)
        W_expanded = W.unsqueeze(0)

        P_rules = FuzzyLogicOperators.log_product_t_norm(
            L_expanded.expand(B, self.num_classes, self.num_rules, 2 * K_active),
            W_expanded.expand(B, self.num_classes, self.num_rules, 2 * K_active),
        )

        Y_hat = FuzzyLogicOperators.log_product_t_conorm(P_rules)

        if return_intermediates:
            return Y_hat, {
                "L": L,
                "omega_active": omega_active,
                "W": W,
                "P_rules": P_rules,
            }
        return Y_hat

    def compute_logic_loss(
        self,
        K_active: int,
        lambda_l1: float = 1e-3,
        lambda_conflict: float = 1e-2,
        target_active: float = 3.0,
    ) -> dict:
        """Compute logic regularization: targeted sparsity + conflict penalty."""
        omega_active = self.Omega[:, :, :2 * K_active]  # [C, R, 2K]
        sig = torch.sigmoid(omega_active)

        # Targeted sparsity: penalise deviation from target_active literals per rule
        active_per_rule = sig.sum(dim=-1)  # [C, R]
        L_sparse = ((active_per_rule - target_active) ** 2).mean()

        # Conflict: same concept should not have both positive and negative active
        K = K_active
        W_plus = sig[:, :, :K]
        W_minus = sig[:, :, K:2*K]
        L_conflict = (W_plus * W_minus).sum() / (self.num_classes * self.num_rules * K + 1e-8)

        L_total = lambda_l1 * L_sparse + lambda_conflict * L_conflict

        return {
            "L_sparse": L_sparse,
            "L_conflict": L_conflict,
            "L_total": L_total,
        }

    def expand_classes(self, num_new_classes: int, current_k_active: int = None) -> None:
        """
        增量扩展类别数
        
        冻结旧类别的 Omega，扩展新类别的行
        """
        old_C = self.num_classes
        new_C = old_C + num_new_classes

        # 冻结旧类别
        self.frozen_class_mask[:old_C] = True

        old_omega = self.Omega.data  # [old_C, R, 2M]
        new_omega = self._sparse_init_omega(
            num_new_classes, self.num_rules, 2 * self.max_concepts,
            self.init_active_per_rule,
            active_range=2 * current_k_active if current_k_active else None,
        ).to(device=old_omega.device, dtype=old_omega.dtype)

        expanded = torch.cat([old_omega, new_omega], dim=0)  # [new_C, R, 2M]
        self.Omega = nn.Parameter(expanded)

        # 扩展冻结掩码
        new_mask = torch.zeros(new_C, dtype=torch.bool, device=self.frozen_class_mask.device)
        new_mask[:old_C] = True  # 旧类别冻结
        self.frozen_class_mask = new_mask

        self.num_classes = new_C

    def apply_gradient_mask(self):
        """将冻结类别的梯度清零"""
        if self.Omega.grad is not None and self.frozen_class_mask.any():
            frozen = self.frozen_class_mask  # [C]
            self.Omega.grad.data[frozen] = 0.0

    def extra_repr(self) -> str:
        return (
            f"num_classes={self.num_classes}, "
            f"num_rules={self.num_rules}, "
            f"max_concepts={self.max_concepts}"
        )
