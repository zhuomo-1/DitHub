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
    ):
        super().__init__()
        self.max_concepts = max_concepts  # M
        self.num_classes = num_classes    # C_total
        self.num_rules = num_rules       # R
        self.init_scale = init_scale

        # 逻辑权重掩码 Ω ∈ R^{C_total × R × 2M}
        # 使用 2M 是因为双极性扩展 (正/负文字)
        self.Omega = nn.Parameter(
            torch.randn(num_classes, num_rules, 2 * max_concepts) * init_scale
        )

        # 冻结掩码: 标记哪些类别的 Omega 行已冻结 (旧类别)
        self.register_buffer(
            "frozen_class_mask",
            torch.zeros(num_classes, dtype=torch.bool),
        )

    def _bipolar_expand(self, C: torch.Tensor) -> torch.Tensor:
        """
        双极性映射
        C [B, K] → L [B, 2K]
        前 K 维: 属性存在 (C)
        后 K 维: 属性不存在 (1-C)
        """
        return torch.cat([C, 1.0 - C], dim=-1)  # [B, 2K]

    def _binary_ste(self, omega: torch.Tensor) -> torch.Tensor:
        """
        Binary STE 离散化
        
        前向: W = (σ(Ω) > 0.5).float()  — 稀疏布尔掩码
        反向: 连续的 Sigmoid 梯度
        
        实现: W = STE(σ(Ω)) = (σ(Ω)>0.5).float() - σ(Ω).detach() + σ(Ω)
        """
        sig = torch.sigmoid(omega)
        return BinarySTE.apply(sig) - sig.detach() + sig

    def forward(
        self,
        C: torch.Tensor,
        K_active: int,
    ) -> torch.Tensor:
        """
        DNF 前向推理
        
        Args:
            C: 概念激活值 [B, K_active]，来自 OrthogonalConceptDict
            K_active: 当前激活概念数
        
        Returns:
            Y_hat: 类别概率 [B, C_total]
        """
        B = C.shape[0]
        
        # Step 1: 双极性扩展
        L = self._bipolar_expand(C)  # [B, 2*K_active]

        # Step 2: 截取活跃维度的 Omega 并 STE 离散化
        # Omega[:, :, :2*K_active] → 仅使用与当前激活概念对应的部分
        omega_active = self.Omega[:, :, :2 * K_active]  # [C, R, 2K]
        W = self._binary_ste(omega_active)               # [C, R, 2K]

        # Step 3: 合取 (AND over literals within each rule)
        # 对每条规则：LogSpaceAnd(L, W)
        # L: [B, 2K] → 扩展为 [B, 1, 1, 2K] 以便广播
        # W: [C, R, 2K]
        L_expanded = L.unsqueeze(1).unsqueeze(1)  # [B, 1, 1, 2K]
        W_expanded = W.unsqueeze(0)                # [1, C, R, 2K]

        # 对数空间合取
        P_rules = FuzzyLogicOperators.log_product_t_norm(
            L_expanded.expand(B, self.num_classes, self.num_rules, 2 * K_active),
            W_expanded.expand(B, self.num_classes, self.num_rules, 2 * K_active),
        )  # [B, C, R]

        # Step 4: 析取 (OR over rules for each class)
        Y_hat = FuzzyLogicOperators.log_product_t_conorm(P_rules)  # [B, C]

        return Y_hat

    def compute_logic_loss(
        self,
        K_active: int,
        lambda_l1: float = 1e-3,
        lambda_conflict: float = 1e-2,
    ) -> dict:
        """
        计算逻辑正则化损失
        
        Returns:
            dict: {
                'L_L1': 稀疏惩罚,
                'L_conflict': 互斥惩罚,
                'L_total': 加权总和,
            }
        """
        # 仅对活跃维度计算
        omega_active = self.Omega[:, :, :2 * K_active]  # [C, R, 2K]
        sig = torch.sigmoid(omega_active)

        # --- 稀疏惩罚 L_L1 ---
        # 鼓励逻辑掩码稀疏，减少每条规则使用的文字数
        L_L1 = sig.abs().mean()

        # --- 互斥惩罚 L_conflict ---
        # 同一规则不应对同一属性同时选择肯定和否定
        # W_plus = sig[:, :, :K]  (肯定部分)
        # W_minus = sig[:, :, K:]  (否定部分)
        K = K_active
        W_plus = sig[:, :, :K]       # [C, R, K]
        W_minus = sig[:, :, K:2*K]   # [C, R, K]
        L_conflict = (W_plus * W_minus).sum() / (self.num_classes * self.num_rules * K + 1e-8)

        L_total = lambda_l1 * L_L1 + lambda_conflict * L_conflict

        return {
            "L_L1": L_L1,
            "L_conflict": L_conflict,
            "L_total": L_total,
        }

    def expand_classes(self, num_new_classes: int) -> None:
        """
        增量扩展类别数
        
        冻结旧类别的 Omega，扩展新类别的行
        """
        old_C = self.num_classes
        new_C = old_C + num_new_classes

        # 冻结旧类别
        self.frozen_class_mask[:old_C] = True

        # 扩展 Omega
        old_omega = self.Omega.data  # [old_C, R, 2M]
        new_omega = torch.randn(
            num_new_classes, self.num_rules, 2 * self.max_concepts,
            device=old_omega.device, dtype=old_omega.dtype,
        ) * self.init_scale

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
