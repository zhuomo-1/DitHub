# -*- coding: utf-8 -*-
"""
可微 DNF 规则引擎 (Differentiable DNF Engine) v2
=================================================
核心改进（解决 Y_hat 初始化 ≈ 1.0 的 OR 饱和陷阱）:

1. 稀疏确定性初始化: 每条规则只强绑定 2~3 个文字（Ω=+3），其余全关（Ω=-3），
   使初始 P_rule ≈ C1*C2 ≈ 0.25 而非之前的 ≈ 0.5。
2. 退火 sigmoid: 训练早期用软 sigmoid(β·Ω) 代替 STE 硬阈值 0.5，
   打通梯度高速公路；后期 β→∞ 逼近离散逻辑。
3. 目标条件数正则: 不再无差别 L1 打压所有 Ω，而是将每条规则的激活文字数
   拉向目标值 μ_target ≈ 3，防止全开/全关坍塌。
"""

import torch
import torch.nn as nn

from .fuzzy_ops import FuzzyLogicOperators


class DifferentiableDNF(nn.Module):
    """
    可微析取范式 (DNF) 规则引擎 v2

    参数:
        max_concepts (int): 字典总容量 M
        num_classes (int): 初始类别数 C
        num_rules (int): 每类规则数 R (默认 3)
        literals_per_rule (int): 每条规则初始绑定的文字数 (默认 3)
    """

    def __init__(
        self,
        max_concepts: int = 2048,
        num_classes: int = 10,
        num_rules: int = 3,
        literals_per_rule: int = 3,
    ):
        super().__init__()
        self.max_concepts = max_concepts
        self.num_classes = num_classes
        self.num_rules = num_rules
        self.literals_per_rule = literals_per_rule

        omega = self._sparse_init(num_classes, num_rules, 2 * max_concepts, literals_per_rule)
        self.Omega = nn.Parameter(omega)

        self.register_buffer(
            "frozen_class_mask",
            torch.zeros(num_classes, dtype=torch.bool),
        )

        self._train_step = 0
        self._anneal_steps = 500

    @staticmethod
    def _sparse_init(C, R, num_literals, lits_per_rule):
        """
        稀疏确定性初始化:
        - 基础值 -3.0 → sigmoid(-3) ≈ 0.047，STE/soft 判定为 off
        - 每条规则随机选 lits_per_rule 个文字设为 +3.0 → sigmoid(3) ≈ 0.953，判定为 on
        """
        omega = torch.full((C, R, num_literals), -3.0)
        for c in range(C):
            for r in range(R):
                active_idx = torch.randperm(num_literals)[:lits_per_rule]
                omega[c, r, active_idx] = 3.0
        return omega

    def _get_beta(self) -> float:
        """退火温度: 1.0 → 10.0 线性退火"""
        progress = min(1.0, self._train_step / max(1, self._anneal_steps))
        return 1.0 + 9.0 * progress

    def _get_weights(self, omega: torch.Tensor) -> torch.Tensor:
        """
        退火 sigmoid 替代 STE:
        - 训练早期 (β≈1): soft sigmoid，梯度畅通
        - 训练后期 (β≈10): 逼近硬阈值，保持逻辑稀疏性
        - eval 模式: 直接硬阈值
        """
        if not self.training:
            return (torch.sigmoid(omega) > 0.5).float()

        beta = self._get_beta()
        soft_w = torch.sigmoid(omega * beta)

        if beta >= 8.0:
            hard_w = (soft_w > 0.5).float()
            return hard_w - soft_w.detach() + soft_w
        return soft_w

    def _bipolar_expand(self, C: torch.Tensor) -> torch.Tensor:
        """C [B, K] → L [B, 2K]: 前 K = 存在, 后 K = 不存在"""
        return torch.cat([C, 1.0 - C], dim=-1)

    def forward(
        self,
        C: torch.Tensor,
        K_active: int,
    ) -> torch.Tensor:
        """
        Args:
            C: 概念激活值 [B, K_active]
            K_active: 当前激活概念数

        Returns:
            Y_hat: 类别概率 [B, C_total]
        """
        B = C.shape[0]

        L = self._bipolar_expand(C)  # [B, 2*K_active]

        omega_active = self.Omega[:, :, :2 * K_active]  # [C, R, 2K]
        W = self._get_weights(omega_active)               # [C, R, 2K]

        L_expanded = L.unsqueeze(1).unsqueeze(1)  # [B, 1, 1, 2K]
        W_expanded = W.unsqueeze(0)                # [1, C, R, 2K]

        P_rules = FuzzyLogicOperators.log_product_t_norm(
            L_expanded.expand(B, self.num_classes, self.num_rules, 2 * K_active),
            W_expanded.expand(B, self.num_classes, self.num_rules, 2 * K_active),
        )  # [B, C, R]

        Y_hat = FuzzyLogicOperators.log_product_t_conorm(P_rules)  # [B, C]

        return Y_hat

    def step_anneal(self):
        """每个 training step 后调用，推进退火进度"""
        self._train_step += 1

    def compute_logic_loss(
        self,
        K_active: int,
        lambda_l1: float = 1e-3,
        lambda_conflict: float = 1e-2,
    ) -> dict:
        """
        目标条件数正则 + 互斥惩罚

        - 稀疏项: 将每条规则的激活文字数拉向 literals_per_rule (≈3)
        - 互斥项: 同一属性不应同时被选中和否定
        """
        omega_active = self.Omega[:, :, :2 * K_active]
        sig = torch.sigmoid(omega_active)

        # 目标条件数正则: (sum_of_activations - target)^2
        rule_activation_count = sig.sum(dim=-1)  # [C, R]
        mu_target = float(self.literals_per_rule)
        L_sparse = ((rule_activation_count - mu_target) ** 2).mean()

        # 互斥惩罚
        K = K_active
        W_plus = sig[:, :, :K]
        W_minus = sig[:, :, K:2*K]
        L_conflict = (W_plus * W_minus).sum() / (self.num_classes * self.num_rules * K + 1e-8)

        L_total = lambda_l1 * L_sparse + lambda_conflict * L_conflict

        return {
            "L_L1": L_sparse,
            "L_conflict": L_conflict,
            "L_total": L_total,
        }

    def expand_classes(self, num_new_classes: int) -> None:
        old_C = self.num_classes
        new_C = old_C + num_new_classes

        self.frozen_class_mask[:old_C] = True

        old_omega = self.Omega.data
        new_omega = self._sparse_init(
            num_new_classes, self.num_rules,
            2 * self.max_concepts, self.literals_per_rule,
        ).to(old_omega.device, old_omega.dtype)

        expanded = torch.cat([old_omega, new_omega], dim=0)
        self.Omega = nn.Parameter(expanded)

        new_mask = torch.zeros(new_C, dtype=torch.bool, device=self.frozen_class_mask.device)
        new_mask[:old_C] = True
        self.frozen_class_mask = new_mask

        self.num_classes = new_C

    def apply_gradient_mask(self):
        if self.Omega.grad is not None and self.frozen_class_mask.any():
            frozen = self.frozen_class_mask
            self.Omega.grad.data[frozen] = 0.0

    def extra_repr(self) -> str:
        return (
            f"num_classes={self.num_classes}, "
            f"num_rules={self.num_rules}, "
            f"max_concepts={self.max_concepts}, "
            f"beta={self._get_beta():.1f}"
        )
