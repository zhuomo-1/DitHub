# -*- coding: utf-8 -*-
"""
模糊逻辑算子 (Fuzzy Logic Operators)
====================================
对数空间 (Log-space) 实现的乘积三角模 (Product T-norm) 和三角余模 (T-conorm)。
强制使用对数空间运算，防御多重规则叠加导致的数值下溢出。

核心公式:
- LogSpaceAnd(L, W): exp(sum(W · log(L + ε)))  — 合取
- LogSpaceOr(P):     1 - exp(sum(log(1 - P + ε)))  — 析取

参考:
- LTNtorch (third_party_code/LTNtorch/ltn/fuzzy_ops.py:AndProd) — pi_0 数值护栏
- pytorch_explain (third_party_code/pytorch_explain/torch_explain/nn/semantics.py:ProductTNorm)
"""

import torch


class FuzzyLogicOperators:
    """
    对数空间模糊逻辑算子集合
    
    所有方法均为静态方法，无需实例化状态。
    使用对数空间运算替代直接乘法，防止多因子连乘导致的下溢出。
    """

    EPS = 1e-7  # 数值护栏 (参考 LTNtorch 的 pi_0 思想)

    @staticmethod
    def log_product_t_norm(
        literals: torch.Tensor,
        weights: torch.Tensor,
        eps: float = 1e-7,
    ) -> torch.Tensor:
        """
        对数空间乘积三角模 (合取 / AND)
        
        公式: exp(sum(W · log(L + ε)) / num_active)
        
        Empty rules (num_active=0) output 0 instead of 1 to prevent
        the OR gate from being saturated by vacuously true rules.
        """
        L_safe = literals.clamp(min=eps, max=1.0)

        log_L = torch.log(L_safe)
        weighted_log = weights * log_L
        log_sum = weighted_log.sum(dim=-1)

        num_active = weights.sum(dim=-1).clamp(min=1.0)
        result = torch.exp(log_sum / num_active)

        return result

    @staticmethod
    def log_product_t_conorm(
        rule_probs: torch.Tensor,
        eps: float = 1e-7,
    ) -> torch.Tensor:
        """
        对数空间乘积三角余模 (析取 / OR)
        
        公式: 1 - exp(sum(log(1 - P + ε)))
        
        等价于 1 - prod(1-P)，即概率和 (probabilistic sum)。
        在对数空间中防止 (1-P) 连乘下溢出。
        
        参考: LTNtorch OrProbSum: x + y - xy = 1 - (1-x)(1-y)
        
        Args:
            rule_probs: 各规则的真值 [..., R]，值域 (0, 1)
            eps:        数值护栏
        
        Returns:
            析取结果 [...]
        """
        # 数值护栏
        complement = (1.0 - rule_probs).clamp(min=eps, max=1.0)
        
        # 对数空间析取: 1 - exp(sum(log(1-P)))
        log_complement = torch.log(complement)             # [..., R]
        result = 1.0 - torch.exp(log_complement.sum(dim=-1))  # [...]
        
        # clamp 输出到 (0, 1) 防止浮点误差
        result = result.clamp(min=0.0, max=1.0)
        
        return result

    @staticmethod
    def fuzzy_not(x: torch.Tensor) -> torch.Tensor:
        """标准模糊否定: ¬x = 1 - x"""
        return 1.0 - x

    @staticmethod
    def fuzzy_and_pair(
        x: torch.Tensor, 
        y: torch.Tensor, 
        eps: float = 1e-7,
    ) -> torch.Tensor:
        """
        二元乘积合取 (稳定版)
        参考 LTNtorch AndProd 的 pi_0 护栏
        """
        x_safe = x.clamp(min=eps)
        y_safe = y.clamp(min=eps)
        return x_safe * y_safe

    @staticmethod
    def fuzzy_or_pair(
        x: torch.Tensor, 
        y: torch.Tensor, 
        eps: float = 1e-7,
    ) -> torch.Tensor:
        """
        二元概率和析取 (稳定版)
        参考 LTNtorch OrProbSum
        """
        x_safe = x.clamp(max=1.0 - eps)
        y_safe = y.clamp(max=1.0 - eps)
        return x_safe + y_safe - x_safe * y_safe
