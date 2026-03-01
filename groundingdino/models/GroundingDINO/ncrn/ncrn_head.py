# -*- coding: utf-8 -*-
"""
NCRN 分类头 (Neural-Symbolic Concept Reasoning Network Head)
============================================================
顶层封装模块，串联 OrthogonalConceptDict 和 DifferentiableDNF，
对外暴露统一的 forward(Z) 和 incremental_update(Z_new, Y_new) 接口。

作为 DitHub 框架中分类头的 Drop-in Replacement。

架构流:
    Z [B, D] → OrthogonalConceptDict → C [B, K] → DifferentiableDNF → Y_hat [B, C]
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from .concept_dict import OrthogonalConceptDict
from .dnf_engine import DifferentiableDNF


def _auto_device() -> torch.device:
    """设备选择: cuda > mps > cpu"""
    if torch.cuda.is_available():
        return torch.device("cuda")
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


class NCRN_Head(nn.Module):
    """
    神经符号概念推理网络分类头
    
    参数:
        feat_dim (int): 输入 ROI 特征维度 D (默认 256, 与 GroundingDINO 对齐)
        num_classes (int): 初始类别数 C
        total_concepts (int): 字典总容量 M (默认 2048)
        init_active (int): 初始激活概念数 K_active (默认 64)
        num_rules (int): 每类 DNF 规则数 R (默认 8)
        lambda_l1 (float): 稀疏正则化系数
        lambda_conflict (float): 互斥正则化系数
    """

    def __init__(
        self,
        feat_dim: int = 256,
        num_classes: int = 10,
        total_concepts: int = 2048,
        init_active: int = 64,
        num_rules: int = 8,
        lambda_l1: float = 1e-3,
        lambda_conflict: float = 1e-2,
    ):
        super().__init__()

        self.feat_dim = feat_dim
        self.lambda_l1 = lambda_l1
        self.lambda_conflict = lambda_conflict

        # 子模块
        self.concept_dict = OrthogonalConceptDict(
            feat_dim=feat_dim,
            total_concepts=total_concepts,
            init_active=init_active,
        )
        self.dnf = DifferentiableDNF(
            max_concepts=total_concepts,
            num_classes=num_classes,
            num_rules=num_rules,
        )

        # 记录增量学习步骤
        self._task_count = 0

    @property
    def K_active(self) -> int:
        return self.concept_dict.K_active

    @property
    def num_classes(self) -> int:
        return self.dnf.num_classes

    def forward(self, Z: torch.Tensor) -> torch.Tensor:
        """
        前向推理
        
        Args:
            Z: ROI 连续特征 [B, D]
        
        Returns:
            Y_hat: 类别概率 [B, C_total]，值域 (0, 1)
        """
        # Step 1: 概念投影
        C = self.concept_dict(Z)  # [B, K_active]

        # Step 2: DNF 逻辑推理
        Y_hat = self.dnf(C, self.K_active)  # [B, C_total]

        return Y_hat

    def compute_loss(
        self,
        Z: torch.Tensor,
        Y: torch.Tensor,
    ) -> dict:
        """
        计算完整损失 = 分类损失 + 逻辑正则化损失
        
        Args:
            Z: ROI 特征 [B, D]
            Y: 目标标签 [B, C_total]，单热或多热编码
        
        Returns:
            dict: {
                'L_BCE': 分类损失,
                'L_L1': 稀疏惩罚,
                'L_conflict': 互斥惩罚,
                'L_total': 总损失,
                'Y_hat': 预测概率,
            }
        """
        Y_hat = self.forward(Z)  # [B, C]

        # 分类损失 (BCE)
        L_BCE = F.binary_cross_entropy(
            Y_hat.clamp(1e-7, 1.0 - 1e-7),
            Y.float(),
            reduction="mean",
        )

        # 逻辑正则化损失
        logic_loss = self.dnf.compute_logic_loss(
            K_active=self.K_active,
            lambda_l1=self.lambda_l1,
            lambda_conflict=self.lambda_conflict,
        )

        L_total = L_BCE + logic_loss["L_total"]

        return {
            "L_BCE": L_BCE,
            "L_L1": logic_loss["L_L1"],
            "L_conflict": logic_loss["L_conflict"],
            "L_total": L_total,
            "Y_hat": Y_hat,
        }

    @torch.no_grad()
    def incremental_update(
        self,
        Z_new: torch.Tensor,
        Y_new: torch.Tensor,
        num_new_concepts: int = 16,
    ) -> dict:
        """
        增量学习更新接口
        
        策略: "绝对冻结旧参数 + 释放新字典容量 + 学习新逻辑掩码"
        
        Args:
            Z_new: 新任务样本特征 [N, D]
            Y_new: 新任务标签 [N, C_new]
            num_new_concepts: 新增概念数量
        
        Returns:
            dict: 更新信息
        """
        self._task_count += 1
        info = {"task_id": self._task_count}

        # Step 1: 冻结旧概念字典参数
        old_K = self.K_active
        old_C = self.num_classes

        # Step 2: 字典扩容 (SVD 零空间投影)
        actual_new = self.concept_dict.expand_via_nullspace(
            Z_new, num_new_concepts=num_new_concepts,
        )
        info["new_concepts"] = actual_new
        info["K_active"] = self.K_active

        # Step 3: 扩展 DNF 的类别数
        num_new_classes = Y_new.shape[1] if Y_new.dim() > 1 else 1
        self.dnf.expand_classes(num_new_classes)
        info["new_classes"] = num_new_classes
        info["total_classes"] = self.num_classes

        return info

    def apply_gradient_mask(self):
        """
        在每步反向传播后调用，确保旧参数梯度为零
        
        使用方法:
            loss.backward()
            ncrn_head.apply_gradient_mask()
            optimizer.step()
        """
        self.concept_dict.apply_gradient_mask()
        self.dnf.apply_gradient_mask()

    def get_logic_explanation(self, class_idx: int = None) -> dict:
        """
        获取逻辑解释（人类可读的规则描述）
        
        Args:
            class_idx: 指定类别索引，None 则返回所有类别
        
        Returns:
            dict: 每个类别的活跃规则描述
        """
        explanations = {}
        K = self.K_active

        with torch.no_grad():
            sig = torch.sigmoid(self.dnf.Omega[:, :, :2 * K])
            W = (sig > 0.5).float()

        classes = range(self.num_classes) if class_idx is None else [class_idx]

        for c in classes:
            rules = []
            for r in range(self.dnf.num_rules):
                w = W[c, r]  # [2K]
                pos_active = (w[:K] > 0.5).nonzero(as_tuple=True)[0].tolist()
                neg_active = (w[K:] > 0.5).nonzero(as_tuple=True)[0].tolist()

                if pos_active or neg_active:
                    rule_desc = {
                        "positive_concepts": pos_active,
                        "negative_concepts": neg_active,
                    }
                    rules.append(rule_desc)

            explanations[c] = rules

        return explanations

    def extra_repr(self) -> str:
        return (
            f"feat_dim={self.feat_dim}, "
            f"num_classes={self.num_classes}, "
            f"K_active={self.K_active}, "
            f"task_count={self._task_count}"
        )
