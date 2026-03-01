# -*- coding: utf-8 -*-
"""
NCRN 单元测试
=============
验证四个核心模块的正确性:
1. OrthogonalConceptDict: 前向形状、值域、零空间投影正交性
2. FuzzyLogicOperators: 对数空间算子数值稳定性
3. DifferentiableDNF: STE 前向离散/反向连续、逻辑损失
4. NCRN_Head: 端到端前向反向、增量更新冻结性
"""

import sys
import os
import argparse

# 添加项目根目录到路径
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import torch
import torch.nn.functional as F
import numpy as np


def get_device():
    """设备选择: cuda > mps > cpu"""
    if torch.cuda.is_available():
        return torch.device("cuda")
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


# =====================================================================
# Test 1: OrthogonalConceptDict
# =====================================================================
def test_concept_dict_forward():
    """验证前向输出形状和值域"""
    from groundingdino.models.GroundingDINO.ncrn.concept_dict import OrthogonalConceptDict
    
    device = get_device()
    D, M, K = 256, 512, 64
    B = 8

    model = OrthogonalConceptDict(feat_dim=D, total_concepts=M, init_active=K).to(device)
    Z = torch.randn(B, D, device=device)
    C = model(Z)

    # 形状检查
    assert C.shape == (B, K), f"Expected shape ({B}, {K}), got {C.shape}"
    
    # 值域检查 [0, 1]
    assert C.min() >= 0.0, f"Min value {C.min()} < 0"
    assert C.max() <= 1.0, f"Max value {C.max()} > 1"

    # 梯度存在性
    C.sum().backward()
    assert model.P_total.grad is not None, "P_total should have gradients"

    print("✅ test_concept_dict_forward PASSED")


def test_concept_dict_expand_nullspace():
    """验证零空间投影的正交性"""
    from groundingdino.models.GroundingDINO.ncrn.concept_dict import OrthogonalConceptDict
    
    device = get_device()
    D, M, K = 128, 256, 32
    
    model = OrthogonalConceptDict(feat_dim=D, total_concepts=M, init_active=K).to(device)
    
    # 获取旧字典
    P_old = model.P_active.detach().clone()  # [32, 128]
    
    # 模拟新样本特征
    Z_new = torch.randn(20, D, device=device)
    num_added = model.expand_via_nullspace(Z_new, num_new_concepts=16)
    
    assert num_added > 0, "Should have added new concepts"
    assert model.K_active == K + num_added, f"K_active should be {K + num_added}, got {model.K_active}"
    
    # 正交性验证: 新概念与旧概念的内积应接近 0
    P_new = model.P_total[K:K + num_added].detach()  # [num_added, D]
    cross_dot = torch.mm(P_new, P_old.t())  # [num_added, K]
    max_dot = cross_dot.abs().max().item()
    
    assert max_dot < 0.05, f"Max cross dot product {max_dot} too large (should be ~0)"
    
    # 冻结掩码检查
    assert model.frozen_mask[:K].all(), "Old concepts should be frozen"
    assert not model.frozen_mask[K + num_added:].any(), "Unused concepts should not be frozen"
    
    print(f"✅ test_concept_dict_expand_nullspace PASSED (max_dot={max_dot:.6f}, added={num_added})")


# =====================================================================
# Test 2: FuzzyLogicOperators
# =====================================================================
def test_fuzzy_ops_numerical_stability():
    """验证对数空间算子在极端值下不产生 NaN"""
    from groundingdino.models.GroundingDINO.ncrn.fuzzy_ops import FuzzyLogicOperators
    
    ops = FuzzyLogicOperators
    
    # 极端值测试
    extreme_values = torch.tensor([0.0, 1e-8, 0.5, 1.0 - 1e-8, 1.0])
    weights = torch.ones_like(extreme_values)
    
    # T-norm (AND)
    result_and = ops.log_product_t_norm(extreme_values.unsqueeze(0), weights.unsqueeze(0))
    assert not torch.isnan(result_and).any(), f"T-norm produced NaN: {result_and}"
    assert not torch.isinf(result_and).any(), f"T-norm produced Inf: {result_and}"
    
    # T-conorm (OR)
    result_or = ops.log_product_t_conorm(extreme_values.unsqueeze(0))
    assert not torch.isnan(result_or).any(), f"T-conorm produced NaN: {result_or}"
    assert not torch.isinf(result_or).any(), f"T-conorm produced Inf: {result_or}"
    
    # 全 0 输入
    zeros = torch.zeros(1, 10)
    result_and_0 = ops.log_product_t_norm(zeros, torch.ones_like(zeros))
    assert not torch.isnan(result_and_0).any(), "T-norm on zeros produced NaN"
    
    # 全 1 输入
    ones = torch.ones(1, 10)
    result_and_1 = ops.log_product_t_norm(ones, torch.ones_like(ones))
    assert not torch.isnan(result_and_1).any(), "T-norm on ones produced NaN"
    assert result_and_1.item() > 0.99, f"AND of all 1s should be ~1, got {result_and_1.item()}"
    
    # 梯度测试
    x = torch.tensor([0.3, 0.7, 0.5], requires_grad=True)
    w = torch.tensor([1.0, 1.0, 0.0])
    result = ops.log_product_t_norm(x.unsqueeze(0), w.unsqueeze(0))
    result.backward()
    assert not torch.isnan(x.grad).any(), "T-norm gradient contains NaN"
    
    print("✅ test_fuzzy_ops_numerical_stability PASSED")


def test_fuzzy_ops_semantics():
    """验证模糊算子的语义正确性"""
    from groundingdino.models.GroundingDINO.ncrn.fuzzy_ops import FuzzyLogicOperators
    
    ops = FuzzyLogicOperators
    
    # AND(a, b) ≈ a * b (当权重全为 1)
    a, b = 0.6, 0.8
    L = torch.tensor([[a, b]])
    W = torch.tensor([[1.0, 1.0]])
    result = ops.log_product_t_norm(L, W).item()
    expected = a * b
    assert abs(result - expected) < 0.01, f"AND({a}, {b}) = {result}, expected ~{expected}"
    
    # OR(a, b) = a + b - ab
    rules = torch.tensor([[a, b]])
    result_or = ops.log_product_t_conorm(rules).item()
    expected_or = a + b - a * b
    assert abs(result_or - expected_or) < 0.01, f"OR({a}, {b}) = {result_or}, expected ~{expected_or}"
    
    # 权重为 0 时应忽略该文字
    L2 = torch.tensor([[0.1, 0.9]])
    W2 = torch.tensor([[0.0, 1.0]])
    result2 = ops.log_product_t_norm(L2, W2).item()
    assert abs(result2 - 0.9) < 0.01, f"AND with W=[0,1] should give ~0.9, got {result2}"
    
    print("✅ test_fuzzy_ops_semantics PASSED")


# =====================================================================
# Test 3: DifferentiableDNF
# =====================================================================
def test_dnf_forward():
    """验证 DNF 前向: STE 离散化和输出形状"""
    from groundingdino.models.GroundingDINO.ncrn.dnf_engine import DifferentiableDNF
    
    device = get_device()
    M, C, R, K = 256, 5, 4, 32
    B = 8
    
    model = DifferentiableDNF(max_concepts=M, num_classes=C, num_rules=R).to(device)
    C_input = torch.rand(B, K, device=device)  # 模拟概念激活
    
    Y_hat = model(C_input, K_active=K)
    
    # 形状检查
    assert Y_hat.shape == (B, C), f"Expected shape ({B}, {C}), got {Y_hat.shape}"
    
    # 值域检查 [0, 1]
    assert Y_hat.min() >= 0.0, f"Min value {Y_hat.min()} < 0"
    assert Y_hat.max() <= 1.0, f"Max value {Y_hat.max()} > 1"
    
    # 反向传播梯度存在性 (STE 应保证梯度流通)
    loss = Y_hat.sum()
    loss.backward()
    assert model.Omega.grad is not None, "Omega should have gradients (STE)"
    assert not torch.isnan(model.Omega.grad).any(), "Omega gradient contains NaN"
    
    print("✅ test_dnf_forward PASSED")


def test_dnf_logic_loss():
    """验证逻辑正则化损失计算"""
    from groundingdino.models.GroundingDINO.ncrn.dnf_engine import DifferentiableDNF
    
    device = get_device()
    model = DifferentiableDNF(max_concepts=128, num_classes=3, num_rules=4).to(device)
    
    losses = model.compute_logic_loss(K_active=32)
    
    assert "L_L1" in losses, "Missing L_L1"
    assert "L_conflict" in losses, "Missing L_conflict"
    assert "L_total" in losses, "Missing L_total"
    
    # 所有损失应为非负标量
    for key, val in losses.items():
        assert val.dim() == 0, f"{key} should be scalar, got dim={val.dim()}"
        assert val.item() >= 0, f"{key} should be non-negative, got {val.item()}"
        assert not torch.isnan(val), f"{key} is NaN"
    
    print("✅ test_dnf_logic_loss PASSED")


# =====================================================================
# Test 4: NCRN_Head
# =====================================================================
def test_ncrn_head_end_to_end():
    """端到端前向反向跑通"""
    from groundingdino.models.GroundingDINO.ncrn.ncrn_head import NCRN_Head
    
    device = get_device()
    D, C, M, K = 128, 5, 256, 32
    B = 4
    
    model = NCRN_Head(
        feat_dim=D, num_classes=C, total_concepts=M, 
        init_active=K, num_rules=4,
    ).to(device)
    
    Z = torch.randn(B, D, device=device)
    Y = torch.zeros(B, C, device=device)
    Y[range(B), torch.randint(0, C, (B,))] = 1.0  # one-hot
    
    # 前向
    result = model.compute_loss(Z, Y)
    
    assert "L_BCE" in result
    assert "L_total" in result
    assert "Y_hat" in result
    assert result["Y_hat"].shape == (B, C)
    
    # 反向
    result["L_total"].backward()
    model.apply_gradient_mask()
    
    # 所有参数应有梯度
    for name, param in model.named_parameters():
        if param.requires_grad:
            assert param.grad is not None, f"{name} has no gradient"
    
    print(f"✅ test_ncrn_head_end_to_end PASSED (L_total={result['L_total'].item():.4f})")


def test_incremental_update():
    """验证增量更新后旧参数冻结"""
    from groundingdino.models.GroundingDINO.ncrn.ncrn_head import NCRN_Head
    
    device = get_device()
    D, C, M, K = 128, 3, 256, 32
    
    model = NCRN_Head(
        feat_dim=D, num_classes=C, total_concepts=M,
        init_active=K, num_rules=4,
    ).to(device)
    
    # 记录旧状态
    old_P = model.concept_dict.P_active.detach().clone()
    old_K = model.K_active
    old_C = model.num_classes
    
    # 增量更新
    Z_new = torch.randn(10, D, device=device)
    Y_new = torch.zeros(10, 2, device=device)  # 2 个新类别
    Y_new[range(10), torch.randint(0, 2, (10,))] = 1.0
    
    info = model.incremental_update(Z_new, Y_new, num_new_concepts=8)
    
    # 验证扩展
    assert model.K_active > old_K, "K_active should have increased"
    assert model.num_classes == old_C + 2, f"num_classes should be {old_C + 2}"
    
    # 验证旧参数冻结
    assert model.concept_dict.frozen_mask[:old_K].all(), "Old concepts should be frozen"
    assert model.dnf.frozen_class_mask[:old_C].all(), "Old classes should be frozen"
    
    # 验证旧字典值未变
    P_old_after = model.concept_dict.P_total[:old_K].detach()
    diff = (P_old_after - old_P).abs().max().item()
    assert diff < 1e-6, f"Old dictionary changed! max diff={diff}"
    
    # 验证梯度掩码生效
    Z_test = torch.randn(4, D, device=device)
    Y_test = torch.zeros(4, model.num_classes, device=device)
    Y_test[:, -1] = 1.0
    
    model.zero_grad()
    result = model.compute_loss(Z_test, Y_test)
    result["L_total"].backward()
    model.apply_gradient_mask()
    
    # 旧字典行的梯度应为 0
    old_grad = model.concept_dict.P_total.grad[:old_K]
    assert old_grad.abs().max() < 1e-10, f"Old concept grads should be zero, max={old_grad.abs().max()}"
    
    # 旧类别 Omega 的梯度应为 0
    old_omega_grad = model.dnf.Omega.grad[:old_C]
    assert old_omega_grad.abs().max() < 1e-10, f"Old class grads should be zero, max={old_omega_grad.abs().max()}"
    
    print(f"✅ test_incremental_update PASSED (K: {old_K}→{model.K_active}, C: {old_C}→{model.num_classes})")


def test_logic_explanation():
    """验证逻辑解释接口"""
    from groundingdino.models.GroundingDINO.ncrn.ncrn_head import NCRN_Head
    
    device = get_device()
    model = NCRN_Head(feat_dim=64, num_classes=3, total_concepts=128, init_active=16, num_rules=4).to(device)
    
    explanations = model.get_logic_explanation()
    assert len(explanations) == 3, f"Expected 3 classes, got {len(explanations)}"
    
    for c, rules in explanations.items():
        assert isinstance(rules, list), f"Class {c} rules should be a list"
    
    print("✅ test_logic_explanation PASSED")


# =====================================================================
# Smoke Test with COCO 128
# =====================================================================
def test_smoke_coco128(data_dir: str):
    """用 COCO 128 数据集做冒烟测试"""
    import json
    from groundingdino.models.GroundingDINO.ncrn.ncrn_head import NCRN_Head
    
    device = get_device()
    print(f"\n🔥 Smoke test with COCO 128 dataset on {device}")
    
    # 读取 annotations 获取类别数
    ann_path = os.path.join(data_dir, "train", "_annotations.coco.json")
    if not os.path.exists(ann_path):
        print(f"⚠️  Annotation file not found: {ann_path}, skipping smoke test")
        return
    
    with open(ann_path, "r") as f:
        annotations = json.load(f)
    
    num_classes = len(annotations["categories"])
    print(f"   Found {num_classes} classes, {len(annotations['images'])} images")
    
    # 模拟 ROI 特征 (真实场景中由 backbone+decoder 生成)
    D = 256  # GroundingDINO 的 hidden_dim
    B = 16
    
    model = NCRN_Head(
        feat_dim=D, num_classes=num_classes, total_concepts=512,
        init_active=64, num_rules=8,
    ).to(device)
    
    # 模拟训练步骤
    Z = torch.randn(B, D, device=device)
    Y = torch.zeros(B, num_classes, device=device)
    for i in range(B):
        Y[i, torch.randint(0, num_classes, (1,))] = 1.0
    
    result = model.compute_loss(Z, Y)
    result["L_total"].backward()
    model.apply_gradient_mask()
    
    print(f"   L_total={result['L_total'].item():.4f}, "
          f"L_BCE={result['L_BCE'].item():.4f}, "
          f"L_L1={result['L_L1'].item():.6f}")
    
    # 模拟增量学习
    Z_new = torch.randn(5, D, device=device)
    Y_new = torch.zeros(5, 3, device=device)  # 新增 3 个类
    info = model.incremental_update(Z_new, Y_new, num_new_concepts=8)
    print(f"   Incremental: K {info.get('K_active')}, Classes {info.get('total_classes')}")
    
    # 增量后再次前向
    model.zero_grad()
    Z2 = torch.randn(B, D, device=device)
    Y2 = torch.zeros(B, model.num_classes, device=device)
    Y2[:, -1] = 1.0
    result2 = model.compute_loss(Z2, Y2)
    result2["L_total"].backward()
    model.apply_gradient_mask()
    
    print(f"   After incremental: L_total={result2['L_total'].item():.4f}")
    print("✅ test_smoke_coco128 PASSED")


# =====================================================================
# Main
# =====================================================================
def main():
    parser = argparse.ArgumentParser(description="NCRN Unit Tests")
    parser.add_argument(
        "--smoke-test", action="store_true",
        help="Run smoke test with COCO 128 dataset",
    )
    parser.add_argument(
        "--data-dir", type=str,
        default="/Volumes/SSD512/Code/dataset/COCO 128.v2-640x640.coco",
        help="Path to COCO 128 dataset",
    )
    args = parser.parse_args()
    
    device = get_device()
    print(f"🧪 NCRN Unit Tests — Device: {device}\n")
    
    # 运行所有单元测试
    test_concept_dict_forward()
    test_concept_dict_expand_nullspace()
    test_fuzzy_ops_numerical_stability()
    test_fuzzy_ops_semantics()
    test_dnf_forward()
    test_dnf_logic_loss()
    test_ncrn_head_end_to_end()
    test_incremental_update()
    test_logic_explanation()
    
    if args.smoke_test:
        test_smoke_coco128(args.data_dir)
    
    print(f"\n🎉 All tests passed!")


if __name__ == "__main__":
    main()
