"""
Unit tests for Phase 8: Task Extensions (Additive Auxiliary Heads and Multi-Task Loss).
Tests:
1. Auxiliary heads default disabled (backward compatibility).
2. Forward pass with auxiliary heads enabled produces direction, severity, and PK regression outputs.
3. Multi-task loss computation with fully labeled data and valid gradients.
4. Multi-task loss masking on missing/unannotated data (direction=-1, severity=-1, pk=NaN).
5. Mixed batch handling with selective loss masking.
"""

import math
import numpy as np
import pytest
import torch

from src.models.interaction_model import ReliabilityConditionedInteractionModel


def test_auxiliary_heads_disabled_by_default():
    """Verify auxiliary heads are None and omitted from output by default."""
    model = ReliabilityConditionedInteractionModel(
        graph_dim=16,
        evidence_dim=16,
        fused_dim=16,
        hidden_dim=16,
        enable_auxiliary_heads=False,
    )
    assert model.direction_head is None
    assert model.severity_head is None
    assert model.pk_regression_head is None

    h = torch.randn(4, 16)
    d = torch.randn(4, 16)
    ev = torch.randn(4, 16)

    out = model(h, d, ev)
    assert "logits" in out
    assert "probs" in out
    assert "direction_logits" not in out
    assert "severity_logits" not in out
    assert "pk_fold_change" not in out


def test_auxiliary_heads_enabled_forward():
    """Verify auxiliary heads produce correct output shapes when enabled."""
    model = ReliabilityConditionedInteractionModel(
        graph_dim=16,
        evidence_dim=16,
        fused_dim=16,
        hidden_dim=16,
        enable_auxiliary_heads=True,
        num_severity_classes=3,
    )
    assert model.direction_head is not None
    assert model.severity_head is not None
    assert model.pk_regression_head is not None

    batch_size = 4
    h = torch.randn(batch_size, 16)
    d = torch.randn(batch_size, 16)
    ev = torch.randn(batch_size, 16)

    out = model(h, d, ev)
    assert out["direction_logits"].shape == (batch_size, 1)
    assert out["severity_logits"].shape == (batch_size, 3)
    assert out["pk_fold_change"].shape == (batch_size, 1)


def test_multitask_loss_fully_labeled():
    """Verify multi-task loss computation and gradient propagation on fully labeled targets."""
    model = ReliabilityConditionedInteractionModel(
        graph_dim=16,
        evidence_dim=16,
        fused_dim=16,
        hidden_dim=16,
        enable_auxiliary_heads=True,
        num_severity_classes=3,
    )

    batch_size = 6
    h = torch.randn(batch_size, 16)
    d = torch.randn(batch_size, 16)
    ev = torch.randn(batch_size, 16)
    out = model(h, d, ev)

    target_int = torch.tensor([1, 0, 1, 0, 1, 0])
    target_dir = torch.tensor([0, 1, 1, 0, 0, 1])  # 0: decreased, 1: increased
    target_sev = torch.tensor([0, 1, 2, 0, 1, 2])  # 0: mild, 1: moderate, 2: severe
    target_pk = torch.tensor([-0.693, 0.405, 1.200, -0.223, -1.050, 0.850])  # log fold changes

    losses = model.compute_multitask_loss(
        predictions=out,
        target_interaction=target_int,
        target_direction=target_dir,
        target_severity=target_sev,
        target_pk_fold_change=target_pk,
    )

    assert "loss_total" in losses
    assert losses["loss_total"].item() > 0.0
    assert losses["loss_interaction"].item() > 0.0
    assert losses["loss_direction"].item() > 0.0
    assert losses["loss_severity"].item() > 0.0
    assert losses["loss_pk"].item() > 0.0

    # Ensure gradients propagate to all heads
    losses["loss_total"].backward()
    for head in [model.link_predictor, model.direction_head, model.severity_head, model.pk_regression_head]:
        for param in head.parameters():
            if param.requires_grad:
                assert param.grad is not None
                assert not torch.isnan(param.grad).any()


def test_multitask_loss_masking_all_unlabeled():
    """Verify auxiliary losses gracefully mask to 0 when targets are unannotated or NaN."""
    model = ReliabilityConditionedInteractionModel(
        graph_dim=16,
        evidence_dim=16,
        fused_dim=16,
        hidden_dim=16,
        enable_auxiliary_heads=True,
    )

    batch_size = 4
    h = torch.randn(batch_size, 16)
    d = torch.randn(batch_size, 16)
    ev = torch.randn(batch_size, 16)
    out = model(h, d, ev)

    target_int = torch.tensor([1, 0, 1, 0])
    target_dir = torch.tensor([-1, -1, -1, -1])  # All unlabeled
    target_sev = torch.tensor([-1, -1, -1, -1])  # All unlabeled
    target_pk = torch.tensor([float("nan"), float("nan"), float("nan"), float("nan")])  # All missing

    losses = model.compute_multitask_loss(
        predictions=out,
        target_interaction=target_int,
        target_direction=target_dir,
        target_severity=target_sev,
        target_pk_fold_change=target_pk,
    )

    # Auxiliary losses must be exactly 0.0
    assert losses["loss_direction"].item() == 0.0
    assert losses["loss_severity"].item() == 0.0
    assert losses["loss_pk"].item() == 0.0
    # Total loss should match interaction loss exactly (weight = 1.0)
    assert math.isclose(losses["loss_total"].item(), losses["loss_interaction"].item(), abs_tol=1e-5)


def test_multitask_loss_mixed_masking():
    """Verify multi-task loss is computed correctly on partially labeled batches."""
    model = ReliabilityConditionedInteractionModel(
        graph_dim=16,
        evidence_dim=16,
        fused_dim=16,
        hidden_dim=16,
        enable_auxiliary_heads=True,
    )

    batch_size = 4
    h = torch.randn(batch_size, 16)
    d = torch.randn(batch_size, 16)
    ev = torch.randn(batch_size, 16)
    out = model(h, d, ev)

    target_int = torch.tensor([1, 0, 1, 0])
    target_dir = torch.tensor([0, -1, 1, -1])  # 2 labeled, 2 unlabeled
    target_sev = torch.tensor([-1, 1, -1, 2])  # 2 unlabeled, 2 labeled
    target_pk = torch.tensor([float("nan"), 0.5, float("nan"), -0.8])  # 2 missing, 2 present

    losses = model.compute_multitask_loss(
        predictions=out,
        target_interaction=target_int,
        target_direction=target_dir,
        target_severity=target_sev,
        target_pk_fold_change=target_pk,
    )

    assert losses["loss_direction"].item() > 0.0
    assert losses["loss_severity"].item() > 0.0
    assert losses["loss_pk"].item() > 0.0
    assert not torch.isnan(losses["loss_total"])
