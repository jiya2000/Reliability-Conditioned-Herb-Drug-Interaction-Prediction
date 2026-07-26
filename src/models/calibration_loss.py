"""
Adversarial Reliability Calibration (ARC) Loss

★ NOVEL CONTRIBUTION 1 ★

Ensures the reliability score R is calibrated — when R predicts
high reliability, the cross-attention should actually produce better
link predictions, and vice versa.

This addresses the "stretch goal" from the implementation plan (Week 11):
providing the reliability scorer with an explicit training signal beyond
just end-to-end backprop, giving R an interpretable inductive bias.

Three components:
1. Calibration Loss: Pearson correlation between R and per-sample accuracy
2. ECE (Expected Calibration Error) monitoring
3. Reliability regularization to prevent R collapse to constant values

Loss formulation:
    L_total = L_link + λ₁·L_calibration + λ₂·L_diversity + λ₃·L_ordering

    L_calibration = -Correlation(R, correctness)
    L_diversity   = -Var(R)  (prevents R from collapsing to a constant)
    L_ordering    = Σ max(0, R_low_quality - R_high_quality + margin)
                    (ordinal constraint: clinical trials should score > forums)
"""

from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from loguru import logger


class CalibrationLoss(nn.Module):
    """
    Adversarial Reliability Calibration loss.

    Combines three objectives to ensure R is meaningful:
    1. Correlation loss: R should correlate with prediction accuracy
    2. Diversity loss: R should not collapse to a constant
    3. Ordering loss: R should respect source quality hierarchy

    Usage:
        cal_loss = CalibrationLoss(lambda_cal=0.1, lambda_div=0.01)
        loss = cal_loss(reliability_scores, predictions, labels, metadata)
    """

    def __init__(
        self,
        lambda_calibration: float = 0.1,
        lambda_diversity: float = 0.01,
        lambda_ordering: float = 0.05,
        ordering_margin: float = 0.1,
    ):
        super().__init__()
        self.lambda_calibration = lambda_calibration
        self.lambda_diversity = lambda_diversity
        self.lambda_ordering = lambda_ordering
        self.ordering_margin = ordering_margin

        # Source type quality ordering (higher index = lower expected R)
        # Clinical trial > peer_reviewed > textbook > case_report >
        # traditional_medicine > health_forum > social_media
        self.quality_ordering = {
            1: 0.95,  # clinical_trial
            2: 0.85,  # peer_reviewed
            3: 0.75,  # textbook
            4: 0.65,  # case_report
            7: 0.40,  # traditional_medicine
            5: 0.25,  # health_forum
            6: 0.15,  # social_media
            0: 0.30,  # unknown
        }

        logger.info(
            f"CalibrationLoss: λ_cal={lambda_calibration}, "
            f"λ_div={lambda_diversity}, λ_ord={lambda_ordering}"
        )

    def pearson_correlation(
        self, x: torch.Tensor, y: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute Pearson correlation between two tensors.

        Returns a value in [-1, 1]. Returns 0 if either has zero variance.
        """
        x_mean = x.mean()
        y_mean = y.mean()
        x_centered = x - x_mean
        y_centered = y - y_mean

        cov = (x_centered * y_centered).mean()
        x_std = x_centered.pow(2).mean().sqrt()
        y_std = y_centered.pow(2).mean().sqrt()

        denom = x_std * y_std
        if denom < 1e-8:
            return torch.tensor(0.0, device=x.device)

        return cov / denom

    def calibration_loss(
        self,
        reliability_scores: torch.Tensor,
        predictions: torch.Tensor,
        labels: torch.Tensor,
    ) -> torch.Tensor:
        """
        Calibration loss: R should correlate with per-sample correctness.

        When R is high, predictions should be more accurate.
        When R is low, we expect more errors.

        Args:
            reliability_scores: (batch,) R values
            predictions: (batch,) predicted probabilities
            labels: (batch,) ground truth labels

        Returns:
            Negative correlation (minimize this to maximize correlation)
        """
        # Per-sample accuracy: how close is the prediction to the label
        # Use soft accuracy (1 - |pred - label|) for differentiability
        correctness = 1.0 - (predictions - labels).abs()

        # We want R to correlate positively with correctness
        correlation = self.pearson_correlation(
            reliability_scores.squeeze(), correctness
        )

        # Minimize negative correlation
        return -correlation

    def diversity_loss(
        self, reliability_scores: torch.Tensor
    ) -> torch.Tensor:
        """
        Diversity loss: prevent R from collapsing to a constant.

        Without this, R can learn to always output the same value
        (e.g., always 0.5), which makes the gating meaningless.

        Returns:
            Negative variance of R (minimize to maximize diversity)
        """
        return -reliability_scores.var()

    def ordering_loss(
        self,
        reliability_scores: torch.Tensor,
        metadata: torch.Tensor,
    ) -> torch.Tensor:
        """
        Ordering loss: R should respect source quality hierarchy.

        Clinical trial evidence should get higher R than health forum evidence.
        This provides weak supervision without explicit R labels.

        Args:
            reliability_scores: (batch, 1) R values
            metadata: (batch, 5) with column 4 being source type code
        """
        source_types = metadata[:, 4].long()
        R = reliability_scores.squeeze()

        total_loss = torch.tensor(0.0, device=R.device)
        num_pairs = 0

        # Create pairwise ordering constraints
        for i in range(len(R)):
            for j in range(i + 1, min(i + 5, len(R))):  # Limit pairs for efficiency
                st_i = source_types[i].item()
                st_j = source_types[j].item()

                expected_i = self.quality_ordering.get(st_i, 0.3)
                expected_j = self.quality_ordering.get(st_j, 0.3)

                if expected_i > expected_j + 0.1:
                    # i should have higher R than j
                    violation = F.relu(
                        R[j] - R[i] + self.ordering_margin
                    )
                    total_loss = total_loss + violation
                    num_pairs += 1
                elif expected_j > expected_i + 0.1:
                    # j should have higher R than i
                    violation = F.relu(
                        R[i] - R[j] + self.ordering_margin
                    )
                    total_loss = total_loss + violation
                    num_pairs += 1

        if num_pairs > 0:
            total_loss = total_loss / num_pairs

        return total_loss

    def forward(
        self,
        reliability_scores: torch.Tensor,
        predictions: torch.Tensor,
        labels: torch.Tensor,
        metadata: Optional[torch.Tensor] = None,
    ) -> tuple[torch.Tensor, dict[str, float]]:
        """
        Compute the full ARC loss.

        Args:
            reliability_scores: (batch, 1) R scores from reliability scorer
            predictions: (batch,) predicted probabilities
            labels: (batch,) ground truth binary labels
            metadata: (batch, 5) optional metadata for ordering loss

        Returns:
            total_loss: scalar loss tensor
            breakdown: dict with individual loss components
        """
        R = reliability_scores.squeeze()

        # 1. Calibration
        L_cal = self.calibration_loss(R, predictions, labels)

        # 2. Diversity
        L_div = self.diversity_loss(R)

        # 3. Ordering (if metadata available)
        L_ord = torch.tensor(0.0, device=R.device)
        if metadata is not None:
            L_ord = self.ordering_loss(reliability_scores, metadata)

        total = (
            self.lambda_calibration * L_cal
            + self.lambda_diversity * L_div
            + self.lambda_ordering * L_ord
        )

        breakdown = {
            "calibration_loss": L_cal.item(),
            "diversity_loss": L_div.item(),
            "ordering_loss": L_ord.item(),
            "arc_total_loss": total.item(),
        }

        return total, breakdown


class ExpectedCalibrationError(nn.Module):
    """
    Computes Expected Calibration Error (ECE) for monitoring.

    ECE measures how well R's values correspond to actual accuracy.
    Not used for training (non-differentiable), but for evaluation.
    """

    def __init__(self, n_bins: int = 10):
        super().__init__()
        self.n_bins = n_bins

    @torch.no_grad()
    def forward(
        self,
        reliability_scores: torch.Tensor,
        predictions: torch.Tensor,
        labels: torch.Tensor,
    ) -> tuple[float, list[dict]]:
        """
        Compute ECE.

        Returns:
            ece: scalar ECE value
            bin_stats: per-bin statistics for plotting
        """
        R = reliability_scores.squeeze().cpu().numpy()
        correctness = (
            1.0 - (predictions - labels).abs()
        ).cpu().numpy()

        bin_boundaries = [i / self.n_bins for i in range(self.n_bins + 1)]
        bin_stats = []
        ece = 0.0

        for b in range(self.n_bins):
            lo, hi = bin_boundaries[b], bin_boundaries[b + 1]
            mask = (R >= lo) & (R < hi)

            if mask.sum() == 0:
                bin_stats.append({
                    "bin_lo": lo, "bin_hi": hi,
                    "count": 0, "avg_R": 0, "avg_accuracy": 0,
                })
                continue

            avg_R = R[mask].mean()
            avg_acc = correctness[mask].mean()
            count = mask.sum()

            gap = abs(avg_R - avg_acc)
            ece += gap * count / len(R)

            bin_stats.append({
                "bin_lo": lo, "bin_hi": hi,
                "count": int(count),
                "avg_R": float(avg_R),
                "avg_accuracy": float(avg_acc),
            })

        return float(ece), bin_stats
