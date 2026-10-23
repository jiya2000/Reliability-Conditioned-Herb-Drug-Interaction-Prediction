"""
Calibration and Abstention Metrics Module.

Implements:
1. Temperature Scaling for post-hoc logit calibration.
2. Expected Calibration Error (ECE) and Maximum Calibration Error (MCE).
3. Brier Score & Brier Skill Score.
4. Selective Classification & Abstention Metrics: Coverage-versus-Risk curves,
   selective accuracy, and risk reduction under abstention.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from loguru import logger


@dataclass
class CalibrationMetrics:
    """Evaluation metrics for prediction confidence and probability calibration."""
    expected_calibration_error: float  # ECE (binned)
    max_calibration_error: float       # MCE
    brier_score: float                 # MSE between probs and binary labels
    log_loss: float                    # Binary cross-entropy
    temperature: float = 1.0


@dataclass
class AbstentionEvaluationResult:
    """Results from evaluating coverage vs selective risk under abstention."""
    coverage: float                    # Fraction of predictions not abstained
    abstention_rate: float             # Fraction of predictions abstained
    selective_accuracy: float          # Accuracy on retained (non-abstained) predictions
    selective_precision: float
    selective_recall: float
    selective_f1: float
    retained_count: int
    abstained_count: int
    risk_reduction: float              # Error reduction compared to full non-abstaining model


class TemperatureScaler(nn.Module):
    """
    Post-hoc temperature scaling to calibrate binary classification logits.
    Fits a single positive scalar T on validation logits without changing ranking (maintains ROC-AUC).
    """

    def __init__(self):
        super().__init__()
        self.temperature = nn.Parameter(torch.ones(1) * 1.5)

    def forward(self, logits: torch.Tensor) -> torch.Tensor:
        """Scale logits by temperature T."""
        # Use softplus to ensure T > 0 with smooth gradients everywhere
        T = torch.nn.functional.softplus(self.temperature) + 1e-3
        return logits / T

    def fit(
        self,
        val_logits: Union[torch.Tensor, np.ndarray],
        val_labels: Union[torch.Tensor, np.ndarray],
        lr: float = 0.01,
        max_iter: int = 100,
    ) -> float:
        """
        Fit temperature parameter on validation set logits using NLL loss.
        """
        if isinstance(val_logits, np.ndarray):
            logits_t = torch.tensor(val_logits, dtype=torch.float32)
        else:
            logits_t = val_logits.detach().float()

        if isinstance(val_labels, np.ndarray):
            labels_t = torch.tensor(val_labels, dtype=torch.float32)
        else:
            labels_t = val_labels.detach().float()

        if labels_t.dim() == 1:
            labels_t = labels_t.unsqueeze(1)
        if logits_t.dim() == 1:
            logits_t = logits_t.unsqueeze(1)

        optimizer = optim.LBFGS([self.temperature], lr=lr, max_iter=max_iter)
        criterion = nn.BCEWithLogitsLoss()

        def eval_loss():
            optimizer.zero_grad()
            scaled = self.forward(logits_t)
            loss = criterion(scaled, labels_t)
            loss.backward()
            return loss

        optimizer.step(eval_loss)
        t_val = float(self.temperature.item())
        logger.info(f"Fitted TemperatureScaler: optimal temperature T = {t_val:.4f}")
        return t_val


def compute_calibration_metrics(
    probs: Union[np.ndarray, torch.Tensor],
    labels: Union[np.ndarray, torch.Tensor],
    num_bins: int = 10,
    temperature: float = 1.0,
) -> CalibrationMetrics:
    """
    Compute ECE, MCE, Brier score, and log loss for binary predictions.
    """
    if isinstance(probs, torch.Tensor):
        probs = probs.detach().cpu().numpy().flatten()
    else:
        probs = np.asarray(probs, dtype=np.float64).flatten()

    if isinstance(labels, torch.Tensor):
        labels = labels.detach().cpu().numpy().flatten()
    else:
        labels = np.asarray(labels, dtype=np.float64).flatten()

    probs = np.clip(probs, 1e-7, 1.0 - 1e-7)

    # 1. Brier Score
    brier_score = float(np.mean((probs - labels) ** 2))

    # 2. Log Loss (Binary Cross-Entropy)
    log_loss = float(-np.mean(labels * np.log(probs) + (1.0 - labels) * np.log(1.0 - probs)))

    # 3. Expected Calibration Error (ECE) and MCE
    bin_boundaries = np.linspace(0.0, 1.0, num_bins + 1)
    ece = 0.0
    mce = 0.0
    total_samples = len(probs)

    for i in range(num_bins):
        bin_lower = bin_boundaries[i]
        bin_upper = bin_boundaries[i + 1]
        in_bin = (probs >= bin_lower) & (probs < bin_upper) if i < num_bins - 1 else (probs >= bin_lower) & (probs <= bin_upper)
        bin_count = np.sum(in_bin)

        if bin_count > 0:
            bin_acc = float(np.mean(labels[in_bin]))
            bin_conf = float(np.mean(probs[in_bin]))
            bin_error = abs(bin_acc - bin_conf)
            ece += (bin_count / total_samples) * bin_error
            mce = max(mce, bin_error)

    return CalibrationMetrics(
        expected_calibration_error=float(ece),
        max_calibration_error=float(mce),
        brier_score=brier_score,
        log_loss=log_loss,
        temperature=temperature,
    )


def evaluate_abstention(
    predictions: Union[np.ndarray, List[int]],
    labels: Union[np.ndarray, List[int]],
    abstain_flags: Union[np.ndarray, List[bool]],
) -> AbstentionEvaluationResult:
    """
    Evaluate prediction performance on the selective subset of non-abstained predictions.
    """
    preds = np.asarray(predictions)
    y_true = np.asarray(labels)
    abstained = np.asarray(abstain_flags, dtype=bool)

    total_samples = len(y_true)
    retained_mask = ~abstained
    retained_count = int(np.sum(retained_mask))
    abstained_count = int(np.sum(abstained))

    coverage = float(retained_count / total_samples) if total_samples > 0 else 0.0
    abstention_rate = 1.0 - coverage

    # Base error without abstention (binary 0/1 prediction)
    base_preds = np.where(preds >= 0, preds, 0)
    base_error = float(np.mean(base_preds != y_true)) if total_samples > 0 else 0.0

    if retained_count == 0:
        return AbstentionEvaluationResult(
            coverage=0.0,
            abstention_rate=1.0,
            selective_accuracy=0.0,
            selective_precision=0.0,
            selective_recall=0.0,
            selective_f1=0.0,
            retained_count=0,
            abstained_count=abstained_count,
            risk_reduction=0.0,
        )

    ret_preds = preds[retained_mask]
    ret_labels = y_true[retained_mask]

    acc = float(np.mean(ret_preds == ret_labels))
    selective_risk = 1.0 - acc

    # Precision, recall, F1
    tp = int(np.sum((ret_preds == 1) & (ret_labels == 1)))
    fp = int(np.sum((ret_preds == 1) & (ret_labels == 0)))
    fn = int(np.sum((ret_preds == 0) & (ret_labels == 1)))

    prec = float(tp / (tp + fp)) if (tp + fp) > 0 else 0.0
    rec = float(tp / (tp + fn)) if (tp + fn) > 0 else 0.0
    f1 = float(2 * prec * rec / (prec + rec)) if (prec + rec) > 0 else 0.0
    risk_reduction = float(base_error - selective_risk)

    return AbstentionEvaluationResult(
        coverage=coverage,
        abstention_rate=abstention_rate,
        selective_accuracy=acc,
        selective_precision=prec,
        selective_recall=rec,
        selective_f1=f1,
        retained_count=retained_count,
        abstained_count=abstained_count,
        risk_reduction=risk_reduction,
    )
