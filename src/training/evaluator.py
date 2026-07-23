"""
HDI Model Evaluator

Computes comprehensive evaluation metrics:
- AUC-ROC, AUC-PR
- Precision, Recall, F1
- MRR (Mean Reciprocal Rank)
- Hits@K
"""

from __future__ import annotations

from typing import Optional

import numpy as np
import torch
from loguru import logger

from src.models.hdi_model import HDIModel


class HDIEvaluator:
    """
    Evaluator for HDI link prediction.

    Computes all metrics specified in the implementation plan:
    AUC-ROC, AUC-PR, Precision, Recall, F1, MRR, Hits@10.

    Usage:
        evaluator = HDIEvaluator(model)
        results = evaluator.evaluate(test_loader, graph_data)
    """

    def __init__(
        self,
        model: HDIModel,
        threshold: float = 0.5,
        device: str = "auto",
    ):
        self.model = model
        self.threshold = threshold

        if device == "auto":
            self.device = torch.device(
                "cuda" if torch.cuda.is_available() else "cpu"
            )
        else:
            self.device = torch.device(device)

        self.model = self.model.to(self.device)

    @torch.no_grad()
    def evaluate(
        self,
        loader,
        graph_data: dict,
        return_predictions: bool = False,
    ) -> dict:
        """
        Run full evaluation on a data loader.

        Args:
            loader: DataLoader with test/val data
            graph_data: Dict with graph tensors
            return_predictions: If True, include raw predictions in output

        Returns:
            Dict with all computed metrics
        """
        self.model.eval()

        node_features = graph_data["node_features"].to(self.device)
        edge_index = graph_data["edge_index"].to(self.device)
        edge_type = graph_data["edge_type"].to(self.device)

        all_labels = []
        all_probs = []
        all_logits = []

        for batch in loader:
            source_indices = batch["source_indices"].to(self.device)
            target_indices = batch["target_indices"].to(self.device)
            labels = batch["labels"]
            metadata = batch["metadata"].to(self.device)
            evidence_texts = batch.get("evidence_texts", None)

            output = self.model(
                node_features=node_features,
                edge_index=edge_index,
                edge_type=edge_type,
                source_indices=source_indices,
                target_indices=target_indices,
                metadata=metadata,
                evidence_texts=evidence_texts,
            )

            all_labels.append(labels.numpy())
            all_probs.append(output["probabilities"].cpu().numpy())
            all_logits.append(output["logits"].cpu().numpy())

        labels = np.concatenate(all_labels)
        probs = np.concatenate(all_probs)
        logits = np.concatenate(all_logits)
        preds = (probs >= self.threshold).astype(int)

        # Compute metrics
        results = {
            "auc_roc": self._auc_roc(labels, probs),
            "auc_pr": self._auc_pr(labels, probs),
            "precision": self._precision(labels, preds),
            "recall": self._recall(labels, preds),
            "f1": self._f1(labels, preds),
            "mrr": self._mrr(labels, probs),
            "hits@10": self._hits_at_k(labels, probs, k=10),
            "hits@5": self._hits_at_k(labels, probs, k=5),
            "hits@1": self._hits_at_k(labels, probs, k=1),
            "num_samples": len(labels),
            "num_positives": int(labels.sum()),
            "num_negatives": int((1 - labels).sum()),
            "threshold": self.threshold,
        }

        if return_predictions:
            results["predictions"] = {
                "labels": labels,
                "probabilities": probs,
                "predicted_labels": preds,
            }

        # Log results
        logger.info("Evaluation Results:")
        for key, value in results.items():
            if isinstance(value, float):
                logger.info(f"  {key}: {value:.4f}")
            elif isinstance(value, int):
                logger.info(f"  {key}: {value}")

        return results

    @staticmethod
    def _auc_roc(labels: np.ndarray, probs: np.ndarray) -> float:
        """Compute Area Under ROC Curve."""
        try:
            from sklearn.metrics import roc_auc_score
            if len(np.unique(labels)) < 2:
                return 0.0
            return float(roc_auc_score(labels, probs))
        except ImportError:
            # Manual AUC computation
            return HDIEvaluator._manual_auc(labels, probs)

    @staticmethod
    def _auc_pr(labels: np.ndarray, probs: np.ndarray) -> float:
        """Compute Area Under Precision-Recall Curve."""
        try:
            from sklearn.metrics import average_precision_score
            if len(np.unique(labels)) < 2:
                return 0.0
            return float(average_precision_score(labels, probs))
        except ImportError:
            return 0.0

    @staticmethod
    def _precision(labels: np.ndarray, preds: np.ndarray) -> float:
        """Compute precision."""
        tp = ((preds == 1) & (labels == 1)).sum()
        fp = ((preds == 1) & (labels == 0)).sum()
        return float(tp / (tp + fp)) if (tp + fp) > 0 else 0.0

    @staticmethod
    def _recall(labels: np.ndarray, preds: np.ndarray) -> float:
        """Compute recall."""
        tp = ((preds == 1) & (labels == 1)).sum()
        fn = ((preds == 0) & (labels == 1)).sum()
        return float(tp / (tp + fn)) if (tp + fn) > 0 else 0.0

    @staticmethod
    def _f1(labels: np.ndarray, preds: np.ndarray) -> float:
        """Compute F1 score."""
        p = HDIEvaluator._precision(labels, preds)
        r = HDIEvaluator._recall(labels, preds)
        return 2 * p * r / (p + r) if (p + r) > 0 else 0.0

    @staticmethod
    def _mrr(labels: np.ndarray, probs: np.ndarray) -> float:
        """
        Compute Mean Reciprocal Rank.

        For each positive, rank all candidates by probability.
        MRR = 1/|Q| * Σ 1/rank_i
        """
        positive_mask = labels == 1
        if not positive_mask.any():
            return 0.0

        # Sort by probability descending
        sorted_indices = np.argsort(-probs)
        ranks = np.zeros_like(sorted_indices)
        ranks[sorted_indices] = np.arange(1, len(sorted_indices) + 1)

        positive_ranks = ranks[positive_mask]
        reciprocal_ranks = 1.0 / positive_ranks

        return float(reciprocal_ranks.mean())

    @staticmethod
    def _hits_at_k(
        labels: np.ndarray, probs: np.ndarray, k: int = 10
    ) -> float:
        """
        Compute Hits@K.

        Fraction of positive edges ranked in top K by probability.
        """
        if labels.sum() == 0:
            return 0.0

        top_k_indices = np.argsort(-probs)[:k]
        hits = labels[top_k_indices].sum()
        total_positives = labels.sum()

        return float(hits / total_positives)

    @staticmethod
    def _manual_auc(labels: np.ndarray, probs: np.ndarray) -> float:
        """Manual AUC-ROC (when sklearn is unavailable)."""
        sorted_idx = np.argsort(-probs)
        sorted_labels = labels[sorted_idx]

        tp = 0
        fp = 0
        auc = 0.0
        total_pos = labels.sum()
        total_neg = len(labels) - total_pos

        if total_pos == 0 or total_neg == 0:
            return 0.0

        for label in sorted_labels:
            if label == 1:
                tp += 1
            else:
                fp += 1
                auc += tp

        return auc / (total_pos * total_neg)
