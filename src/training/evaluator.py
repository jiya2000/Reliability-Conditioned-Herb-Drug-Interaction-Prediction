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
        all_source_indices = []

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
            all_source_indices.append(batch["source_indices"].numpy())

        labels = np.concatenate(all_labels)
        probs = np.concatenate(all_probs)
        logits = np.concatenate(all_logits)
        source_indices = np.concatenate(all_source_indices)
        preds = (probs >= self.threshold).astype(int)

        # Compute metrics
        results = {
            "auc_roc": self._auc_roc(labels, probs),
            "auc_pr": self._auc_pr(labels, probs),
            "precision": self._precision(labels, preds),
            "recall": self._recall(labels, preds),
            "f1": self._f1(labels, preds),
            "mrr": self._mrr_per_query(labels, probs, source_indices),
            "hits@10": self._hits_at_k_per_query(labels, probs, source_indices, k=10),
            "hits@5": self._hits_at_k_per_query(labels, probs, source_indices, k=5),
            "hits@1": self._hits_at_k_per_query(labels, probs, source_indices, k=1),
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
    def _mrr_per_query(
        labels: np.ndarray,
        probs: np.ndarray,
        source_indices: np.ndarray,
    ) -> float:
        """
        Compute Mean Reciprocal Rank (per-query).

        For each source (query) node, rank all its candidates by probability
        and find the rank of the first true positive.
        MRR = 1/|Q| * Σ 1/rank_i

        This is the standard link prediction MRR — ranking is done within
        each query's candidate set, not globally across all samples.
        """
        unique_sources = np.unique(source_indices)
        reciprocal_ranks = []

        for src in unique_sources:
            mask = source_indices == src
            src_labels = labels[mask]
            src_probs = probs[mask]

            if src_labels.sum() == 0:
                continue  # No positives for this query

            # Rank candidates for this query by probability (descending)
            sorted_idx = np.argsort(-src_probs)
            sorted_labels = src_labels[sorted_idx]

            # Find rank of first positive
            positive_positions = np.where(sorted_labels == 1)[0]
            if len(positive_positions) > 0:
                # Rank is 1-indexed
                first_rank = positive_positions[0] + 1
                reciprocal_ranks.append(1.0 / first_rank)

        if not reciprocal_ranks:
            return 0.0

        return float(np.mean(reciprocal_ranks))

    @staticmethod
    def _hits_at_k_per_query(
        labels: np.ndarray,
        probs: np.ndarray,
        source_indices: np.ndarray,
        k: int = 10,
    ) -> float:
        """
        Compute Hits@K (per-query).

        For each source (query) node, check if any true positive is
        ranked in the top K candidates. Returns the fraction of queries
        where at least one positive appears in the top K.
        """
        unique_sources = np.unique(source_indices)
        hits = 0
        queries_with_positives = 0

        for src in unique_sources:
            mask = source_indices == src
            src_labels = labels[mask]
            src_probs = probs[mask]

            if src_labels.sum() == 0:
                continue  # No positives for this query

            queries_with_positives += 1

            # Top K candidates for this query
            top_k_idx = np.argsort(-src_probs)[:k]
            if src_labels[top_k_idx].sum() > 0:
                hits += 1

        if queries_with_positives == 0:
            return 0.0

        return float(hits / queries_with_positives)

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

    # --- ★ NOVEL: Stratified Evaluation ★ ---

    @torch.no_grad()
    def evaluate_stratified(
        self,
        loader,
        graph_data: dict,
        n_strata: int = 4,
    ) -> dict:
        """
        ★ Stratified evaluation by reliability score.

        Splits predictions into R-strata and computes metrics per stratum.
        This is essential for the paper to show how model performance
        varies with evidence quality.

        Args:
            loader: DataLoader
            graph_data: Graph tensors
            n_strata: Number of reliability strata

        Returns:
            Dict with per-stratum and overall metrics
        """
        self.model.eval()

        node_features = graph_data["node_features"].to(self.device)
        edge_index = graph_data["edge_index"].to(self.device)
        edge_type = graph_data["edge_type"].to(self.device)

        all_labels = []
        all_probs = []
        all_R = []
        all_metadata = []

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
            all_metadata.append(metadata.cpu().numpy())

            if "reliability_scores" in output and output["reliability_scores"] is not None:
                all_R.append(output["reliability_scores"].cpu().numpy().squeeze())
            else:
                # Use biomedical quality as proxy for R
                all_R.append(metadata[:, 2].cpu().numpy())

        labels = np.concatenate(all_labels)
        probs = np.concatenate(all_probs)
        R_scores = np.concatenate(all_R)
        metadata_all = np.concatenate(all_metadata)

        # Overall metrics
        preds = (probs >= self.threshold).astype(int)
        overall = {
            "auc_roc": self._auc_roc(labels, probs),
            "f1": self._f1(labels, preds),
            "precision": self._precision(labels, preds),
            "recall": self._recall(labels, preds),
            "n_samples": len(labels),
        }

        # Stratified by R
        boundaries = np.linspace(0, 1, n_strata + 1)
        strata_results = {}

        for i in range(n_strata):
            lo, hi = boundaries[i], boundaries[i + 1]
            mask = (R_scores >= lo) & (R_scores < hi + 1e-6)

            if mask.sum() < 5:
                strata_results[f"R_{lo:.2f}_{hi:.2f}"] = {
                    "n_samples": int(mask.sum()),
                    "note": "insufficient samples",
                }
                continue

            s_labels = labels[mask]
            s_probs = probs[mask]
            s_preds = (s_probs >= self.threshold).astype(int)

            strata_results[f"R_{lo:.2f}_{hi:.2f}"] = {
                "auc_roc": self._auc_roc(s_labels, s_probs),
                "f1": self._f1(s_labels, s_preds),
                "precision": self._precision(s_labels, s_preds),
                "recall": self._recall(s_labels, s_preds),
                "n_samples": int(mask.sum()),
                "avg_R": float(R_scores[mask].mean()),
            }

        # Stratified by source type
        source_types = metadata_all[:, 4].astype(int)
        source_names = {
            0: "unknown", 1: "clinical_trial", 2: "peer_reviewed",
            3: "case_report", 4: "textbook", 5: "health_forum",
            6: "social_media", 7: "traditional",
        }
        source_results = {}

        for st_code, st_name in source_names.items():
            mask = source_types == st_code
            if mask.sum() < 3:
                continue

            s_labels = labels[mask]
            s_probs = probs[mask]
            s_preds = (s_probs >= self.threshold).astype(int)

            source_results[st_name] = {
                "auc_roc": self._auc_roc(s_labels, s_probs),
                "f1": self._f1(s_labels, s_preds),
                "n_samples": int(mask.sum()),
                "avg_R": float(R_scores[mask].mean()),
            }

        # Log
        logger.info("Stratified Evaluation Results:")
        logger.info(f"  Overall: AUC={overall['auc_roc']:.4f}, F1={overall['f1']:.4f}")
        for name, metrics in strata_results.items():
            if "auc_roc" in metrics:
                logger.info(
                    f"  {name}: AUC={metrics['auc_roc']:.4f}, "
                    f"F1={metrics['f1']:.4f}, n={metrics['n_samples']}"
                )

        return {
            "overall": overall,
            "by_reliability": strata_results,
            "by_source": source_results,
        }

    def get_calibration_data(
        self,
        loader,
        graph_data: dict,
        n_bins: int = 10,
    ) -> dict:
        """
        Generate data for calibration plots.

        Computes per-bin accuracy vs confidence for reliability
        calibration diagrams.
        """
        self.model.eval()

        node_features = graph_data["node_features"].to(self.device)
        edge_index = graph_data["edge_index"].to(self.device)
        edge_type = graph_data["edge_type"].to(self.device)

        all_labels = []
        all_probs = []
        all_R = []

        with torch.no_grad():
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

                if "reliability_scores" in output and output["reliability_scores"] is not None:
                    all_R.append(output["reliability_scores"].cpu().numpy().squeeze())

        labels = np.concatenate(all_labels)
        probs = np.concatenate(all_probs)

        # Prediction calibration (confidence vs accuracy)
        pred_bins = []
        boundaries = np.linspace(0, 1, n_bins + 1)
        for i in range(n_bins):
            lo, hi = boundaries[i], boundaries[i + 1]
            mask = (probs >= lo) & (probs < hi + 1e-6)
            if mask.sum() == 0:
                continue
            pred_bins.append({
                "confidence_lo": lo,
                "confidence_hi": hi,
                "avg_confidence": float(probs[mask].mean()),
                "accuracy": float(labels[mask].mean()),
                "count": int(mask.sum()),
            })

        result = {
            "prediction_calibration": pred_bins,
            "ece": self._compute_ece(probs, labels, n_bins),
        }

        # R calibration (if available)
        if all_R:
            R_scores = np.concatenate(all_R)
            correctness = 1.0 - np.abs(probs - labels)
            R_bins = []
            for i in range(n_bins):
                lo, hi = boundaries[i], boundaries[i + 1]
                mask = (R_scores >= lo) & (R_scores < hi + 1e-6)
                if mask.sum() == 0:
                    continue
                R_bins.append({
                    "R_lo": lo,
                    "R_hi": hi,
                    "avg_R": float(R_scores[mask].mean()),
                    "avg_correctness": float(correctness[mask].mean()),
                    "count": int(mask.sum()),
                })
            result["reliability_calibration"] = R_bins

        return result

    @staticmethod
    def _compute_ece(
        probs: np.ndarray, labels: np.ndarray, n_bins: int = 10
    ) -> float:
        """Compute Expected Calibration Error."""
        boundaries = np.linspace(0, 1, n_bins + 1)
        ece = 0.0
        for i in range(n_bins):
            lo, hi = boundaries[i], boundaries[i + 1]
            mask = (probs >= lo) & (probs < hi + 1e-6)
            if mask.sum() == 0:
                continue
            avg_conf = probs[mask].mean()
            avg_acc = labels[mask].mean()
            ece += mask.sum() / len(probs) * abs(avg_conf - avg_acc)
        return float(ece)

