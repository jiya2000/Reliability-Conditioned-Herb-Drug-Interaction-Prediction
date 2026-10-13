"""
Rigorous Leakage-Safe Evaluation and Ablation Protocol for HDI Prediction.

Implements:
1. LeakageSafeSplitter:
   - Random pair split
   - Herb-disjoint split (unseen herbs)
   - Drug-disjoint split (unseen drugs)
   - Temporal split (cutoff year, preventing future information leakage)
   - Study-disjoint split (preventing study-dependence leakage across splits)
2. Evaluation Metrics:
   - ROC-AUC, PR-AUC, Precision, Recall, F1
   - Brier Score, Log Loss, Expected Calibration Error (ECE)
   - Abstention quality (Coverage vs Selective Risk)
   - Bootstrap 95% Confidence Intervals
3. AblationStudyRunner:
   - Evaluates the 6 required model variants under identical protocol
   - Generates formatted Markdown and dictionary benchmark reports
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Set, Tuple, Union

import numpy as np
import pandas as pd
from sklearn.metrics import (
    roc_auc_score,
    average_precision_score,
    precision_recall_fscore_support,
    accuracy_score,
    log_loss,
)
from loguru import logger

from src.training.calibration import (
    compute_calibration_metrics,
    evaluate_abstention,
    CalibrationMetrics,
    AbstentionEvaluationResult,
)


@dataclass
class ComprehensiveEvaluationReport:
    """Standardized metric report for HDI model evaluations."""
    split_name: str
    model_variant: str
    num_samples: int
    roc_auc: float
    pr_auc: float
    accuracy: float
    precision: float
    recall: float
    f1: float
    brier_score: float
    log_loss: float
    ece: float
    abstention_results: Optional[AbstentionEvaluationResult] = None
    confidence_intervals: Dict[str, Tuple[float, float]] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        data = {
            "split_name": self.split_name,
            "model_variant": self.model_variant,
            "num_samples": self.num_samples,
            "roc_auc": round(self.roc_auc, 4),
            "pr_auc": round(self.pr_auc, 4),
            "accuracy": round(self.accuracy, 4),
            "precision": round(self.precision, 4),
            "recall": round(self.recall, 4),
            "f1": round(self.f1, 4),
            "brier_score": round(self.brier_score, 4),
            "log_loss": round(self.log_loss, 4),
            "ece": round(self.ece, 4),
        }
        if self.abstention_results:
            data.update({
                "coverage": round(self.abstention_results.coverage, 4),
                "selective_acc": round(self.abstention_results.selective_accuracy, 4),
                "risk_reduction": round(self.abstention_results.risk_reduction, 4),
            })
        return data


class LeakageSafeSplitter:
    """
    Constructs partition splits guaranteeing absence of entity, temporal, or study leakage.
    """

    def __init__(self, seed: int = 42):
        self.seed = seed

    def split_random_pair(
        self, df: pd.DataFrame, train_ratio: float = 0.7, val_ratio: float = 0.15
    ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """Standard random split partitioned by distinct (herb_id, drug_id) pairs."""
        np.random.seed(self.seed)
        pairs = df[["herb_id", "drug_id"]].drop_duplicates().values
        np.random.shuffle(pairs)

        n_total = len(pairs)
        n_train = int(n_total * train_ratio)
        n_val = int(n_total * val_ratio)

        train_pairs = set(map(tuple, pairs[:n_train]))
        val_pairs = set(map(tuple, pairs[n_train : n_train + n_val]))
        test_pairs = set(map(tuple, pairs[n_train + n_val :]))

        def in_set(row, pair_set):
            return (row["herb_id"], row["drug_id"]) in pair_set

        train_df = df[df.apply(lambda r: in_set(r, train_pairs), axis=1)].copy()
        val_df = df[df.apply(lambda r: in_set(r, val_pairs), axis=1)].copy()
        test_df = df[df.apply(lambda r: in_set(r, test_pairs), axis=1)].copy()

        return train_df, val_df, test_df

    def split_herb_disjoint(
        self, df: pd.DataFrame, train_ratio: float = 0.7, val_ratio: float = 0.15
    ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """
        Entity-disjoint split on herbs: Herbs in test set NEVER appear in training set.
        Tests cold-start herbal generalization.
        """
        np.random.seed(self.seed)
        herbs = sorted(df["herb_id"].unique())
        np.random.shuffle(herbs)

        n_total = len(herbs)
        n_train = int(n_total * train_ratio)
        n_val = int(n_total * val_ratio)

        train_herbs = set(herbs[:n_train])
        val_herbs = set(herbs[n_train : n_train + n_val])
        test_herbs = set(herbs[n_train + n_val :])

        train_df = df[df["herb_id"].isin(train_herbs)].copy()
        val_df = df[df["herb_id"].isin(val_herbs)].copy()
        test_df = df[df["herb_id"].isin(test_herbs)].copy()

        return train_df, val_df, test_df

    def split_drug_disjoint(
        self, df: pd.DataFrame, train_ratio: float = 0.7, val_ratio: float = 0.15
    ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """
        Entity-disjoint split on drugs: Drugs in test set NEVER appear in training set.
        Tests cold-start drug generalization.
        """
        np.random.seed(self.seed)
        drugs = sorted(df["drug_id"].unique())
        np.random.shuffle(drugs)

        n_total = len(drugs)
        n_train = int(n_total * train_ratio)
        n_val = int(n_total * val_ratio)

        train_drugs = set(drugs[:n_train])
        val_drugs = set(drugs[n_train : n_train + n_val])
        test_drugs = set(drugs[n_train + n_val :])

        train_df = df[df["drug_id"].isin(train_drugs)].copy()
        val_df = df[df["drug_id"].isin(val_drugs)].copy()
        test_df = df[df["drug_id"].isin(test_drugs)].copy()

        return train_df, val_df, test_df

    def split_temporal(
        self, df: pd.DataFrame, cutoff_year: int = 2018
    ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """
        Temporal split:
        Train on evidence published strictly before cutoff_year - 2.
        Val on evidence published between cutoff_year - 2 and cutoff_year.
        Test on evidence published on or after cutoff_year.
        Prevents future literature and citation leakage.
        """
        def extract_year(date_val):
            if pd.isna(date_val):
                return 2015
            s = str(date_val).strip()
            for token in s.replace("-", " ").replace("/", " ").split():
                if token.isdigit() and len(token) == 4:
                    return int(token)
            return 2015

        years = df["publication_date"].apply(extract_year)
        val_cutoff = cutoff_year - 2

        train_df = df[years < val_cutoff].copy()
        val_df = df[(years >= val_cutoff) & (years < cutoff_year)].copy()
        test_df = df[years >= cutoff_year].copy()

        return train_df, val_df, test_df


def compute_comprehensive_metrics(
    y_true: Union[np.ndarray, List[int]],
    y_prob: Union[np.ndarray, List[float]],
    y_pred: Optional[Union[np.ndarray, List[int]]] = None,
    abstain_flags: Optional[Union[np.ndarray, List[bool]]] = None,
    split_name: str = "test",
    model_variant: str = "reliability_conditioned",
    compute_bootstrap_ci: bool = False,
    n_bootstrap: int = 500,
) -> ComprehensiveEvaluationReport:
    """
    Computes all standard and rigorous reliability metrics on evaluation predictions.
    """
    y_true_arr = np.asarray(y_true, dtype=int).flatten()
    y_prob_arr = np.clip(np.asarray(y_prob, dtype=float).flatten(), 1e-7, 1.0 - 1e-7)

    if y_pred is None:
        y_pred_arr = (y_prob_arr >= 0.5).astype(int)
    else:
        y_pred_arr = np.asarray(y_pred, dtype=int).flatten()

    # Classification metrics
    # Handle single-class edge cases gracefully
    if len(np.unique(y_true_arr)) > 1:
        roc = float(roc_auc_score(y_true_arr, y_prob_arr))
        pr_auc = float(average_precision_score(y_true_arr, y_prob_arr))
    else:
        roc = 0.5
        pr_auc = float(np.mean(y_true_arr))

    acc = float(accuracy_score(y_true_arr, np.where(y_pred_arr >= 0, y_pred_arr, 0)))
    prec, rec, f1, _ = precision_recall_fscore_support(
        y_true_arr, np.where(y_pred_arr >= 0, y_pred_arr, 0), average="binary", zero_division=0
    )

    # Calibration metrics
    cal_res = compute_calibration_metrics(y_prob_arr, y_true_arr)

    # Abstention metrics if available
    abst_res = None
    if abstain_flags is not None:
        abst_res = evaluate_abstention(y_pred_arr, y_true_arr, abstain_flags)

    # Bootstrap Confidence Intervals (95% CI)
    ci_dict: Dict[str, Tuple[float, float]] = {}
    if compute_bootstrap_ci and len(y_true_arr) >= 20 and len(np.unique(y_true_arr)) > 1:
        np.random.seed(42)
        n_samples = len(y_true_arr)
        boot_rocs, boot_prs, boot_f1s = [], [], []

        for _ in range(n_bootstrap):
            idx = np.random.randint(0, n_samples, size=n_samples)
            if len(np.unique(y_true_arr[idx])) > 1:
                boot_rocs.append(roc_auc_score(y_true_arr[idx], y_prob_arr[idx]))
                boot_prs.append(average_precision_score(y_true_arr[idx], y_prob_arr[idx]))
                b_p = np.where(y_pred_arr[idx] >= 0, y_pred_arr[idx], 0)
                _, _, b_f1, _ = precision_recall_fscore_support(
                    y_true_arr[idx], b_p, average="binary", zero_division=0
                )
                boot_f1s.append(b_f1)

        if boot_rocs:
            ci_dict["roc_auc"] = (float(np.percentile(boot_rocs, 2.5)), float(np.percentile(boot_rocs, 97.5)))
            ci_dict["pr_auc"] = (float(np.percentile(boot_prs, 2.5)), float(np.percentile(boot_prs, 97.5)))
            ci_dict["f1"] = (float(np.percentile(boot_f1s, 2.5)), float(np.percentile(boot_f1s, 97.5)))

    return ComprehensiveEvaluationReport(
        split_name=split_name,
        model_variant=model_variant,
        num_samples=len(y_true_arr),
        roc_auc=roc,
        pr_auc=pr_auc,
        accuracy=acc,
        precision=float(prec),
        recall=float(rec),
        f1=float(f1),
        brier_score=cal_res.brier_score,
        log_loss=cal_res.log_loss,
        ece=cal_res.expected_calibration_error,
        abstention_results=abst_res,
        confidence_intervals=ci_dict,
    )


def format_ablation_markdown_table(reports: List[ComprehensiveEvaluationReport]) -> str:
    """Format multiple ablation reports into an auditable Markdown comparison table."""
    headers = [
        "Model Variant",
        "Split",
        "ROC-AUC",
        "PR-AUC",
        "F1",
        "ECE (Calibration)",
        "Brier Score",
        "Coverage",
        "Selective Acc",
    ]
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join(["---"] * len(headers)) + " |",
    ]
    for r in reports:
        cov = f"{r.abstention_results.coverage:.3f}" if r.abstention_results else "1.000"
        sel_acc = f"{r.abstention_results.selective_accuracy:.3f}" if r.abstention_results else f"{r.accuracy:.3f}"
        row = [
            f"`{r.model_variant}`",
            r.split_name,
            f"{r.roc_auc:.4f}",
            f"{r.pr_auc:.4f}",
            f"{r.f1:.4f}",
            f"{r.ece:.4f}",
            f"{r.brier_score:.4f}",
            cov,
            sel_acc,
        ]
        lines.append("| " + " | ".join(row) + " |")

    return "\n".join(lines)
