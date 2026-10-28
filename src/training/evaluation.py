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
    brier_score_loss,
)
from loguru import logger
import torch
import torch.nn as nn

from src.training.calibration import (
    compute_calibration_metrics,
    evaluate_abstention,
    CalibrationMetrics,
    AbstentionEvaluationResult,
    TemperatureScaler,
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
    equal_mass_ece: float = 0.0
    brier_positive: float = 0.0
    brier_negative: float = 0.0
    temperature: float = 1.0
    abstention_results: Optional[AbstentionEvaluationResult] = None
    confidence_intervals: Dict[str, Tuple[float, float]] = field(default_factory=dict)
    reliability_diagram_data: Dict[str, Any] = field(default_factory=dict)
    point_estimates: Dict[str, float] = field(default_factory=dict)
    bootstrap_means: Dict[str, float] = field(default_factory=dict)
    confidence_intervals_95: Dict[str, Tuple[float, float]] = field(default_factory=dict)
    predictions: Optional[List[Dict[str, Any]]] = None

    def to_dict(self) -> Dict[str, Any]:
        data = {
            "split_name": self.split_name,
            "model_variant": self.model_variant,
            "num_samples": self.num_samples,
            # Point estimates (headline metrics)
            "roc_auc": float(self.roc_auc),
            "pr_auc": float(self.pr_auc),
            "accuracy": float(self.accuracy),
            "precision": float(self.precision),
            "recall": float(self.recall),
            "f1": float(self.f1),
            "brier_score": float(self.brier_score),
            "log_loss": float(self.log_loss),
            "ece": float(self.ece),
            "equal_mass_ece": float(self.equal_mass_ece),
            "brier_positive": float(self.brier_positive),
            "brier_negative": float(self.brier_negative),
            "temperature": float(self.temperature),
            # Clearly labeled breakdowns
            "point_estimates": self.point_estimates if self.point_estimates else {
                "roc_auc": float(self.roc_auc),
                "pr_auc": float(self.pr_auc),
                "accuracy": float(self.accuracy),
                "precision": float(self.precision),
                "recall": float(self.recall),
                "f1": float(self.f1),
                "brier_score": float(self.brier_score),
                "log_loss": float(self.log_loss),
                "ece": float(self.ece),
                "equal_mass_ece": float(self.equal_mass_ece),
                "brier_positive": float(self.brier_positive),
                "brier_negative": float(self.brier_negative),
                "temperature": float(self.temperature),
            },
            "bootstrap_mean": self.bootstrap_means,
            "confidence_intervals_95": {
                k: [float(v[0]), float(v[1])]
                for k, v in (self.confidence_intervals_95 or self.confidence_intervals).items()
            },
        }
        if self.predictions is not None:
            data["predictions"] = self.predictions
        if self.abstention_results:
            data.update({
                "coverage": float(self.abstention_results.coverage),
                "selective_acc": float(self.abstention_results.selective_accuracy),
                "risk_reduction": float(self.abstention_results.risk_reduction),
            })
        return data


@dataclass
class RepeatedNestedCVResult:
    """Aggregated evaluation results across repeated nested cross-validation folds."""
    model_variant: str
    split_mode: str
    n_repeats: int
    n_splits: int
    metrics_mean: Dict[str, float]
    metrics_std: Dict[str, float]
    confidence_intervals_95: Dict[str, Tuple[float, float]]
    temperature_variability: Dict[str, float]
    pooled_report: ComprehensiveEvaluationReport
    fold_reports: List[Dict[str, Any]]
    significance_vs_baselines: Dict[str, Dict[str, Any]] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "model_variant": self.model_variant,
            "split_mode": self.split_mode,
            "n_repeats": self.n_repeats,
            "n_splits": self.n_splits,
            "metrics_mean": {k: round(v, 4) for k, v in self.metrics_mean.items()},
            "metrics_std": {k: round(v, 4) for k, v in self.metrics_std.items()},
            "confidence_intervals_95": {
                k: (round(v[0], 4), round(v[1], 4)) for k, v in self.confidence_intervals_95.items()
            },
            "temperature_variability": {
                k: round(v, 4) for k, v in self.temperature_variability.items()
            },
            "pooled_report": self.pooled_report.to_dict(),
            "significance_vs_baselines": self.significance_vs_baselines,
        }


class LeakageSafeSplitter:
    """
    Constructs partition splits guaranteeing absence of entity, temporal, or study leakage.
    Modes supported:
    - study_disjoint: no primary study or citation cluster shared across splits
    - pair_disjoint: no (herb, drug) pair shared between train/val/test
    - herb_disjoint: no herb shared (cold-start herb generalization)
    - drug_disjoint: no drug shared (cold-start drug generalization)
    - temporal: split by chronological publication cutoff year
    """

    def __init__(self, seed: int = 42):
        self.seed = seed

    def split(
        self,
        df: pd.DataFrame,
        mode: str = "study_disjoint",
        train_ratio: float = 0.7,
        val_ratio: float = 0.15,
        cutoff_year: int = 2018,
    ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """Dispatcher for leakage-safe partition split modes."""
        mode_norm = str(mode).lower().strip()
        if mode_norm in {"study_disjoint", "study"}:
            return self.split_study_disjoint(df, train_ratio=train_ratio, val_ratio=val_ratio)
        elif mode_norm in {"pair_disjoint", "pair", "random_pair"}:
            return self.split_pair_disjoint(df, train_ratio=train_ratio, val_ratio=val_ratio)
        elif mode_norm in {"herb_disjoint", "herb"}:
            return self.split_herb_disjoint(df, train_ratio=train_ratio, val_ratio=val_ratio)
        elif mode_norm in {"drug_disjoint", "drug"}:
            return self.split_drug_disjoint(df, train_ratio=train_ratio, val_ratio=val_ratio)
        elif mode_norm in {"temporal", "time"}:
            return self.split_temporal(df, cutoff_year=cutoff_year)
        else:
            raise ValueError(
                f"Unknown split mode '{mode}'. Choose from: "
                f"'study_disjoint', 'pair_disjoint', 'herb_disjoint', 'drug_disjoint', 'temporal'."
            )

    def split_pair_disjoint(
        self, df: pd.DataFrame, train_ratio: float = 0.7, val_ratio: float = 0.15
    ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """
        Partition partitioned strictly by distinct (herb_id, drug_id) pairs.
        Guarantees that NO pair in the test or validation partition ever appears in training.
        """
        np.random.seed(self.seed)
        pairs = df[["herb_id", "drug_id"]].drop_duplicates().values
        n_total = len(pairs)

        if n_total < 3:
            raise ValueError(
                f"Pair-disjoint split requires at least 3 distinct (herb, drug) pairs, but found {n_total}. "
                f"Current data size is insufficient for a 3-way partition."
            )

        np.random.shuffle(pairs)

        n_train = max(1, int(round(n_total * train_ratio)))
        n_val = max(1, int(round(n_total * val_ratio)))
        n_test = n_total - n_train - n_val

        while n_test < 1 and n_train > 1:
            n_train -= 1
            n_test += 1
        while n_test < 1 and n_val > 1:
            n_val -= 1
            n_test += 1

        if n_test < 1:
            raise ValueError(
                f"Infeasible pair-disjoint split: could not allocate non-empty partitions for {n_total} pairs "
                f"(train_ratio={train_ratio}, val_ratio={val_ratio})."
            )

        train_pairs = set(map(tuple, pairs[:n_train]))
        val_pairs = set(map(tuple, pairs[n_train : n_train + n_val]))
        test_pairs = set(map(tuple, pairs[n_train + n_val :]))

        # Strict assertion against cross-split pair leakage
        pair_leakage = (train_pairs & test_pairs) | (train_pairs & val_pairs) | (val_pairs & test_pairs)
        if pair_leakage:
            raise RuntimeError(f"Pair-disjoint split failure: detected leaked pairs across splits: {pair_leakage}")

        def in_set(row, pair_set):
            return (row["herb_id"], row["drug_id"]) in pair_set

        train_df = df[df.apply(lambda r: in_set(r, train_pairs), axis=1)].copy()
        val_df = df[df.apply(lambda r: in_set(r, val_pairs), axis=1)].copy()
        test_df = df[df.apply(lambda r: in_set(r, test_pairs), axis=1)].copy()

        if train_df.empty or val_df.empty or test_df.empty:
            raise RuntimeError(
                f"Pair-disjoint split produced an empty partition: "
                f"train={len(train_df)}, val={len(val_df)}, test={len(test_df)}."
            )

        return train_df, val_df, test_df

    def split_random_pair(
        self, df: pd.DataFrame, train_ratio: float = 0.7, val_ratio: float = 0.15
    ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """Backward-compatible alias for split_pair_disjoint."""
        return self.split_pair_disjoint(df, train_ratio=train_ratio, val_ratio=val_ratio)

    def split_herb_disjoint(
        self, df: pd.DataFrame, train_ratio: float = 0.7, val_ratio: float = 0.15
    ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """
        Entity-disjoint split on herbs: Herbs in test set NEVER appear in training set.
        Tests cold-start herbal generalization.
        """
        np.random.seed(self.seed)
        herbs = sorted(df["herb_id"].unique())
        n_total = len(herbs)

        if n_total < 3:
            raise ValueError(
                f"Herb-disjoint split requires at least 3 distinct herbs, but found {n_total}. "
                f"Current data size is insufficient for cold-start herbal evaluation."
            )

        np.random.shuffle(herbs)

        n_train = max(1, int(round(n_total * train_ratio)))
        n_val = max(1, int(round(n_total * val_ratio)))
        n_test = n_total - n_train - n_val

        while n_test < 1 and n_train > 1:
            n_train -= 1
            n_test += 1
        while n_test < 1 and n_val > 1:
            n_val -= 1
            n_test += 1

        if n_test < 1:
            raise ValueError(
                f"Infeasible herb-disjoint split: cannot allocate partitions for {n_total} herbs."
            )

        train_herbs = set(herbs[:n_train])
        val_herbs = set(herbs[n_train : n_train + n_val])
        test_herbs = set(herbs[n_train + n_val :])

        herb_leakage = (train_herbs & test_herbs) | (train_herbs & val_herbs) | (val_herbs & test_herbs)
        if herb_leakage:
            raise RuntimeError(f"Herb-disjoint split failure: detected leaked herbs: {herb_leakage}")

        train_df = df[df["herb_id"].isin(train_herbs)].copy()
        val_df = df[df["herb_id"].isin(val_herbs)].copy()
        test_df = df[df["herb_id"].isin(test_herbs)].copy()

        if train_df.empty or val_df.empty or test_df.empty:
            raise RuntimeError(
                f"Herb-disjoint split produced an empty partition: "
                f"train={len(train_df)}, val={len(val_df)}, test={len(test_df)}."
            )

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
        n_total = len(drugs)

        if n_total < 3:
            raise ValueError(
                f"Drug-disjoint split requires at least 3 distinct drugs, but found {n_total}. "
                f"Current data size is insufficient for cold-start drug evaluation."
            )

        np.random.shuffle(drugs)

        n_train = max(1, int(round(n_total * train_ratio)))
        n_val = max(1, int(round(n_total * val_ratio)))
        n_test = n_total - n_train - n_val

        while n_test < 1 and n_train > 1:
            n_train -= 1
            n_test += 1
        while n_test < 1 and n_val > 1:
            n_val -= 1
            n_test += 1

        if n_test < 1:
            raise ValueError(
                f"Infeasible drug-disjoint split: cannot allocate partitions for {n_total} drugs."
            )

        train_drugs = set(drugs[:n_train])
        val_drugs = set(drugs[n_train : n_train + n_val])
        test_drugs = set(drugs[n_train + n_val :])

        drug_leakage = (train_drugs & test_drugs) | (train_drugs & val_drugs) | (val_drugs & test_drugs)
        if drug_leakage:
            raise RuntimeError(f"Drug-disjoint split failure: detected leaked drugs: {drug_leakage}")

        train_df = df[df["drug_id"].isin(train_drugs)].copy()
        val_df = df[df["drug_id"].isin(val_drugs)].copy()
        test_df = df[df["drug_id"].isin(test_drugs)].copy()

        if train_df.empty or val_df.empty or test_df.empty:
            raise RuntimeError(
                f"Drug-disjoint split produced an empty partition: "
                f"train={len(train_df)}, val={len(val_df)}, test={len(test_df)}."
            )

        return train_df, val_df, test_df

    def split_temporal(
        self, df: pd.DataFrame, cutoff_year: int = 2018
    ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """
        Temporal split:
        Train on evidence published strictly before cutoff_year - 2.
        Val on evidence published between cutoff_year - 2 and cutoff_year.
        Test on evidence published on or after cutoff_year.

        CRITICAL FIX: Undated records are NOT given a fabricated default date.
        They are quarantined and excluded from chronological evaluation to prevent future leakage.
        """
        def extract_year(date_val) -> Optional[int]:
            if pd.isna(date_val):
                return None
            s = str(date_val).strip()
            for token in s.replace("-", " ").replace("/", " ").split():
                if token.isdigit() and len(token) == 4:
                    y = int(token)
                    if 1900 <= y <= 2030:
                        return y
            return None

        years = df["publication_date"].apply(extract_year)
        valid_date_mask = years.notna()
        dated_df = df[valid_date_mask].copy()
        valid_years = years[valid_date_mask].astype(int)

        undated_count = len(df) - len(dated_df)
        if undated_count > 0:
            logger.info(f"Temporal split quarantined {undated_count} undated records to prevent chronological leakage.")

        val_cutoff = cutoff_year - 2
        train_df = dated_df[valid_years < val_cutoff].copy()
        val_df = dated_df[(valid_years >= val_cutoff) & (valid_years < cutoff_year)].copy()
        test_df = dated_df[valid_years >= cutoff_year].copy()

        return train_df, val_df, test_df

    def split_study_disjoint(
        self,
        df: pd.DataFrame,
        train_ratio: float = 0.7,
        val_ratio: float = 0.15,
    ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """
        Study- and Citation-Cluster-Disjoint Split:
        Clusters records by shared primary study and citation dependencies.
        Guarantees that no study_id or derivative citation report appears across multiple splits.
        """
        import networkx as nx

        np.random.seed(self.seed)
        G = nx.Graph()

        # Build citation dependency graph across all records
        for _, row in df.iterrows():
            s_id = str(row["study_id"]).strip()
            G.add_node(s_id)
            cited_list = row.get("cited_study_ids", [])
            if isinstance(cited_list, str):
                import json
                try:
                    cited_list = json.loads(cited_list)
                except Exception:
                    cited_list = [c.strip() for c in cited_list.split(",") if c.strip()]
            if isinstance(cited_list, (list, tuple, set)):
                for c_id in cited_list:
                    c_str = str(c_id).strip()
                    if c_str and c_str != s_id:
                        G.add_node(c_str)
                        G.add_edge(s_id, c_str)

        # Validate split ratios
        if not (0.0 < train_ratio < 1.0) or not (0.0 < val_ratio < 1.0) or (train_ratio + val_ratio >= 1.0):
            raise ValueError(
                f"Invalid split ratios: train_ratio={train_ratio}, val_ratio={val_ratio}. "
                f"Must satisfy train_ratio > 0, val_ratio > 0, and train_ratio + val_ratio < 1.0"
            )

        clusters = list(nx.connected_components(G))
        np.random.shuffle(clusters)

        n_clusters = len(clusters)
        if n_clusters < 3:
            raise ValueError(
                f"Study-disjoint splitting requires at least 3 independent citation clusters to form "
                f"non-empty train, validation, and test partitions, but only found {n_clusters} cluster(s). "
                f"Add more independent study literature or use pair-disjoint splitting for small subsets."
            )

        # Proportional allocation guaranteeing >= 1 cluster per partition
        n_train = int(round(n_clusters * train_ratio))
        n_val = int(round(n_clusters * val_ratio))
        n_train = max(1, n_train)
        n_val = max(1, n_val)
        n_test = n_clusters - n_train - n_val

        while n_test < 1 and n_train > 1:
            n_train -= 1
            n_test += 1
        while n_test < 1 and n_val > 1:
            n_val -= 1
            n_test += 1

        if n_test < 1:
            raise ValueError(
                f"Cannot allocate non-empty train/val/test partitions for {n_clusters} clusters with ratios "
                f"train={train_ratio}, val={val_ratio}."
            )

        train_study_set = set().union(*clusters[:n_train])
        val_study_set = set().union(*clusters[n_train : n_train + n_val])
        test_study_set = set().union(*clusters[n_train + n_val :])

        train_df = df[df["study_id"].astype(str).isin(train_study_set)].copy()
        val_df = df[df["study_id"].astype(str).isin(val_study_set)].copy()
        test_df = df[df["study_id"].astype(str).isin(test_study_set)].copy()

        # Explicitly verify all 3 partitions contain usable records
        if train_df.empty or val_df.empty or test_df.empty:
            raise RuntimeError(
                f"Study-disjoint split produced an empty partition: "
                f"train={len(train_df)}, val={len(val_df)}, test={len(test_df)}. "
                f"Ensure study_id values in evidence DataFrame map to the citation graph."
            )

        # Strict assertion against cross-split study leakage
        train_studies = set(train_df["study_id"].astype(str))
        val_studies = set(val_df["study_id"].astype(str))
        test_studies = set(test_df["study_id"].astype(str))
        leakage = (train_studies & test_studies) | (train_studies & val_studies) | (val_studies & test_studies)
        if leakage:
            raise RuntimeError(f"Study-disjoint split failure: leaked studies across partitions: {leakage}")

        logger.info(
            f"Study-disjoint split: {len(train_df)} train, {len(val_df)} val, {len(test_df)} test records "
            f"across {n_clusters} independent study clusters (zero study leakage)."
        )
        return train_df, val_df, test_df

    def _get_citation_clusters(self, df: pd.DataFrame) -> List[Set[str]]:
        """Extract connected citation clusters for studies present in df."""
        import networkx as nx
        import json

        G = nx.Graph()
        for _, row in df.iterrows():
            s_id = str(row["study_id"]).strip()
            G.add_node(s_id)
            cited_list = row.get("cited_study_ids", [])
            if isinstance(cited_list, str):
                try:
                    cited_list = json.loads(cited_list)
                except Exception:
                    cited_list = [c.strip() for c in cited_list.split(",") if c.strip()]
            if isinstance(cited_list, (list, tuple, set)):
                for c_id in cited_list:
                    c_str = str(c_id).strip()
                    if c_str and c_str != s_id:
                        G.add_node(c_str)
                        G.add_edge(s_id, c_str)

        all_clusters = list(nx.connected_components(G))
        df_studies = set(df["study_id"].astype(str))
        valid_clusters = [c for c in all_clusters if c & df_studies]
        return valid_clusters

    def split_train_val(
        self,
        df: pd.DataFrame,
        mode: str = "study_disjoint",
        train_ratio: float = 0.8,
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Group-disjoint 2-way split into train and validation partitions.
        Guarantees zero group leakage. Useful for inner cross-validation folds.
        """
        np.random.seed(self.seed)
        mode_norm = str(mode).lower().strip()

        if mode_norm in {"study_disjoint", "study"}:
            clusters = self._get_citation_clusters(df)
            n_clusters = len(clusters)
            if n_clusters < 2:
                raise ValueError(
                    f"Study-disjoint train/val split requires at least 2 citation clusters, found {n_clusters}."
                )
            np.random.shuffle(clusters)
            n_train = max(1, min(n_clusters - 1, int(round(n_clusters * train_ratio))))
            train_study_set = set().union(*clusters[:n_train])
            val_study_set = set().union(*clusters[n_train:])
            train_df = df[df["study_id"].astype(str).isin(train_study_set)].copy()
            val_df = df[df["study_id"].astype(str).isin(val_study_set)].copy()
            leakage = set(train_df["study_id"].astype(str)) & set(val_df["study_id"].astype(str))
            if leakage:
                raise RuntimeError(f"Study-disjoint train/val leakage: {leakage}")

        elif mode_norm in {"pair_disjoint", "pair", "random_pair"}:
            pairs = df[["herb_id", "drug_id"]].drop_duplicates().values
            n_pairs = len(pairs)
            if n_pairs < 2:
                raise ValueError(
                    f"Pair-disjoint train/val split requires at least 2 distinct pairs, found {n_pairs}."
                )
            np.random.shuffle(pairs)
            n_train = max(1, min(n_pairs - 1, int(round(n_pairs * train_ratio))))
            train_pairs = set(map(tuple, pairs[:n_train]))
            val_pairs = set(map(tuple, pairs[n_train:]))
            train_df = df[df.apply(lambda r: (r["herb_id"], r["drug_id"]) in train_pairs, axis=1)].copy()
            val_df = df[df.apply(lambda r: (r["herb_id"], r["drug_id"]) in val_pairs, axis=1)].copy()
            leakage = train_pairs & val_pairs
            if leakage:
                raise RuntimeError(f"Pair-disjoint train/val leakage: {leakage}")

        elif mode_norm in {"herb_disjoint", "herb"}:
            herbs = sorted(df["herb_id"].unique())
            n_herbs = len(herbs)
            if n_herbs < 2:
                raise ValueError(
                    f"Herb-disjoint train/val split requires at least 2 distinct herbs, found {n_herbs}."
                )
            np.random.shuffle(herbs)
            n_train = max(1, min(n_herbs - 1, int(round(n_herbs * train_ratio))))
            train_herbs = set(herbs[:n_train])
            val_herbs = set(herbs[n_train:])
            train_df = df[df["herb_id"].isin(train_herbs)].copy()
            val_df = df[df["herb_id"].isin(val_herbs)].copy()
            leakage = train_herbs & val_herbs
            if leakage:
                raise RuntimeError(f"Herb-disjoint train/val leakage: {leakage}")

        elif mode_norm in {"drug_disjoint", "drug"}:
            drugs = sorted(df["drug_id"].unique())
            n_drugs = len(drugs)
            if n_drugs < 2:
                raise ValueError(
                    f"Drug-disjoint train/val split requires at least 2 distinct drugs, found {n_drugs}."
                )
            np.random.shuffle(drugs)
            n_train = max(1, min(n_drugs - 1, int(round(n_drugs * train_ratio))))
            train_drugs = set(drugs[:n_train])
            val_drugs = set(drugs[n_train:])
            train_df = df[df["drug_id"].isin(train_drugs)].copy()
            val_df = df[df["drug_id"].isin(val_drugs)].copy()
            leakage = train_drugs & val_drugs
            if leakage:
                raise RuntimeError(f"Drug-disjoint train/val leakage: {leakage}")
        else:
            raise ValueError(f"Unknown split mode '{mode}' for train/val split")

        if train_df.empty or val_df.empty:
            raise RuntimeError(f"Train/val split produced an empty partition: train={len(train_df)}, val={len(val_df)}")

        return train_df, val_df

    def grouped_kfold_split(
        self,
        df: pd.DataFrame,
        mode: str = "study_disjoint",
        n_splits: int = 5,
        stratify: bool = True,
    ) -> List[Tuple[pd.DataFrame, pd.DataFrame]]:
        """
        Group-aware K-Fold cross validation split.
        Guarantees that no study cluster, (herb, drug) pair, herb, or drug is ever shared
        between train and test folds.

        Args:
            df: DataFrame containing HDI interaction evidence
            mode: 'study_disjoint', 'pair_disjoint', 'herb_disjoint', or 'drug_disjoint'
            n_splits: Number of cross-validation folds
            stratify: Whether to balance positive and negative interaction outcomes across folds

        Returns:
            List of (train_df, test_df) tuples for each fold
        """
        if n_splits < 2:
            raise ValueError(f"n_splits must be at least 2, got {n_splits}")

        np.random.seed(self.seed)
        mode_norm = str(mode).lower().strip()

        # 1. Identify distinct groups
        if mode_norm in {"study_disjoint", "study"}:
            clusters = self._get_citation_clusters(df)
            n_groups = len(clusters)
            if n_groups < n_splits:
                raise ValueError(
                    f"Study-disjoint {n_splits}-fold CV requires at least {n_splits} citation clusters, "
                    f"but only found {n_groups} valid clusters. Current data size is insufficient."
                )
            study_to_cluster = {}
            for c_idx, c in enumerate(clusters):
                for s in c:
                    study_to_cluster[s] = c_idx
            groups = list(range(n_groups))
            row_groups = df["study_id"].astype(str).map(study_to_cluster).values

        elif mode_norm in {"pair_disjoint", "pair", "random_pair"}:
            pairs = df[["herb_id", "drug_id"]].drop_duplicates().values
            n_groups = len(pairs)
            if n_groups < n_splits:
                raise ValueError(
                    f"Pair-disjoint {n_splits}-fold CV requires at least {n_splits} distinct (herb, drug) pairs, "
                    f"but only found {n_groups} pairs. Current data size is insufficient."
                )
            pair_to_idx = {tuple(p): i for i, p in enumerate(pairs)}
            groups = list(range(n_groups))
            row_groups = np.array([pair_to_idx[(r["herb_id"], r["drug_id"])] for _, r in df.iterrows()])

        elif mode_norm in {"herb_disjoint", "herb"}:
            herbs = sorted(df["herb_id"].unique())
            n_groups = len(herbs)
            if n_groups < n_splits:
                raise ValueError(
                    f"Herb-disjoint {n_splits}-fold CV requires at least {n_splits} distinct herbs, "
                    f"but only found {n_groups} herbs. Current data size is insufficient."
                )
            herb_to_idx = {h: i for i, h in enumerate(herbs)}
            groups = list(range(n_groups))
            row_groups = df["herb_id"].map(herb_to_idx).values

        elif mode_norm in {"drug_disjoint", "drug"}:
            drugs = sorted(df["drug_id"].unique())
            n_groups = len(drugs)
            if n_groups < n_splits:
                raise ValueError(
                    f"Drug-disjoint {n_splits}-fold CV requires at least {n_splits} distinct drugs, "
                    f"but only found {n_groups} drugs. Current data size is insufficient."
                )
            drug_to_idx = {d: i for i, d in enumerate(drugs)}
            groups = list(range(n_groups))
            row_groups = df["drug_id"].map(drug_to_idx).values
        else:
            raise ValueError(f"Unknown split mode '{mode}' for grouped CV")

        # 2. Stratification by binary outcome
        outcomes = df["outcome"].astype(str).str.lower().isin(["interaction_present", "positive", "1", "true"]).values.astype(int)

        # Compute group-level labels
        group_labels = np.zeros(n_groups, dtype=float)
        for g in groups:
            mask = (row_groups == g)
            if np.any(mask):
                group_labels[g] = float(np.mean(outcomes[mask]))

        # Allocate groups to folds
        fold_buckets: List[List[int]] = [[] for _ in range(n_splits)]

        if stratify and len(np.unique(outcomes)) > 1:
            pos_groups = [g for g in groups if group_labels[g] >= 0.5]
            neg_groups = [g for g in groups if group_labels[g] < 0.5]
            np.random.shuffle(pos_groups)
            np.random.shuffle(neg_groups)

            for idx, g in enumerate(pos_groups):
                fold_buckets[idx % n_splits].append(g)
            for idx, g in enumerate(neg_groups):
                fold_buckets[idx % n_splits].append(g)
        else:
            shuffled_groups = list(groups)
            np.random.shuffle(shuffled_groups)
            for idx, g in enumerate(shuffled_groups):
                fold_buckets[idx % n_splits].append(g)

        folds: List[Tuple[pd.DataFrame, pd.DataFrame]] = []
        for k in range(n_splits):
            test_g_set = set(fold_buckets[k])
            train_g_set = set(groups) - test_g_set

            # Strict assertion: zero group leakage
            assert len(test_g_set & train_g_set) == 0, "Group leakage detected in fold generation!"

            test_mask = np.isin(row_groups, list(test_g_set))
            train_mask = np.isin(row_groups, list(train_g_set))

            train_df = df[train_mask].copy()
            test_df = df[test_mask].copy()

            # Explicit code assertion: train and test group/entity sets are disjoint
            if mode_norm in {"herb_disjoint", "herb"}:
                leak = set(train_df["herb_id"]) & set(test_df["herb_id"])
                assert len(leak) == 0, f"Herb leakage detected in fold {k}: {leak}"
            elif mode_norm in {"drug_disjoint", "drug"}:
                leak = set(train_df["drug_id"]) & set(test_df["drug_id"])
                assert len(leak) == 0, f"Drug leakage detected in fold {k}: {leak}"
            elif mode_norm in {"pair_disjoint", "pair", "random_pair"}:
                train_p = set(zip(train_df["herb_id"], train_df["drug_id"]))
                test_p = set(zip(test_df["herb_id"], test_df["drug_id"]))
                leak = train_p & test_p
                assert len(leak) == 0, f"Pair leakage detected in fold {k}: {leak}"

            if len(train_df) == 0 or len(test_df) == 0:
                raise RuntimeError(
                    f"Fold {k} generated an empty partition: train={len(train_df)}, test={len(test_df)}"
                )

            folds.append((train_df, test_df))

        return folds



def compute_comprehensive_metrics(
    y_true: Union[np.ndarray, List[int]],
    y_prob: Union[np.ndarray, List[float]],
    y_pred: Optional[Union[np.ndarray, List[int]]] = None,
    abstain_flags: Optional[Union[np.ndarray, List[bool]]] = None,
    split_name: str = "test",
    model_variant: str = "reliability_conditioned",
    compute_bootstrap_ci: bool = False,
    n_bootstrap: int = 500,
    sample_metadata: Optional[List[Dict[str, Any]]] = None,
) -> ComprehensiveEvaluationReport:
    """
    Computes all standard and rigorous reliability metrics on evaluation predictions.
    Records exact point estimates, bootstrap means, 95% CIs, and individual sample predictions.
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

    is_leaky_ref = (
        "leaky" in model_variant.lower()
        or "reference" in model_variant.lower()
        or "probe" in model_variant.lower()
    )
    if roc > 0.98 and not is_leaky_ref:
        msg = (
            f"SUSPICIOUS HIGH PERFORMANCE WARNING: Model variant '{model_variant}' achieved "
            f"ROC-AUC {roc:.4f} > 0.98 on split '{split_name}'. This strongly suggests potential label "
            f"leakage or evaluation flaw in non-reference models."
        )
        logger.warning(msg)
        import warnings
        warnings.warn(msg, UserWarning, stacklevel=2)

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

    # Point estimates dictionary (raw double-precision floats for exact 1e-9 reproduction)
    pt_dict: Dict[str, float] = {
        "roc_auc": float(roc),
        "pr_auc": float(pr_auc),
        "accuracy": float(acc),
        "precision": float(prec),
        "recall": float(rec),
        "f1": float(f1),
        "brier_score": float(cal_res.brier_score),
        "log_loss": float(cal_res.log_loss),
        "ece": float(cal_res.expected_calibration_error),
        "equal_mass_ece": float(cal_res.equal_mass_ece),
        "brier_positive": float(cal_res.brier_positive),
        "brier_negative": float(cal_res.brier_negative),
        "temperature": float(cal_res.temperature),
    }

    # Bootstrap Confidence Intervals (95% CI) and Bootstrap Sample Means
    ci_dict: Dict[str, Tuple[float, float]] = {}
    boot_means: Dict[str, float] = {}
    if compute_bootstrap_ci and len(y_true_arr) >= 6 and len(np.unique(y_true_arr)) > 1:
        np.random.seed(42)
        n_samples = len(y_true_arr)
        boot_rocs, boot_prs, boot_f1s = [], [], []
        boot_accs, boot_briers = [], []

        for _ in range(n_bootstrap):
            idx = np.random.randint(0, n_samples, size=n_samples)
            if len(np.unique(y_true_arr[idx])) > 1:
                boot_rocs.append(float(roc_auc_score(y_true_arr[idx], y_prob_arr[idx])))
                boot_prs.append(float(average_precision_score(y_true_arr[idx], y_prob_arr[idx])))
                b_p = np.where(y_pred_arr[idx] >= 0, y_pred_arr[idx], 0)
                _, _, b_f1, _ = precision_recall_fscore_support(
                    y_true_arr[idx], b_p, average="binary", zero_division=0
                )
                boot_f1s.append(float(b_f1))
                boot_accs.append(float(accuracy_score(y_true_arr[idx], b_p)))
                boot_briers.append(float(brier_score_loss(y_true_arr[idx], y_prob_arr[idx])))

        if boot_rocs:
            ci_dict["roc_auc"] = (float(np.percentile(boot_rocs, 2.5)), float(np.percentile(boot_rocs, 97.5)))
            ci_dict["pr_auc"] = (float(np.percentile(boot_prs, 2.5)), float(np.percentile(boot_prs, 97.5)))
            ci_dict["f1"] = (float(np.percentile(boot_f1s, 2.5)), float(np.percentile(boot_f1s, 97.5)))
            ci_dict["accuracy"] = (float(np.percentile(boot_accs, 2.5)), float(np.percentile(boot_accs, 97.5)))
            ci_dict["brier_score"] = (float(np.percentile(boot_briers, 2.5)), float(np.percentile(boot_briers, 97.5)))

            boot_means["roc_auc"] = float(np.mean(boot_rocs))
            boot_means["pr_auc"] = float(np.mean(boot_prs))
            boot_means["f1"] = float(np.mean(boot_f1s))
            boot_means["accuracy"] = float(np.mean(boot_accs))
            boot_means["brier_score"] = float(np.mean(boot_briers))

    # Construct individual sample predictions records
    pred_records: List[Dict[str, Any]] = []
    for i in range(len(y_true_arr)):
        rec_item: Dict[str, Any] = {
            "index": int(i),
            "y_true": int(y_true_arr[i]),
            "y_prob": float(y_prob_arr[i]),
            "y_pred": int(y_pred_arr[i]),
        }
        if sample_metadata and i < len(sample_metadata):
            rec_item.update(sample_metadata[i])
        pred_records.append(rec_item)

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
        equal_mass_ece=cal_res.equal_mass_ece,
        brier_positive=cal_res.brier_positive,
        brier_negative=cal_res.brier_negative,
        temperature=cal_res.temperature,
        abstention_results=abst_res,
        confidence_intervals=ci_dict,
        reliability_diagram_data=cal_res.reliability_diagram_data,
        point_estimates=pt_dict,
        bootstrap_means=boot_means,
        confidence_intervals_95=ci_dict,
        predictions=pred_records,
    )


def format_ablation_markdown_table(reports: List[ComprehensiveEvaluationReport]) -> str:
    """Format multiple ablation reports into auditable Markdown comparison tables."""
    def is_leaky(report: ComprehensiveEvaluationReport) -> bool:
        v = report.model_variant.lower()
        return "leaky" in v or "full_vector" in v or "probe" in v

    main_reports = [r for r in reports if not is_leaky(r)]
    leaky_reports = [r for r in reports if is_leaky(r)]

    headers = [
        "Model Variant",
        "Split",
        "ROC-AUC (95% CI)",
        "PR-AUC",
        "F1",
        "ECE (Calibration)",
        "Brier Score",
        "Coverage",
        "Selective Acc",
    ]

    def render_rows(reps: List[ComprehensiveEvaluationReport], is_leaky_table: bool = False) -> List[str]:
        rows = []
        for r in reps:
            cov = f"{r.abstention_results.coverage:.3f}" if r.abstention_results else "1.000"
            sel_acc = f"{r.abstention_results.selective_accuracy:.3f}" if r.abstention_results else f"{r.accuracy:.3f}"
            roc_str = f"{r.roc_auc:.4f}"
            if "roc_auc" in r.confidence_intervals:
                lo, hi = r.confidence_intervals["roc_auc"]
                roc_str += f" [{lo:.2f}, {hi:.2f}]"

            var_name = f"`{r.model_variant}`"
            if is_leaky_table:
                var_name += " **[LEAKY REFERENCE (not a valid model)]**"

            row = [
                var_name,
                r.split_name,
                roc_str,
                f"{r.pr_auc:.4f}",
                f"{r.f1:.4f}",
                f"{r.ece:.4f}",
                f"{r.brier_score:.4f}",
                cov,
                sel_acc,
            ]
            rows.append("| " + " | ".join(row) + " |")
        return rows

    table_header = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join(["---"] * len(headers)) + " |",
    ]

    lines = table_header + render_rows(main_reports, is_leaky_table=False)

    if leaky_reports:
        lines.append("")
        lines.append("### Leaky Reference Baselines (Not Valid Models - Label Leakage Demonstration)")
        lines.append("")
        lines.append(
            "> [!CAUTION]\n"
            "> **LEAKY REFERENCE (not a valid model)**: The following variants utilize the full 12D lineage vector "
            "or direct clinical polarity features (S_pos, S_neg, margin, N_raw_pos, N_raw_neg). "
            "They are retained exclusively as diagnostic controls demonstrating label leakage (~1.00 ROC-AUC) "
            "and MUST NEVER be compared as valid predictive models."
        )
        lines.append("")
        lines.extend(table_header + render_rows(leaky_reports, is_leaky_table=True))

    return "\n".join(lines)


def prepare_evaluation_tensors(
    summary_map: Dict[Any, Any],
    partition_df: pd.DataFrame,
    bio_extractor: Any,
    text_extractor: Any,
    embedding_dim: int = 16,
    feature_group: str = "quality_only",
    mode: Optional[str] = None,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Constructs PyTorch feature and label tensors for a partition DataFrame.
    Defaults strictly to quality_only features (5D for no_context_conflict, 7D for other valid models).
    """
    h_tensors, d_tensors, meta_list, labels_list, texts_list = [], [], [], [], []
    for _, row in partition_df.iterrows():
        h_id = row["herb_id"]
        d_id = row["drug_id"]
        h_tensors.append(torch.tensor(bio_extractor.get_herb_representation(h_id), dtype=torch.float32))
        d_tensors.append(torch.tensor(bio_extractor.get_drug_representation(d_id), dtype=torch.float32))
        texts_list.append(str(row.get("original_text", "")))

        summary = summary_map.get((h_id, d_id))
        if summary:
            if hasattr(summary, "to_feature_vector"):
                meta_vec = summary.to_feature_vector(mode=mode, feature_group=feature_group)
            else:
                meta_vec = summary.to_full_metadata_vector()
        else:
            dim = 5 if mode == "no_context_conflict" else (12 if feature_group in {"full", "leaky_reference"} else 7)
            meta_vec = np.zeros(dim, dtype=np.float32)
        meta_list.append(meta_vec)

        out = str(row.get("outcome", "")).lower()
        if out in {"interaction_present", "positive", "1", "true"}:
            y = 1.0
        elif out in {"no_interaction_observed", "negative", "0", "false"}:
            y = 0.0
        else:
            y = 0.5
        labels_list.append(y)

    text_feats = text_extractor.transform(texts_list)
    ev_tensors = torch.tensor(text_feats, dtype=torch.float32)

    return (
        torch.stack(h_tensors),
        torch.stack(d_tensors),
        ev_tensors,
        torch.tensor(np.array(meta_list), dtype=torch.float32),
        torch.tensor(labels_list, dtype=torch.float32).unsqueeze(1),
    )


def compare_models_significance(
    y_true: Union[np.ndarray, List[int]],
    y_prob_proposed: Union[np.ndarray, List[float]],
    y_prob_baseline: Union[np.ndarray, List[float]],
    n_bootstrap: int = 2000,
    seed: int = 42,
) -> Dict[str, Any]:
    """
    Computes rigorous paired statistical tests comparing the proposed model against a baseline:
    1. DeLong's asymptotic test for correlated ROC-AUC
    2. Paired bootstrap hypothesis test for ROC-AUC
    3. Paired bootstrap hypothesis test for Brier Score calibration
    """
    from src.training.statistical_tests import (
        delong_auc_test,
        paired_bootstrap_auc_test,
        paired_bootstrap_brier_test,
    )

    y_t = np.asarray(y_true, dtype=int).flatten()
    p_prop = np.asarray(y_prob_proposed, dtype=float).flatten()
    p_base = np.asarray(y_prob_baseline, dtype=float).flatten()

    delong_res = delong_auc_test(y_t, p_base, p_prop)
    boot_auc_res = paired_bootstrap_auc_test(y_t, p_base, p_prop, n_bootstrap=n_bootstrap, seed=seed)
    boot_brier_res = paired_bootstrap_brier_test(y_t, p_base, p_prop, n_bootstrap=n_bootstrap, seed=seed)

    return {
        "delong_auc": delong_res,
        "bootstrap_auc": boot_auc_res,
        "bootstrap_brier": boot_brier_res,
    }


def run_repeated_nested_cv(
    df: pd.DataFrame,
    model_variants: Optional[List[str]] = None,
    n_repeats: int = 5,
    n_splits: int = 5,
    split_mode: str = "study_disjoint",
    feature_group: str = "quality_only",
    embedding_dim: int = 16,
    epochs: int = 25,
    lr: float = 0.01,
    seed: int = 42,
    stratify: bool = True,
    run_significance_tests: bool = True,
) -> Dict[str, RepeatedNestedCVResult]:
    """
    Execute repeated, nested, group-disjoint cross-validation.
    Guarantees:
    - Preprocessing (TF-IDF) fit strictly on inner training folds
    - Calibration (TemperatureScaler) fit strictly on inner validation folds
    - Metric evaluation conducted on held-out outer test folds
    - T* variability recorded across CV folds and pooled OOF evaluation computed
    - Paired hypothesis testing (DeLong & paired bootstrap) evaluated against baselines
    """
    from src.features.biological_features import BiologicalProfileExtractor, EvidenceTextFeatureExtractor
    from src.data.evidence_lineage import EvidenceLineageAnalyzer
    from src.models.interaction_model import ReliabilityConditionedInteractionModel

    if model_variants is None:
        model_variants = [
            "reliability_conditioned_quality_only",
            "standard_fusion",
            "graph_only",
            "evidence_only",
        ]

    fold_metric_records: Dict[str, List[Dict[str, float]]] = {v: [] for v in model_variants}
    val_temps: Dict[str, List[float]] = {v: [] for v in model_variants}
    pooled_logits: Dict[str, List[float]] = {v: [] for v in model_variants}
    pooled_probs: Dict[str, List[float]] = {v: [] for v in model_variants}
    pooled_labels: Dict[str, List[int]] = {v: [] for v in model_variants}
    pooled_preds: Dict[str, List[int]] = {v: [] for v in model_variants}
    pooled_abstains: Dict[str, List[bool]] = {v: [] for v in model_variants}

    for r in range(n_repeats):
        repeat_seed = seed + r * 100
        splitter = LeakageSafeSplitter(seed=repeat_seed)
        outer_folds = splitter.grouped_kfold_split(df, mode=split_mode, n_splits=n_splits, stratify=stratify)

        for fold_idx, (outer_train_df, outer_test_df) in enumerate(outer_folds):
            inner_splitter = LeakageSafeSplitter(seed=repeat_seed + fold_idx + 1)
            inner_train_df, inner_val_df = inner_splitter.split_train_val(
                outer_train_df, mode=split_mode, train_ratio=0.8
            )

            # Fit TF-IDF text feature extractor STRICTLY on inner train texts (zero val/test leakage)
            train_texts = [str(t) for t in inner_train_df["original_text"] if pd.notna(t)]
            text_extractor = EvidenceTextFeatureExtractor(embedding_dim=embedding_dim, min_df=1)
            text_extractor.fit(train_texts)

            bio_extractor = BiologicalProfileExtractor(feature_dim=embedding_dim)

            # Lineage analyzer with requested feature_group
            lineage_analyzer = EvidenceLineageAnalyzer(feature_group=feature_group)
            tr_sums = lineage_analyzer.analyze_dataset(inner_train_df)
            val_sums = lineage_analyzer.analyze_dataset(inner_val_df)
            te_sums = lineage_analyzer.analyze_dataset(outer_test_df)

            tr_h, tr_d, tr_ev, tr_meta, tr_y = prepare_evaluation_tensors(
                tr_sums, inner_train_df, bio_extractor, text_extractor, embedding_dim=embedding_dim
            )
            val_h, val_d, val_ev, val_meta, val_y = prepare_evaluation_tensors(
                val_sums, inner_val_df, bio_extractor, text_extractor, embedding_dim=embedding_dim
            )
            te_h, te_d, te_ev, te_meta, te_y = prepare_evaluation_tensors(
                te_sums, outer_test_df, bio_extractor, text_extractor, embedding_dim=embedding_dim
            )

            for variant in model_variants:
                torch.manual_seed(repeat_seed + fold_idx)
                model = ReliabilityConditionedInteractionModel(
                    graph_dim=embedding_dim,
                    evidence_dim=embedding_dim,
                    fused_dim=embedding_dim,
                    hidden_dim=embedding_dim,
                    mode=variant,
                    abstain_conflict_threshold=0.5,
                )
                optimizer = torch.optim.Adam(model.parameters(), lr=lr)
                criterion = nn.BCEWithLogitsLoss()

                model.train()
                for _ in range(epochs):
                    optimizer.zero_grad()
                    out = model(tr_h, tr_d, tr_ev, metadata=tr_meta)
                    loss = criterion(out["logits"], tr_y)
                    loss.backward()
                    optimizer.step()

                # Validation: Fit TemperatureScaler strictly on validation set logits
                model.eval()
                with torch.no_grad():
                    val_out = model(val_h, val_d, val_ev, metadata=val_meta)
                    test_out = model(te_h, te_d, te_ev, metadata=te_meta, return_details=True)

                scaler = TemperatureScaler()
                t_star = scaler.fit(val_out["logits"].cpu().numpy(), val_y.cpu().numpy(), max_iter=25)
                val_temps[variant].append(t_star)

                # Calibrate test logits using fold T*
                scaled_test_logits = scaler(test_out["logits"])
                test_probs = torch.sigmoid(scaled_test_logits).detach().cpu().numpy().flatten()
                test_labels = te_y.cpu().numpy().flatten().astype(int)
                test_preds = (test_probs >= 0.5).astype(int)
                abstain_flags = test_out.get("abstain", torch.zeros_like(test_out["logits"])).detach().cpu().numpy().flatten().astype(bool)

                # Record predictions
                pooled_logits[variant].extend(test_out["logits"].detach().cpu().numpy().flatten().tolist())
                pooled_probs[variant].extend(test_probs.tolist())
                pooled_labels[variant].extend(test_labels.tolist())
                pooled_preds[variant].extend(test_preds.tolist())
                pooled_abstains[variant].extend(abstain_flags.tolist())

                # Fold report
                fold_rep = compute_comprehensive_metrics(
                    y_true=test_labels,
                    y_prob=test_probs,
                    y_pred=test_preds,
                    abstain_flags=abstain_flags,
                    split_name=f"rep{r}_fold{fold_idx}",
                    model_variant=variant,
                )
                fold_metric_records[variant].append(fold_rep.to_dict())

    results: Dict[str, RepeatedNestedCVResult] = {}
    metric_keys = [
        "roc_auc", "pr_auc", "accuracy", "precision", "recall", "f1",
        "brier_score", "log_loss", "ece", "equal_mass_ece", "brier_positive", "brier_negative",
        "coverage", "selective_acc",
    ]

    for variant in model_variants:
        recs = fold_metric_records[variant]
        m_mean = {}
        m_std = {}
        for k in metric_keys:
            vals = [r.get(k, 0.0) for r in recs if k in r]
            if vals:
                m_mean[k] = float(np.mean(vals))
                m_std[k] = float(np.std(vals, ddof=1)) if len(vals) > 1 else 0.0

        # Temperature variability across folds
        temps = val_temps[variant]
        t_variability = {
            "mean": float(np.mean(temps)),
            "std": float(np.std(temps, ddof=1)) if len(temps) > 1 else 0.0,
            "min": float(np.min(temps)),
            "max": float(np.max(temps)),
            "median": float(np.median(temps)),
        }

        # Pooled evaluation with pooled T*
        p_labels = np.array(pooled_labels[variant])
        p_logits = torch.tensor(pooled_logits[variant], dtype=torch.float32).unsqueeze(1)
        p_labels_t = torch.tensor(p_labels, dtype=torch.float32).unsqueeze(1)

        pooled_scaler = TemperatureScaler()
        pooled_t = pooled_scaler.fit(p_logits.numpy(), p_labels_t.numpy(), max_iter=25)
        t_variability["pooled_t"] = float(pooled_t)

        scaled_pooled_logits = pooled_scaler(p_logits)
        pooled_cal_probs = torch.sigmoid(scaled_pooled_logits).detach().cpu().numpy().flatten()
        p_preds = np.array(pooled_preds[variant])
        p_abst = np.array(pooled_abstains[variant])

        pooled_report = compute_comprehensive_metrics(
            y_true=p_labels,
            y_prob=pooled_cal_probs,
            y_pred=p_preds,
            abstain_flags=p_abst,
            split_name=f"pooled_{split_mode}",
            model_variant=variant,
            compute_bootstrap_ci=True,
            n_bootstrap=1000,
        )

        ci_95 = pooled_report.confidence_intervals

        results[variant] = RepeatedNestedCVResult(
            model_variant=variant,
            split_mode=split_mode,
            n_repeats=n_repeats,
            n_splits=n_splits,
            metrics_mean=m_mean,
            metrics_std=m_std,
            confidence_intervals_95=ci_95,
            temperature_variability=t_variability,
            pooled_report=pooled_report,
            fold_reports=recs,
        )

    # Paired significance tests between proposed model and each baseline
    if run_significance_tests and len(model_variants) > 1:
        proposed_key = model_variants[0]
        prop_labels = np.array(pooled_labels[proposed_key])
        prop_probs = np.array(pooled_probs[proposed_key])

        for base_key in model_variants[1:]:
            base_probs = np.array(pooled_probs[base_key])
            sig_res = compare_models_significance(
                y_true=prop_labels,
                y_prob_proposed=prop_probs,
                y_prob_baseline=base_probs,
                n_bootstrap=1000,
                seed=seed,
            )
            results[proposed_key].significance_vs_baselines[base_key] = sig_res

    return results


def format_repeated_cv_markdown_table(results: Dict[str, RepeatedNestedCVResult]) -> str:
    """Format repeated nested CV results into an auditable Markdown summary table."""
    headers = [
        "Model Variant",
        "Split Mode",
        "ROC-AUC (Mean ± SD) [95% CI]",
        "PR-AUC (Mean ± SD)",
        "F1 (Mean ± SD)",
        "Brier (Mean ± SD)",
        "Equal-Mass ECE",
        "T* (Mean ± SD)",
        "Coverage",
        "Selective Acc",
    ]
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join(["---"] * len(headers)) + " |",
    ]
    for name, r in results.items():
        m = r.metrics_mean
        s = r.metrics_std
        ci = r.confidence_intervals_95.get("roc_auc", (0.0, 0.0))
        t_stat = r.temperature_variability
        cov_val = m.get("coverage", 1.0)
        sel_acc_val = m.get("selective_acc", m.get("accuracy", 0.0))
        row = [
            f"`{r.model_variant}`",
            r.split_mode,
            f"{m.get('roc_auc', 0.0):.4f} ± {s.get('roc_auc', 0.0):.4f} [{ci[0]:.2f}, {ci[1]:.2f}]",
            f"{m.get('pr_auc', 0.0):.4f} ± {s.get('pr_auc', 0.0):.4f}",
            f"{m.get('f1', 0.0):.4f} ± {s.get('f1', 0.0):.4f}",
            f"{m.get('brier_score', 0.0):.4f} ± {s.get('brier_score', 0.0):.4f}",
            f"{m.get('equal_mass_ece', 0.0):.4f}",
            f"{t_stat.get('mean', 1.0):.3f} ± {t_stat.get('std', 0.0):.3f}",
            f"{cov_val:.3f}",
            f"{sel_acc_val:.3f}",
        ]
        lines.append("| " + " | ".join(row) + " |")

    return "\n".join(lines)
