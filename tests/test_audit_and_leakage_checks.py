"""
Phase 0 Audit and Leakage Detection Test Suite.
Encodes empirical findings from reports/audit.md:
1. Verifies that train and test partitions in study_disjoint split have zero pair overlap.
2. Quantifies entity overlap (herbs and drugs).
3. Verifies that S_pos/S_neg/margin encode label information when computed on labeled datasets.
4. Verifies the behavior of missing-literature vectors.
"""

import pytest
import numpy as np
import pandas as pd
import torch

from src.data.public_data_loader import PublicDataLoader
from src.training.evaluation import LeakageSafeSplitter
from src.data.evidence_lineage import (
    EvidenceRecord,
    EvidenceLineageAnalyzer,
    build_pair_summaries_from_dataframe,
)
from src.models.reliability_scorer import ReliabilityScorer
from src.models.interaction_model import ReliabilityConditionedInteractionModel


@pytest.fixture
def real_benchmark_df():
    loader = PublicDataLoader(allow_synthetic=False)
    return loader.load_hdi_evidence("data/benchmark/verified_botanical_hdi.csv")


def test_audit_study_disjoint_zero_pair_overlap(real_benchmark_df):
    """
    Audit Finding: Ensure that in the current study_disjoint split,
    there is strictly ZERO pair overlap between train and test partitions.
    """
    splitter = LeakageSafeSplitter(seed=42)
    train_df, val_df, test_df = splitter.split_study_disjoint(real_benchmark_df, train_ratio=0.6, val_ratio=0.2)
    
    train_pairs = set(zip(train_df["herb_id"], train_df["drug_id"]))
    test_pairs = set(zip(test_df["herb_id"], test_df["drug_id"]))
    
    overlap = train_pairs.intersection(test_pairs)
    assert len(overlap) == 0, f"Found unexpected pair overlap between train and test: {overlap}"


def test_audit_study_disjoint_entity_overlap(real_benchmark_df):
    """
    Audit Finding: In study_disjoint mode, all 13 herbs in the test set
    exist in the train set (100% herb overlap), meaning it is not herb-disjoint.
    """
    splitter = LeakageSafeSplitter(seed=42)
    train_df, _, test_df = splitter.split_study_disjoint(real_benchmark_df, train_ratio=0.6, val_ratio=0.2)
    
    train_herbs = set(train_df["herb_id"])
    test_herbs = set(test_df["herb_id"])
    
    # All test herbs are present in train (13/13)
    assert test_herbs.issubset(train_herbs)
    
    train_drugs = set(train_df["drug_id"])
    test_drugs = set(test_df["drug_id"])
    unseen_drugs = test_drugs - train_drugs
    # Exactly 5 drugs in test are unseen in train
    assert len(unseen_drugs) == 5


def test_audit_label_adjacency_in_lineage_features():
    """
    Audit Finding: S_pos, S_neg, and margin directly mirror the outcome label.
    A record with 'interaction_present' produces S_pos > 0 and S_neg == 0.
    A record with 'no_interaction_observed' produces S_pos == 0 and S_neg > 0.
    """
    rec_pos = EvidenceRecord(
        herb_id="H1", drug_id="D1", outcome="interaction_present",
        evidence_type="clinical", source_id="S1", study_id="ST1",
        extraction_confidence=1.0,
    )
    rec_neg = EvidenceRecord(
        herb_id="H2", drug_id="D2", outcome="no_interaction_observed",
        evidence_type="clinical", source_id="S2", study_id="ST2",
        extraction_confidence=1.0,
    )
    
    analyzer = EvidenceLineageAnalyzer()
    sum_pos = analyzer.aggregate_pair_evidence("H1", "D1", [rec_pos])
    sum_neg = analyzer.aggregate_pair_evidence("H2", "D2", [rec_neg])
    
    # S_pos and S_neg strictly separate by outcome label
    assert sum_pos.independent_positive_strength > 0.0
    assert sum_pos.independent_negative_strength == 0.0
    assert sum_pos.net_evidence_margin > 0.0
    
    assert sum_neg.independent_positive_strength == 0.0
    assert sum_neg.independent_negative_strength > 0.0
    assert sum_neg.net_evidence_margin < 0.0


def test_audit_dag_clustering_derivatives():
    """
    Audit Finding: When cited_study_ids connects two studies,
    the DAG clusters them into one connected component, discounting the derivative.
    """
    rec_primary = EvidenceRecord(
        herb_id="H1", drug_id="D1", outcome="interaction_present",
        evidence_type="clinical", source_id="P1", study_id="STUDY_1",
        cited_study_ids=[], extraction_confidence=1.0,
    )
    rec_derivative = EvidenceRecord(
        herb_id="H1", drug_id="D1", outcome="interaction_present",
        evidence_type="review", source_id="P2", study_id="STUDY_2",
        cited_study_ids=["STUDY_1"], extraction_confidence=1.0,
    )
    
    analyzer = EvidenceLineageAnalyzer(echo_discount_factor=0.15)
    summary = analyzer.aggregate_pair_evidence("H1", "D1", [rec_primary, rec_derivative])
    
    # 2 records clustered into 1 independent cluster
    assert summary.total_records == 2
    assert summary.num_independent_positive_clusters == 1
    # Derivative discounted: primary (1.0) + derivative (0.15 * 0.20)
    assert summary.independent_positive_strength < 2.0


def test_target_index_hygiene_raises_key_error():
    """
    Code Hygiene Audit: Verify that TARGET_TO_IDX raises KeyError on invalid
    or uncanonical target names, preventing silent fallback bugs.
    """
    from src.features.biological_features import TARGET_TO_IDX

    with pytest.raises(KeyError):
        _ = TARGET_TO_IDX["CYP99A99"]

    with pytest.raises(KeyError):
        _ = TARGET_TO_IDX["UNKNOWN_RECEPTOR"]

    # Verify canonical targets resolve without error
    canonical_keys = ["CYP3A4", "ABCB1", "CYP2C9", "CYP1A2", "CYP2E1"]
    for k in canonical_keys:
        assert k in TARGET_TO_IDX
        assert isinstance(TARGET_TO_IDX[k], int)


def test_cv_helper_disjoint_group_assertions(real_benchmark_df):
    """
    Code Hygiene Audit: Explicit assertion that train and test group sets
    are strictly disjoint across all folds in grouped_kfold_split.
    """
    splitter = LeakageSafeSplitter(seed=42)

    # 1. Herb disjoint
    herb_folds = splitter.grouped_kfold_split(real_benchmark_df, mode="herb_disjoint", n_splits=5)
    for fold_idx, (tr_df, te_df) in enumerate(herb_folds):
        train_herbs = set(tr_df["herb_id"])
        test_herbs = set(te_df["herb_id"])
        intersection = train_herbs & test_herbs
        assert len(intersection) == 0, f"Herb fold {fold_idx} has overlapping herbs: {intersection}"

    # 2. Drug disjoint
    drug_folds = splitter.grouped_kfold_split(real_benchmark_df, mode="drug_disjoint", n_splits=5)
    for fold_idx, (tr_df, te_df) in enumerate(drug_folds):
        train_drugs = set(tr_df["drug_id"])
        test_drugs = set(te_df["drug_id"])
        intersection = train_drugs & test_drugs
        assert len(intersection) == 0, f"Drug fold {fold_idx} has overlapping drugs: {intersection}"

    # 3. Pair disjoint
    pair_folds = splitter.grouped_kfold_split(real_benchmark_df, mode="pair_disjoint", n_splits=5)
    for fold_idx, (tr_df, te_df) in enumerate(pair_folds):
        train_pairs = set(zip(tr_df["herb_id"], tr_df["drug_id"]))
        test_pairs = set(zip(te_df["herb_id"], te_df["drug_id"]))
        intersection = train_pairs & test_pairs
        assert len(intersection) == 0, f"Pair fold {fold_idx} has overlapping pairs: {intersection}"

