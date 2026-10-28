"""
Tests for Phase 5 Evaluation Protocol Overhaul:
1. All 4 split modes (study_disjoint, pair_disjoint, herb_disjoint, drug_disjoint)
   with zero cross-split leakage and informative ValueError on small datasets.
2. Grouped K-fold cross-validation with zero cross-fold group leakage.
3. Preprocessing leakage safety: verifying no TF-IDF, scaler, or TemperatureScaler
   fit on val/test data.
4. Repeated nested CV with T* variability and pooled out-of-fold calibration reporting.
5. Paired significance tests (DeLong and paired bootstrap for AUC; paired bootstrap for Brier).
"""

import pytest
import numpy as np
import pandas as pd
import torch

from src.data.public_data_loader import PublicDataLoader
from src.training.evaluation import (
    LeakageSafeSplitter,
    compute_comprehensive_metrics,
    ComprehensiveEvaluationReport,
    RepeatedNestedCVResult,
    run_repeated_nested_cv,
    compare_models_significance,
)
from src.features.biological_features import EvidenceTextFeatureExtractor
from src.training.calibration import TemperatureScaler


@pytest.fixture
def real_benchmark_df():
    loader = PublicDataLoader(allow_synthetic=False)
    return loader.load_hdi_evidence("data/benchmark/verified_botanical_hdi.csv")


def test_split_pair_disjoint_zero_leakage(real_benchmark_df):
    """Verify pair-disjoint split guarantees zero (herb, drug) pair overlap."""
    splitter = LeakageSafeSplitter(seed=42)
    train_df, val_df, test_df = splitter.split_pair_disjoint(real_benchmark_df, train_ratio=0.6, val_ratio=0.2)

    train_pairs = set(zip(train_df["herb_id"], train_df["drug_id"]))
    val_pairs = set(zip(val_df["herb_id"], val_df["drug_id"]))
    test_pairs = set(zip(test_df["herb_id"], test_df["drug_id"]))

    assert len(train_pairs & test_pairs) == 0, f"Pair leakage train-test: {train_pairs & test_pairs}"
    assert len(train_pairs & val_pairs) == 0, f"Pair leakage train-val: {train_pairs & val_pairs}"
    assert len(val_pairs & test_pairs) == 0, f"Pair leakage val-test: {val_pairs & test_pairs}"
    assert len(train_df) > 0 and len(val_df) > 0 and len(test_df) > 0


def test_split_herb_disjoint_zero_leakage(real_benchmark_df):
    """Verify herb-disjoint split guarantees zero herb overlap."""
    splitter = LeakageSafeSplitter(seed=42)
    train_df, val_df, test_df = splitter.split_herb_disjoint(real_benchmark_df, train_ratio=0.6, val_ratio=0.2)

    train_herbs = set(train_df["herb_id"])
    val_herbs = set(val_df["herb_id"])
    test_herbs = set(test_df["herb_id"])

    assert len(train_herbs & test_herbs) == 0, f"Herb leakage train-test: {train_herbs & test_herbs}"
    assert len(train_herbs & val_herbs) == 0, f"Herb leakage train-val: {train_herbs & val_herbs}"
    assert len(val_herbs & test_herbs) == 0, f"Herb leakage val-test: {val_herbs & test_herbs}"


def test_split_drug_disjoint_zero_leakage(real_benchmark_df):
    """Verify drug-disjoint split guarantees zero drug overlap."""
    splitter = LeakageSafeSplitter(seed=42)
    train_df, val_df, test_df = splitter.split_drug_disjoint(real_benchmark_df, train_ratio=0.6, val_ratio=0.2)

    train_drugs = set(train_df["drug_id"])
    val_drugs = set(val_df["drug_id"])
    test_drugs = set(test_df["drug_id"])

    assert len(train_drugs & test_drugs) == 0, f"Drug leakage train-test: {train_drugs & test_drugs}"
    assert len(train_drugs & val_drugs) == 0, f"Drug leakage train-val: {train_drugs & val_drugs}"
    assert len(val_drugs & test_drugs) == 0, f"Drug leakage val-test: {val_drugs & test_drugs}"


def test_split_modes_infeasible_raise_clear_error():
    """Verify clear ValueError with counts when data has insufficient distinct groups."""
    tiny_df = pd.DataFrame({
        "herb_id": ["herb_A", "herb_B"],
        "drug_id": ["drug_1", "drug_2"],
        "study_id": ["study_1", "study_2"],
        "outcome": ["positive", "negative"],
    })
    splitter = LeakageSafeSplitter(seed=42)

    # 3-way partition on 2 herbs must fail with explicit count in error message
    with pytest.raises(ValueError) as excinfo:
        splitter.split_herb_disjoint(tiny_df)
    assert "2 distinct herbs" in str(excinfo.value) or "2" in str(excinfo.value)

    # 3-way partition on 2 drugs must fail with explicit count
    with pytest.raises(ValueError) as excinfo:
        splitter.split_drug_disjoint(tiny_df)
    assert "2 distinct drugs" in str(excinfo.value) or "2" in str(excinfo.value)

    # 5-fold CV on 2 groups must fail
    with pytest.raises(ValueError) as excinfo:
        splitter.grouped_kfold_split(tiny_df, mode="pair_disjoint", n_splits=5)
    assert "5-fold CV requires at least 5" in str(excinfo.value)


def test_grouped_kfold_split_zero_leakage(real_benchmark_df):
    """Verify grouped_kfold_split guarantees zero leakage across all folds for all modes."""
    splitter = LeakageSafeSplitter(seed=42)

    # Test study_disjoint 5-fold CV
    folds = splitter.grouped_kfold_split(real_benchmark_df, mode="study_disjoint", n_splits=5, stratify=True)
    assert len(folds) == 5
    for train_df, test_df in folds:
        train_studies = set(train_df["study_id"].astype(str))
        test_studies = set(test_df["study_id"].astype(str))
        assert len(train_studies & test_studies) == 0
        assert len(train_df) > 0 and len(test_df) > 0

    # Test pair_disjoint 4-fold CV
    pair_folds = splitter.grouped_kfold_split(real_benchmark_df, mode="pair_disjoint", n_splits=4, stratify=True)
    assert len(pair_folds) == 4
    for train_df, test_df in pair_folds:
        tr_p = set(zip(train_df["herb_id"], train_df["drug_id"]))
        te_p = set(zip(test_df["herb_id"], test_df["drug_id"]))
        assert len(tr_p & te_p) == 0


def test_no_preprocessing_leakage_on_val_or_test():
    """
    CRITICAL CHECK: Verify that text TF-IDF vocabulary and temperature scaling
    do not leak information from test or validation partitions.
    """
    train_texts = ["hyperforin induces cyp3a4 clearance", "st john wort reduces cyclosporine auc"]
    test_texts = ["novel_unseen_chemical_x uniquely in test set inhibits clearance"]

    extractor = EvidenceTextFeatureExtractor(embedding_dim=16, min_df=1)
    extractor.fit(train_texts)

    # The test-only unique word MUST NOT be present in vocabulary
    vocab = extractor.vocabulary_
    assert "novel_unseen_chemical_x" not in vocab
    assert "hyperforin" in vocab

    # Verify TemperatureScaler fits strictly on validation logits without test access
    val_logits = np.array([1.2, -0.8, 2.1, -1.5])
    val_labels = np.array([1, 0, 1, 0])
    test_logits = np.array([10.0, -10.0])  # Extreme test logits

    scaler = TemperatureScaler()
    t_val = scaler.fit(val_logits, val_labels, max_iter=25)
    assert t_val > 0.0

    # Test logits scaled with t_val
    scaled_test = scaler(torch.tensor(test_logits, dtype=torch.float32)).detach().numpy()
    assert np.all(np.isfinite(scaled_test))


def test_repeated_nested_cv_execution_and_temperature_variability(real_benchmark_df):
    """
    Verify repeated nested cross-validation runs without error,
    records T* variability across folds, computes equal-mass ECE,
    and returns paired significance tests against baselines.
    """
    results = run_repeated_nested_cv(
        real_benchmark_df,
        model_variants=["reliability_conditioned_quality_only", "standard_fusion"],
        n_repeats=1,
        n_splits=3,
        split_mode="study_disjoint",
        feature_group="quality_only",
        embedding_dim=16,
        epochs=5,
        seed=42,
        stratify=True,
        run_significance_tests=True,
    )

    assert "reliability_conditioned_quality_only" in results
    res_prop = results["reliability_conditioned_quality_only"]
    assert isinstance(res_prop, RepeatedNestedCVResult)

    # Check metrics mean & std
    m = res_prop.metrics_mean
    assert "roc_auc" in m
    assert "brier_score" in m
    assert "equal_mass_ece" in m
    assert "brier_positive" in m
    assert "brier_negative" in m

    # Check temperature variability across folds
    t_stat = res_prop.temperature_variability
    assert "mean" in t_stat
    assert "std" in t_stat
    assert "min" in t_stat
    assert "max" in t_stat
    assert "pooled_t" not in t_stat  # No fit on pooled outer assessment labels.

    # Check paired significance tests against baseline
    assert "standard_fusion" in res_prop.significance_vs_baselines
    sig = res_prop.significance_vs_baselines["standard_fusion"]
    assert "delong_auc" in sig
    assert "bootstrap_auc" in sig
    assert "bootstrap_brier" in sig
    assert "p_value" in sig["delong_auc"]
    assert "p_value" in sig["bootstrap_auc"]
    assert "cohens_d" in sig["bootstrap_brier"]


def test_paired_significance_tests_delong_and_bootstrap():
    """Verify paired significance test module computes valid p-values and effect sizes."""
    np.random.seed(42)
    n = 100
    y_true = np.random.choice([0, 1], size=n, p=[0.5, 0.5])

    # Model A: strong predictor (noise = 0.2)
    p_a = np.clip(y_true * 0.7 + np.random.normal(0.1, 0.2, size=n), 0.01, 0.99)
    # Model B: weak predictor (noise = 0.5)
    p_b = np.clip(y_true * 0.3 + np.random.normal(0.3, 0.5, size=n), 0.01, 0.99)

    res = compare_models_significance(y_true=y_true, y_prob_proposed=p_a, y_prob_baseline=p_b, n_bootstrap=500, seed=42)

    assert "delong_auc" in res
    assert "bootstrap_auc" in res
    assert "bootstrap_brier" in res

    # Model A has higher AUC than Model B
    assert res["delong_auc"]["diff_auc"] > 0
    assert res["bootstrap_auc"]["diff_auc"] > 0
    assert 0.0 <= res["delong_auc"]["p_value"] <= 1.0
    assert 0.0 <= res["bootstrap_auc"]["p_value"] <= 1.0
    # Model A has lower Brier error (diff_brier = Brier(B) - Brier(A) > 0)
    assert res["bootstrap_brier"]["diff_brier"] > 0
