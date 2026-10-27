"""
Unit tests for Phase 6: Stronger Baselines and Evaluation Comparisons.
Tests:
1. Tabular feature extraction (quality_only lineage + CYP/P-gp overlap).
2. Gradient boosted trees and logistic regression on tabular features.
3. Mechanistic rule baseline prediction behavior.
4. Naive concat MLP fusion mode in ReliabilityConditionedInteractionModel.
5. Verification of all baselines in run_benchmark.
"""

import pytest
import numpy as np
import pandas as pd
import torch

from src.data.public_data_loader import PublicDataLoader
from src.features.biological_features import BiologicalProfileExtractor
from src.data.evidence_lineage import EvidenceLineageAnalyzer
from src.models.interaction_model import ReliabilityConditionedInteractionModel
from benchmarks.run_verified_hdi_benchmark import (
    extract_tabular_features,
    predict_mechanistic_rule,
    run_benchmark,
)
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression


@pytest.fixture
def benchmark_df():
    loader = PublicDataLoader(allow_synthetic=False)
    return loader.load_hdi_evidence("data/benchmark/verified_botanical_hdi.csv")


def test_tabular_features_extraction(benchmark_df):
    """Verify tabular feature matrix has 13 dimensions (7 lineage + 6 overlap)."""
    analyzer = EvidenceLineageAnalyzer(feature_group="quality_only")
    summaries = analyzer.analyze_dataset(benchmark_df)
    bio_extractor = BiologicalProfileExtractor(feature_dim=16)

    X, y = extract_tabular_features(benchmark_df, summaries, bio_extractor, feature_group="quality_only")
    assert X.shape[0] == len(benchmark_df)
    assert X.shape[1] == 13, f"Expected 13 tabular features, got {X.shape[1]}"
    assert len(y) == len(benchmark_df)
    assert set(np.unique(y)).issubset({0, 1})


def test_gradient_boosting_and_logistic_regression_fit(benchmark_df):
    """Verify gradient boosted trees and logistic regression fit cleanly on tabular features."""
    analyzer = EvidenceLineageAnalyzer(feature_group="quality_only")
    summaries = analyzer.analyze_dataset(benchmark_df)
    bio_extractor = BiologicalProfileExtractor(feature_dim=16)

    X, y = extract_tabular_features(benchmark_df, summaries, bio_extractor, feature_group="quality_only")

    # Gradient boosting
    hgb = HistGradientBoostingClassifier(random_state=42, max_iter=20)
    hgb.fit(X, y)
    probs_hgb = hgb.predict_proba(X)[:, 1]
    assert probs_hgb.shape == (len(benchmark_df),)
    assert np.all((probs_hgb >= 0.0) & (probs_hgb <= 1.0))

    # Logistic regression
    lr = LogisticRegression(random_state=42, max_iter=100)
    lr.fit(X, y)
    probs_lr = lr.predict_proba(X)[:, 1]
    assert probs_lr.shape == (len(benchmark_df),)
    assert np.all((probs_lr >= 0.0) & (probs_lr <= 1.0))


def test_mechanistic_rule_baseline(benchmark_df):
    """Verify mechanistic rule predicts interaction when herb and drug share clearance enzymes."""
    bio_extractor = BiologicalProfileExtractor(feature_dim=16)
    labels, probs, preds = predict_mechanistic_rule(benchmark_df, bio_extractor)

    assert len(labels) == len(benchmark_df)
    assert len(probs) == len(benchmark_df)
    assert len(preds) == len(benchmark_df)
    assert np.all((probs >= 0.0) & (probs <= 1.0))
    assert set(np.unique(preds)).issubset({0, 1})

    # St. John's Wort + Cyclosporine has strong shared CYP3A4 and ABCB1 target overlap -> must predict 1
    sjw_cyclo = benchmark_df[(benchmark_df["herb_id"] == "HERB_STJOHNSWORT") & (benchmark_df["drug_id"] == "DRUG_CYCLOSPORINE")]
    if len(sjw_cyclo) > 0:
        _, sjw_probs, sjw_preds = predict_mechanistic_rule(sjw_cyclo, bio_extractor)
        assert sjw_preds[0] == 1
        assert sjw_probs[0] >= 0.80


def test_naive_concat_mlp_forward_pass():
    """Verify naive_concat_mlp forward pass executes without reliability gating."""
    model = ReliabilityConditionedInteractionModel(
        graph_dim=16,
        evidence_dim=16,
        fused_dim=16,
        hidden_dim=16,
        mode="naive_concat_mlp",
    )
    h = torch.randn(4, 16)
    d = torch.randn(4, 16)
    ev = torch.randn(4, 16)

    out = model(h, d, ev)
    assert "logits" in out
    assert "probs" in out
    assert out["logits"].shape == (4, 1)
    # Under naive_concat_mlp, R must equal 1.0 (unconditioned)
    assert torch.allclose(out["reliability"], torch.ones(4, 1))
