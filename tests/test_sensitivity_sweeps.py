"""Unit tests verifying sensitivity analysis sweeps produce non-identical tensors.

Mandated by TASK 2:
Add a test that fails if two different sweep values produce byte-identical tensors
when they should differ.
"""

from pathlib import Path
import pytest
import torch
import pandas as pd
import numpy as np

from src.data.evidence_lineage import EvidenceLineageAnalyzer
from src.models.context_compatibility import (
    ContextCompatibilityScorer,
    SPECIES_PRIORS,
    PREPARATION_PRIORS,
    ROUTE_PRIORS,
)
from src.features.biological_features import BiologicalProfileExtractor, EvidenceTextFeatureExtractor
from benchmarks.sensitivity_analysis import prepare_partition_tensors


@pytest.fixture(scope="module")
def sample_benchmark_df():
    """Load benchmark dataset for tensor building tests."""
    df_path = Path("data/benchmark/verified_botanical_hdi.csv")
    assert df_path.exists(), f"Benchmark data file missing at {df_path}"
    df = pd.read_csv(df_path)
    return df.head(30)


@pytest.fixture(scope="module")
def feature_extractors(sample_benchmark_df):
    """Instantiate biological and text feature extractors."""
    bio_extractor = BiologicalProfileExtractor(feature_dim=16)
    train_texts = [str(t) for t in sample_benchmark_df["original_text"] if pd.notna(t)]
    text_extractor = EvidenceTextFeatureExtractor(embedding_dim=16, min_df=1)
    text_extractor.fit(train_texts)
    return bio_extractor, text_extractor


def test_context_weight_sweep_produces_distinct_tensors(sample_benchmark_df, feature_extractors):
    """Assert that scaling context compatibility weights produces non-identical metadata tensors."""
    bio_extractor, text_extractor = feature_extractors

    # 1. Low context weight scale (0.50x)
    scale_low = 0.50
    species_low = {k: min(1.0, max(0.05, v * scale_low)) for k, v in SPECIES_PRIORS.items()}
    prep_low = {k: min(1.0, max(0.05, v * scale_low)) for k, v in PREPARATION_PRIORS.items()}
    route_low = {k: min(1.0, max(0.05, v * scale_low)) for k, v in ROUTE_PRIORS.items()}
    scorer_low = ContextCompatibilityScorer(
        learnable_refinement=False,
        custom_species_priors=species_low,
        custom_prep_priors=prep_low,
        custom_route_priors=route_low,
    )
    analyzer_low = EvidenceLineageAnalyzer(echo_discount_factor=0.15, context_scorer=scorer_low)
    summaries_low = analyzer_low.analyze_dataset(sample_benchmark_df)
    tensors_low = prepare_partition_tensors(
        sample_benchmark_df, summaries_low, bio_extractor, text_extractor, feature_group="quality_only"
    )

    # 2. High context weight scale (1.50x)
    scale_high = 1.50
    species_high = {k: min(1.0, max(0.05, v * scale_high)) for k, v in SPECIES_PRIORS.items()}
    prep_high = {k: min(1.0, max(0.05, v * scale_high)) for k, v in PREPARATION_PRIORS.items()}
    route_high = {k: min(1.0, max(0.05, v * scale_high)) for k, v in ROUTE_PRIORS.items()}
    scorer_high = ContextCompatibilityScorer(
        learnable_refinement=False,
        custom_species_priors=species_high,
        custom_prep_priors=prep_high,
        custom_route_priors=route_high,
    )
    analyzer_high = EvidenceLineageAnalyzer(echo_discount_factor=0.15, context_scorer=scorer_high)
    summaries_high = analyzer_high.analyze_dataset(sample_benchmark_df)
    tensors_high = prepare_partition_tensors(
        sample_benchmark_df, summaries_high, bio_extractor, text_extractor, feature_group="quality_only"
    )

    meta_low = tensors_low[3]
    meta_high = tensors_high[3]

    # Must NOT be byte-identical or all-close
    assert not torch.equal(meta_low, meta_high), (
        "Metadata tensors for context weight scale 0.50x and 1.50x are byte-identical! "
        "ContextCompatibilityScorer was not properly plumbed into tensor generation."
    )
    assert not torch.allclose(meta_low, meta_high, atol=1e-4), (
        "Metadata tensors for context weight scale 0.50x and 1.50x are numerically identical within 1e-4."
    )

    # Check context compatibility score column specifically (column index 2 in quality_only mode)
    # quality_only: [citation_echo_ratio, mean_quality, context_compat, contradiction, metadata_missingness, rho_reach, f_m_CYP]
    assert not torch.equal(meta_low[:, 2], meta_high[:, 2]), (
        "Context compatibility feature column (index 2) is identical between 0.50x and 1.50x sweeps!"
    )
    assert not torch.allclose(meta_low[:, 2], meta_high[:, 2], atol=1e-3), (
        "Context compatibility feature column is numerically close within 1e-3 between 0.50x and 1.50x sweeps!"
    )


def test_alpha_sweep_produces_distinct_tensors_in_full_mode(sample_benchmark_df, feature_extractors):
    """Assert that varying echo discount alpha produces non-identical tensors in 'full' feature mode."""
    bio_extractor, text_extractor = feature_extractors

    # 1. Low alpha (0.05)
    analyzer_low = EvidenceLineageAnalyzer(echo_discount_factor=0.05)
    summaries_low = analyzer_low.analyze_dataset(sample_benchmark_df)
    tensors_low = prepare_partition_tensors(
        sample_benchmark_df, summaries_low, bio_extractor, text_extractor, feature_group="full"
    )

    # 2. High alpha (1.00)
    analyzer_high = EvidenceLineageAnalyzer(echo_discount_factor=1.00)
    summaries_high = analyzer_high.analyze_dataset(sample_benchmark_df)
    tensors_high = prepare_partition_tensors(
        sample_benchmark_df, summaries_high, bio_extractor, text_extractor, feature_group="full"
    )

    meta_low = tensors_low[3]
    meta_high = tensors_high[3]

    assert not torch.equal(meta_low, meta_high), (
        "Metadata tensors for alpha=0.05 and alpha=1.00 are byte-identical in full feature mode! "
        "Alpha parameter is not affecting polarity feature construction."
    )
    assert not torch.allclose(meta_low, meta_high, atol=1e-4), (
        "Metadata tensors for alpha=0.05 and alpha=1.00 are numerically identical within 1e-4."
    )


def test_sensitivity_figures_and_results_exist():
    """Verify generated sensitivity figures and JSON result artifacts exist and are non-empty."""
    fig_dir = Path("reports/figures")
    for fig_name in [
        "sensitivity_alpha.png",
        "sensitivity_context_weights.png",
        "sensitivity_thresholds.png",
    ]:
        p = fig_dir / fig_name
        assert p.exists(), f"Figure {p} does not exist"
        assert p.stat().st_size > 1000, f"Figure {p} is suspiciously small ({p.stat().st_size} bytes)"

    json_path = Path("reports/sensitivity_analysis_results.json")
    assert json_path.exists(), f"Results JSON {json_path} does not exist"
    assert json_path.stat().st_size > 500, f"Results JSON {json_path} is suspiciously small"
