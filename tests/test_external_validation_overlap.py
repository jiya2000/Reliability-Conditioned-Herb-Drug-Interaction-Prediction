"""Unit tests for External Validation Overlap Diagnostics and Abstention Analysis.

Mandated by TASK 4:
In benchmarks/external_validation.py compute and log pair-, herb-, drug-, and study-level overlap
between the external cohort and the training data (the audit found Curcuma+Midazolam and
Curcuma+Talinolol overlap). Save the overlap table to reports/. Add an option to EXCLUDE overlapping
pairs and report metrics for overlap-excluded and full sets separately. Also report why the model
abstains on 5/5 external pairs (which condition triggers abstention: epistemic uncertainty,
conflict, or conformal set size) and whether abstention is appropriate or a miscalibrated threshold.
"""

from pathlib import Path
import json
import pytest
import pandas as pd

from benchmarks.external_validation import (
    compute_overlap_diagnostics,
    run_external_validation,
)
from src.data.public_data_loader import PublicDataLoader


def test_overlap_diagnostics_computation():
    """Verify pair-, herb-, drug-, and study-level overlap computation against external template."""
    loader = PublicDataLoader(allow_synthetic=False)
    train_df = loader.load_hdi_evidence("data/benchmark/verified_botanical_hdi.csv")
    ext_df = loader.load_hdi_evidence("data/external/napdi_template.csv")

    diagnostics = compute_overlap_diagnostics(train_df, ext_df)

    # Pair level assertions
    pair_overlap = diagnostics["pair_level"]["overlapping_pairs"]
    assert "HERB_CURCUMA + DRUG_MIDAZOLAM" in pair_overlap
    assert "HERB_CURCUMA + DRUG_TALINOLOL" in pair_overlap
    assert "HERB_GOLDENSEAL + DRUG_METFORMIN" in pair_overlap
    assert "HERB_GOLDENSEAL + DRUG_MIDAZOLAM" in pair_overlap
    assert len(pair_overlap) == 4

    # Novel pair assertion
    novel_pairs = diagnostics["pair_level"]["novel_non_overlapping_pairs"]
    assert "HERB_KRATOM + DRUG_OXYCODONE" in novel_pairs
    assert len(novel_pairs) == 1

    # Herb level assertions
    assert "HERB_CURCUMA" in diagnostics["herb_level"]["overlapping_herbs"]
    assert "HERB_GOLDENSEAL" in diagnostics["herb_level"]["overlapping_herbs"]
    assert "HERB_KRATOM" in diagnostics["herb_level"]["novel_herbs"]

    # Study and Source level assertions: must be strictly 0 (zero citation leakage)
    assert diagnostics["study_level"]["n_overlap"] == 0
    assert diagnostics["source_level"]["n_overlap"] == 0


def test_external_validation_execution_and_artifacts(tmp_path):
    """Test full execution generating overlap files and separate cohort metrics."""
    out_json = tmp_path / "ext_val.json"
    out_md = tmp_path / "ext_val.md"
    overlap_json = tmp_path / "overlap.json"
    overlap_md = tmp_path / "overlap.md"

    res = run_external_validation(
        benchmark_csv="data/benchmark/verified_botanical_hdi.csv",
        external_csv="data/external/napdi_template.csv",
        output_json=str(out_json),
        output_md=str(out_md),
        overlap_json=str(overlap_json),
        overlap_md=str(overlap_md),
        exclude_overlap=False,
        seed=42,
    )

    assert res is not None
    assert "full_cohort_metrics" in res
    assert "overlap_excluded_metrics" in res
    assert "curcuma_excluded_metrics" in res
    assert "overlap_diagnostics" in res
    assert "abstention_analysis" in res

    # Check sample counts
    assert res["full_cohort_metrics"]["n_samples"] == 5
    assert res["overlap_excluded_metrics"]["n_samples"] == 1
    assert res["curcuma_excluded_metrics"]["n_samples"] == 3

    # Check abstention root cause
    abstention = res["abstention_analysis"]
    trigger_dist = abstention["summary_trigger_distribution"]
    assert trigger_dist["conformal_set_size_2"] == 5, "All 5 external records must trigger conformal_set_size_2"
    assert "APPROPRIATE PROTECTIVE BEHAVIOR" in abstention["clinical_assessment"]

    # Check file persistence
    assert out_json.exists()
    assert out_md.exists()
    assert overlap_json.exists()
    assert overlap_md.exists()

    with open(overlap_md) as f:
        md_text = f.read()
        assert "HERB_CURCUMA + DRUG_MIDAZOLAM" in md_text
        assert "HERB_CURCUMA + DRUG_TALINOLOL" in md_text
        assert "Disjoint Studies" in md_text
