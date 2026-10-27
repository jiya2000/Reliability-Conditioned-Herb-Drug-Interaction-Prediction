"""
Unit tests for External Validation and Generalization Benchmark (Phase 10).
"""

from __future__ import annotations

import json
from pathlib import Path
import pytest
import pandas as pd

from benchmarks.external_validation import run_external_validation
from src.data.public_data_loader import PublicDataLoader


def test_external_validation_schema_and_loader(tmp_path):
    """Verify that external validation CSV conforms strictly to schema and loads cleanly."""
    ext_csv = Path("data/external/napdi_template.csv")
    assert ext_csv.exists(), "Template file data/external/napdi_template.csv must exist"

    loader = PublicDataLoader(allow_synthetic=False)
    df = loader.load_hdi_evidence(
        file_path=str(ext_csv),
        validate=True,
        allow_synthetic=False,
    )
    assert len(df) >= 3, f"Expected at least 3 records in template, got {len(df)}"

    required_cols = [
        "herb_id",
        "drug_id",
        "outcome",
        "evidence_type",
        "source_id",
        "extraction_confidence",
    ]
    for col in required_cols:
        assert col in df.columns, f"Missing required column {col} in external template"

    # Verify entities are non-empty
    assert df["herb_id"].str.startswith("HERB_").all()
    assert df["drug_id"].str.startswith("DRUG_").all()


def test_external_validation_graceful_skip():
    """Verify that if external validation file is missing, the script gracefully returns None (exit 0)."""
    res = run_external_validation(
        external_csv="data/external/non_existent_file_xyz.csv",
    )
    assert res is None, "Expected None return for missing external file"


def test_external_validation_execution(tmp_path):
    """Test full execution of external validation on template without refitting."""
    out_json = tmp_path / "ext_val_test.json"
    out_md = tmp_path / "ext_val_test.md"

    res = run_external_validation(
        benchmark_csv="data/benchmark/verified_botanical_hdi.csv",
        external_csv="data/external/napdi_template.csv",
        output_json=str(out_json),
        output_md=str(out_md),
        seed=42,
    )

    assert res is not None, "External validation should produce result dict"
    assert "metrics" in res
    assert "conformal_q_hat" in res
    assert "record_details" in res
    assert len(res["record_details"]) == res["n_samples"]

    # Check metrics keys
    metrics = res["metrics"]
    for k in ["brier_score", "coverage", "abstention_rate", "expected_calibration_error"]:
        assert k in metrics
        assert isinstance(metrics[k], float)

    # Check reliability distribution
    assert "reliability_distribution" in res
    assert "mean" in res["reliability_distribution"]

    # Check files created
    assert out_json.exists()
    assert out_md.exists()

    with open(out_json) as f:
        loaded = json.load(f)
        assert loaded["headline_model"] == res["headline_model"]

    with open(out_md) as f:
        md_text = f.read()
        assert "# External Validation Benchmark Report" in md_text
        assert "ZERO-REFITTING" in md_text
