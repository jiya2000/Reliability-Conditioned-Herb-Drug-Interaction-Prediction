"""
Unit tests for PublicDataLoader, strict real-data mode enforcement, and dataset readiness auditing.
"""

import pytest
import pandas as pd
from pathlib import Path
from src.data.public_data_loader import PublicDataLoader, BiologicalGraphData


def test_manifest_loading_and_rules():
    loader = PublicDataLoader()
    manifest = loader.manifest
    assert "rules" in manifest
    assert manifest["rules"].get("unknown_is_negative") is False
    assert manifest["rules"].get("retain_source_references") is True
    assert manifest["rules"].get("prohibit_synthetic_in_production_benchmarks") is True


def test_real_data_mode_fails_on_missing_files(tmp_path):
    """
    P0 Fix: Ensure missing real datasets raise explicit FileNotFoundError
    instead of silently falling back to synthetic graphs.
    """
    empty_data_dir = tmp_path / "empty_data"
    empty_data_dir.mkdir()

    loader = PublicDataLoader(data_dir=str(empty_data_dir), allow_synthetic=False)

    # 1. Missing Dryad TCM files must raise FileNotFoundError
    with pytest.raises(FileNotFoundError) as exc_dryad:
        loader.load_dryad_tcm()
    assert "Dryad TCM network files not found" in str(exc_dryad.value)
    assert "datadryad.org" in str(exc_dryad.value)

    # 2. Missing HERB database files must raise FileNotFoundError
    with pytest.raises(FileNotFoundError) as exc_herb:
        loader.load_herb_database()
    assert "HERB database files not found" in str(exc_herb.value)
    assert "herb.ac.cn" in str(exc_herb.value)

    # 3. Missing HDI evidence must raise FileNotFoundError
    with pytest.raises(FileNotFoundError) as exc_ev:
        loader.load_hdi_evidence()
    assert "Genuine HDI evidence file not found" in str(exc_ev.value)

    # 4. Readiness check must raise RuntimeError
    with pytest.raises(RuntimeError) as exc_ready:
        loader.verify_dataset_readiness()
    assert "readiness check failed" in str(exc_ready.value)


def test_explicit_synthetic_demo_mode():
    """
    Verify that synthetic data is ONLY generated when allow_synthetic=True is explicitly set.
    """
    loader = PublicDataLoader(allow_synthetic=True)
    kg_data = loader.load_dryad_tcm()
    assert isinstance(kg_data, BiologicalGraphData)
    assert kg_data.metadata.get("is_synthetic") is True

    herb_kg = loader.load_herb_database()
    assert isinstance(herb_kg, BiologicalGraphData)
    assert herb_kg.metadata.get("is_synthetic") is True

    ev_df = loader.load_hdi_evidence()
    assert len(ev_df) > 0


def test_sample_evidence_generation():
    loader = PublicDataLoader(allow_synthetic=True)
    df = loader.generate_sample_hdi_evidence_table(num_pairs=10)
    assert len(df) >= 10
    assert "herb_id" in df.columns
    assert "drug_id" in df.columns
    assert "outcome" in df.columns
    assert "study_id" in df.columns
    assert "source_url_or_doi" in df.columns

    # Verify that positive, negative, and unknown outcomes are all represented
    outcomes = set(df["outcome"].unique())
    assert "positive" in outcomes
    assert "negative" in outcomes
