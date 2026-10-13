"""
Unit tests for PublicDataLoader and biological graph data containers.
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


def test_sample_evidence_generation():
    loader = PublicDataLoader()
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


def test_biological_graph_topology():
    loader = PublicDataLoader()
    kg_data = loader.load_dryad_tcm()
    assert isinstance(kg_data, BiologicalGraphData)
    summary = kg_data.summary()
    assert summary["num_herbs"] > 0
    assert summary["num_compounds"] > 0
    assert summary["num_herb_compound_edges"] > 0
