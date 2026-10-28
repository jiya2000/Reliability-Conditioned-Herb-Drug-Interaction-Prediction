"""
Unit Tests for Phase 4 Scaffolding & Labeling File Schemas.

Covers:
1. Aggregation model logit = logit_prior + sum_i R_i * delta_i
   - With zero studies, output equals the prior exactly.
   - delta from study findings only, R from design + lineage features only.
2. Reliability target:
   - agree iff (o=+1 and Y=1) or (o=-1 and Y=0); o=0 excluded.
   - Both ReliabilityTarget and load_consensus_labels raise RuntimeError
     while consensus_labels.csv has no validated rows.
3. Labeling file schema:
   - Exactly 8 columns: herb_id, drug_id, consensus_label, severity,
     source_name, source_url_or_citation, retrieval_date, notes
   - Fails if any other column exists (no strata, outcome, evidence_source).
   - Last 6 columns empty.
   - Stratification metadata strictly in private gitignored directory.
"""

import os
from pathlib import Path
import pytest
import numpy as np
import pandas as pd
import torch

from src.models.aggregation_model import (
    ReliabilityAggregationModel,
    ReliabilityTarget,
    load_consensus_labels,
)

EXPECTED_LABELING_COLUMNS = [
    "herb_id",
    "drug_id",
    "consensus_label",
    "severity",
    "source_name",
    "source_url_or_citation",
    "retrieval_date",
    "notes",
]


# ==============================================================================
# 1. Aggregation Model Scaffolding Tests
# ==============================================================================

def test_aggregation_model_zero_studies_equals_prior_exactly():
    """
    With zero studies, output must equal the prior exactly.
    Tested for multiple prior values and batch sizes.
    """
    for prior_val in [-1.5, 0.0, 0.75, 2.3]:
        model = ReliabilityAggregationModel(logit_prior=prior_val, delta_dim=1, design_dim=5)
        
        # Test via forward_no_studies
        out_no_studies = model.forward_no_studies(batch_size=4)
        expected = torch.tensor([prior_val] * 4, dtype=torch.float32)
        assert torch.allclose(out_no_studies, expected), (
            f"forward_no_studies output {out_no_studies} did not match prior {expected}"
        )
        
        # Test via forward with max_studies=0
        findings_zero = torch.zeros((4, 0, 1), dtype=torch.float32)
        design_zero = torch.zeros((4, 0, 5), dtype=torch.float32)
        out_zero, info = model(findings_zero, design_zero)
        assert torch.allclose(out_zero, expected), (
            f"forward(max_studies=0) output {out_zero} did not match prior {expected}"
        )
        assert info["logit_prior"] == pytest.approx(prior_val)


def test_aggregation_model_logit_formula():
    """
    Assert that model computes logit = logit_prior + sum_i R_i * delta_i.
    """
    model = ReliabilityAggregationModel(logit_prior=0.5, delta_dim=2, design_dim=3)
    batch_size = 2
    max_studies = 3
    
    findings = torch.randn(batch_size, max_studies, 2)
    design = torch.randn(batch_size, max_studies, 3)
    study_mask = torch.tensor([[1.0, 1.0, 0.0], [1.0, 0.0, 0.0]])
    
    logits, info = model(findings, design, study_mask=study_mask)
    R = info["R"]
    delta = info["delta"]
    
    # Manual calculation with mask
    expected_logits = 0.5 + (R * delta * study_mask).sum(dim=-1)
    assert torch.allclose(logits, expected_logits, atol=1e-5), (
        f"Aggregated logit {logits} did not match expected formula {expected_logits}"
    )


# ==============================================================================
# 2. Reliability Target Tests
# ==============================================================================

def test_reliability_target_agree_formula():
    """
    agree iff (o=+1 and Y=1) or (o=-1 and Y=0); o=0 excluded.
    """
    rt = ReliabilityTarget(enforce_validated_labels=False)
    
    # Study directions: [+1, -1, 0, +1, -1]
    # Consensus ground truth: [1, 0, 1, 0, 1]
    # Expected:
    # (+1, 1) -> agree (1.0)
    # (-1, 0) -> agree (1.0)
    # (0, 1)  -> excluded (o=0)
    # (+1, 0) -> disagree (0.0)
    # (-1, 1) -> disagree (0.0)
    o = np.array([1, -1, 0, 1, -1])
    Y = np.array([1, 0, 1, 0, 1])
    
    targets, valid_mask = rt.compute_target(o, Y)
    
    np.testing.assert_array_equal(valid_mask, np.array([True, True, False, True, True]))
    np.testing.assert_array_equal(targets, np.array([1.0, 1.0, 0.0, 0.0], dtype=np.float32))


def test_reliability_target_and_loader_raise_error_when_no_validated_rows():
    """
    Both ReliabilityTarget and load_consensus_labels must raise a clear error
    while consensus_labels.csv has no validated rows.
    """
    consensus_path = "data/benchmark/consensus_labels.csv"
    assert os.path.exists(consensus_path), f"Missing {consensus_path}"
    
    # 1. load_consensus_labels must raise RuntimeError
    with pytest.raises(RuntimeError, match="consensus_labels.csv has no validated rows"):
        load_consensus_labels(consensus_path)
        
    # 2. ReliabilityTarget instantiation must raise RuntimeError when enforcing validated labels
    with pytest.raises(RuntimeError, match="consensus_labels.csv has no validated rows"):
        ReliabilityTarget(consensus_path=consensus_path, enforce_validated_labels=True)


# ==============================================================================
# 3. Labeling File Schema Tests (Item 0 and Item 3e)
# ==============================================================================

def test_consensus_labels_pairs_to_label_schema():
    """
    Assert data/benchmark/consensus_labels_pairs_to_label.csv:
    - Exactly 8 columns: herb_id, drug_id, consensus_label, severity,
      source_name, source_url_or_citation, retrieval_date, notes
    - Fails if contains any other column (e.g. no strata, outcome, evidence_source)
    - Last 6 columns must be empty
    - Exactly 120 unique benchmark pairs
    """
    path = "data/benchmark/consensus_labels_pairs_to_label.csv"
    assert os.path.exists(path), f"Missing {path}"
    
    df = pd.read_csv(path)
    actual_cols = list(df.columns)
    assert actual_cols == EXPECTED_LABELING_COLUMNS, (
        f"Columns in {path} must be exactly {EXPECTED_LABELING_COLUMNS}, got {actual_cols}"
    )
    
    # Ensure forbidden leakage columns do NOT exist
    forbidden_cols = {"strata", "outcome", "evidence_source", "study_status", "direction"}
    present_forbidden = set(actual_cols) & forbidden_cols
    assert not present_forbidden, f"Forbidden columns present in labeling file: {present_forbidden}"
    
    # Last 6 columns empty
    last6 = EXPECTED_LABELING_COLUMNS[2:]
    for col in last6:
        non_empty = df[col].notna() & (df[col].astype(str).str.strip() != "")
        assert not non_empty.any(), f"Column {col} in {path} has non-empty entries!"
        
    # Unique benchmark pairs check
    assert len(df) == 120, f"Expected 120 rows, got {len(df)}"
    assert len(df.drop_duplicates(subset=["herb_id", "drug_id"])) == 120, (
        "Pairs in labeling file must be strictly unique!"
    )


def test_consensus_labels_random_unstudied_schema():
    """
    Assert data/benchmark/consensus_labels_random_unstudied.csv:
    - Exactly same 8 columns
    - Fails if contains any other column (no study_status, strata, etc.)
    - Last 6 columns must be empty
    """
    path = "data/benchmark/consensus_labels_random_unstudied.csv"
    assert os.path.exists(path), f"Missing {path}"
    
    df = pd.read_csv(path)
    actual_cols = list(df.columns)
    assert actual_cols == EXPECTED_LABELING_COLUMNS, (
        f"Columns in {path} must be exactly {EXPECTED_LABELING_COLUMNS}, got {actual_cols}"
    )
    
    # Last 6 columns empty
    last6 = EXPECTED_LABELING_COLUMNS[2:]
    for col in last6:
        non_empty = df[col].notna() & (df[col].astype(str).str.strip() != "")
        assert not non_empty.any(), f"Column {col} in {path} has non-empty entries!"


def test_private_stratification_metadata_is_gitignored():
    """
    Move any stratification metadata to data/benchmark/private/pair_strata.csv (gitignored).
    Verify private directory contains stratification files and is gitignored.
    """
    private_dir = Path("data/benchmark/private")
    assert private_dir.exists(), "data/benchmark/private directory does not exist"
    
    pair_strata = private_dir / "pair_strata.csv"
    assert pair_strata.exists(), "pair_strata.csv does not exist in private directory"
    
    gitignore_path = Path(".gitignore")
    assert gitignore_path.exists(), ".gitignore does not exist"
    gitignore_content = gitignore_path.read_text()
    
    assert "data/benchmark/private/" in gitignore_content or "private/" in gitignore_content, (
        "data/benchmark/private/ must be in .gitignore"
    )
