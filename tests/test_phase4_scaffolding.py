"""
Unit Tests for Phase 4 Scaffolding & Labeling File Schemas.

Covers:
1. Aggregation model logit = logit_prior + sum_i R_i * delta_i (Item 4a)
   - With zero studies (and all-missing studies), output equals the prior exactly (Item 4b).
   - delta from study findings only, R from design + lineage features only.
2. Reliability target (Item 4c):
   - agree iff (o=+1 and Y=1) or (o=-1 and Y=0); o=0 excluded.
   - Raises ConsensusLabelsNotPopulatedError while consensus_labels.csv has no validated rows.
   - Training entry point calls loader.
3. Task B baselines (Item 4d):
   - Majority vote, quality-weighted vote, echo-discounted vote, LR on pair-level features.
4. SYNTHETIC_FIXTURE end-to-end pipeline test (Item 4e):
   - Planted R structure recovered with AUC > 0.8.
   - Calibration code runs, aggregation reduces to prior at k=0.
   - Nothing from SYNTHETIC_FIXTURE is written to results/.
5. Labeling file schema & precondition checks (Item 0a, 0b, 3e):
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
import torch.nn as nn
from sklearn.metrics import roc_auc_score

from src.models.aggregation import (
    ReliabilityAggregationModel,
    ReliabilityTarget,
    ConsensusLabelsNotPopulatedError,
    load_consensus_labels,
    majority_vote_baseline,
    quality_weighted_vote_baseline,
    echo_discounted_vote_baseline,
    train_pair_level_lr_aggregate,
    train_reliability_model_entrypoint,
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
# 1. Aggregation Model Scaffolding Tests (Item 4a, 4b)
# ==============================================================================

def test_aggregation_model_zero_studies_equals_prior_exactly():
    """
    With zero studies (and all-missing studies), output must equal the prior exactly.
    Tested for multiple prior values and batch sizes.
    """
    for prior_val in [-1.5, 0.0, 0.75, 2.3]:
        model = ReliabilityAggregationModel(logit_prior=prior_val, delta_dim=1, design_dim=7)
        
        # Test 1: via forward_no_studies
        out_no_studies = model.forward_no_studies(batch_size=4)
        expected = torch.tensor([prior_val] * 4, dtype=torch.float32)
        assert torch.allclose(out_no_studies, expected), (
            f"forward_no_studies output {out_no_studies} did not match prior {expected}"
        )
        
        # Test 2: via forward with max_studies=0
        findings_zero = torch.zeros((4, 0, 1), dtype=torch.float32)
        design_zero = torch.zeros((4, 0, 7), dtype=torch.float32)
        out_zero, info = model(findings_zero, design_zero)
        assert torch.allclose(out_zero, expected), (
            f"forward(max_studies=0) output {out_zero} did not match prior {expected}"
        )
        assert info["logit_prior"] == pytest.approx(prior_val)

        # Test 3: with studies present but all masked out (all-missing studies)
        findings_masked = torch.randn(4, 3, 1)
        design_masked = torch.randn(4, 3, 7)
        study_mask_all_zeros = torch.zeros(4, 3)
        out_all_masked, _ = model(findings_masked, design_masked, study_mask=study_mask_all_zeros)
        assert torch.allclose(out_all_masked, expected), (
            f"forward with all-zero mask output {out_all_masked} did not match prior {expected}"
        )


def test_aggregation_model_logit_formula():
    """
    Assert that model computes logit = logit_prior + sum_i R_i * delta_i.
    """
    model = ReliabilityAggregationModel(logit_prior=0.5, delta_dim=2, design_dim=7)
    batch_size = 2
    max_studies = 3
    
    findings = torch.randn(batch_size, max_studies, 2)
    design = torch.randn(batch_size, max_studies, 7)
    study_mask = torch.tensor([[1.0, 1.0, 0.0], [1.0, 0.0, 0.0]])
    
    logits, info = model(findings, design, study_mask=study_mask)
    R = info["R"]
    delta = info["delta"]
    
    expected_logits = 0.5 + (R * delta * study_mask).sum(dim=-1)
    assert torch.allclose(logits, expected_logits, atol=1e-5), (
        f"Aggregated logit {logits} did not match expected formula {expected_logits}"
    )


# ==============================================================================
# 2. Reliability Target Tests (Item 4c)
# ==============================================================================

def test_reliability_target_agree_formula():
    """
    agree iff (o=+1 and Y=1) or (o=-1 and Y=0); o=0 excluded.
    """
    rt = ReliabilityTarget(enforce_validated_labels=False)
    
    o = np.array([1, -1, 0, 1, -1])
    Y = np.array([1, 0, 1, 0, 1])
    
    targets, valid_mask = rt.compute_target(o, Y)
    
    np.testing.assert_array_equal(valid_mask, np.array([True, True, False, True, True]))
    np.testing.assert_array_equal(targets, np.array([1.0, 1.0, 0.0, 0.0], dtype=np.float32))


def test_reliability_loader_and_training_entrypoint_raise_consensus_error():
    """
    Loader raises ConsensusLabelsNotPopulatedError while consensus_labels.csv has no validated rows.
    The training entrypoint must call the loader and fail loudly.
    """
    consensus_path = "data/benchmark/consensus_labels.csv"
    assert os.path.exists(consensus_path), f"Missing {consensus_path}"
    
    # 1. load_consensus_labels must raise ConsensusLabelsNotPopulatedError
    with pytest.raises(ConsensusLabelsNotPopulatedError, match="0 validated rows"):
        load_consensus_labels(consensus_path)
        
    # 2. train_reliability_model_entrypoint must fail with ConsensusLabelsNotPopulatedError
    with pytest.raises(ConsensusLabelsNotPopulatedError):
        train_reliability_model_entrypoint(consensus_path=consensus_path)


# ==============================================================================
# 3. Task B Baselines (Item 4d)
# ==============================================================================

def test_task_b_baselines():
    """Test majority vote, quality-weighted vote, and echo-discounted vote."""
    # Pair 1: two positive studies
    # Pair 2: one positive, two negative studies
    # Pair 3: tie (one pos, one neg)
    study_dirs = [[1, 1], [1, -1, -1], [1, -1]]
    q_weights = [[0.8, 0.9], [0.9, 0.4, 0.4], [0.8, 0.8]]
    echo_ratios = [[0.1, 0.0], [0.0, 0.5, 0.5], [0.0, 0.0]]

    # 1. Majority vote
    maj = majority_vote_baseline(study_dirs)
    np.testing.assert_array_equal(maj, np.array([1.0, 0.0, 0.5], dtype=np.float32))

    # 2. Quality weighted vote
    # Pair 2: pos has 1*0.9 = 0.9, negs have -1*0.4 + -1*0.4 = -0.8 -> sum = +0.1 > 0 -> 1.0
    qw = quality_weighted_vote_baseline(study_dirs, q_weights)
    np.testing.assert_array_equal(qw, np.array([1.0, 1.0, 0.5], dtype=np.float32))

    # 3. Echo discounted vote
    # Pair 1: pos
    ed = echo_discounted_vote_baseline(study_dirs, q_weights, echo_ratios)
    assert ed[0] == 1.0

    # 4. Pair-level LR aggregate
    X_dummy = np.array([[1.0, 0.5], [-1.0, 0.2], [0.8, 0.9], [-0.5, 0.1]])
    y_dummy = np.array([1, 0, 1, 0])
    clf = train_pair_level_lr_aggregate(X_dummy, y_dummy)
    preds = clf.predict(X_dummy)
    assert len(preds) == 4


# ==============================================================================
# 4. SYNTHETIC_FIXTURE End-to-End Pipeline Test (Item 4e, R6)
# ==============================================================================

def test_synthetic_fixture_end_to_end_reliability_pipeline(tmp_path):
    """
    R6 & 4e: End-to-end pipeline test on a SYNTHETIC_FIXTURE.
    Generated ONLY inside test, with planted known R structure.
    Verify:
    - R training recovers the planted structure (AUC of R > 0.8 on fixture).
    - Calibration code runs.
    - Aggregation reduces to the prior at k=0.
    - Nothing from this fixture is written to results/ or reports/.
    """
    torch.manual_seed(42)
    np.random.seed(42)

    n_samples = 200
    design_dim = 7
    SYNTHETIC_FIXTURE_design = torch.randn(n_samples, design_dim)
    
    # Plant a known true linear weight for reliability: feature 0 is strongly positive
    true_w = torch.tensor([2.5, -0.2, 0.1, 0.0, -0.1, 0.0, 0.0])
    true_scores = torch.sigmoid(SYNTHETIC_FIXTURE_design @ true_w).numpy()
    
    # Ground truth binary targets generated with planted probability
    planted_targets = (np.random.rand(n_samples) < true_scores).astype(np.float32)

    # Train reliability head on SYNTHETIC_FIXTURE
    model = ReliabilityAggregationModel(logit_prior=0.0, delta_dim=1, design_dim=design_dim)
    optimizer = torch.optim.Adam(model.reliability_head.parameters(), lr=0.05, weight_decay=1e-4)
    loss_fn = nn.BCELoss()

    targets_t = torch.tensor(planted_targets, dtype=torch.float32).unsqueeze(1)
    
    model.reliability_head.train()
    for _ in range(80):
        optimizer.zero_grad()
        pred_R = model.reliability_head(SYNTHETIC_FIXTURE_design)
        loss = loss_fn(pred_R, targets_t)
        loss.backward()
        optimizer.step()

    model.reliability_head.eval()
    with torch.no_grad():
        learned_R = model.reliability_head(SYNTHETIC_FIXTURE_design).squeeze().numpy()

    # Verify R training recovers the planted structure: AUC > 0.8
    r_auc = roc_auc_score(planted_targets, learned_R)
    assert r_auc > 0.80, f"Learned R AUC ({r_auc:.4f}) failed to exceed 0.80 on SYNTHETIC_FIXTURE!"

    # Calibration scaling test
    from scripts.run_phase5_task_a import fit_temperature_scaling
    logits_cal = np.log(np.clip(learned_R, 1e-5, 1 - 1e-5) / (1 - np.clip(learned_R, 1e-5, 1 - 1e-5)))
    temp = fit_temperature_scaling(logits_cal, planted_targets.astype(int))
    assert temp > 0.0, f"Temperature {temp} must be positive"

    # Aggregation reduces to prior at k=0
    out_k0 = model.forward_no_studies(batch_size=5)
    assert torch.allclose(out_k0, torch.zeros(5))

    # Assert that results/ and reports/ contain ZERO occurrences of SYNTHETIC_FIXTURE
    for dir_to_check in ["results", "reports"]:
        for root, _, files in os.walk(dir_to_check):
            for file in files:
                if file.endswith((".json", ".md", ".csv")):
                    content = Path(os.path.join(root, file)).read_text(errors="ignore")
                    assert "SYNTHETIC_FIXTURE" not in content, (
                        f"SYNTHETIC_FIXTURE leaked into {os.path.join(root, file)}!"
                    )


def test_no_synthetic_fixture_in_results_or_reports():
    """
    Permanent CI Test (R6, 5f(g)):
    Assert that 'SYNTHETIC_FIXTURE' is strictly forbidden from appearing in results/ or reports/.
    """
    for dir_to_check in ["results", "reports"]:
        for root, _, files in os.walk(dir_to_check):
            for file in files:
                if file.endswith((".json", ".md", ".csv")):
                    file_path = os.path.join(root, file)
                    content = Path(file_path).read_text(errors="ignore")
                    assert "SYNTHETIC_FIXTURE" not in content, (
                        f"Rule R6 Violation: SYNTHETIC_FIXTURE found in {file_path}!"
                    )


# ==============================================================================
# 5. Labeling File Schema Tests (Item 0a, 0b, 3e)
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
