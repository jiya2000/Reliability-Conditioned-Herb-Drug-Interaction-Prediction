"""
End-to-end integration and uncertainty calibration tests.

Covers:
1. Real-data smoke test with explicit data-source reporting and strict real-mode enforcement.
2. End-to-end training and evaluation pipeline with study-disjoint splits and validation-exclusive temperature scaling.
3. MC dropout epistemic uncertainty verification: deterministic eval behavior, repeated stochastic passes,
   and epistemic uncertainty calibration against held-out conflicting evidence.
"""

import pytest
import torch
import numpy as np
import pandas as pd
from pathlib import Path

from src.data.public_data_loader import PublicDataLoader, BiologicalGraphData
from src.data.evidence_schema import EvidenceRecord, validate_dataframe, export_canonical_table
from src.data.evidence_lineage import EvidenceLineageAnalyzer, PairEvidenceSummary
from src.models.interaction_model import ReliabilityConditionedInteractionModel
from src.models.reliability_scorer import ReliabilityScorer
from src.training.calibration import (
    TemperatureScaler,
    compute_calibration_metrics,
    evaluate_abstention,
)
from src.training.evaluation import (
    LeakageSafeSplitter,
    compute_comprehensive_metrics,
    format_ablation_markdown_table,
)


def test_real_data_smoke_test_with_source_reporting(tmp_path):
    """
    Smoke test: verify that when genuine public files are present,
    the loader reads real files, reports real sources, marks is_synthetic=False,
    and never invokes the synthetic generator.
    """
    real_data_dir = tmp_path / "data"
    raw_dryad = real_data_dir / "raw" / "dryad_tcm"
    raw_herb = real_data_dir / "raw" / "HERB"
    processed = real_data_dir / "processed"
    raw_dryad.mkdir(parents=True)
    raw_herb.mkdir(parents=True)
    processed.mkdir(parents=True)

    # 1. Create Dryad files
    hc_df = pd.DataFrame({
        "herb_id": ["H001", "H002", "H003"],
        "compound_id": ["C101", "C102", "C103"],
    })
    ht_df = pd.DataFrame({
        "herb_id": ["H001", "H002"],
        "target_id": ["T201", "T202"],
    })
    hc_df.to_csv(raw_dryad / "herb_compound_network.csv", index=False)
    ht_df.to_csv(raw_dryad / "herb_target_network.csv", index=False)

    # 2. Create HERB database files
    h_info = pd.DataFrame({"herb_id": ["H001", "H002"], "herb_name": ["Ginseng", "Licorice"]})
    hi_pairs = pd.DataFrame({"herb_id": ["H001", "H002"], "compound_id": ["C101", "C102"]})
    h_info.to_csv(raw_herb / "HERB_herb_info.csv", index=False)
    hi_pairs.to_csv(raw_herb / "HERB_herb_ingredient_pairs.csv", index=False)

    # 3. Create Canonical HDI Evidence table
    ev_df = pd.DataFrame([
        {
            "herb_id": "H001",
            "drug_id": "D001",
            "outcome": "positive",
            "evidence_type": "clinical",
            "source_id": "PUBMED_12345",
            "study_id": "STUDY_12345",
            "cited_study_ids": "[]",
            "publication_date": "2020-01-01",
            "extraction_confidence": 0.95,
            "source_url_or_doi": "https://doi.org/10.1000/182",
        },
        {
            "herb_id": "H002",
            "drug_id": "D002",
            "outcome": "negative",
            "evidence_type": "in_vitro",
            "source_id": "PUBMED_67890",
            "study_id": "STUDY_67890",
            "cited_study_ids": "[]",
            "publication_date": "2021-05-12",
            "extraction_confidence": 0.90,
            "source_url_or_doi": "https://doi.org/10.1000/183",
        },
    ])
    ev_df.to_csv(processed / "hdi_evidence_canonical.csv", index=False)

    # Instantiate strictly in real mode
    loader = PublicDataLoader(data_dir=str(real_data_dir), allow_synthetic=False)

    # Verify readiness check passes
    readiness = loader.verify_dataset_readiness()
    assert readiness["dryad_tcm_ready"] is True
    assert readiness["herb_database_ready"] is True
    assert readiness["hdi_evidence_ready"] is True

    # Load Dryad
    dryad_data = loader.load_dryad_tcm()
    assert isinstance(dryad_data, BiologicalGraphData)
    assert dryad_data.metadata.get("is_synthetic") is False
    assert dryad_data.metadata.get("source") == "Dryad_TCM"
    assert len(dryad_data.herbs) == 3
    assert len(dryad_data.compounds) == 3

    # Load HERB
    herb_data = loader.load_herb_database()
    assert isinstance(herb_data, BiologicalGraphData)
    assert herb_data.metadata.get("is_synthetic") is False
    assert herb_data.metadata.get("source") == "HERB_database"
    assert len(herb_data.herbs) == 2

    # Load Evidence
    loaded_ev = loader.load_hdi_evidence()
    assert len(loaded_ev) == 2
    assert "H001" in loaded_ev["herb_id"].values


def test_synthetic_pipeline_integration_and_validation_exclusive_calibration():
    """
    SYNTHETIC INTEGRATION TEST:
    Verifies computational wiring, multi-epoch training, study-disjoint partitioning,
    and validation-exclusive temperature calibration on mock tensor inputs.
    
    CRITICAL DISTINCTION: This test validates software execution integrity only.
    It does NOT evaluate or establish genuine botanical HDI prediction, clinical
    generalization, or real-world calibration quality (which requires verified real benchmarks).
    """
    np.random.seed(42)
    torch.manual_seed(42)

    # Build synthetic evidence records across distinct study clusters
    records = []
    n_pairs = 30
    for i in range(n_pairs):
        h_id = f"HERB_{i % 8:03d}"
        d_id = f"DRUG_{i % 5:03d}"
        study_id = f"STUDY_{i // 3:03d}"  # 3 records per study cluster
        outcome = "positive" if (i % 2 == 0) else "negative"
        records.append({
            "herb_id": h_id,
            "drug_id": d_id,
            "outcome": outcome,
            "evidence_type": "clinical" if i % 2 == 0 else "in_vitro",
            "source_id": f"PMID_{1000 + i}",
            "study_id": study_id,
            "cited_study_ids": [],
            "publication_date": f"{2015 + (i % 8)}-06-01",
            "extraction_confidence": 0.9,
            "source_url_or_doi": f"https://doi.org/10.1000/{i}",
        })

    ev_df = pd.DataFrame(records)
    splitter = LeakageSafeSplitter(seed=42)

    # Study-disjoint split
    train_df, val_df, test_df = splitter.split_study_disjoint(ev_df, train_ratio=0.6, val_ratio=0.2)
    assert len(train_df) > 0
    assert len(val_df) > 0
    assert len(test_df) > 0

    # Ensure no study leakage
    assert len(set(train_df["study_id"]) & set(test_df["study_id"])) == 0

    # Create model
    feat_dim = 16
    model = ReliabilityConditionedInteractionModel(
        graph_dim=feat_dim,
        evidence_dim=feat_dim,
        fused_dim=feat_dim,
        hidden_dim=feat_dim,
        mode="reliability_conditioned",
    )
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    criterion = torch.nn.BCEWithLogitsLoss()

    def get_batch(df):
        n = len(df)
        herb_emb = torch.randn(n, feat_dim)
        drug_emb = torch.randn(n, feat_dim)
        ev_feat = torch.randn(n, feat_dim)
        # 8D lineage metadata
        meta = torch.zeros(n, 8)
        meta[:, 0] = torch.tensor([1.5 if o == "positive" else 0.0 for o in df["outcome"]])
        meta[:, 1] = torch.tensor([1.5 if o == "negative" else 0.0 for o in df["outcome"]])
        meta[:, 4] = 0.9
        meta[:, 6] = meta[:, 0] - meta[:, 1]
        labels = torch.tensor([1.0 if o == "positive" else 0.0 for o in df["outcome"]]).unsqueeze(1)
        return herb_emb, drug_emb, ev_feat, meta, labels

    # Simple training loop for 5 epochs
    model.train()
    for _ in range(5):
        optimizer.zero_grad()
        h_emb, d_emb, e_feat, meta, y = get_batch(train_df)
        out = model(h_emb, d_emb, e_feat, metadata=meta)
        loss = criterion(out["logits"], y)
        loss.backward()
        optimizer.step()

    model.eval()
    with torch.no_grad():
        # Validation logits and labels
        vh_emb, vd_emb, ve_feat, vmeta, vy = get_batch(val_df)
        val_out = model(vh_emb, vd_emb, ve_feat, metadata=vmeta)
        val_logits = val_out["logits"].cpu().numpy()
        val_labels = vy.cpu().numpy()

        # Test logits and labels
        th_emb, td_emb, te_feat, tmeta, ty = get_batch(test_df)
        test_out = model(th_emb, td_emb, te_feat, metadata=tmeta)
        test_logits = test_out["logits"].cpu().numpy()
        test_labels = ty.cpu().numpy()

    # Temperature Scaling: Fit strictly on validation partition ONLY
    scaler = TemperatureScaler()
    initial_t = scaler.temperature.item()
    opt_t = scaler.fit(val_logits, val_labels, max_iter=20)
    assert opt_t > 0.0

    # Scale test logits using validation-fitted temperature
    test_scaled_logits = scaler(torch.tensor(test_logits, dtype=torch.float32)).detach().numpy()
    test_cal_probs = 1.0 / (1.0 + np.exp(-test_scaled_logits))

    # Evaluate test partition metrics
    report = compute_comprehensive_metrics(
        y_true=test_labels,
        y_prob=test_cal_probs,
        split_name="test_study_disjoint",
        model_variant="reliability_conditioned",
    )
    assert report.num_samples == len(test_df)
    assert 0.0 <= report.ece <= 1.0
    assert 0.0 <= report.brier_score <= 1.0


def test_leaky_reference_mc_dropout_conflict_heuristic():
    """
    Legacy full-vector heuristic test, not clinical uncertainty validation.
    Quality-only invariance is covered in test_evaluation_integrity.py.
    1. Deterministic evaluation behavior: model.eval() without MC sampling produces zero variance.
    2. Repeated stochastic passes: forward_with_uncertainty produces non-zero uncertainty.
    3. Epistemic uncertainty is significantly higher on conflicting evidence compared to concordant evidence.
    """
    feat_dim = 16
    scorer = ReliabilityScorer(
        hidden_dim=feat_dim,
        mc_dropout_rate=0.20,
        mc_samples=15,
        feature_group="full",  # Explicit outcome-dependent reference, not primary R.
    )
    scorer.eval()

    # Concordant positive evidence:
    # 3 independent positive studies, 0 negative, 0 echo, 0 conflict, positive margin
    meta_concordant = torch.tensor([[3.0, 0.0, 0.0, 0.95, 0.95, 0.0, 3.0, 0.0]], dtype=torch.float32)

    # Conflicting ambiguous evidence:
    # Equal positive & negative studies, high contradiction (0.95), zero margin
    meta_conflicting = torch.tensor([[2.0, 2.0, 0.1, 0.80, 0.80, 0.95, 0.0, 0.1]], dtype=torch.float32)

    # 1. Deterministic evaluation behavior
    with torch.no_grad():
        r1, _ = scorer(meta_concordant)
        r2, _ = scorer(meta_concordant)
    # Consecutive deterministic forward calls in eval mode must produce identical values
    assert torch.allclose(r1, r2), "Deterministic evaluation in eval() mode must produce identical outputs"

    # 2. Repeated stochastic predictions with MC Dropout
    with torch.no_grad():
        r_mean_concord, r_unc_concord, _ = scorer.forward_with_uncertainty(meta_concordant, n_samples=20)
        r_mean_conflict, r_unc_conflict, _ = scorer.forward_with_uncertainty(meta_conflicting, n_samples=20)

    # Non-zero epistemic uncertainty
    assert r_unc_concord.item() > 0.0, "MC dropout must produce non-zero epistemic uncertainty"
    assert r_unc_conflict.item() > 0.0, "MC dropout must produce non-zero epistemic uncertainty"

    # 3. Conflicting evidence must have strictly higher epistemic uncertainty than concordant evidence
    u_concord = r_unc_concord.item()
    u_conflict = r_unc_conflict.item()
    assert u_conflict > u_concord, (
        f"Conflicting scientific evidence must have higher epistemic uncertainty ({u_conflict:.4f}) "
        f"than concordant evidence ({u_concord:.4f})"
    )

    # 4. Reliability of concordant evidence must be higher than conflicting evidence
    assert r_mean_concord.item() > r_mean_conflict.item(), (
        f"Concordant evidence reliability ({r_mean_concord.item():.4f}) must exceed "
        f"conflicted evidence reliability ({r_mean_conflict.item():.4f})"
    )
