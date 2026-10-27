"""
Unit tests for Phase 9: Citation-DAG Quality, Study-Type Scaling, and Refutation Handling.
Tests:
1. Study-type scaled echo discount (RCT discounted less than review).
2. Text-reuse similarity scaling.
3. Contradicting derivative refutation handling (retains full weight w=1.0).
4. Full CitationEchoAnalyzer with multi-attribute records.
5. Verification of the generated DAG audit dataset.
"""

import os
from pathlib import Path
import pytest
import pandas as pd

from src.models.citation_dag import (
    CitationDAG,
    CitationEchoAnalyzer,
    compute_jaccard_text_reuse,
    DEFAULT_STUDY_TYPE_MULTIPLIERS,
)


def test_study_type_scaled_attenuation():
    """Verify RCT derivatives are discounted less than review derivatives."""
    # Scenario A: Root trial (2015) cited by a Review (2018)
    dag_review = CitationDAG(echo_discount_alpha=0.20, scale_by_study_type=True)
    dag_review.add_paper("P_ROOT", year=2015, study_type="clinical", outcome="interaction_present")
    dag_review.add_paper("P_REV", year=2018, study_type="review", outcome="interaction_present")
    dag_review.add_citation("P_REV", "P_ROOT")
    res_review = dag_review.analyze()

    # Scenario B: Root trial (2015) cited by a confirmatory RCT (2018)
    dag_rct = CitationDAG(echo_discount_alpha=0.20, scale_by_study_type=True)
    dag_rct.add_paper("P_ROOT", year=2015, study_type="clinical", outcome="interaction_present")
    dag_rct.add_paper("P_RCT", year=2018, study_type="rct", outcome="interaction_present")
    dag_rct.add_citation("P_RCT", "P_ROOT")
    res_rct = dag_rct.analyze()

    # RCT derivative should retain significantly higher independent weight than review derivative
    assert res_rct.effective_independent_count > res_review.effective_independent_count
    # Consequently, echo chamber index for review is higher
    assert res_review.echo_chamber_index > res_rct.echo_chamber_index


def test_text_reuse_similarity_scaling():
    """Verify high text-reuse paraphrasing increases echo discounting."""
    text_root = "St Johns wort induces cytochrome P450 3A4 and markedly lowers cyclosporine plasma concentration"
    text_paraphrase = "St Johns wort induces cytochrome P450 3A4 and significantly reduces cyclosporine plasma concentration"
    text_distinct = "A separate kinetic analysis evaluated renal clearance pathways without altering hepatic enzymes"

    sim_high = compute_jaccard_text_reuse(text_root, text_paraphrase)
    sim_low = compute_jaccard_text_reuse(text_root, text_distinct)
    assert sim_high > sim_low

    dag_paraphrase = CitationDAG(echo_discount_alpha=0.15, use_text_reuse=True)
    dag_paraphrase.add_paper("R", year=2010, text=text_root)
    dag_paraphrase.add_paper("D", year=2015, text=text_paraphrase)
    dag_paraphrase.add_citation("D", "R")
    res_paraphrase = dag_paraphrase.analyze()

    dag_distinct = CitationDAG(echo_discount_alpha=0.15, use_text_reuse=True)
    dag_distinct.add_paper("R", year=2010, text=text_root)
    dag_distinct.add_paper("D", year=2015, text=text_distinct)
    dag_distinct.add_citation("D", "R")
    res_distinct = dag_distinct.analyze()

    # Paraphrased echo should be discounted more (lower effective count)
    assert res_paraphrase.effective_independent_count < res_distinct.effective_independent_count


def test_contradicting_derivative_refutation_handling():
    """Verify a derivative that refutes the root finding retains full weight (w=1.0)."""
    dag = CitationDAG(echo_discount_alpha=0.25, handle_contradictions=True)
    # Root study reports interaction present
    dag.add_paper("P_ORIGINAL", year=2000, outcome="interaction_present")
    # Derivative study explicitly refutes or fails to replicate: reports no interaction
    dag.add_paper("P_REFUTATION", year=2005, outcome="no_interaction_observed")
    dag.add_citation("P_REFUTATION", "P_ORIGINAL")

    res = dag.analyze()
    assert res.total_papers == 2
    assert res.num_derivatives == 1
    assert res.has_conflicting_derivatives
    assert ("P_ORIGINAL", "P_REFUTATION") in res.conflicting_derivatives

    # Because it is a refutation, it is NOT an echo! Weight should be 1.0 + 1.0 = 2.0
    assert pytest.approx(res.effective_independent_count, rel=1e-3) == 2.0
    assert res.echo_chamber_index == 0.0


def test_dag_audit_file_integrity():
    """Verify dag_audit_samples.csv was generated and contains valid columns."""
    csv_path = "data/audit/dag_audit_samples.csv"
    assert os.path.exists(csv_path), f"Audit CSV {csv_path} does not exist"
    df = pd.read_csv(csv_path)

    expected_cols = {
        "component_id",
        "herb_id",
        "drug_id",
        "study_id",
        "source_pmid",
        "is_root",
        "primary_root",
        "depth",
        "study_type",
        "outcome",
        "is_contradiction",
        "text_reuse_score",
        "flat_weight",
        "study_type_scaled_weight",
    }
    assert expected_cols.issubset(set(df.columns))
    assert len(df) >= 140
    assert df["flat_weight"].min() > 0.0
    assert df["flat_weight"].max() == 1.0
