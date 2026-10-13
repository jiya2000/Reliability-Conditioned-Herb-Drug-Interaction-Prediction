"""
Unit tests for evidence lineage, citation echo detection, context compatibility, and contradiction resolution.
"""

import pytest
import numpy as np
from src.data.evidence_schema import EvidenceRecord, EvidenceOutcome, EvidenceType
from src.data.evidence_lineage import EvidenceLineageAnalyzer, PairEvidenceSummary


def test_citation_echo_vs_independent_studies():
    analyzer = EvidenceLineageAnalyzer(echo_discount_factor=0.15)

    # Scenario A: 5 papers, but 4 merely cite the first (Echo Chamber)
    primary_id = "STUDY_PRIMARY"
    echo_records = [
        EvidenceRecord(
            herb_id="H1",
            drug_id="D1",
            outcome="positive",
            evidence_type="clinical",
            source_id="PAPER_1",
            study_id=primary_id,
            cited_study_ids=[],
            species="human",
        )
    ]
    for i in range(2, 6):
        echo_records.append(
            EvidenceRecord(
                herb_id="H1",
                drug_id="D1",
                outcome="positive",
                evidence_type="case_report",
                source_id=f"PAPER_{i}",
                study_id=f"DERIV_{i}",
                cited_study_ids=[primary_id],
                species="human",
            )
        )

    # Scenario B: 5 truly independent studies
    indep_records = []
    for i in range(1, 6):
        indep_records.append(
            EvidenceRecord(
                herb_id="H1",
                drug_id="D1",
                outcome="positive",
                evidence_type="clinical",
                source_id=f"INDEP_PAPER_{i}",
                study_id=f"INDEP_STUDY_{i}",
                cited_study_ids=[],
                species="human",
            )
        )

    summary_echo = analyzer.aggregate_pair_evidence("H1", "D1", echo_records)
    summary_indep = analyzer.aggregate_pair_evidence("H1", "D1", indep_records)

    # Key Assertion 1: Echo chamber must have high citation_echo_ratio
    assert summary_echo.citation_echo_ratio > 0.0
    assert summary_indep.citation_echo_ratio == 0.0

    # Key Assertion 2: 5 truly independent studies must yield significantly higher evidentiary strength
    assert summary_indep.independent_positive_strength > summary_echo.independent_positive_strength
    assert summary_echo.num_independent_positive_clusters == 1
    assert summary_indep.num_independent_positive_clusters == 5


def test_context_compatibility_scoring():
    analyzer = EvidenceLineageAnalyzer()

    rec_human = EvidenceRecord(
        herb_id="H1", drug_id="D1", outcome="positive", evidence_type="clinical",
        source_id="S1", study_id="ST1", species="human", route="oral", preparation="standardized extract"
    )
    rec_rat = EvidenceRecord(
        herb_id="H1", drug_id="D1", outcome="positive", evidence_type="animal",
        source_id="S2", study_id="ST2", species="rat", route="ip", preparation="crude powder"
    )
    rec_vitro = EvidenceRecord(
        herb_id="H1", drug_id="D1", outcome="positive", evidence_type="in_vitro",
        source_id="S3", study_id="ST3", species="caco-2 cell line", route=None, preparation=None
    )

    c_human = analyzer.compute_context_compatibility(rec_human)
    c_rat = analyzer.compute_context_compatibility(rec_rat)
    c_vitro = analyzer.compute_context_compatibility(rec_vitro)

    assert c_human > c_rat > c_vitro
    assert 0.0 < c_vitro < 1.0


def test_explicit_contradiction_detection():
    analyzer = EvidenceLineageAnalyzer()

    # Pair with unanimous positive findings
    pos_recs = [
        EvidenceRecord(
            herb_id="H1", drug_id="D1", outcome="positive", evidence_type="clinical",
            source_id="S1", study_id="ST1", species="human"
        ),
        EvidenceRecord(
            herb_id="H1", drug_id="D1", outcome="positive", evidence_type="animal",
            source_id="S2", study_id="ST2", species="rat"
        ),
    ]
    summary_pos = analyzer.aggregate_pair_evidence("H1", "D1", pos_recs)
    assert summary_pos.contradiction_strength == 0.0
    assert summary_pos.net_evidence_margin > 0.0

    # Pair with strong conflict: one clinical positive, one rigorous clinical negative
    conflict_recs = [
        EvidenceRecord(
            herb_id="H2", drug_id="D2", outcome="positive", evidence_type="clinical",
            source_id="S1", study_id="ST1", species="human"
        ),
        EvidenceRecord(
            herb_id="H2", drug_id="D2", outcome="negative", evidence_type="clinical",
            source_id="S2", study_id="ST2", species="human"
        ),
    ]
    summary_conf = analyzer.aggregate_pair_evidence("H2", "D2", conflict_recs)
    # Contradiction strength must be near 1.0 for balanced strong clashing evidence
    assert summary_conf.contradiction_strength > 0.8
    assert abs(summary_conf.net_evidence_margin) < 0.2

    # Feature vector shape and integrity
    vec = summary_conf.to_feature_vector()
    assert isinstance(vec, np.ndarray)
    assert vec.shape == (8,)
