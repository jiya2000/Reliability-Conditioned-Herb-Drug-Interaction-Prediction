"""
Unit tests for canonical evidence schema, strict data validation, and null/NaN rejection.
"""

import pytest
import numpy as np
import pandas as pd
from src.data.evidence_schema import (
    EvidenceRecord,
    EvidenceOutcome,
    EvidenceType,
    validate_dataframe,
    export_canonical_table,
    load_canonical_table,
)


def test_valid_evidence_record():
    rec = EvidenceRecord(
        herb_id="HERB_001",
        drug_id="DRUG_001",
        outcome="positive",
        evidence_type="clinical",
        source_id="PUBMED_10534248",
        study_id="STUDY_001",
        cited_study_ids=["STUDY_OLD"],
        dose="900 mg/day",
        preparation="extract",
        route="oral",
        species="human",
        extraction_confidence=0.95,
        source_url_or_doi="https://doi.org/10.1016/sample",
    )
    is_valid, errors = rec.validate()
    assert is_valid
    assert len(errors) == 0
    assert rec.outcome == "positive"
    assert rec.cited_study_ids == ["STUDY_OLD"]


def test_invalid_outcome_rejection():
    rec = EvidenceRecord(
        herb_id="HERB_001",
        drug_id="DRUG_001",
        outcome="alleged_interaction",  # not in allowed outcomes
        evidence_type="clinical",
        source_id="SOURCE_01",
        study_id="STUDY_01",
    )
    is_valid, errors = rec.validate()
    assert not is_valid
    assert any("outcome" in err for err in errors)


def test_unknown_is_not_negative():
    """Verify that unknown is represented distinctly from negative."""
    rec_unk = EvidenceRecord(
        herb_id="HERB_001",
        drug_id="DRUG_001",
        outcome="unknown",
        evidence_type="inconclusive",
        source_id="SOURCE_UNK",
        study_id="STUDY_UNK",
    )
    rec_neg = EvidenceRecord(
        herb_id="HERB_001",
        drug_id="DRUG_001",
        outcome="negative",
        evidence_type="clinical",
        source_id="SOURCE_NEG",
        study_id="STUDY_NEG",
    )
    assert rec_unk.outcome != rec_neg.outcome
    assert rec_unk.outcome == EvidenceOutcome.UNKNOWN.value
    assert rec_neg.outcome == EvidenceOutcome.NEGATIVE.value


def test_null_nan_string_rejection():
    """Verify that null, NaN, or string 'nan' / 'None' are caught and rejected as invalid identifiers."""
    rec_nan = EvidenceRecord(
        herb_id="nan",
        drug_id="DRUG_001",
        outcome="positive",
        evidence_type="clinical",
        source_id="S1",
        study_id="ST1",
    )
    is_valid, errors = rec_nan.validate()
    assert not is_valid
    assert any("herb_id" in e for e in errors)

    rec_none = EvidenceRecord(
        herb_id="HERB_001",
        drug_id="None",
        outcome="positive",
        evidence_type="clinical",
        source_id="S1",
        study_id="ST1",
    )
    is_valid, errors = rec_none.validate()
    assert not is_valid
    assert any("drug_id" in e for e in errors)


def test_dataframe_validation_with_rejected_rows():
    records = [
        # Valid row 0
        {
            "herb_id": "HERB_01",
            "drug_id": "DRUG_01",
            "outcome": "positive",
            "evidence_type": "clinical",
            "source_id": "S1",
            "study_id": "ST1",
        },
        # Invalid row 1 (NaN herb_id)
        {
            "herb_id": np.nan,
            "drug_id": "DRUG_02",
            "outcome": "positive",
            "evidence_type": "clinical",
            "source_id": "S2",
            "study_id": "ST2",
        },
        # Invalid row 2 (string 'nan' drug_id)
        {
            "herb_id": "HERB_03",
            "drug_id": "nan",
            "outcome": "positive",
            "evidence_type": "clinical",
            "source_id": "S3",
            "study_id": "ST3",
        },
        # Invalid row 3 (bad outcome)
        {
            "herb_id": "HERB_04",
            "drug_id": "DRUG_04",
            "outcome": "unsupported_label",
            "evidence_type": "clinical",
            "source_id": "S4",
            "study_id": "ST4",
        },
    ]
    df = pd.DataFrame(records)
    cleaned_df, errors = validate_dataframe(df)

    # Exactly 1 valid row should pass
    assert len(cleaned_df) == 1
    assert cleaned_df.iloc[0]["herb_id"] == "HERB_01"

    # Exactly 3 rows should be rejected with clear error reports
    assert len(errors) == 3
    assert any("Row 1 rejected" in e for e in errors)
    assert any("Row 2 rejected" in e for e in errors)
    assert any("Row 3 rejected" in e for e in errors)


def test_dataframe_export_and_roundtrip(tmp_path):
    records = [
        {
            "herb_id": "H1",
            "drug_id": "D1",
            "outcome": "positive",
            "evidence_type": "clinical",
            "source_id": "S1",
            "study_id": "ST1",
            "cited_study_ids": ["ST0"],
        },
        {
            "herb_id": "H2",
            "drug_id": "D2",
            "outcome": "negative",
            "evidence_type": "animal",
            "source_id": "S2",
            "study_id": "ST2",
            "cited_study_ids": [],
        },
    ]
    df = pd.DataFrame(records)
    cleaned_df, errors = validate_dataframe(df)
    assert len(errors) == 0
    assert len(cleaned_df) == 2

    # Test export and reload
    csv_file = str(tmp_path / "test_evidence.csv")
    export_canonical_table(cleaned_df, csv_file)
    reloaded_df = load_canonical_table(csv_file)
    assert len(reloaded_df) == 2
    assert reloaded_df.iloc[0]["cited_study_ids"] == ["ST0"]


def test_strict_traceability_and_source_reference_validation():
    """
    P1 Fix: Verify that strict traceability rejects missing or malformed DOIs/PMIDs.
    """
    # 1. Missing source_url_or_doi
    rec_no_doi = EvidenceRecord(
        herb_id="HERB_001",
        drug_id="DRUG_001",
        outcome="positive",
        evidence_type="clinical",
        source_id="PUBMED_10534248",
        study_id="STUDY_001",
        source_url_or_doi=None,
    )
    is_valid, errors = rec_no_doi.validate(strict_traceability=True)
    assert not is_valid
    assert any("source_url_or_doi cannot be empty" in e for e in errors)

    # 2. Malformed non-traceable string
    rec_bad_doi = EvidenceRecord(
        herb_id="HERB_001",
        drug_id="DRUG_001",
        outcome="positive",
        evidence_type="clinical",
        source_id="LOCAL_SOURCE",
        study_id="STUDY_001",
        source_url_or_doi="arbitrary_fake_reference",
    )
    is_valid, errors = rec_bad_doi.validate(strict_traceability=True)
    assert not is_valid
    assert any("Unverifiable source reference" in e for e in errors)

    # 3. Valid traceable DOI
    rec_valid_doi = EvidenceRecord(
        herb_id="HERB_001",
        drug_id="DRUG_001",
        outcome="positive",
        evidence_type="clinical",
        source_id="PUBMED_10534248",
        study_id="STUDY_001",
        source_url_or_doi="https://doi.org/10.1016/s0140-6736(99)05018-9",
    )
    is_valid, errors = rec_valid_doi.validate(strict_traceability=True)
    assert is_valid
    assert len(errors) == 0


def test_validate_dataframe_all_invalid():
    """
    Verify that when all rows are invalid, validate_dataframe cleanly returns
    an empty DataFrame and captures row-by-row error logs.
    """
    records = [
        {"herb_id": "H1", "drug_id": "D1", "outcome": "bad_label", "evidence_type": "clinical", "source_id": "S1", "study_id": "ST1"},
        {"herb_id": "H2", "drug_id": "D2", "outcome": "positive", "evidence_type": "bad_type", "source_id": "S2", "study_id": "ST2"},
    ]
    df = pd.DataFrame(records)
    cleaned_df, errors = validate_dataframe(df)
    assert cleaned_df.empty
    assert len(errors) == 2


def test_consensus_pair_record_validation():
    """Verify consensus validation, empty-table handling, and design extraction."""
    from src.data.evidence_schema import (
        ConsensusPairRecord,
        validate_consensus_labels_table,
        load_consensus_labels_table,
        extract_study_finding,
        extract_study_design_fields,
        StudyFinding,
    )
    # 1. Valid record
    rec = ConsensusPairRecord(
        herb_id="HERB_HYPERICUM",
        drug_id="DRUG_CYCLOSPORINE",
        consensus_label="interaction",
        source_name="Stockley's Drug Interactions",
        source_url_or_citation="Stockley 10th Ed.",
        retrieval_date="2026-10-28",
    )
    is_valid, errors = rec.validate()
    assert is_valid
    assert len(errors) == 0

    # 2. Invalid record (bad label)
    bad_rec = ConsensusPairRecord(
        herb_id="HERB_HYPERICUM",
        drug_id="DRUG_CYCLOSPORINE",
        consensus_label="alleged_interaction",
    )
    is_valid, errors = bad_rec.validate()
    assert not is_valid
    assert any("Invalid consensus_label" in e for e in errors)

    # 3. Empty template is valid but has no labels; no legacy targets are created.
    df_empty, labels_unavailable = load_consensus_labels_table("data/benchmark/consensus_labels.csv")
    assert labels_unavailable
    assert len(df_empty) == 0

    # 4. Study design extractor (must exclude outcome)
    row = {
        "species": "human",
        "evidence_type": "rct",
        "route": "oral",
        "dose": "300 mg TID",
        "original_text": "Significantly increased clearance of cyclosporine.",
        "outcome": "positive",
    }
    design = extract_study_design_fields(row)
    assert design["species"] == "human"
    assert design["assay"] == "rct"
    assert "original_text" not in design
    assert "outcome" not in design

    # 5. Finding extraction
    assert extract_study_finding("positive") == StudyFinding.POSITIVE
    assert extract_study_finding("no_interaction_observed") == StudyFinding.NEGATIVE
    assert extract_study_finding("conflicting") == StudyFinding.NULL

