"""
Evidence Schema for Reliability- and Context-Aware Herb–Drug Interaction Prediction.

Defines the canonical interaction record schema and validation rules.
Granularity: Exactly ONE row per evidence record (not one row per herb–drug pair).

Crucial rule:
Unknown is represented as the absence of sufficient evidence, never as a negative interaction.
Negative outcome requires an explicit negative finding reported by a source.
"""

from __future__ import annotations

from dataclasses import dataclass, field, asdict
from enum import Enum
from typing import Any, Dict, List, Optional, Set, Tuple
import json
import pandas as pd
from loguru import logger


class EvidenceOutcome(str, Enum):
    """Permitted canonical outcome labels."""
    POSITIVE = "positive"          # Source confirms an interaction exists
    NEGATIVE = "negative"          # Source explicitly evaluated and found NO significant interaction
    CONFLICTING = "conflicting"    # Single source reports equivocal or mixed results
    UNKNOWN = "unknown"            # Absence of sufficient evidence / unverified candidate


class EvidenceType(str, Enum):
    """Hierarchy of biological / clinical evidence types."""
    CLINICAL = "clinical"              # Randomized controlled trial, human cohort, clinical pharmacokinetic study
    CASE_REPORT = "case_report"        # Documented human clinical case report
    ANIMAL = "animal"                  # In vivo rodent / mammalian pharmacokinetic or pharmacodynamic study
    IN_VITRO = "in_vitro"              # In vitro CYP inhibition, Caco-2 permeability, transporter assay
    MECHANISTIC = "mechanistic"        # Target binding, pathway overlap, structural homology
    IN_SILICO = "in_silico"            # Molecular docking, computational prediction
    INCONCLUSIVE = "inconclusive"      # Unspecified or inconclusive study design


VALID_OUTCOMES: Set[str] = {e.value for e in EvidenceOutcome}
VALID_EVIDENCE_TYPES: Set[str] = {e.value for e in EvidenceType}


@dataclass
class EvidenceRecord:
    """
    Canonical evidence record representing an individual piece of interaction evidence.
    """
    herb_id: str
    drug_id: str
    outcome: str
    evidence_type: str
    source_id: str
    study_id: str
    cited_study_ids: List[str] = field(default_factory=list)
    dose: Optional[str] = None
    preparation: Optional[str] = None
    route: Optional[str] = None
    species: Optional[str] = None
    population: Optional[str] = None
    publication_date: Optional[str] = None
    extraction_confidence: float = 1.0
    source_url_or_doi: Optional[str] = None
    original_text: Optional[str] = None
    extraction_method: str = "manual_curation"
    raw_label: Optional[str] = None

    def __post_init__(self):
        # Normalize string values
        self.herb_id = str(self.herb_id).strip()
        self.drug_id = str(self.drug_id).strip()
        self.outcome = str(self.outcome).strip().lower()
        self.evidence_type = str(self.evidence_type).strip().lower()
        self.source_id = str(self.source_id).strip()
        self.study_id = str(self.study_id).strip() if self.study_id else self.source_id

        # Normalize cited_study_ids
        if isinstance(self.cited_study_ids, str):
            try:
                parsed = json.loads(self.cited_study_ids)
                self.cited_study_ids = [str(x).strip() for x in parsed if str(x).strip()]
            except Exception:
                # Comma or semicolon separated
                parts = [p.strip() for p in self.cited_study_ids.replace(";", ",").split(",")]
                self.cited_study_ids = [p for p in parts if p]
        elif isinstance(self.cited_study_ids, (list, tuple, set)):
            self.cited_study_ids = [str(x).strip() for x in self.cited_study_ids if str(x).strip()]
        else:
            self.cited_study_ids = []

        # Validate extraction confidence
        try:
            self.extraction_confidence = float(self.extraction_confidence)
            self.extraction_confidence = max(0.0, min(1.0, self.extraction_confidence))
        except (ValueError, TypeError):
            self.extraction_confidence = 0.5

    def validate(self) -> Tuple[bool, List[str]]:
        """Validate instance fields according to the canonical evidence contract."""
        errors: List[str] = []

        if not self.herb_id:
            errors.append("herb_id cannot be empty")
        if not self.drug_id:
            errors.append("drug_id cannot be empty")

        if self.outcome not in VALID_OUTCOMES:
            errors.append(
                f"outcome '{self.outcome}' is invalid; must be one of {sorted(VALID_OUTCOMES)}"
            )

        if self.evidence_type not in VALID_EVIDENCE_TYPES:
            errors.append(
                f"evidence_type '{self.evidence_type}' is invalid; must be one of {sorted(VALID_EVIDENCE_TYPES)}"
            )

        if not self.source_id:
            errors.append("source_id cannot be empty (traceability requirement)")

        if not self.study_id:
            errors.append("study_id cannot be empty (lineage tracking requirement)")

        return len(errors) == 0, errors

    def to_dict(self) -> Dict[str, Any]:
        """Convert record to dictionary, serializing lists to JSON string or list."""
        data = asdict(self)
        return data


def validate_dataframe(df: pd.DataFrame) -> Tuple[pd.DataFrame, List[str]]:
    """
    Validate a DataFrame of evidence records.

    Ensures all required columns exist, types and value constraints are met,
    and returns a cleaned DataFrame alongside validation error messages.
    """
    required_cols = [
        "herb_id",
        "drug_id",
        "outcome",
        "evidence_type",
        "source_id",
        "study_id",
    ]
    errors: List[str] = []

    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        errors.append(f"Missing mandatory columns: {missing}")
        return df, errors

    cleaned_records: List[EvidenceRecord] = []
    for idx, row in df.iterrows():
        try:
            rec = EvidenceRecord(
                herb_id=str(row["herb_id"]),
                drug_id=str(row["drug_id"]),
                outcome=str(row["outcome"]),
                evidence_type=str(row["evidence_type"]),
                source_id=str(row["source_id"]),
                study_id=str(row.get("study_id", row["source_id"])),
                cited_study_ids=row.get("cited_study_ids", []),
                dose=row.get("dose") if pd.notna(row.get("dose")) else None,
                preparation=row.get("preparation") if pd.notna(row.get("preparation")) else None,
                route=row.get("route") if pd.notna(row.get("route")) else None,
                species=row.get("species") if pd.notna(row.get("species")) else None,
                population=row.get("population") if pd.notna(row.get("population")) else None,
                publication_date=str(row["publication_date"]) if pd.notna(row.get("publication_date")) else None,
                extraction_confidence=row.get("extraction_confidence", 1.0),
                source_url_or_doi=str(row["source_url_or_doi"]) if pd.notna(row.get("source_url_or_doi")) else None,
                original_text=str(row["original_text"]) if pd.notna(row.get("original_text")) else None,
                extraction_method=str(row.get("extraction_method", "manual_curation")),
                raw_label=str(row.get("raw_label", "")) if pd.notna(row.get("raw_label")) else None,
            )
            is_valid, rec_errors = rec.validate()
            if not is_valid:
                errors.append(f"Row {idx} validation failure: {rec_errors}")
            else:
                cleaned_records.append(rec)
        except Exception as e:
            errors.append(f"Row {idx} exception during parsing: {e}")

    if not cleaned_records:
        cleaned_df = pd.DataFrame(columns=required_cols)
    else:
        cleaned_df = pd.DataFrame([r.to_dict() for r in cleaned_records])

    return cleaned_df, errors


def export_canonical_table(df: pd.DataFrame, output_path: str) -> None:
    """Save cleaned evidence table to CSV or Parquet with serialized JSON list columns."""
    save_df = df.copy()
    if "cited_study_ids" in save_df.columns:
        save_df["cited_study_ids"] = save_df["cited_study_ids"].apply(
            lambda x: json.dumps(x) if isinstance(x, (list, tuple, set)) else str(x)
        )

    if output_path.endswith(".parquet"):
        save_df.to_parquet(output_path, index=False)
    else:
        save_df.to_csv(output_path, index=False)
    logger.info(f"Exported {len(save_df)} canonical evidence records to {output_path}")


def load_canonical_table(input_path: str) -> pd.DataFrame:
    """Load canonical evidence records from CSV or Parquet."""
    if input_path.endswith(".parquet"):
        df = pd.read_parquet(input_path)
    else:
        df = pd.read_csv(input_path)

    if "cited_study_ids" in df.columns:
        def parse_citations(val):
            if pd.isna(val) or val == "":
                return []
            if isinstance(val, list):
                return val
            try:
                parsed = json.loads(val)
                if isinstance(parsed, list):
                    return parsed
            except Exception:
                pass
            return [p.strip() for p in str(val).split(",") if p.strip()]

        df["cited_study_ids"] = df["cited_study_ids"].apply(parse_citations)

    return df
