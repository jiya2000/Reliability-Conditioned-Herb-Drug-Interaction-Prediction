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


def _clean_str(val: Any) -> Optional[str]:
    """Helper to detect and eliminate null, NaN, None, or empty string representations."""
    if val is None or pd.isna(val):
        return None
    s = str(val).strip()
    if s.lower() in {"", "nan", "none", "null", "undefined", "."}:
        return None
    return s


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
        # Normalize and validate string values against null/nan leakage
        cleaned_herb = _clean_str(self.herb_id)
        self.herb_id = cleaned_herb if cleaned_herb is not None else ""

        cleaned_drug = _clean_str(self.drug_id)
        self.drug_id = cleaned_drug if cleaned_drug is not None else ""

        self.outcome = str(self.outcome).strip().lower() if self.outcome else ""
        self.evidence_type = str(self.evidence_type).strip().lower() if self.evidence_type else ""

        cleaned_source = _clean_str(self.source_id)
        self.source_id = cleaned_source if cleaned_source is not None else ""

        cleaned_study = _clean_str(self.study_id)
        self.study_id = cleaned_study if cleaned_study is not None else self.source_id

        # Normalize cited_study_ids
        if isinstance(self.cited_study_ids, str):
            try:
                parsed = json.loads(self.cited_study_ids)
                self.cited_study_ids = [s for x in parsed if (s := _clean_str(x))]
            except Exception:
                parts = [p.strip() for p in self.cited_study_ids.replace(";", ",").split(",")]
                self.cited_study_ids = [s for p in parts if (s := _clean_str(p))]
        elif isinstance(self.cited_study_ids, (list, tuple, set)):
            self.cited_study_ids = [s for x in self.cited_study_ids if (s := _clean_str(x))]
        else:
            self.cited_study_ids = []

        # Validate extraction confidence
        try:
            self.extraction_confidence = float(self.extraction_confidence)
            self.extraction_confidence = max(0.0, min(1.0, self.extraction_confidence))
        except (ValueError, TypeError):
            self.extraction_confidence = 0.5

        # Clean optional strings
        self.dose = _clean_str(self.dose)
        self.preparation = _clean_str(self.preparation)
        self.route = _clean_str(self.route)
        self.species = _clean_str(self.species)
        self.population = _clean_str(self.population)
        self.publication_date = _clean_str(self.publication_date)
        self.source_url_or_doi = _clean_str(self.source_url_or_doi)
        self.original_text = _clean_str(self.original_text)

    def validate(
        self,
        strict_traceability: bool = False,
        require_registered: bool = False,
    ) -> Tuple[bool, List[str]]:
        """
        Validate instance fields according to the canonical evidence contract.
        If strict_traceability=True, requires a valid, resolvable DOI, PubMed URL, or PMID.
        """
        errors: List[str] = []

        if not self.herb_id:
            errors.append("herb_id cannot be null, empty, or 'NaN'")
        if not self.drug_id:
            errors.append("drug_id cannot be null, empty, or 'NaN'")

        if self.outcome not in VALID_OUTCOMES:
            errors.append(
                f"outcome '{self.outcome}' is invalid; must be one of {sorted(VALID_OUTCOMES)}"
            )

        if self.evidence_type not in VALID_EVIDENCE_TYPES:
            errors.append(
                f"evidence_type '{self.evidence_type}' is invalid; must be one of {sorted(VALID_EVIDENCE_TYPES)}"
            )

        if not self.source_id:
            errors.append("source_id cannot be empty or null (traceability requirement)")

        if not self.study_id:
            errors.append("study_id cannot be empty or null (lineage tracking requirement)")

        if strict_traceability:
            if not self.source_url_or_doi:
                errors.append("source_url_or_doi cannot be empty under strict traceability mode")
            else:
                from src.data.source_registry import DEFAULT_SOURCE_REGISTRY
                is_valid_ref, reason = DEFAULT_SOURCE_REGISTRY.verify_reference(
                    self.source_id, self.source_url_or_doi, require_registered=require_registered
                )
                if not is_valid_ref:
                    errors.append(f"Unverifiable source reference: {reason}")

        return len(errors) == 0, errors

    def to_dict(self) -> Dict[str, Any]:
        """Convert record to dictionary, serializing lists to JSON string or list."""
        data = asdict(self)
        return data


def validate_dataframe(
    df: pd.DataFrame,
    strict_traceability: bool = False,
    require_registered: bool = False,
) -> Tuple[pd.DataFrame, List[str]]:
    """
    Validate a DataFrame of evidence records.

    Ensures all required columns exist, null/nan values are caught before conversion,
    validates traceable references when strict_traceability=True,
    and returns a cleaned DataFrame alongside validation error messages for rejected rows.
    """
    required_cols = [
        "herb_id",
        "drug_id",
        "outcome",
        "evidence_type",
        "source_id",
        "study_id",
    ]
    if strict_traceability:
        required_cols.append("source_url_or_doi")

    errors: List[str] = []

    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        errors.append(f"Missing mandatory columns: {missing}")
        return pd.DataFrame(columns=required_cols), errors

    cleaned_records: List[EvidenceRecord] = []
    for idx, row in df.iterrows():
        # Check required fields for explicit nulls before construction
        null_fields = [c for c in required_cols if _clean_str(row.get(c)) is None]
        if null_fields:
            errors.append(f"Row {idx} rejected: mandatory fields contain null/NaN values: {null_fields}")
            continue

        try:
            rec = EvidenceRecord(
                herb_id=row["herb_id"],
                drug_id=row["drug_id"],
                outcome=row["outcome"],
                evidence_type=row["evidence_type"],
                source_id=row["source_id"],
                study_id=row.get("study_id", row["source_id"]),
                cited_study_ids=row.get("cited_study_ids", []),
                dose=row.get("dose"),
                preparation=row.get("preparation"),
                route=row.get("route"),
                species=row.get("species"),
                population=row.get("population"),
                publication_date=row.get("publication_date"),
                extraction_confidence=row.get("extraction_confidence", 1.0),
                source_url_or_doi=row.get("source_url_or_doi"),
                original_text=row.get("original_text"),
                extraction_method=str(row.get("extraction_method", "manual_curation")),
                raw_label=row.get("raw_label"),
            )
            is_valid, rec_errors = rec.validate(
                strict_traceability=strict_traceability,
                require_registered=require_registered,
            )
            if not is_valid:
                errors.append(f"Row {idx} rejected: {rec_errors}")
            else:
                cleaned_records.append(rec)
        except Exception as e:
            errors.append(f"Row {idx} rejected with exception during parsing: {e}")

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
