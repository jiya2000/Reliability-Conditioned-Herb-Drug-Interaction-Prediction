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
import re
from datetime import date
from urllib.parse import urlparse
import os
import pandas as pd
from loguru import logger


class EvidenceOutcome(str, Enum):
    """
    Permitted canonical outcome labels separating interaction presence from exposure direction.
    """
    INTERACTION_PRESENT = "interaction_present"      # Interaction verified (exposure increased, decreased, or toxicity altered)
    NO_INTERACTION_OBSERVED = "no_interaction_observed"  # Rigorously evaluated; confirmed NO interaction
    POSITIVE = "positive"                            # Standard alias for INTERACTION_PRESENT
    NEGATIVE = "negative"                            # Standard alias for NO_INTERACTION_OBSERVED
    CONFLICTING = "conflicting"                      # Conflicting or equivocal findings
    UNKNOWN = "unknown"                              # Insufficient or absent evidence


class EffectDirection(str, Enum):
    """Specific directional and pharmacological mechanism of interaction."""
    DECREASED_EXPOSURE = "decreased_exposure"        # e.g., CYP/P-gp induction lowering drug exposure
    INCREASED_EXPOSURE = "increased_exposure"        # e.g., CYP/P-gp inhibition increasing drug exposure
    INCREASED_TOXICITY = "increased_toxicity"        # e.g., additive bleeding, adverse event
    DECREASED_EFFECT = "decreased_effect"            # e.g., antagonistic pharmacodynamic outcome
    NO_EFFECT = "no_effect"                          # True negative control finding
    CONTRADICTORY = "contradictory"
    UNSPECIFIED = "unspecified"


class ClinicalSignificance(str, Enum):
    """Clinical severity and impact level of the observed interaction."""
    MAJOR = "major"              # Critical risk (organ rejection, hemorrhage, hospitalization, life-threatening)
    MODERATE = "moderate"        # Clinically meaningful change requiring dose adjustment or close monitoring
    MINOR = "minor"              # Statistically detectable change without major clinical consequence
    NONE = "none"                # No interaction / negative clinical trial
    UNCERTAIN = "uncertain"      # Equivocal or uncharacterized


class EvidenceType(str, Enum):
    """Hierarchy of biological / clinical evidence types."""
    META_ANALYSIS = "meta_analysis"    # Systematic review and meta-analysis of clinical trials
    RCT = "rct"                        # Randomized controlled trial in humans
    CLINICAL = "clinical"              # Clinical cohort, pharmacokinetic trial, controlled human study
    OBSERVATIONAL = "observational"    # Observational study, registry, pharmacoepidemiology
    CASE_REPORT = "case_report"        # Documented human clinical case report
    ANIMAL = "animal"                  # In vivo rodent / mammalian pharmacokinetic or pharmacodynamic study
    IN_VITRO = "in_vitro"              # In vitro CYP inhibition, Caco-2 permeability, transporter assay
    MECHANISTIC = "mechanistic"        # Target binding, pathway overlap, structural homology
    IN_SILICO = "in_silico"            # Molecular docking, computational prediction
    INCONCLUSIVE = "inconclusive"      # Unspecified or inconclusive study design


VALID_OUTCOMES: Set[str] = {e.value for e in EvidenceOutcome}
VALID_EFFECT_DIRECTIONS: Set[str] = {e.value for e in EffectDirection}
VALID_CLINICAL_SIGNIFICANCES: Set[str] = {e.value for e in ClinicalSignificance}
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
    effect_direction: str = EffectDirection.UNSPECIFIED.value
    clinical_significance: Optional[str] = None
    # Phase 3 Optional Pharmacokinetic Metadata Fields
    extract_standardization: Optional[str] = None  # e.g., "0.3% hyperforin", "95% curcuminoids"
    treatment_duration: Optional[str] = None       # e.g., "14 days", "single dose", "4 weeks"
    interaction_direction: Optional[str] = None    # "induction", "inhibition", "none", "unspecified"
    time_dependence: Optional[str] = None          # "reversible", "mechanism_based", "time_dependent", "tdi"

    def __post_init__(self):
        # Normalize and validate string values against null/nan leakage
        cleaned_herb = _clean_str(self.herb_id)
        self.herb_id = cleaned_herb if cleaned_herb is not None else ""

        cleaned_drug = _clean_str(self.drug_id)
        self.drug_id = cleaned_drug if cleaned_drug is not None else ""

        raw_out = str(self.outcome).strip().lower() if self.outcome else ""
        if raw_out in {"interaction_present", "positive", "no_interaction_observed", "negative", "conflicting", "unknown"}:
            self.outcome = raw_out
        elif raw_out == "no_effect":
            self.outcome = EvidenceOutcome.NO_INTERACTION_OBSERVED.value
        elif raw_out == "contradictory":
            self.outcome = EvidenceOutcome.CONFLICTING.value
        elif raw_out == "inconclusive":
            self.outcome = EvidenceOutcome.UNKNOWN.value
        else:
            self.outcome = raw_out

        raw_eff = str(self.effect_direction).strip().lower() if self.effect_direction else "unspecified"
        self.effect_direction = raw_eff if raw_eff in VALID_EFFECT_DIRECTIONS else EffectDirection.UNSPECIFIED.value
        self.clinical_significance = _clean_str(self.clinical_significance)

        self.evidence_type = str(self.evidence_type).strip().lower() if self.evidence_type else ""

        cleaned_source = _clean_str(self.source_id)
        self.source_id = cleaned_source if cleaned_source is not None else ""

        cleaned_study = _clean_str(self.study_id)
        self.study_id = cleaned_study if cleaned_study is not None else self.source_id

        # Phase 3 optional PK fields normalization
        self.extract_standardization = _clean_str(self.extract_standardization)
        self.treatment_duration = _clean_str(self.treatment_duration)
        self.interaction_direction = _clean_str(self.interaction_direction)
        self.time_dependence = _clean_str(self.time_dependence)

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

    @property
    def is_interaction(self) -> Optional[bool]:
        """Binary classification ground truth: True for interaction, False for no interaction, None for uncertain."""
        if self.outcome in {EvidenceOutcome.INTERACTION_PRESENT.value, EvidenceOutcome.POSITIVE.value}:
            return True
        elif self.outcome in {EvidenceOutcome.NO_INTERACTION_OBSERVED.value, EvidenceOutcome.NEGATIVE.value}:
            return False
        return None

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
                effect_direction=row.get("effect_direction", EffectDirection.UNSPECIFIED.value),
                clinical_significance=row.get("clinical_significance"),
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


# =============================================================================
# Pair-Level Ground-Truth Consensus Schema & Study-Design Decoupling
# =============================================================================

class ConsensusLabel(str, Enum):
    """
    Expert consensus interaction label for a herb-drug pair from authoritative
    pharmacology compendia (NaPDI, Natural Medicines, Stockley's, DrugBank).
    Decoupled from individual study findings.
    """
    INTERACTION = "interaction"
    NO_INTERACTION = "no_interaction"
    DISPUTED = "disputed"


VALID_CONSENSUS_LABELS: Set[str] = {c.value for c in ConsensusLabel}


CONSENSUS_REQUIRED_COLUMNS = (
    "herb_id", "drug_id", "consensus_label", "source_name",
    "source_url_or_citation", "retrieval_date",
)


class ConsensusLabelsNotPopulatedError(RuntimeError):
    """Consensus-dependent training/evaluation cannot run without curated labels."""


def _consensus_text(value: Any) -> Optional[str]:
    if not isinstance(value, str):
        return None
    text = _clean_str(value)
    if text is None or text.lower() in {"unverified", "unknown", "n/a", "na", "tbd", "todo"}:
        return None
    return text


@dataclass
class ConsensusPairRecord:
    """External pair verdict; validation checks structure, not source authenticity."""
    herb_id: str
    drug_id: str
    consensus_label: str
    severity: Optional[str] = None
    direction: Optional[str] = None
    source_name: Optional[str] = None
    source_url_or_citation: Optional[str] = None
    retrieval_date: Optional[str] = None
    notes: Optional[str] = None

    def validate(self) -> Tuple[bool, List[str]]:
        errors = []
        for field_name, prefix in (("herb_id", "HERB_"), ("drug_id", "DRUG_")):
            value = _consensus_text(getattr(self, field_name))
            if value is None or re.fullmatch(prefix + r"[A-Z0-9]+(?:_[A-Z0-9]+)*", value) is None:
                errors.append(f"Invalid {field_name}: expected {prefix} identifier")
        label = _consensus_text(self.consensus_label)
        if label is None or label.lower() not in VALID_CONSENSUS_LABELS:
            errors.append(f"Invalid consensus_label '{self.consensus_label}'")
        for field_name in ("source_name", "source_url_or_citation", "retrieval_date"):
            if _consensus_text(getattr(self, field_name)) is None:
                errors.append(f"Missing required provenance: {field_name}")
        citation = _consensus_text(self.source_url_or_citation)
        if citation and ("://" in citation or citation.lower().startswith(("http:", "https:"))):
            try:
                parsed = urlparse(citation)
                valid_url = (parsed.scheme in {"http", "https"} and parsed.hostname
                             and not re.search(r"\s", citation))
                parsed.port  # Reject malformed port numbers as well.
            except ValueError:
                valid_url = False
            if not valid_url:
                errors.append("Invalid source_url_or_citation URL")
        elif citation and citation.lower().startswith(("pmid:", "doi:", "10.")):
            if citation.lower().startswith("pmid:"):
                valid_reference = re.fullmatch(r"PMID:\s*[1-9][0-9]*", citation, re.IGNORECASE)
            else:
                valid_reference = re.fullmatch(r"(?:doi:\s*)?10\.\d{4,9}/\S+", citation, re.IGNORECASE)
            if not valid_reference:
                errors.append("Invalid source_url_or_citation identifier")
        retrieval = _consensus_text(self.retrieval_date)
        if retrieval:
            try:
                if date.fromisoformat(retrieval).isoformat() != retrieval:
                    raise ValueError
            except ValueError:
                errors.append("Invalid retrieval_date: expected YYYY-MM-DD")
        return not errors, errors


def validate_consensus_labels_table(df: pd.DataFrame) -> Tuple[bool, List[str]]:
    """Validate all rows; a schema-bearing empty table is valid, blank labels are not.

    Bibliographic citations are allowed alongside HTTP(S) URLs. Nonempty source
    metadata does not authenticate a clinical verdict or establish independence.
    """
    missing = [c for c in CONSENSUS_REQUIRED_COLUMNS if c not in df.columns]
    if missing:
        return False, [f"Missing required columns in consensus table: {missing}"]
    if df.columns.duplicated().any():
        return False, ["Duplicate columns in consensus table"]
    errors = []
    pairs = df[["herb_id", "drug_id"]].apply(lambda col: col.map(_consensus_text))
    if pairs.duplicated().any():
        errors.append("Duplicate herb-drug pairs in consensus table")
    for idx, row in df.iterrows():
        rec = ConsensusPairRecord(**{
            name: row.get(name) for name in ConsensusPairRecord.__dataclass_fields__
        })
        _, row_errors = rec.validate()
        errors.extend(f"Row {idx}: {error}" for error in row_errors)
    return not errors, errors


def load_consensus_labels_table(
    path: str = "data/benchmark/consensus_labels.csv", *, require_labels: bool = False,
) -> Tuple[pd.DataFrame, bool]:
    """Load validated consensus data without synthesizing targets or legacy fallback.

    The second return value indicates labels are unavailable (kept for tuple API
    compatibility); it never authorizes use of study outcomes as consensus truth.
    Training/evaluation consumers must pass require_labels=True.
    Missing files and malformed files raise; only a valid header-only table is empty.
    """
    df = pd.read_csv(path)
    valid, errors = validate_consensus_labels_table(df)
    if not valid:
        raise ValueError("Consensus table validation failed:\n" + "\n".join(errors[:10]))
    if df.empty and require_labels:
        raise ConsensusLabelsNotPopulatedError(
            f"{path} has 0 validated rows. Consensus-dependent training and evaluation "
            "are BLOCKED until external labels are curated; no legacy fallback is allowed."
        )
    df = df.copy()
    for column in CONSENSUS_REQUIRED_COLUMNS:
        df[column] = df[column].map(lambda value: value.strip())
    df["consensus_label"] = df["consensus_label"].str.lower()
    return df, df.empty


class StudyFinding(str, Enum):
    """Individual study observation direction."""
    POSITIVE = "positive"
    NEGATIVE = "negative"
    NULL = "null"


def extract_study_finding(outcome_str: Any) -> StudyFinding:
    """Normalize raw study finding string to StudyFinding enum."""
    s = str(outcome_str).lower().strip()
    if s in {"interaction_present", "positive", "1", "true"}:
        return StudyFinding.POSITIVE
    elif s in {"no_interaction_observed", "negative", "0", "false"}:
        return StudyFinding.NEGATIVE
    return StudyFinding.NULL


def extract_study_design_fields(row: Dict[str, Any] | pd.Series) -> Dict[str, Any]:
    """
    Extract strictly methodological/design fields from a study record.
    Guaranteed to contain zero study outcome conclusions.
    """
    return {
        "species": _clean_str(row.get("species", "unknown")),
        "assay": _clean_str(row.get("evidence_type", "unknown")),
        "route": _clean_str(row.get("route", "oral")),
        "preparation": _clean_str(row.get("preparation", "standardized_extract")),
        "dose": _clean_str(row.get("dose")),
        "population": _clean_str(row.get("population")),
        "extract_standardization": _clean_str(row.get("extract_standardization")),
    }
