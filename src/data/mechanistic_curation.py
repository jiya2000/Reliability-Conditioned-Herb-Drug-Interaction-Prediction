"""
Mechanistic Feature Blind-Curation Pipeline (Phase 3).

Enforces strict provenance and integrity checks for pharmacokinetic target features:
- All values must come from verifiable external literature / databases.
- Any unverified or unsourced entity-target entry must be tagged 'UNVERIFIED'
  with the numeric value left empty.
- Prevents unverified curation rows from entering model training or benchmark evaluation.
"""

import os
import csv
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
import numpy as np
import pandas as pd


CANONICAL_TARGETS = [
    "CYP3A4",
    "CYP2D6",
    "CYP2C9",
    "CYP2C19",
    "CYP1A2",
    "CYP2E1",
    "ABCB1",    # P-glycoprotein efflux transporter
    "SLCO1B1",  # OATP1B1 hepatic uptake transporter
    "NR1I2",    # PXR nuclear receptor
]

CURATION_COLUMNS = [
    "entity",
    "target",
    "status",
    "value_ic50_uM",
    "value_ki_uM",
    "value_fold_induction",
    "value_fm",
    "value_substrate_flag",
    "exposure_conc_uM",
    "exposure_source",
    "assay_or_unit",
    "source",
    "citation",
    "retrieval_date",
    "notes",
]


class UnverifiedCurationError(RuntimeError):
    """Raised when attempting to execute model training or evaluation on unverified curation data."""
    pass


def load_mechanistic_curation(
    csv_path: str = "data/curation/mechanistic_features.csv"
) -> pd.DataFrame:
    """
    Load and validate the blind-curation mechanistic features table.
    """
    path = Path(csv_path)
    if not path.exists():
        raise FileNotFoundError(f"Mechanistic curation file not found at: {csv_path}")

    # pandas otherwise silently renames duplicate CSV headers (e.g. entity.1).
    with path.open(newline="", encoding="utf-8-sig") as handle:
        header = next(csv.reader(handle), [])
    if len(header) != len(set(header)):
        raise ValueError("Duplicate mechanistic curation columns")
    df = pd.read_csv(csv_path)
    validate_curation_dataframe(df)
    return df


def _canonical_identifier(value: Any, *, kind: str) -> str:
    """Strip boundary whitespace only; never coerce or repair ambiguous IDs."""
    if not isinstance(value, str):
        raise ValueError(f"Invalid {kind} identifier: expected a string")
    identifier = value.strip()
    if not identifier or any(c.isspace() for c in identifier):
        raise ValueError(f"Invalid {kind} identifier: empty or internal whitespace")
    if kind != "target":
        prefixes = ("HERB_", "DRUG_") if kind == "entity" else (kind,)
        if not any(identifier.startswith(p) and len(identifier) > len(p) for p in prefixes):
            raise ValueError(f"Invalid {kind} identifier: requires a role prefix and nonempty suffix")
    return identifier


def validate_curation_dataframe(df: pd.DataFrame) -> None:
    """
    Validate that the curation table contains required columns and conforms to integrity rules.
    """
    if not df.columns.is_unique:
        raise ValueError("Duplicate mechanistic curation columns")
    for col in CURATION_COLUMNS:
        if col not in df.columns:
            raise ValueError(f"Mechanistic curation table is missing required column: '{col}'")

    keys = pd.DataFrame({
        "entity": df["entity"].map(lambda v: _canonical_identifier(v, kind="entity")),
        "target": df["target"].map(lambda v: _canonical_identifier(v, kind="target")),
    })
    if keys.duplicated(["entity", "target"]).any():
        raise ValueError("Duplicate entity-target rows require explicit curation resolution")
    numeric = {
        "value_ic50_uM": (0, None, False), "value_ki_uM": (0, None, False),
        "value_fold_induction": (1, None, True), "value_fm": (0, 1, True),
        "value_substrate_flag": (0, 1, True), "exposure_conc_uM": (0, None, True),
    }
    for _, row in df.iterrows():
        if row["status"] not in {"measured", "tested_no_effect", "not_tested"}:
            raise ValueError("Invalid curation status")
        values = {}
        for col, (lower, upper, inclusive) in numeric.items():
            raw = row[col]
            if pd.isna(raw) or (isinstance(raw, str) and not raw.strip()):
                continue
            try:
                value = float(raw)
            except (TypeError, ValueError) as exc:
                raise ValueError(f"Invalid numeric value in {col}") from exc
            if (not np.isfinite(value) or value < lower or
                    (not inclusive and value == lower) or
                    (upper is not None and value > upper) or
                    (col == "value_substrate_flag" and value not in (0, 1))):
                raise ValueError(f"Out-of-range value in {col}")
            values[col] = value
        measurements = set(values) - {"exposure_conc_uM"}
        if row["status"] != "measured" and measurements:
            raise ValueError("Only measured rows may contain mechanistic values")
        inhibition = bool({"value_ic50_uM", "value_ki_uM"} & measurements)
        if inhibition and values.get("value_fold_induction", 1) > 1:
            raise ValueError("Conflicting inhibition and induction require curation resolution")
        ent = _canonical_identifier(row["entity"], kind="entity")
        if not ent.startswith(("HERB_", "DRUG_")):
            raise ValueError("Entity must use HERB_ or DRUG_ identifier")
        if ent.startswith("HERB_") and measurements & {"value_fm", "value_substrate_flag"}:
            raise ValueError("Drug-only measurements supplied for herb")
        if ent.startswith("DRUG_") and measurements & {
                "value_ic50_uM", "value_ki_uM", "value_fold_induction"}:
            raise ValueError("Herb modulation measurements supplied for drug")


def is_curation_verified(df: pd.DataFrame) -> Tuple[bool, int, int]:
    """
    Check whether all rows in the curation table are verified.
    
    A row is verified if and only if:
    1. 'source' is not empty and not 'UNVERIFIED'.
    2. 'citation' is not empty and not 'UNVERIFIED'.
    3. Status-specific validation:
       - 'measured': At least one quantitative or substrate value column must be populated.
       - 'tested_no_effect': Valid with external source/citation confirming no measurable effect.
       - 'not_tested': Allowed ONLY if accompanied by a citation or database lookup note
         verifying that an external search was conducted and no experimental data was found.
    
    Returns:
        (is_verified, verified_count, unverified_count)
    """
    validate_curation_dataframe(df)

    is_unverified_source = df["source"].astype(str).str.upper().str.strip().isin(["UNVERIFIED", "", "NAN", "NONE"])
    is_unverified_cite = df["citation"].astype(str).str.upper().str.strip().isin(["UNVERIFIED", "", "NAN", "NONE"])
    
    # Check measured rows
    status_measured = df["status"].astype(str).str.strip() == "measured"
    has_value_ic50 = df["value_ic50_uM"].apply(lambda x: str(x).strip() != "" and str(x).strip().lower() != "nan" if pd.notna(x) else False)
    has_value_ki = df["value_ki_uM"].apply(lambda x: str(x).strip() != "" and str(x).strip().lower() != "nan" if pd.notna(x) else False)
    has_value_fold = df["value_fold_induction"].apply(lambda x: str(x).strip() != "" and str(x).strip().lower() != "nan" if pd.notna(x) else False)
    has_value_fm = df["value_fm"].apply(lambda x: str(x).strip() != "" and str(x).strip().lower() != "nan" if pd.notna(x) else False)
    has_value_substrate = df["value_substrate_flag"].apply(lambda x: str(x).strip() not in ["", "nan", "none"] if pd.notna(x) else False)
    
    has_any_value = has_value_ic50 | has_value_ki | has_value_fold | has_value_fm | has_value_substrate
    empty_measured = status_measured & (~has_any_value)

    # Check not_tested rows: allowed only if a citation or note confirms lookup found no data
    status_not_tested = df["status"].astype(str).str.strip() == "not_tested"
    notes_str = df["notes"].fillna("").astype(str).str.lower().str.strip()
    has_lookup_note = notes_str.apply(lambda n: any(w in n for w in ["lookup", "searched", "no data", "not found", "checked", "verified absent"]))
    invalid_not_tested = status_not_tested & (is_unverified_cite & ~has_lookup_note)

    unverified_mask = is_unverified_source | is_unverified_cite | empty_measured | invalid_not_tested

    unverified_count = int(unverified_mask.sum())
    verified_count = int((~unverified_mask).sum())
    is_verified = (unverified_count == 0 and verified_count > 0)

    return is_verified, verified_count, unverified_count


def extract_curated_feature_matrix(
    df_pairs: pd.DataFrame,
    curation_df: Optional[pd.DataFrame] = None,
    allow_unverified: bool = False,
) -> np.ndarray:
    """
    Extract signed target-activity features; see docs/mechanistic_features.md.

    Missing pathways remain NaN. Output uses CANONICAL_TARGETS followed by
    sum(abs(pathways)), which is NaN if any pathway is unknown.
    allow_unverified is retained for API compatibility but never bypasses provenance.
    
    Strict Guard:
    Raises UnverifiedCurationError if any rows in curation_df are unverified,
    preventing unverified numbers from entering the model.
    """
    if curation_df is None:
        curation_df = load_mechanistic_curation()

    verified, n_ver, n_unver = is_curation_verified(curation_df)
    if not verified:
        raise UnverifiedCurationError(
            f"Cannot extract new mechanistic features: curation table contains {n_unver} UNVERIFIED rows "
            f"and only {n_ver} verified entries. New-feature model evaluation is strictly blocked until "
            f"all entities have human-verified sources and citations."
        )

    if not df_pairs.columns.is_unique:
        raise ValueError("Duplicate pair table columns")
    required_pairs = {"herb_id", "drug_id"}
    if not required_pairs.issubset(df_pairs.columns):
        raise ValueError("Pair table requires herb_id and drug_id")
    pair_keys = df_pairs.loc[:, ["herb_id", "drug_id"]].copy()
    for col, prefix in (("herb_id", "HERB_"), ("drug_id", "DRUG_")):
        pair_keys[col] = pair_keys[col].map(lambda v: _canonical_identifier(v, kind=prefix))

    rows = {(_canonical_identifier(r["entity"], kind="entity"),
             _canonical_identifier(r["target"], kind="target")): r
            for _, r in curation_df.iterrows()}

    def number(row, column):
        value = row[column]
        return np.nan if pd.isna(value) or str(value).strip() == "" else float(value)

    def modulation(row):
        if row is None or row["status"] == "not_tested":
            return np.nan
        if row["status"] == "tested_no_effect":
            return 0.0
        if any(np.isfinite(number(row, c)) for c in ("value_ic50_uM", "value_ki_uM")):
            return -1.0
        fold = number(row, "value_fold_induction")
        return np.nan if np.isnan(fold) else float(fold > 1)

    features = []
    for _, pair in pair_keys.iterrows():
        effects = []
        for target in CANONICAL_TARGETS:
            herb = rows.get((pair["herb_id"], target))
            drug = rows.get((pair["drug_id"], target))
            weight = np.nan
            if drug is not None and drug["status"] == "measured":
                # Clearance fraction has quantitative meaning; substrate status
                # is only a binary fallback, never an invented fraction.
                weight = number(drug, "value_fm")
                if np.isnan(weight):
                    weight = number(drug, "value_substrate_flag")
            elif drug is not None and drug["status"] == "tested_no_effect":
                weight = 0.0
            effect = modulation(herb)
            # A known null factor proves no contribution even if the other is unknown.
            effects.append(0.0 if effect == 0 or weight == 0 else effect * weight)
        effects = np.asarray(effects, dtype=np.float32)
        features.append(np.concatenate([effects, [np.abs(effects).sum()]]))
    return np.asarray(features, dtype=np.float32).reshape(-1, len(CANONICAL_TARGETS) + 1)
