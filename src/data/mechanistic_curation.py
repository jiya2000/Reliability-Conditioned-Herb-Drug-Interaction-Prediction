"""
Mechanistic Feature Blind-Curation Pipeline (Phase 3).

Enforces strict provenance and integrity checks for pharmacokinetic target features:
- All values must come from verifiable external literature / databases.
- Any unverified or unsourced entity-target entry must be tagged 'UNVERIFIED'
  with the numeric value left empty.
- Prevents unverified curation rows from entering model training or benchmark evaluation.
"""

import os
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

    df = pd.read_csv(csv_path)
    validate_curation_dataframe(df)
    return df


def validate_curation_dataframe(df: pd.DataFrame) -> None:
    """
    Validate that the curation table contains required columns and conforms to integrity rules.
    """
    for col in CURATION_COLUMNS:
        if col not in df.columns:
            raise ValueError(f"Mechanistic curation table is missing required column: '{col}'")

    # Check for empty entities or targets
    if df["entity"].isna().any() or (df["entity"].astype(str).str.strip() == "").any():
        raise ValueError("Mechanistic curation table contains rows with missing 'entity'")
    if df["target"].isna().any() or (df["target"].astype(str).str.strip() == "").any():
        raise ValueError("Mechanistic curation table contains rows with missing 'target'")


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
    Extract pairwise mechanistic interaction feature matrix from the verified curation table.
    
    Strict Guard:
    Raises UnverifiedCurationError if any rows in curation_df are unverified,
    preventing unverified numbers from entering the model.
    """
    if curation_df is None:
        curation_df = load_mechanistic_curation()

    verified, n_ver, n_unver = is_curation_verified(curation_df)
    if not verified and not allow_unverified:
        raise UnverifiedCurationError(
            f"Cannot extract new mechanistic features: curation table contains {n_unver} UNVERIFIED rows "
            f"and only {n_ver} verified entries. New-feature model evaluation is strictly blocked until "
            f"all entities have human-verified sources and citations."
        )

    # If verified, build entity -> target -> value map
    entity_map: Dict[str, Dict[str, float]] = {}
    for _, row in curation_df.iterrows():
        ent = str(row["entity"]).strip()
        tgt = str(row["target"]).strip()
        val = row["value"]
        if pd.notna(val) and str(val).strip() != "":
            try:
                num_val = float(val)
                if ent not in entity_map:
                    entity_map[ent] = {}
                entity_map[ent][tgt] = num_val
            except ValueError:
                continue

    target_to_idx = {t: i for i, t in enumerate(CANONICAL_TARGETS)}
    n_targets = len(CANONICAL_TARGETS)

    # Compute Hadamard product interaction vector for each pair
    features = []
    for _, row in df_pairs.iterrows():
        herb = str(row["herb_id"]).strip()
        drug = str(row["drug_id"]).strip()

        h_vec = np.zeros(n_targets, dtype=np.float32)
        d_vec = np.zeros(n_targets, dtype=np.float32)

        for tgt, val in entity_map.get(herb, {}).items():
            if tgt in target_to_idx:
                h_vec[target_to_idx[tgt]] = val
        for tgt, val in entity_map.get(drug, {}).items():
            if tgt in target_to_idx:
                d_vec[target_to_idx[tgt]] = val

        interaction_vec = h_vec * d_vec
        overlap_total = float(np.sum(np.abs(interaction_vec)))
        # Vector: [interaction_vec (9-D), overlap_total (1-D)] -> 10-D
        feat = np.concatenate([interaction_vec, [overlap_total]])
        features.append(feat)

    return np.array(features, dtype=np.float32)
