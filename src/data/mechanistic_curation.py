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
    "value",
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
    
    Returns:
        (is_verified, verified_count, unverified_count)
    """
    validate_curation_dataframe(df)

    is_unverified_source = df["source"].astype(str).str.upper().str.strip() == "UNVERIFIED"
    is_unverified_cite = df["citation"].astype(str).str.upper().str.strip() == "UNVERIFIED"
    has_empty_value = df["value"].isna() | (df["value"].astype(str).str.strip() == "")

    unverified_mask = is_unverified_source | is_unverified_cite | has_empty_value

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
