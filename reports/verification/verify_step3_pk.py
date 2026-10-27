"""
Step 3 Verification Script: Pharmacokinetics & IVIVE Reachability.

Checks:
1. Hepatic & Intestinal formulas vs 3 hand-computed numeric examples.
2. Routing logic: CYP3A4/P-gp intestinal uses gut formula, hepatic uses plasma formula. Branch recorded in metadata.
3. Missing dose/MW degrades gracefully with missing_dose_or_mw=True, without crashing or silent 1.0.
4. Optional schema fields validate against existing benchmark CSV without breaking PublicDataLoader.
"""

from __future__ import annotations

import sys
from pathlib import Path
import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.models.pk_reachability import (
    PKReachabilityScreen,
    InVitroFinding,
    ReachabilityResult,
    HERBAL_PK_PARAMETERS,
)
from src.data.public_data_loader import PublicDataLoader
from src.data.evidence_schema import EvidenceRecord


def verify_numeric_examples():
    print("--- 1. Verification of Hepatic & Intestinal Formulas vs Hand-Computed Examples ---")
    scorer = PKReachabilityScreen()

    # Hand Example 1: Intestinal Berberine on P-gp / CYP3A4
    # Dose: 500 mg, MW: 336.4 g/mol, potency: 2.0 uM, site: intestinal
    # Hand calculation:
    # I_gut = (500 * 4000.0) / 336.4 = 2,000,000 / 336.4 = 5945.30321 uM
    # Ratio = 5945.30321 / 2.0 = 2972.6516
    # Threshold = 10.0 -> is_reachable = True, rho_reach = 1.0
    f1 = InVitroFinding(
        herb_constituent="berberine",
        enzyme_transporter="P-gp",
        interaction_type="inhibition",
        potency_value=2.0,
        potency_unit="uM",
        site_of_action="intestinal",
        dose_mg=500.0,
        molecular_weight=336.4,
    )
    res1 = scorer.screen(f1)
    hand_i_gut = (500.0 * 4000.0) / 336.4
    hand_ratio = hand_i_gut / 2.0
    print(f"Example 1 (Intestinal Berberine):")
    print(f"  Hand I_gut: {hand_i_gut:.4f} uM | Code I_gut: {res1.i_gut_um:.4f} uM")
    print(f"  Hand Ratio: {hand_ratio:.4f} | Code Ratio: {res1.ratio:.4f}")
    print(f"  Branch used: {res1.branch_used} | Reachable: {res1.is_reachable} | rho_reach: {res1.rho_reach}")
    assert abs(res1.i_gut_um - hand_i_gut) < 1e-4
    assert abs(res1.ratio - hand_ratio) < 1e-4
    assert res1.is_reachable is True
    assert res1.branch_used == "intestinal"
    assert res1.rho_reach == 1.0

    # Hand Example 2: Hepatic Hyperforin on CYP3A4
    # Hyperforin: Cmax_total = 0.5 uM, fu = 0.01 -> Cu,max = 0.005 uM
    # Reported potency: IC50 = 5.0 uM (weak in vitro assay)
    # Hand calculation:
    # Ratio = 0.005 / 5.0 = 0.0010
    # Threshold = 0.1 (FDA reversible CYP inhibition) -> is_reachable = False
    # rho_reach = min(1.0, 0.001) = 0.0010
    f2 = InVitroFinding(
        herb_constituent="hyperforin",
        enzyme_transporter="CYP3A4",
        interaction_type="inhibition",
        potency_value=5.0,
        potency_unit="uM",
        site_of_action="hepatic",
    )
    res2 = scorer.screen(f2)
    hand_cu = 0.005
    hand_ratio2 = 0.005 / 5.0
    print(f"\nExample 2 (Hepatic Hyperforin):")
    print(f"  Hand Cu,max: {hand_cu:.4f} uM | Code Cu,max: {res2.cu_max_human:.4f} uM")
    print(f"  Hand Ratio : {hand_ratio2:.4f} | Code Ratio : {res2.ratio:.4f}")
    print(f"  Branch used: {res2.branch_used} | Reachable: {res2.is_reachable} | rho_reach: {res2.rho_reach}")
    assert abs(res2.cu_max_human - hand_cu) < 1e-4
    assert abs(res2.ratio - hand_ratio2) < 1e-4
    assert res2.is_reachable is False
    assert res2.branch_used == "hepatic"
    assert abs(res2.rho_reach - 0.0010) < 1e-4

    # Hand Example 3: Missing dose & MW degradation on intestinal route
    # Unlisted herb constituent, dose_mg=None, molecular_weight=None
    # Defaults in code: dose_mg = 10.0, MW = 400.0
    # I_gut = (10.0 * 4000.0) / 400.0 = 100.0 uM
    # Potency: 25.0 uM
    # Ratio = 100.0 / 25.0 = 4.0
    # Threshold = 10.0 -> is_reachable = False (4.0 < 10.0)
    # missing_dose_or_mw = True, confidence = 'low'
    f3 = InVitroFinding(
        herb_constituent="unknown_novel_phyto",
        enzyme_transporter="intestinal_cyp3a4",
        interaction_type="inhibition",
        potency_value=25.0,
        potency_unit="uM",
        dose_mg=None,
        molecular_weight=None,
    )
    res3 = scorer.screen(f3)
    hand_i_gut3 = (10.0 * 4000.0) / 400.0
    hand_ratio3 = hand_i_gut3 / 25.0
    print(f"\nExample 3 (Missing Dose & MW Graceful Degradation):")
    print(f"  Hand I_gut: {hand_i_gut3:.4f} uM | Code I_gut: {res3.i_gut_um:.4f} uM")
    print(f"  Hand Ratio: {hand_ratio3:.4f} | Code Ratio: {res3.ratio:.4f}")
    print(f"  Branch: {res3.branch_used} | Reachable: {res3.is_reachable}")
    print(f"  missing_dose_or_mw flag: {res3.missing_dose_or_mw} | confidence: {res3.confidence}")
    assert abs(res3.i_gut_um - hand_i_gut3) < 1e-4
    assert abs(res3.ratio - hand_ratio3) < 1e-4
    assert res3.missing_dose_or_mw is True
    assert res3.confidence == "low"
    assert res3.branch_used == "intestinal"
    print("PASS: All 3 hand-computed numeric examples match code output exactly.")


def verify_routing_and_metadata():
    print("\n--- 2. Routing Logic and Metadata Verification ---")
    scorer = PKReachabilityScreen()

    # Route 1: CYP3A4 in Caco-2 preparation without explicit site_of_action
    f_gut = InVitroFinding(
        herb_constituent="hyperforin",
        enzyme_transporter="CYP3A4",
        interaction_type="inhibition",
        potency_value=0.5,
        preparation="caco-2",
        dose_mg=300.0,
        molecular_weight=536.6,
    )
    res_gut = scorer.screen(f_gut)
    print(f"Caco-2 CYP3A4 auto-routing branch: {res_gut.branch_used}")
    assert res_gut.branch_used == "intestinal"

    # Route 2: CYP2D6 in human liver microsomes
    f_hep = InVitroFinding(
        herb_constituent="berberine",
        enzyme_transporter="CYP2D6",
        interaction_type="inhibition",
        potency_value=1.5,
        preparation="microsomes",
    )
    res_hep = scorer.screen(f_hep)
    print(f"Microsomal CYP2D6 auto-routing branch: {res_hep.branch_used}")
    assert res_hep.branch_used == "hepatic"
    print("PASS: Site-of-action routing correctly directs intestinal vs hepatic pathways and records branch in metadata.")


def verify_schema_validation():
    print("\n--- 3. Optional Schema Fields Validation with PublicDataLoader ---")
    loader = PublicDataLoader(allow_synthetic=False)
    df = loader.load_hdi_evidence("data/benchmark/verified_botanical_hdi.csv", validate=True)
    print(f"Successfully loaded and validated {len(df)} records with PublicDataLoader.")

    records = [
        EvidenceRecord(
            herb_id=str(row["herb_id"]),
            drug_id=str(row["drug_id"]),
            outcome=str(row["outcome"]),
            evidence_type=str(row["evidence_type"]),
            source_id=str(row["source_id"]),
            study_id=str(row["study_id"]),
            cited_study_ids=row["cited_study_ids"] if isinstance(row["cited_study_ids"], list) else [],
            dose=str(row["dose"]) if pd.notna(row.get("dose")) else None,
            preparation=str(row["preparation"]) if pd.notna(row.get("preparation")) else None,
            route=str(row["route"]) if pd.notna(row.get("route")) else None,
            species=str(row["species"]) if pd.notna(row.get("species")) else None,
            source_url_or_doi=str(row["source_url_or_doi"]) if pd.notna(row.get("source_url_or_doi")) else None,
            original_text=str(row["original_text"]) if pd.notna(row.get("original_text")) else None,
            extract_standardization=str(row["extract_standardization"]) if pd.notna(row.get("extract_standardization")) else None,
            treatment_duration=str(row["treatment_duration"]) if pd.notna(row.get("treatment_duration")) else None,
            interaction_direction=str(row["interaction_direction"]) if pd.notna(row.get("interaction_direction")) else None,
            time_dependence=str(row["time_dependence"]) if pd.notna(row.get("time_dependence")) else None,
        )
        for _, row in df.iterrows()
    ]
    print(f"Converted into {len(records)} canonical EvidenceRecord objects.")

    first = records[0]
    print(f"Sample Record 0: herb={first.herb_id}, drug={first.drug_id}, outcome={first.outcome}")
    print(f"Optional PK fields in EvidenceRecord:")
    print(f"  extract_standardization: {first.extract_standardization}")
    print(f"  treatment_duration     : {first.treatment_duration}")
    print(f"  interaction_direction  : {first.interaction_direction}")
    print(f"  time_dependence        : {first.time_dependence}")

    # Assert schema fields exist on EvidenceRecord
    assert hasattr(first, "extract_standardization")
    assert hasattr(first, "treatment_duration")
    assert hasattr(first, "interaction_direction")
    assert hasattr(first, "time_dependence")
    print("PASS: Schema extensions validate cleanly against benchmark CSV without loader regression.")


if __name__ == "__main__":
    verify_numeric_examples()
    verify_routing_and_metadata()
    verify_schema_validation()
