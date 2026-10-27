"""
Unit tests for Pharmacokinetic Reachability Screening (PRS) site-of-action branching.
Verifies worked numerical examples for:
1. Hepatic/systemic reachability (unbound plasma Cmax)
2. Intestinal CYP3A4 / P-gp reachability (FDA gut-lumen concentration formula)
3. Missing dose/MW graceful fallback and audit flagging
"""

import pytest
import math
from src.models.pk_reachability import (
    PKReachabilityScreen,
    InVitroFinding,
    ReachabilityResult,
    HERBAL_PK_PARAMETERS,
)


def test_hepatic_reachability_worked_numeric_example():
    """
    Worked Example 1 (Hepatic/Systemic):
    Hyperforin CYP3A4 induction:
    - Target: Hepatic CYP3A4 (hepatic site)
    - Human achievable unbound Cmax: 0.005 uM (from literature/database)
    - In vitro EC50: 0.1 uM
    - Ratio = 0.005 / 0.1 = 0.05
    - Threshold for CYP induction = 0.05
    - Expected: reachable = True, rho_reach = 0.05, penalty_factor = 1.0, branch_used = 'hepatic'
    """
    screen = PKReachabilityScreen()
    finding = InVitroFinding(
        herb_constituent="hyperforin",
        enzyme_transporter="CYP3A4",
        interaction_type="induction",
        potency_value=0.1,
        study_species="human",
        preparation="hepatocytes",
        site_of_action="hepatic",
    )
    result = screen.screen(finding)

    assert result.branch_used == "hepatic"
    assert not result.missing_dose_or_mw
    assert math.isclose(result.cu_max_human, 0.005, rel_tol=1e-3)
    assert math.isclose(result.ratio, 0.05, rel_tol=1e-3)
    assert math.isclose(result.rho_reach, 0.05, rel_tol=1e-3)
    assert result.is_reachable is True
    assert result.penalty_factor == 1.0


def test_hepatic_unreachable_penalty():
    """
    Worked Example 1b (Hepatic Unreachable):
    Curcumin CYP3A4 inhibition:
    - Target: Hepatic CYP3A4
    - Human achievable unbound Cmax: 0.0001 uM (extremely low bioavailability)
    - In vitro Ki: 5.0 uM
    - Ratio = 0.0001 / 5.0 = 0.00002
    - Threshold = 0.1
    - Expected: reachable = False, rho_reach = 0.00002, penalty_factor = 0.25
    """
    screen = PKReachabilityScreen()
    finding = InVitroFinding(
        herb_constituent="curcumin",
        enzyme_transporter="CYP3A4",
        interaction_type="inhibition",
        potency_value=5.0,
        study_species="human",
        preparation="microsomes",
        site_of_action="hepatic",
    )
    result = screen.screen(finding)

    assert result.branch_used == "hepatic"
    assert math.isclose(result.cu_max_human, 0.0001, rel_tol=1e-3)
    assert math.isclose(result.ratio, 2e-5, rel_tol=1e-3)
    assert result.is_reachable is False
    assert result.penalty_factor == 0.25


def test_intestinal_reachability_worked_numeric_example():
    """
    Worked Example 2 (Intestinal CYP3A4 / P-gp per FDA Guidance):
    Hyperforin at intestinal lumen:
    - Dose: 9.0 mg (active hyperforin in 300 mg extract at 3%)
    - MW: 536.6 g/mol
    - Volume: 250 mL (0.25 L, FDA water volume)
    - I_gut = (dose_mg * 4000.0) / MW = (9.0 * 4000) / 536.6 = 36000 / 536.6 = 67.08908 uM
    - In vitro IC50: 2.0 uM
    - Ratio = 67.08908 / 2.0 = 33.5445
    - FDA Guidance threshold for intestinal transporter/CYP: 10.0
    - Ratio >= 10.0 -> is_reachable = True, penalty_factor = 1.0
    - rho_reach = min(1.0, 33.5445) = 1.0
    """
    screen = PKReachabilityScreen()
    finding = InVitroFinding(
        herb_constituent="hyperforin",
        enzyme_transporter="P-gp",
        interaction_type="inhibition",
        potency_value=2.0,
        site_of_action="intestinal",
        dose_mg=9.0,
        molecular_weight=536.6,
    )
    result = screen.screen(finding)

    assert result.branch_used == "intestinal"
    assert not result.missing_dose_or_mw
    assert result.i_gut_um is not None
    assert math.isclose(result.i_gut_um, 36000.0 / 536.6, rel_tol=1e-3)
    assert math.isclose(result.ratio, (36000.0 / 536.6) / 2.0, rel_tol=1e-3)
    assert math.isclose(result.rho_reach, 1.0, rel_tol=1e-3)
    assert result.is_reachable is True
    assert result.penalty_factor == 1.0


def test_intestinal_auto_routing_and_missing_dose_fallback():
    """
    Worked Example 3 (Auto-routing and Graceful Missingness Fallback):
    - Enzyme: 'intestinal_cyp3a4' -> auto-routes to intestinal branch
    - Dose and MW omitted -> missing_dose_or_mw is True, confidence flagged as 'low'
    - Produces valid, non-crashing numerical reachability estimates
    """
    screen = PKReachabilityScreen()
    finding = InVitroFinding(
        herb_constituent="unknown_phyto_compound",
        enzyme_transporter="intestinal_CYP3A4",
        interaction_type="inhibition",
        potency_value=50.0,
        # dose_mg and molecular_weight omitted
    )
    result = screen.screen(finding)

    assert result.branch_used == "intestinal"
    assert result.missing_dose_or_mw is True
    assert result.confidence == "low"
    assert result.i_gut_um is not None
    assert result.i_gut_um > 0.0
    assert not math.isnan(result.ratio)
    assert not math.isnan(result.rho_reach)
    assert "dose/MW imputed; missingness flagged" in result.reasoning
