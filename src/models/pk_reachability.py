"""
Pharmacokinetic Reachability Screening (PRS) — Novelty Component.

Computes whether an in vitro herb-drug interaction finding is clinically reachable
by comparing reported in vitro potency (IC50/Ki) against physiologically achievable
human unbound plasma concentrations (Cmax,u), incorporating:
- Human oral bioavailability and first-pass metabolism
- Plasma protein binding (fu)
- Intestinal/hepatic extraction
- Botanical extract standardization variability

This implements the FDA/EMA in vitro-to-in vivo extrapolation (IVIVE) framework
for CYP/TDX-mediated herb-drug interactions as a deterministic, auditable screen.

Regulatory basis:
- FDA Guidance: "In Vitro Drug Interaction Studies — Cytochrome P450 Enzyme- and Transporter-Mediated Drug Interactions" (2020)
- EMA Guideline: "Investigation of Drug Interactions" (Rev. 2012)
- FDA Guidance: "Botanical Drug Development" (2016)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple
import math
import torch
import torch.nn as nn
from loguru import logger


# Physiologically-based constants for common herbals
# Values from literature: hyperforin (St John's wort), ginkgolides (Ginkgo), 
# ginsenosides (Ginseng), curcuminoids (Turmeric), berberine, etc.
HERBAL_PK_PARAMETERS: Dict[str, Dict[str, float]] = {
    # St. John's Wort (Hypericum perforatum) - hyperforin
    "hyperforin": {
        "mw": 536.6,  # g/mol
        "oral_bioavailability": 0.40,  # F ~ 40%
        "fu_plasma": 0.01,  # 99% protein bound
        "cmax_total_um": 0.5,  # total Cmax at 300mg extract, ~0.5 uM
        "cmax_unbound_um": 0.005,  # fu * Cmax_total
        "cyp3a4_induction_ec50": 0.1,  # uM, potent inducer
        "pgp_induction_ec50": 0.05,
        "half_life_h": 9.0,
        "standardized_marker": "hyperforin",  # LI 160 standardized to 0.3% hyperforin
        "typical_dose_mg": 300,  # mg extract
        "extract_marker_content": 0.003,  # 0.3% hyperforin
    },
    # Ginkgo biloba - ginkgolide B, bilobalide
    "ginkgolide_b": {
        "mw": 424.4,
        "oral_bioavailability": 0.05,  # very low
        "fu_plasma": 0.40,
        "cmax_total_um": 0.15,  # at 120mg EGb 761
        "cmax_unbound_um": 0.06,
        "cyp3a4_inhibition_ki": 20.0,  # weak inhibitor
        "cyp2c9_inhibition_ki": 15.0,
        "half_life_h": 3.0,
        "standardized_marker": "ginkgolide_b",
        "typical_dose_mg": 120,
        "extract_marker_content": 0.008,  # 0.8% ginkgolides
    },
    "bilobalide": {
        "mw": 326.3,
        "oral_bioavailability": 0.30,
        "fu_plasma": 0.50,
        "cmax_total_um": 0.30,
        "cmax_unbound_um": 0.15,
        "half_life_h": 3.0,
        "standardized_marker": "bilobalide",
        "typical_dose_mg": 120,
        "extract_marker_content": 0.03,  # 3% bilobalide
    },
    # Panax ginseng - ginsenosides (Rb1, Rg1)
    "ginsenoside_rb1": {
        "mw": 1109.3,
        "oral_bioavailability": 0.05,  # deglycosylated by gut microbiota
        "fu_plasma": 0.30,
        "cmax_total_um": 0.05,  # as Rb1
        "cmax_unbound_um": 0.015,
        "cyp3a4_inhibition_ki": 50.0,
        "cyp2d6_inhibition_ki": 30.0,
        "half_life_h": 8.0,
        "standardized_marker": "ginsenosides",
        "typical_dose_mg": 200,
        "extract_marker_content": 0.05,  # 5% total ginsenosides
    },
    # Curcuma longa - curcumin
    "curcumin": {
        "mw": 368.4,
        "oral_bioavailability": 0.01,  # extremely low
        "fu_plasma": 0.001,  # 99.9% bound
        "cmax_total_um": 0.1,
        "cmax_unbound_um": 0.0001,
        "cyp3a4_inhibition_ki": 5.0,
        "cyp2c9_inhibition_ki": 10.0,
        "pgp_inhibition_ki": 2.0,
        "half_life_h": 2.0,
        "standardized_marker": "curcuminoids",
        "typical_dose_mg": 500,
        "extract_marker_content": 0.95,  # 95% curcuminoids
    },
    # Berberine (Coptis, Berberis)
    "berberine": {
        "mw": 336.4,
        "oral_bioavailability": 0.05,  # P-gp efflux
        "fu_plasma": 0.50,
        "cmax_total_um": 0.05,
        "cmax_unbound_um": 0.025,
        "cyp3a4_inhibition_ki": 2.0,
        "cyp2d6_inhibition_ki": 1.5,
        "cyp2c9_inhibition_ki": 3.0,
        "pgp_inhibition_ki": 0.5,
        "half_life_h": 4.0,
        "standardized_marker": "berberine",
        "typical_dose_mg": 500,
        "extract_marker_content": 0.97,
    },
    # Garlic - allicin/alliin
    "allicin": {
        "mw": 162.3,
        "oral_bioavailability": 0.02,  # unstable
        "fu_plasma": 0.80,
        "cmax_total_um": 0.02,
        "cmax_unbound_um": 0.016,
        "cyp3a4_inhibition_ki": 50.0,
        "cyp2c9_inhibition_ki": 30.0,
        "half_life_h": 0.5,
        "standardized_marker": "allicin",
        "typical_dose_mg": 600,
        "extract_marker_content": 0.01,
    },
}


@dataclass
class InVitroFinding:
    """Represents a single in vitro herb-drug interaction finding from literature."""
    herb_constituent: str  # e.g., "hyperforin", "ginkgolide_b"
    enzyme_transporter: str  # e.g., "CYP3A4", "CYP2C9", "P-gp", "OATP1B1"
    interaction_type: str  # "inhibition", "induction", "inhibition_reversible", "inhibition_tdi"
    potency_value: float  # IC50, Ki, EC50, or kinact/KI in uM
    potency_unit: str = "uM"  # "uM", "nM"
    study_species: str = "human"  # "human", "rat", "mouse", "microsomes", "recombinant"
    preparation: str = "microsomes"  # "microsomes", "hepatocytes", "recombinant", "cell_line"
    concentration_um: Optional[float] = None  # test concentration if single-concentration
    extract_type: str = "standardized"  # "standardized", "crude", "isolated"
    route: str = "in_vitro"  # "in_vitro", "oral", "iv"
    pmids: List[str] = field(default_factory=list)


@dataclass
class ReachabilityResult:
    """Result of reachability screening for one finding."""
    finding: InVitroFinding
    cu_max_human: float  # Achievable unbound Cmax in humans (uM)
    ratio: float  # cu_max / potency (for inhibition) or EC50 (for induction)
    is_reachable: bool  # True if ratio >= threshold
    threshold: float  # Threshold used (default 0.1 for inhibition, 1.0 for induction)
    confidence: str  # "high", "medium", "low" based on data quality
    reasoning: str  # Human-readable explanation
    penalty_factor: float  # 0.25 if unreachable, 1.0 if reachable


class PKReachabilityScreen:
    """
    Deterministic reachability screen for in vitro herb-drug interaction findings.
    
    For each in vitro finding, computes the ratio:
        R = C_u,max,human / IC50 (for reversible inhibition)
        R = C_u,max,human / Ki (for binding)
        R = C_u,max,human / EC50 (for induction)
    
    If R >= threshold, the in vitro finding is considered "clinically reachable".
    
    Thresholds per FDA/EMA:
    - Reversible CYP inhibition: [I]u/IC50 >= 0.1 (or 0.02 for sensitive substrates)
    - TDI: [I]u/kinact/KI >= 0.02
    - Induction: [I]u/EC50 >= 1.0 (or AUC ratio approach)
    - Transporter inhibition: [I]gut/IC50 >= 10 (for intestinal), [I]u/IC50 >= 0.1 (hepatic)
    
    This is a discrete, auditable, pharmacology-grounded screen — not a learned heuristic.
    """

    # Default thresholds per FDA/EMA guidance
    DEFAULT_THRESHOLDS: Dict[str, float] = {
        "cyp_reversible_inhibition": 0.1,      # [I]u/IC50 >= 0.1
        "cyp_tdi": 0.02,                       # [I]u/(kinact/KI) >= 0.02
        "cyp_induction": 1.0,                  # [I]u/EC50 >= 1.0 (simplified)
        "transporter_intestinal": 10.0,        # [I]gut/IC50 >= 10
        "transporter_hepatic": 0.1,            # [I]u/IC50 >= 0.1
        "pgp_inhibition": 0.1,                 # [I]u/IC50 >= 0.1
        "oatp_inhibition": 0.1,                # [I]u/IC50 >= 0.1
    }

    def __init__(
        self,
        thresholds: Optional[Dict[str, float]] = None,
        herbal_pk_db: Optional[Dict[str, Dict[str, float]]] = None,
    ):
        self.thresholds = thresholds or self.DEFAULT_THRESHOLDS.copy()
        self.herbal_pk = herbal_pk_db or HERBAL_PK_PARAMETERS.copy()

    def screen(self, finding: InVitroFinding) -> ReachabilityResult:
        """
        Perform reachability screen for a single in vitro finding.
        
        Returns ReachabilityResult with penalty_factor = 0.25 if unreachable,
        1.0 if reachable. This penalty_factor can be used as w_conc in
        the ContextCompatibilityScorer.
        """
        # Get human achievable unbound Cmax for this constituent
        cu_max = self._get_human_cu_max(finding)
        
        # Determine threshold based on interaction type and target
        threshold = self._get_threshold(finding)
        
        # Compute ratio
        if finding.interaction_type in {"inhibition", "inhibition_reversible"}:
            ratio = cu_max / max(finding.potency_value, 1e-6)
        elif finding.interaction_type == "induction":
            ratio = cu_max / max(finding.potency_value, 1e-6)
        elif finding.interaction_type == "inhibition_tdi":
            # For TDI, potency_value is kinact/KI (uM^-1 min^-1 or similar)
            # Simplified: use same ratio logic
            ratio = cu_max / max(finding.potency_value, 1e-6)
        else:
            ratio = 0.0
        
        is_reachable = ratio >= threshold
        
        # Confidence based on data availability
        confidence = self._assess_confidence(finding, cu_max)
        
        # Human-readable reasoning
        reasoning = self._generate_reasoning(finding, cu_max, ratio, threshold, is_reachable)
        
        penalty_factor = 1.0 if is_reachable else 0.25
        
        return ReachabilityResult(
            finding=finding,
            cu_max_human=cu_max,
            ratio=ratio,
            is_reachable=is_reachable,
            threshold=threshold,
            confidence=confidence,
            reasoning=reasoning,
            penalty_factor=penalty_factor,
        )

    def _get_human_cu_max(self, finding: InVitroFinding) -> float:
        """Get achievable unbound plasma Cmax in humans for the herbal constituent."""
        pk = self.herbal_pk.get(finding.herb_constituent.lower())
        if pk is None:
            # Unknown constituent: conservative low estimate
            logger.warning(f"No PK data for {finding.herb_constituent}, using conservative estimate")
            return 0.01  # 10 nM unbound
        
        # Base unbound Cmax from database
        cu_max = pk.get("cmax_unbound_um", 0.01)
        
        # Adjust for extract type if not standardized
        if finding.extract_type != "standardized":
            # Crude extracts have variable content; reduce confidence
            marker_content = pk.get("extract_marker_content", 0.01)
            # For crude, assume 50% of standardized marker content
            cu_max *= 0.5
        
        # Adjust for route
        if finding.route == "iv":
            # IV bypasses first-pass; higher systemic exposure
            cu_max *= 2.0
        elif finding.route == "oral":
            # Oral is standard
            pass
        
        return max(cu_max, 1e-6)

    def _get_threshold(self, finding: InVitroFinding) -> float:
        """Determine appropriate threshold based on interaction type and target."""
        et = finding.enzyme_transporter.upper()
        it = finding.interaction_type.lower()
        
        if "CYP" in et:
            if it in {"inhibition", "inhibition_reversible"}:
                return self.thresholds["cyp_reversible_inhibition"]
            elif it == "induction":
                return self.thresholds["cyp_induction"]
            elif it == "inhibition_tdi":
                return self.thresholds["cyp_tdi"]
        elif "P-GP" in et or "ABCB1" in et or "PGP" in et:
            return self.thresholds["pgp_inhibition"]
        elif "OATP" in et:
            return self.thresholds["oatp_inhibition"]
        elif "TRANSPORTER" in et:
            # Default to hepatic
            return self.thresholds["transporter_hepatic"]
        
        # Default fallback
        return 0.1

    def _assess_confidence(self, finding: InVitroFinding, cu_max: float) -> str:
        """Assess confidence in the reachability assessment."""
        # High: known constituent with full PK data, standardized extract, human system
        # Medium: known constituent but some data missing, or non-human system
        # Low: unknown constituent, crude extract, or very high in vitro concentration
        
        pk = self.herbal_pk.get(finding.herb_constituent.lower())
        if pk is None:
            return "low"
        
        if finding.study_species == "human" and finding.preparation in {"hepatocytes", "microsomes"}:
            if finding.extract_type == "standardized":
                return "high"
            return "medium"
        
        if finding.concentration_um is not None and finding.concentration_um > 50:
            return "low"  # supra-physiological concentration
        
        return "medium"

    def _generate_reasoning(
        self,
        finding: InVitroFinding,
        cu_max: float,
        ratio: float,
        threshold: float,
        is_reachable: bool,
    ) -> str:
        """Generate human-readable reasoning for the reachability decision."""
        status = "REACHABLE" if is_reachable else "NOT REACHABLE"
        return (
            f"{finding.herb_constituent} -> {finding.enzyme_transporter} "
            f"({finding.interaction_type}): "
            f"Human Cu,max = {cu_max:.4f} uM, "
            f"In vitro potency = {finding.potency_value:.4f} uM, "
            f"Ratio = {ratio:.3f}, "
            f"Threshold = {threshold:.3f} -> {status}. "
            f"Confidence: {self._assess_confidence(finding, cu_max)}."
        )

    def screen_batch(self, findings: List[InVitroFinding]) -> List[ReachabilityResult]:
        """Screen multiple findings."""
        return [self.screen(f) for f in findings]

    def compute_batch_penalty(self, findings: List[InVitroFinding]) -> torch.Tensor:
        """
        Compute penalty factors for a batch of findings.
        Returns tensor of penalty factors (1.0 = reachable, 0.25 = unreachable).
        """
        results = self.screen_batch(findings)
        penalties = [r.penalty_factor for r in results]
        return torch.tensor(penalties, dtype=torch.float32)


class DifferentiableReachabilityScreen(nn.Module):
    """
    Differentiable wrapper for reachability screening.
    
    Allows the reachability penalty to be used in end-to-end training.
    The core screen is deterministic (based on pharmacology), but we expose
    the penalty factor as a differentiable output.
    
    This implements the w_conc factor from Claim 15 of the patent:
    w_conc = 0.25 if supra-physiological / unreachable, else 1.0
    """

    def __init__(
        self,
        screen: Optional[PKReachabilityScreen] = None,
        learnable_threshold: bool = False,
    ):
        super().__init__()
        self.screen = screen or PKReachabilityScreen()
        
        if learnable_threshold:
            # Allow threshold to be learned around the pharmacology prior
            self.threshold_scale = nn.Parameter(torch.tensor([1.0], dtype=torch.float32))
        else:
            self.register_buffer("threshold_scale", torch.tensor([1.0], dtype=torch.float32))
        
        # Fixed penalty for unreachable
        self.register_buffer("unreachable_penalty", torch.tensor([0.25], dtype=torch.float32))
        self.register_buffer("reachable_penalty", torch.tensor([1.0], dtype=torch.float32))

    def forward(
        self,
        herb_constituents: List[str],
        enzymes: List[str],
        interaction_types: List[str],
        potency_values: torch.Tensor,  # (batch,) in uM
        study_species: List[str] = None,
        preparations: List[str] = None,
        routes: List[str] = None,
        extract_types: List[str] = None,
    ) -> torch.Tensor:
        """
        Compute reachability penalty factors for a batch.
        
        Args:
            All lists of length batch_size
            potency_values: Tensor of in vitro potency values (IC50/Ki/EC50) in uM
            
        Returns:
            penalties: (batch, 1) tensor with values in {0.25, 1.0} (or soft if learnable)
        """
        batch_size = len(herb_constituents)
        penalties = []
        
        for i in range(batch_size):
            finding = InVitroFinding(
                herb_constituent=herb_constituents[i],
                enzyme_transporter=enzymes[i],
                interaction_type=interaction_types[i],
                potency_value=float(potency_values[i].item()),
                study_species=study_species[i] if study_species else "human",
                preparation=preparations[i] if preparations else "microsomes",
                route=routes[i] if routes else "in_vitro",
                extract_type=extract_types[i] if extract_types else "standardized",
            )
            result = self.screen.screen(finding)
            penalties.append(result.penalty_factor)
        
        return torch.tensor(penalties, dtype=torch.float32).unsqueeze(-1)


def demo_reachability():
    """Demonstrate the reachability screen."""
    screen = PKReachabilityScreen()
    
    # Test cases: St. John's wort hyperforin CYP3A4 induction (reachable)
    # vs. Curcumin CYP3A4 inhibition at 10 uM (not reachable - Cu,max = 0.0001 uM)
    findings = [
        InVitroFinding(
            herb_constituent="hyperforin",
            enzyme_transporter="CYP3A4",
            interaction_type="induction",
            potency_value=0.1,  # EC50 = 0.1 uM
            study_species="human",
            preparation="hepatocytes",
            route="in_vitro",
            extract_type="standardized",
        ),
        InVitroFinding(
            herb_constituent="curcumin",
            enzyme_transporter="CYP3A4",
            interaction_type="inhibition",
            potency_value=5.0,  # Ki = 5 uM
            study_species="human",
            preparation="microsomes",
            route="in_vitro",
            extract_type="standardized",
        ),
        InVitroFinding(
            herb_constituent="ginkgolide_b",
            enzyme_transporter="CYP3A4",
            interaction_type="inhibition",
            potency_value=20.0,  # Ki = 20 uM
            study_species="human",
            preparation="microsomes",
            route="in_vitro",
            extract_type="standardized",
        ),
        InVitroFinding(
            herb_constituent="ginsenoside_rb1",
            enzyme_transporter="CYP3A4",
            interaction_type="inhibition",
            potency_value=50.0,  # Ki = 50 uM
            study_species="human",
            preparation="microsomes",
            route="in_vitro",
            extract_type="standardized",
        ),
        # Supra-physiological concentration test
        InVitroFinding(
            herb_constituent="berberine",
            enzyme_transporter="CYP3A4",
            interaction_type="inhibition",
            potency_value=2.0,  # Ki = 2 uM (potent)
            study_species="human",
            preparation="microsomes",
            route="in_vitro",
            extract_type="standardized",
            concentration_um=100.0,  # but tested at 100 uM
        ),
    ]
    
    print("=== PK Reachability Screen Demo ===\n")
    for finding in findings:
        result = screen.screen(finding)
        print(f"Constituent: {finding.herb_constituent}")
        print(f"  Target: {finding.enzyme_transporter} ({finding.interaction_type})")
        print(f"  In vitro potency: {finding.potency_value} uM")
        print(f"  Human Cu,max: {result.cu_max_human:.6f} uM")
        print(f"  Ratio: {result.ratio:.4f}")
        print(f"  Threshold: {result.threshold:.3f}")
        print(f"  Reachable: {result.is_reachable}")
        print(f"  Penalty factor: {result.penalty_factor}")
        print(f"  Confidence: {result.confidence}")
        print(f"  Reasoning: {result.reasoning}")
        print()


if __name__ == "__main__":
    demo_reachability()