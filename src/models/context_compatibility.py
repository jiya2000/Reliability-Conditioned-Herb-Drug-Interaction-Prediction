"""
Pharmacokinetic Context Compatibility (PCC) Module.

NOVELTY CONTRIBUTION 2:
Scores literature evidence for physiological and clinical relevance based on
biological species, botanical preparation quality, administration route, and
physiological concentration plausibility.

Core Components:
1. ContextCompatibilityScorer (PyTorch nn.Module): Differentiable layer that computes
   multiplicative composite context weights:
       W_context = w_species * w_preparation * w_route * w_concentration
   with learnable calibration offsets around physiological priors.
2. ContextMetadataParser: Extracts and standardizes pharmacokinetic attributes from
   text or structured candidate metadata.
3. InVitroArtifactFilter: Identifies non-physiological in vitro artifacts (e.g.,
   CYP inhibition observed only at supra-physiological concentrations > 50 uM).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple, Union
import torch
import torch.nn as nn
from loguru import logger


# Prior physiological weights grounded in clinical pharmacology
# Canonical default standard: human 1.00 / in vivo animal 0.50 / in vitro 0.20
SPECIES_PRIORS: Dict[str, float] = {
    "human": 1.00,
    "clinical_trial": 1.00,
    "volunteer": 1.00,
    "primate": 0.60,
    "rat": 0.50,
    "mouse": 0.50,
    "rabbit": 0.50,
    "dog": 0.50,
    "in_vitro_caco2": 0.20,
    "in_vitro_microsomes": 0.20,
    "in_vitro_cell_line": 0.20,
    "in_silico": 0.10,
    "unknown": 0.35,
}

PREPARATION_PRIORS: Dict[str, float] = {
    "standardized_extract": 1.00,  # Defined active constituent content (e.g. EGb 761)
    "isolated_constituent": 0.90,  # Pure molecule (e.g. isolated berberine, curcumin)
    "crude_decoction": 0.75,       # Traditional aqueous extract / tea
    "unstandardized_powder": 0.65, # Raw dried herb
    "unknown": 0.50,
}

ROUTE_PRIORS: Dict[str, float] = {
    "oral": 1.00,                  # Natural route for herbals in humans
    "parenteral_iv": 0.75,         # Intravenous injection
    "intraperitoneal": 0.65,       # Common in rodent screening
    "subcutaneous": 0.60,
    "topical": 0.50,
    "in_vitro_incubation": 0.30,   # Cell bath
    "unknown": 0.50,
}


@dataclass
class PharmacokineticContext:
    """Pharmacokinetic context parameters for an evidence report."""
    species: str = "human"
    preparation: str = "standardized_extract"
    route: str = "oral"
    concentration_um: Optional[float] = None
    is_supraphysiological: bool = False
    raw_text: Optional[str] = None


class ContextMetadataParser:
    """Extracts and normalizes context attributes from metadata or text."""

    @staticmethod
    def parse_dict(data: Dict[str, Any]) -> PharmacokineticContext:
        """Parses a dictionary into a structured PharmacokineticContext."""
        species = str(data.get("species", data.get("organism", "unknown"))).lower()
        prep = str(data.get("preparation", data.get("extract_type", "unknown"))).lower()
        route = str(data.get("route", data.get("administration", "unknown"))).lower()

        # Normalize species
        norm_species = "unknown"
        if "human" in species or "sapiens" in species or "patient" in species:
            norm_species = "human"
        else:
            for k in SPECIES_PRIORS:
                if k in species:
                    norm_species = k
                    break


        # Normalize preparation
        norm_prep = "unknown"
        if "standard" in prep or "egb" in prep:
            norm_prep = "standardized_extract"
        elif "isolate" in prep or "pure" in prep:
            norm_prep = "isolated_constituent"
        elif "decoct" in prep or "tea" in prep or "water" in prep:
            norm_prep = "crude_decoction"
        elif "powder" in prep or "crude" in prep:
            norm_prep = "unstandardized_powder"

        # Normalize route
        norm_route = "unknown"
        if "oral" in route or "gavage" in route or "ingest" in route:
            norm_route = "oral"
        elif "iv" in route or "intravenous" in route:
            norm_route = "parenteral_iv"
        elif "ip" in route or "intraperitoneal" in route:
            norm_route = "intraperitoneal"
        elif "topical" in route:
            norm_route = "topical"
        elif "vitro" in species or "vitro" in route or "incub" in route:
            norm_route = "in_vitro_incubation"

        # Concentration check
        conc = data.get("concentration_um", data.get("dose_micromolar", None))
        conc_val = float(conc) if conc is not None else None
        supra = False
        if conc_val is not None and ("vitro" in norm_species or norm_route == "in_vitro_incubation"):
            if conc_val > 50.0:  # > 50 uM in vitro is biologically unachievable in vivo
                supra = True

        return PharmacokineticContext(
            species=norm_species,
            preparation=norm_prep,
            route=norm_route,
            concentration_um=conc_val,
            is_supraphysiological=supra,
        )


class ContextCompatibilityScorer(nn.Module):
    """
    Differentiable Pharmacokinetic Context Compatibility Layer.

    Computes composite context score W_context in [0, 1] by combining:
    1. Base physiological priors for species, preparation, and route
    2. Learnable neural refinement weights that adapt to dataset ground truth
    3. Non-physiological in vitro artifact suppression

    Mathematical formulation:
        w_species = clamp(prior_s + delta_s, 0.05, 1.0)
        w_prep    = clamp(prior_p + delta_p, 0.05, 1.0)
        w_route   = clamp(prior_r + delta_r, 0.05, 1.0)
        w_conc    = 0.25 if supra_physiological else 1.0
        W_context = (w_species * w_prep * w_route * w_conc)^(1/3)
    """

    SPECIES_LIST = [
        "human", "clinical_trial", "volunteer", "primate", "rat",
        "mouse", "rabbit", "dog", "in_vitro_caco2", "in_vitro_microsomes",
        "in_vitro_cell_line", "in_silico", "unknown"
    ]
    PREP_LIST = [
        "standardized_extract", "isolated_constituent", "crude_decoction",
        "unstandardized_powder", "unknown"
    ]
    ROUTE_LIST = [
        "oral", "parenteral_iv", "intraperitoneal", "subcutaneous",
        "topical", "in_vitro_incubation", "unknown"
    ]

    def __init__(
        self,
        learnable_refinement: bool = True,
        hidden_dim: int = 32,
        custom_species_priors: Optional[Dict[str, float]] = None,
        custom_prep_priors: Optional[Dict[str, float]] = None,
        custom_route_priors: Optional[Dict[str, float]] = None,
        learn_ordered_weights: bool = False,
    ):
        super().__init__()
        self.learnable = learnable_refinement
        self.learn_ordered_weights = learn_ordered_weights

        species_dict = custom_species_priors or SPECIES_PRIORS
        prep_dict = custom_prep_priors or PREPARATION_PRIORS
        route_dict = custom_route_priors or ROUTE_PRIORS

        # Static prior buffers
        species_prior_tensor = torch.tensor(
            [species_dict.get(s, 0.45) for s in self.SPECIES_LIST], dtype=torch.float32
        )
        prep_prior_tensor = torch.tensor(
            [prep_dict.get(p, 0.50) for p in self.PREP_LIST], dtype=torch.float32
        )
        route_prior_tensor = torch.tensor(
            [route_dict.get(r, 0.50) for r in self.ROUTE_LIST], dtype=torch.float32
        )

        self.register_buffer("species_priors", species_prior_tensor)
        self.register_buffer("prep_priors", prep_prior_tensor)
        self.register_buffer("route_priors", route_prior_tensor)

        # Learnable residual adjustments (delta) initialized to zero
        if learnable_refinement and not learn_ordered_weights:
            self.delta_species = nn.Parameter(torch.zeros(len(self.SPECIES_LIST)))
            self.delta_prep = nn.Parameter(torch.zeros(len(self.PREP_LIST)))
            self.delta_route = nn.Parameter(torch.zeros(len(self.ROUTE_LIST)))
        else:
            self.register_buffer("delta_species", torch.zeros(len(self.SPECIES_LIST)))
            self.register_buffer("delta_prep", torch.zeros(len(self.PREP_LIST)))
            self.register_buffer("delta_route", torch.zeros(len(self.ROUTE_LIST)))

        # Bounded parameterization enforcing ordering constraint: human > animal > in vitro
        if learn_ordered_weights:
            self.u_invitro = nn.Parameter(torch.tensor([-0.5], dtype=torch.float32))
            self.u_animal_delta = nn.Parameter(torch.tensor([0.2], dtype=torch.float32))
            self.u_human_delta = nn.Parameter(torch.tensor([0.2], dtype=torch.float32))

        # MLP projection for continuous embedding context vectors (e.g. from lineage features)
        self.continuous_proj = nn.Sequential(
            nn.Linear(1, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid(),
        )

    def get_effective_species_weights(self) -> torch.Tensor:
        """
        Returns species weight tensor for all species in SPECIES_LIST.
        If learn_ordered_weights=True, guarantees strictly:
            w_human > w_animal > w_in_vitro
        via bounded positive increments.
        """
        if self.learn_ordered_weights:
            # Base in vitro weight in [0.05, 0.40]
            w_invitro = 0.05 + 0.35 * torch.sigmoid(self.u_invitro)
            # Positive increment for animal: [0.05, 0.45]
            delta_animal = 0.05 + 0.40 * torch.sigmoid(self.u_animal_delta)
            w_animal = torch.clamp(w_invitro + delta_animal, max=0.85)
            # Positive increment for human: [0.05, 0.25]
            delta_human = 0.05 + 0.20 * torch.sigmoid(self.u_human_delta)
            w_human = torch.clamp(w_animal + delta_human, max=1.0)

            weights = []
            for s in self.SPECIES_LIST:
                if s in {"human", "clinical_trial", "volunteer"}:
                    weights.append(w_human)
                elif s in {"primate", "rat", "mouse", "rabbit", "dog"}:
                    weights.append(w_animal)
                elif s in {"in_vitro_caco2", "in_vitro_microsomes", "in_vitro_cell_line", "in_silico"}:
                    weights.append(w_invitro)
                else:
                    weights.append((w_animal + w_invitro) / 2.0)
            return torch.cat(weights)
        else:
            return torch.clamp(self.species_priors + self.delta_species, 0.05, 1.0)

    def compute_from_indices(
        self,
        species_idx: torch.Tensor,
        prep_idx: torch.Tensor,
        route_idx: torch.Tensor,
        is_supraphysiological: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Computes context score from categorical feature indices.

        Args:
            species_idx: (batch,) LongTensor indices into SPECIES_LIST
            prep_idx: (batch,) LongTensor indices into PREP_LIST
            route_idx: (batch,) LongTensor indices into ROUTE_LIST
            is_supraphysiological: Optional (batch,) bool or float tensor (1 = supra, 0 = normal)

        Returns:
            W_context: (batch, 1) in [0, 1]
        """
        effective_species = self.get_effective_species_weights()
        w_s = effective_species[species_idx]
        w_p = torch.clamp(self.prep_priors[prep_idx] + self.delta_prep[prep_idx], 0.05, 1.0)
        w_r = torch.clamp(self.route_priors[route_idx] + self.delta_route[route_idx], 0.05, 1.0)

        composite = w_s * w_p * w_r

        if is_supraphysiological is not None:
            # 0.25 penalty factor for non-physiological in vitro concentrations
            penalty = 1.0 - 0.75 * is_supraphysiological.float().view(-1)
            composite = composite * penalty

        # Geometric normalization to [0, 1]
        w_context = torch.pow(composite, 1.0 / 3.0).unsqueeze(-1)
        return torch.clamp(w_context, 0.0, 1.0)

    def forward(
        self,
        context_input: Union[torch.Tensor, List[PharmacokineticContext], Dict[str, Any]],
    ) -> torch.Tensor:
        """
        Flexible forward pass accepting either:
        - 1D/2D Tensor of pre-extracted context compatibility scores in [0, 1]
        - Categorical indices or PharmacokineticContext objects
        """
        if isinstance(context_input, torch.Tensor):
            # If already a scalar/feature column (batch, 1) or (batch,)
            if context_input.dim() == 1:
                context_input = context_input.unsqueeze(-1)
            # Refine via differentiable projection
            return self.continuous_proj(context_input)

        if isinstance(context_input, list) and len(context_input) > 0 and isinstance(context_input[0], PharmacokineticContext):
            device = self.species_priors.device
            s_idx = torch.tensor(
                [self.SPECIES_LIST.index(c.species) if c.species in self.SPECIES_LIST else len(self.SPECIES_LIST)-1 for c in context_input],
                dtype=torch.long, device=device
            )
            p_idx = torch.tensor(
                [self.PREP_LIST.index(c.preparation) if c.preparation in self.PREP_LIST else len(self.PREP_LIST)-1 for c in context_input],
                dtype=torch.long, device=device
            )
            r_idx = torch.tensor(
                [self.ROUTE_LIST.index(c.route) if c.route in self.ROUTE_LIST else len(self.ROUTE_LIST)-1 for c in context_input],
                dtype=torch.long, device=device
            )
            supra = torch.tensor(
                [1.0 if c.is_supraphysiological else 0.0 for c in context_input],
                dtype=torch.float32, device=device
            )
            return self.compute_from_indices(s_idx, p_idx, r_idx, supra)

        # Fallback default
        return torch.tensor([[0.5]], dtype=torch.float32, device=self.species_priors.device)

    def score_record(self, rec: Any) -> float:
        """
        Score an EvidenceRecord or dict using this ContextCompatibilityScorer.
        Evaluates species, preparation, and route priors/weights.
        """
        if hasattr(rec, "to_dict"):
            d = rec.to_dict()
        elif isinstance(rec, dict):
            d = rec
        else:
            d = {
                "species": getattr(rec, "species", None),
                "preparation": getattr(rec, "preparation", None),
                "route": getattr(rec, "route", None),
                "dose": getattr(rec, "dose", None),
            }
        ctx = ContextMetadataParser.parse_dict(d)
        with torch.no_grad():
            res = self.forward([ctx])
            return float(res.squeeze().item())
