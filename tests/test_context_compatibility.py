"""
Tests for Novelty 2: Pharmacokinetic Context Compatibility (PCC) Module.
"""

import pytest
import torch
from src.models.context_compatibility import (
    ContextCompatibilityScorer,
    ContextMetadataParser,
    PharmacokineticContext,
    SPECIES_PRIORS,
    PREPARATION_PRIORS,
    ROUTE_PRIORS,
)


def test_context_metadata_parser():
    # Test oral human clinical trial with standardized extract
    data_human = {
        "species": "Homo sapiens clinical volunteer",
        "preparation": "Standardized Ginkgo EGb 761 extract",
        "route": "Oral tablet",
    }
    ctx = ContextMetadataParser.parse_dict(data_human)
    assert ctx.species == "human"
    assert ctx.preparation == "standardized_extract"
    assert ctx.route == "oral"
    assert not ctx.is_supraphysiological

    # Test supra-physiological in vitro artifact (100 uM incubation)
    data_vitro = {
        "species": "caco-2 cell line in vitro",
        "preparation": "isolated quercetin",
        "route": "in vitro bath incubation",
        "concentration_um": 120.0,
    }
    ctx_v = ContextMetadataParser.parse_dict(data_vitro)
    assert ctx_v.is_supraphysiological


def test_context_compatibility_scorer_indices():
    scorer = ContextCompatibilityScorer(learnable_refinement=True)

    # Index 0: human, standardized, oral -> Should be high (close to 1.0)
    species_idx = torch.tensor([0, 8])  # human, in_vitro_caco2
    prep_idx = torch.tensor([0, 3])     # standardized, unstandardized
    route_idx = torch.tensor([0, 5])    # oral, in_vitro_incubation
    supra = torch.tensor([0.0, 1.0])

    scores = scorer.compute_from_indices(species_idx, prep_idx, route_idx, supra)
    assert scores.shape == (2, 1)

    # Human clinical score should be significantly higher than supra-physiological in vitro score
    assert scores[0].item() > 0.85
    assert scores[1].item() < 0.40


def test_context_compatibility_continuous_tensor():
    scorer = ContextCompatibilityScorer()
    input_features = torch.tensor([[0.9], [0.2], [0.5]])
    refined = scorer(input_features)
    assert refined.shape == (3, 1)
    assert (refined >= 0.0).all() and (refined <= 1.0).all()


def test_context_compatibility_differentiability():
    scorer = ContextCompatibilityScorer(learnable_refinement=True)
    species_idx = torch.tensor([0, 1])
    prep_idx = torch.tensor([0, 0])
    route_idx = torch.tensor([0, 0])

    scores = scorer.compute_from_indices(species_idx, prep_idx, route_idx)
    loss = scores.sum()
    loss.backward()

    assert scorer.delta_species.grad is not None
    assert scorer.delta_prep.grad is not None
    assert scorer.delta_route.grad is not None
