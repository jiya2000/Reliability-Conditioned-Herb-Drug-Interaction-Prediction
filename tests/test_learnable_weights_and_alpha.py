"""
Tests for Phase 4: Learnable parameterization of echo attenuation (alpha)
and pharmacokinetic context weights with strictly enforced ordering constraint:
w_human > w_animal > w_in_vitro.
"""

import pytest
import torch
from src.models.context_compatibility import ContextCompatibilityScorer, SPECIES_PRIORS
from src.models.citation_dag import CitationDAGEchoAttenuator, LearnableEchoAttenuator


def test_learnable_echo_attenuator_bounded_alpha():
    """Verify alpha is learnable and bounded strictly in (0, 1)."""
    attenuator = LearnableEchoAttenuator(initial_alpha=0.15, learnable=True)
    assert 0.0 < float(attenuator.alpha.item()) < 1.0
    assert abs(float(attenuator.alpha.item()) - 0.15) < 1e-3

    depth = torch.tensor([0.0, 1.0, 2.0, 3.0])
    weights = attenuator(depth)
    assert weights[0] == 1.0
    assert weights[1] < weights[0]
    assert weights[2] < weights[1]
    assert weights[3] < weights[2]


def test_context_compatibility_ordering_constraint():
    """
    Verify that ContextCompatibilityScorer with learn_ordered_weights=True
    enforces strictly: w_human > w_animal > w_in_vitro
    both at initialization and under extreme parameter shifts.
    """
    scorer = ContextCompatibilityScorer(learn_ordered_weights=True)
    weights = scorer.get_effective_species_weights()

    # Index map for SPECIES_LIST
    species_list = scorer.SPECIES_LIST
    h_idx = species_list.index("human")
    a_idx = species_list.index("rat")
    v_idx = species_list.index("in_vitro_microsomes")

    w_human = weights[h_idx].item()
    w_animal = weights[a_idx].item()
    w_invitro = weights[v_idx].item()

    assert w_human > w_animal > w_invitro, f"Expected {w_human} > {w_animal} > {w_invitro}"

    # Perturb parameters drastically
    scorer.u_invitro.data.fill_(10.0)      # push invitro high
    scorer.u_animal_delta.data.fill_(-10.0) # push delta low
    scorer.u_human_delta.data.fill_(-10.0)  # push delta low

    perturbed_weights = scorer.get_effective_species_weights()
    p_human = perturbed_weights[h_idx].item()
    p_animal = perturbed_weights[a_idx].item()
    p_invitro = perturbed_weights[v_idx].item()

    assert p_human > p_animal > p_invitro, f"Even perturbed: {p_human} > {p_animal} > {p_invitro}"


def test_custom_configurable_weights():
    """Verify custom prior dictionaries and scaling are supported."""
    custom_species = {k: v * 0.5 for k, v in SPECIES_PRIORS.items()}
    scorer = ContextCompatibilityScorer(
        learn_ordered_weights=False,
        custom_species_priors=custom_species,
    )
    weights = scorer.get_effective_species_weights()
    h_idx = scorer.SPECIES_LIST.index("human")
    assert abs(weights[h_idx].item() - 0.50) < 1e-4
