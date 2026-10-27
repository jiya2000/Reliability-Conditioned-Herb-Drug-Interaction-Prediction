"""
Unit Tests for Phase 2 Outcome Term and Entity Masking.
"""

import pytest
from src.features.text_masking import (
    mask_outcome_language,
    mask_entities,
    mask_outcome_and_entities,
    extract_study_design_text,
)


def test_mask_outcome_language_positive_phrases():
    sample = "St Johns wort markedly decreases cyclosporin blood concentrations leading to acute organ rejection."
    masked = mask_outcome_language(sample)
    # Directional verbs and outcome terms masked
    assert "decreases" not in masked.lower()
    assert "rejection" not in masked.lower()
    assert "[MASK_OUTCOME]" in masked


def test_mask_outcome_language_negative_phrases():
    sample = "Silymarin does not alter indinavir pharmacokinetics in healthy subjects."
    masked = mask_outcome_language(sample)
    assert "does not alter" not in masked.lower()
    assert "[MASK_OUTCOME]" in masked


def test_mask_entities_botanicals_and_drugs():
    sample = "Effect of Ginkgo biloba on warfarin pharmacodynamics and midazolam disposition."
    masked = mask_entities(sample)
    assert "ginkgo" not in masked.lower()
    assert "warfarin" not in masked.lower()
    assert "midazolam" not in masked.lower()
    assert "[HERB]" in masked
    assert "[DRUG]" in masked


def test_mask_outcome_and_entities_combined():
    sample = "Curcumin significantly increases systemic tacrolimus exposure in healthy subjects via P-gp/CYP3A inhibition."
    masked = mask_outcome_and_entities(sample)
    assert "curcumin" not in masked.lower()
    assert "tacrolimus" not in masked.lower()
    assert "increases" not in masked.lower()
    assert "inhibition" not in masked.lower()
    assert "[HERB]" in masked
    assert "[DRUG]" in masked


def test_mask_outcome_and_entities_with_proxies():
    sample = "Effects of milk thistle on midazolam disposition in healthy volunteers showing no change in clearance."
    masked = mask_outcome_and_entities(sample, mask_proxies=True)
    assert "milk thistle" not in masked.lower()
    assert "midazolam" not in masked.lower()
    assert "healthy" not in masked.lower()
    assert "volunteers" not in masked.lower()
    assert "clearance" not in masked.lower()


def test_extract_study_design_text():
    sample = "A randomized controlled trial of oral standardized extract in 24 healthy volunteers."
    design = extract_study_design_text(sample)
    assert "randomized_trial" in design
    assert "oral_administration" in design
    assert "standardized_extract" in design


def test_edge_cases_empty_and_none():
    assert mask_outcome_language("") == ""
    assert mask_outcome_language(None) == ""
    assert mask_entities("") == ""
    assert mask_entities(None) == ""
    assert mask_outcome_and_entities("") == ""
    assert extract_study_design_text("") == ""
