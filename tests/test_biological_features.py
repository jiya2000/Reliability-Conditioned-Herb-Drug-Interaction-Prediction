"""
Unit tests for BiologicalProfileExtractor and EvidenceTextFeatureExtractor.
"""

import pytest
import numpy as np
import torch
from src.features.biological_features import (
    BiologicalProfileExtractor,
    EvidenceTextFeatureExtractor,
    CANONICAL_TARGETS,
    FEATURE_DIM,
)


def test_biological_profile_extractor():
    extractor = BiologicalProfileExtractor(feature_dim=16)

    # 1. Hypericum profile should have high CYP3A4, ABCB1, and NR1I2 values
    h_vec = extractor.get_herb_representation("HERB_HYPERICUM")
    assert isinstance(h_vec, np.ndarray)
    assert len(h_vec) == 16
    cyp3a4_idx = CANONICAL_TARGETS.index("CYP3A4")
    nr1i2_idx = CANONICAL_TARGETS.index("NR1I2")
    assert h_vec[cyp3a4_idx] == 1.0
    assert h_vec[nr1i2_idx] == 1.0

    # 2. Cyclosporine profile should have high CYP3A4 and ABCB1 values
    d_vec = extractor.get_drug_representation("DRUG_CYCLOSPORINE")
    assert isinstance(d_vec, np.ndarray)
    assert len(d_vec) == 16
    assert d_vec[cyp3a4_idx] == 1.0

    # 3. Pair interaction features
    h, d, interaction = extractor.extract_pair_features("HERB_HYPERICUM", "DRUG_CYCLOSPORINE")
    assert interaction[cyp3a4_idx] == 1.0

    # 4. Milk thistle (negative control) should have near zero CYP3A4 modulation
    mt_vec = extractor.get_herb_representation("HERB_MILKTHISTLE")
    assert mt_vec[cyp3a4_idx] < 0.1


def test_evidence_text_feature_extractor_leakage_safety():
    train_texts = [
        "St Johns wort markedly decreases cyclosporin blood concentrations.",
        "Clinical study on P-glycoprotein induction by Hypericum perforatum.",
        "Reduction in oral bioavailability and AUC observed in human volunteers.",
    ]
    test_texts = [
        "Profound drop of cyclosporin trough levels after administration.",
        "Silymarin does not alter indinavir pharmacokinetics in healthy subjects.",
    ]

    extractor = EvidenceTextFeatureExtractor(embedding_dim=16, min_df=1)

    # Unfitted transform must raise RuntimeError
    with pytest.raises(RuntimeError):
        extractor.transform(test_texts)

    # Fit strictly on train
    extractor.fit(train_texts)
    assert extractor._is_fitted
    assert len(extractor.vocabulary_) > 0

    # Transform train and test
    tr_emb = extractor.transform(train_texts)
    te_emb = extractor.transform(test_texts)

    assert tr_emb.shape == (3, 16)
    assert te_emb.shape == (2, 16)

    # Train embeddings with vocabulary overlap should be unit normalized
    for vec in tr_emb:
        assert np.isclose(np.linalg.norm(vec), 1.0, atol=1e-4)
    # Out of vocabulary test embeddings are either unit normalized or zero vector
    for vec in te_emb:
        norm = np.linalg.norm(vec)
        assert np.isclose(norm, 1.0, atol=1e-4) or np.isclose(norm, 0.0, atol=1e-4)
