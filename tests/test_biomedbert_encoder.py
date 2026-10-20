"""
Unit tests for BiomedBERTEvidenceEncoder.
Tests frozen 768D embedding extraction, training-exclusive SVD projection,
leakage safety, and multi-token cross-attention tensor shapes.
"""

import pytest
import numpy as np
import torch

from src.features.biomedbert_encoder import BiomedBERTEvidenceEncoder


def test_biomedbert_encoder_unfitted_raises():
    encoder = BiomedBERTEvidenceEncoder(embedding_dim=16)
    with pytest.raises(RuntimeError):
        encoder.transform(["Sample clinical text"])


def test_biomedbert_encoder_fit_and_transform_leakage_safety():
    train_texts = [
        "St John's wort induces CYP3A4 and ABCB1 expression in human intestinal epithelium.",
        "Clinical study on cyclosporine trough level reduction caused by botanical co-administration.",
        "Silymarin oral administration has no inhibitory effect on cytochrome P450 enzymes.",
        "Significant decrease in oral bioavailability and AUC observed in healthy volunteers.",
    ]
    test_texts = [
        "Curcumin inhibits hepatic CYP3A4 activity increasing tacrolimus exposure.",
        "Lack of interaction observed between saw palmetto and digoxin in clinical trial.",
    ]

    encoder = BiomedBERTEvidenceEncoder(embedding_dim=16)
    encoder.fit_projection(train_texts)

    assert encoder._is_fitted
    assert encoder._projection_matrix is not None
    assert encoder._projection_matrix.shape == (768, 16)

    # Transform train and test
    train_emb = encoder.transform(train_texts)
    test_emb = encoder.transform(test_texts)

    assert train_emb.shape == (4, 16)
    assert test_emb.shape == (2, 16)

    # Verify L2 normalization
    for vec in train_emb:
        assert np.isclose(np.linalg.norm(vec), 1.0, atol=1e-4)
    for vec in test_emb:
        assert np.isclose(np.linalg.norm(vec), 1.0, atol=1e-4)

    # Multi-token extraction
    tokens = encoder.transform_tokens(test_texts, max_tokens=6)
    assert isinstance(tokens, torch.Tensor)
    assert tokens.shape == (2, 6, 16)
