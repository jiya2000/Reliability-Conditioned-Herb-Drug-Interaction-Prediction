"""Feature extraction modules for biological target profiling and evidence representation."""
from src.features.biological_features import (
    BiologicalProfileExtractor,
    EvidenceTextFeatureExtractor,
    CANONICAL_TARGETS,
    FEATURE_DIM,
)

__all__ = [
    "BiologicalProfileExtractor",
    "EvidenceTextFeatureExtractor",
    "CANONICAL_TARGETS",
    "FEATURE_DIM",
]
