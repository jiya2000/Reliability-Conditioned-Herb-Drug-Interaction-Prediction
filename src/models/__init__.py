"""
HDI Prediction — Model Architecture

Core neural network components:
- Heterogeneous GNN encoder (RGCN/HGT)
- Text encoder (PubMedBERT-based)
- Reliability scorer (C, T, B, M, S → R)
- Reliability-gated fusion (core invention)
- Link predictor
- Full end-to-end HDI model
"""

from src.models.gnn_encoder import HeterogeneousGNNEncoder
from src.models.text_encoder import TextEncoder
from src.models.reliability_scorer import ReliabilityScorer
from src.models.cross_attention import ReliabilityGatedFusion
from src.models.link_predictor import LinkPredictor
from src.models.hdi_model import HDIModel
from src.models.interaction_model import ReliabilityConditionedInteractionModel, CalibratedDecision
from src.models.citation_dag import (
    CitationDAG,
    CitationEchoAnalyzer,
    CitationDAGEchoAttenuator,
    CitationDAGResult,
)
from src.models.context_compatibility import (
    ContextCompatibilityScorer,
    ContextMetadataParser,
    PharmacokineticContext,
)
from src.models.conformal_abstention import (
    ConformalTriStateEngine,
    ClinicalDecision,
    ConformalDecision,
    ConformalMetrics,
)

__all__ = [
    "HeterogeneousGNNEncoder",
    "TextEncoder",
    "ReliabilityScorer",
    "ReliabilityGatedFusion",
    "LinkPredictor",
    "HDIModel",
    "ReliabilityConditionedInteractionModel",
    "CalibratedDecision",
    "CitationDAG",
    "CitationEchoAnalyzer",
    "CitationDAGEchoAttenuator",
    "CitationDAGResult",
    "ContextCompatibilityScorer",
    "ContextMetadataParser",
    "PharmacokineticContext",
    "ConformalTriStateEngine",
    "ClinicalDecision",
    "ConformalDecision",
    "ConformalMetrics",
]


