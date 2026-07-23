"""
HDI Prediction — Model Architecture

Core neural network components:
- Heterogeneous GNN encoder (RGCN/HGT)
- Text encoder (PubMedBERT-based)
- Reliability scorer (C, T, B, M, S → R)
- Reliability-conditioned cross-attention (core invention)
- Link predictor
- Full end-to-end HDI model
"""

from src.models.gnn_encoder import HeterogeneousGNNEncoder
from src.models.text_encoder import TextEncoder
from src.models.reliability_scorer import ReliabilityScorer
from src.models.cross_attention import ReliabilityConditionedCrossAttention
from src.models.link_predictor import LinkPredictor
from src.models.hdi_model import HDIModel

__all__ = [
    "HeterogeneousGNNEncoder",
    "TextEncoder",
    "ReliabilityScorer",
    "ReliabilityConditionedCrossAttention",
    "LinkPredictor",
    "HDIModel",
]
