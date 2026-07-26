"""
Full HDI Model

End-to-end model composition:
GNN Encoder + Text Encoder + Reliability Scorer + Cross-Attention + Link Predictor

Supports three ablation modes for the Week 12 critical comparison:
(a) GNN-only baseline — no text, no reliability
(b) Unconditioned cross-attention — text but R fixed to 1.0
(c) Full reliability-conditioned model — the complete system
"""

from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn
from loguru import logger

from src.models.gnn_encoder import HeterogeneousGNNEncoder
from src.models.text_encoder import TextEncoder
from src.models.reliability_scorer import ReliabilityScorer
from src.models.cross_attention import ReliabilityConditionedCrossAttention
from src.models.link_predictor import LinkPredictor


class HDIModel(nn.Module):
    """
    Full Herb-Drug Interaction Prediction Model.

    Architecture (see architecture diagram):
    1. GNN Encoder: Knowledge graph → molecular embeddings
    2. Text Encoder: Evidence text → text embeddings
    3. Reliability Scorer: Metadata (C,T,B,M,S) → R ∈ [0,1]
    4. Cross-Attention: R-gated fusion of molecular + text embeddings
    5. Link Predictor: Fused embeddings → interaction probability

    Three ablation variants:
    - "gnn_only": Only uses GNN embeddings (no text, no R)
    - "unconditioned": Uses cross-attention but R=1.0 (no gating)
    - "full": Complete reliability-conditioned model

    Usage:
        model = HDIModel(config)
        output = model(batch)
        loss = output["loss"]
    """

    def __init__(
        self,
        gnn_input_dim: int = 128,
        gnn_hidden_dim: int = 256,
        gnn_output_dim: int = 128,
        num_relations: int = 6,
        gnn_layers: int = 3,
        text_model_name: str = "microsoft/BiomedNLP-BiomedBERT-base-uncased-abstract",
        text_output_dim: int = 128,
        cross_attention_heads: int = 4,
        cross_attention_dropout: float = 0.1,
        gating_mode: str = "multiplicative",
        link_predictor_layers: int = 2,
        link_predictor_dropout: float = 0.3,
        use_text: bool = True,
        use_reliability: bool = True,
        dropout: float = 0.2,
    ):
        super().__init__()

        self.use_text = use_text
        self.use_reliability = use_reliability
        self.hidden_dim = gnn_output_dim

        # 1. GNN Encoder (always used)
        self.gnn_encoder = HeterogeneousGNNEncoder(
            input_dim=gnn_input_dim,
            hidden_dim=gnn_hidden_dim,
            output_dim=gnn_output_dim,
            num_relations=num_relations,
            num_layers=gnn_layers,
            dropout=dropout,
        )

        # 2. Text Encoder (used in "unconditioned" and "full" modes)
        if use_text:
            self.text_encoder = TextEncoder(
                model_name=text_model_name,
                output_dim=text_output_dim,
            )
        else:
            self.text_encoder = None

        # 3. Reliability Scorer (used only in "full" mode)
        if use_reliability:
            self.reliability_scorer = ReliabilityScorer()
        else:
            self.reliability_scorer = None

        # 4. Cross-Attention (used when text is enabled)
        if use_text:
            self.cross_attention = ReliabilityConditionedCrossAttention(
                hidden_dim=gnn_output_dim,
                num_heads=cross_attention_heads,
                dropout=cross_attention_dropout,
                gating_mode=gating_mode,
            )
        else:
            self.cross_attention = None

        # 5. Link Predictor (always used)
        self.link_predictor = LinkPredictor(
            hidden_dim=gnn_output_dim,
            num_layers=link_predictor_layers,
            dropout=link_predictor_dropout,
        )

        # Loss function
        self.loss_fn = nn.BCEWithLogitsLoss()

        mode = "full" if (use_text and use_reliability) else (
            "unconditioned" if use_text else "gnn_only"
        )
        logger.info(f"HDIModel initialized in '{mode}' mode")

    def forward(
        self,
        node_features: torch.Tensor,
        edge_index: torch.Tensor,
        edge_type: torch.Tensor,
        source_indices: torch.Tensor,
        target_indices: torch.Tensor,
        metadata: Optional[torch.Tensor] = None,
        evidence_texts: Optional[list[str]] = None,
        labels: Optional[torch.Tensor] = None,
    ) -> dict[str, torch.Tensor]:
        """
        Full forward pass.

        Args:
            node_features: (num_nodes, input_dim) initial node features
            edge_index: (2, num_edges) graph structure
            edge_type: (num_edges,) relation type indices
            source_indices: (batch,) source node indices for link prediction
            target_indices: (batch,) target node indices for link prediction
            metadata: (batch, 5) candidate metadata [C, T, B, M, S]
            evidence_texts: List of evidence text strings (batch,)
            labels: (batch,) binary labels for link prediction

        Returns:
            dict with 'logits', 'probabilities', optionally 'loss',
            'reliability_scores', 'attention_weights'
        """
        # 1. GNN encoding: produce node embeddings
        node_embeddings = self.gnn_encoder(
            node_features, edge_index, edge_type
        )  # (num_nodes, hidden_dim)

        # Extract source and target embeddings
        source_emb = node_embeddings[source_indices]  # (batch, hidden_dim)
        target_emb = node_embeddings[target_indices]  # (batch, hidden_dim)

        result = {}

        # 2. Text encoding + cross-attention fusion (if enabled)
        if self.use_text and self.text_encoder is not None and evidence_texts is not None:
            # Encode text evidence
            # Filter empty texts
            non_empty = [t if t else "no evidence" for t in evidence_texts]
            text_emb = self.text_encoder(text_list=non_empty)
            # (batch, hidden_dim)

            # 3. Compute reliability score (if enabled)
            if self.use_reliability and self.reliability_scorer is not None and metadata is not None:
                R, breakdown = self.reliability_scorer(
                    metadata, return_breakdown=True
                )
                result["reliability_scores"] = R
                result["reliability_breakdown"] = breakdown
            else:
                # Unconditioned: R = 1.0
                R = torch.ones(
                    source_emb.size(0), 1, device=source_emb.device
                )
                result["reliability_scores"] = R

            # 4. Cross-attention fusion
            fused_source, attn_weights = self.cross_attention(
                molecular_embedding=source_emb,
                text_embedding=text_emb,
                reliability_score=R,
                return_attention=True,
            )
            result["attention_weights"] = attn_weights

            # Also fuse target if we have text context
            fused_target, _ = self.cross_attention(
                molecular_embedding=target_emb,
                text_embedding=text_emb,
                reliability_score=R,
            )
        else:
            # GNN-only mode: no text fusion
            fused_source = source_emb
            fused_target = target_emb

        # 5. Link prediction
        logits = self.link_predictor(
            fused_source, fused_target
        )  # (batch, 1)

        result["logits"] = logits.squeeze(-1)
        result["probabilities"] = torch.sigmoid(logits).squeeze(-1)
        result["fused_embeddings"] = fused_source  # For contrastive loss

        # Compute loss if labels provided
        if labels is not None:
            result["loss"] = self.loss_fn(
                logits.squeeze(-1), labels.float()
            )

        return result

    @classmethod
    def gnn_only(cls, **kwargs) -> "HDIModel":
        """Create GNN-only baseline (ablation variant a)."""
        return cls(use_text=False, use_reliability=False, **kwargs)

    @classmethod
    def unconditioned(cls, **kwargs) -> "HDIModel":
        """Create unconditioned cross-attention model (ablation variant b)."""
        return cls(use_text=True, use_reliability=False, **kwargs)

    @classmethod
    def full_model(cls, **kwargs) -> "HDIModel":
        """Create full reliability-conditioned model (ablation variant c)."""
        return cls(use_text=True, use_reliability=True, **kwargs)

    def count_parameters(self) -> dict[str, int]:
        """Count trainable parameters per component."""
        counts = {
            "gnn_encoder": sum(
                p.numel() for p in self.gnn_encoder.parameters() if p.requires_grad
            ),
            "link_predictor": sum(
                p.numel() for p in self.link_predictor.parameters() if p.requires_grad
            ),
        }

        if self.text_encoder is not None:
            counts["text_encoder"] = sum(
                p.numel() for p in self.text_encoder.parameters() if p.requires_grad
            )

        if self.reliability_scorer is not None:
            counts["reliability_scorer"] = sum(
                p.numel() for p in self.reliability_scorer.parameters() if p.requires_grad
            )

        if self.cross_attention is not None:
            counts["cross_attention"] = sum(
                p.numel() for p in self.cross_attention.parameters() if p.requires_grad
            )

        counts["total"] = sum(counts.values())
        return counts
