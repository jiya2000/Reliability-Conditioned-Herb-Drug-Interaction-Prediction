"""
Link Predictor

Predicts the probability of an interaction edge between two nodes
based on their fused embeddings.

Takes the output of the cross-attention fusion layer and produces
a binary interaction probability.
"""

from __future__ import annotations

import torch
import torch.nn as nn
from loguru import logger


class LinkPredictor(nn.Module):
    """
    MLP-based link prediction head.

    Takes fused node pair embeddings and predicts interaction probability.

    Supports multiple scoring functions:
    - "mlp": Concatenate + MLP (default)
    - "dot": Dot product
    - "bilinear": Bilinear scoring

    Usage:
        predictor = LinkPredictor(hidden_dim=128)
        logits = predictor(source_emb, target_emb)
    """

    def __init__(
        self,
        hidden_dim: int = 128,
        num_layers: int = 2,
        dropout: float = 0.3,
        scoring: str = "mlp",
    ):
        super().__init__()
        self.scoring = scoring

        if scoring == "mlp":
            layers = []
            input_dim = hidden_dim * 2  # Concatenated source + target

            for i in range(num_layers):
                out_dim = hidden_dim if i < num_layers - 1 else 1
                layers.append(nn.Linear(input_dim, out_dim))
                if i < num_layers - 1:
                    layers.append(nn.GELU())
                    layers.append(nn.Dropout(dropout))
                input_dim = out_dim

            self.mlp = nn.Sequential(*layers)

        elif scoring == "dot":
            # Simple dot product (no learnable parameters)
            self.mlp = None

        elif scoring == "bilinear":
            self.bilinear = nn.Bilinear(hidden_dim, hidden_dim, 1)

        else:
            raise ValueError(f"Unknown scoring: {scoring}")

        logger.info(
            f"LinkPredictor: scoring={scoring}, "
            f"hidden_dim={hidden_dim}, layers={num_layers}"
        )

    def forward(
        self,
        source_embedding: torch.Tensor,
        target_embedding: torch.Tensor,
    ) -> torch.Tensor:
        """
        Predict interaction probability.

        Args:
            source_embedding: (batch, hidden_dim) source node embedding
            target_embedding: (batch, hidden_dim) target node embedding

        Returns:
            logits: (batch, 1) interaction logits (pre-sigmoid)
        """
        if self.scoring == "mlp":
            combined = torch.cat(
                [source_embedding, target_embedding], dim=-1
            )
            return self.mlp(combined)

        elif self.scoring == "dot":
            return (source_embedding * target_embedding).sum(
                dim=-1, keepdim=True
            )

        elif self.scoring == "bilinear":
            return self.bilinear(source_embedding, target_embedding)

        else:
            raise ValueError(f"Unknown scoring: {self.scoring}")

    def predict_proba(
        self,
        source_embedding: torch.Tensor,
        target_embedding: torch.Tensor,
    ) -> torch.Tensor:
        """Return probability (sigmoid of logits)."""
        logits = self.forward(source_embedding, target_embedding)
        return torch.sigmoid(logits)
