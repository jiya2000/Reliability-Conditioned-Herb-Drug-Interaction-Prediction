"""
Heterogeneous GNN Encoder

Graph neural network encoder for the heterogeneous knowledge graph.
Supports multiple architectures:
- RGCN (Relational Graph Convolutional Network) — default
- HGT (Heterogeneous Graph Transformer)

Produces molecular/entity embeddings from the knowledge graph structure.
"""

from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from loguru import logger


class RGCNConv(nn.Module):
    """
    Relational Graph Convolutional layer.

    For heterogeneous graphs with multiple edge types.
    Each edge type has its own weight matrix for message passing.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        num_relations: int,
        num_bases: Optional[int] = None,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.num_relations = num_relations

        # Basis decomposition to reduce parameters
        if num_bases is not None and num_bases < num_relations:
            self.num_bases = num_bases
            self.basis = nn.Parameter(
                torch.Tensor(num_bases, in_channels, out_channels)
            )
            self.att = nn.Parameter(torch.Tensor(num_relations, num_bases))
        else:
            self.num_bases = None
            self.weight = nn.Parameter(
                torch.Tensor(num_relations, in_channels, out_channels)
            )

        self.root = nn.Parameter(torch.Tensor(in_channels, out_channels))
        self.bias = nn.Parameter(torch.Tensor(out_channels))
        self.dropout = nn.Dropout(dropout)

        self.reset_parameters()

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.root)
        nn.init.zeros_(self.bias)
        if self.num_bases is not None:
            nn.init.xavier_uniform_(self.basis)
            nn.init.xavier_uniform_(self.att)
        else:
            nn.init.xavier_uniform_(self.weight)

    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        edge_type: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            x: (num_nodes, in_channels) node features
            edge_index: (2, num_edges) edge source/target indices
            edge_type: (num_edges,) edge type indices

        Returns:
            out: (num_nodes, out_channels) updated node features
        """
        num_nodes = x.size(0)
        out = torch.zeros(num_nodes, self.out_channels, device=x.device)

        # Compute per-relation weight matrices
        if self.num_bases is not None:
            # Basis decomposition: W_r = sum_b att[r,b] * basis[b]
            weight = torch.einsum("rb,bio->rio", self.att, self.basis)
        else:
            weight = self.weight

        # Message passing per relation type
        for r in range(self.num_relations):
            mask = edge_type == r
            if not mask.any():
                continue

            edge_idx_r = edge_index[:, mask]
            src, tgt = edge_idx_r[0], edge_idx_r[1]

            # Message: W_r @ x[src]
            msg = torch.mm(x[src], weight[r])
            msg = self.dropout(msg)

            # Aggregate: sum messages at target nodes
            out.index_add_(0, tgt, msg)

        # Self-loop transformation
        out = out + torch.mm(x, self.root)
        out = out + self.bias

        return out


class HeterogeneousGNNEncoder(nn.Module):
    """
    Multi-layer heterogeneous GNN encoder.

    Stacks multiple RGCN layers with skip connections, layer norm,
    and dropout to produce node embeddings from the knowledge graph.

    The output embeddings are the "molecular embeddings" that enter
    the reliability-conditioned cross-attention layer.

    Usage:
        encoder = HeterogeneousGNNEncoder(
            input_dim=128,
            hidden_dim=256,
            output_dim=128,
            num_relations=6,
            num_layers=3,
        )
        node_embeddings = encoder(x, edge_index, edge_type)
    """

    def __init__(
        self,
        input_dim: int = 128,
        hidden_dim: int = 256,
        output_dim: int = 128,
        num_relations: int = 6,
        num_layers: int = 3,
        num_bases: Optional[int] = None,
        dropout: float = 0.2,
        use_layer_norm: bool = True,
        use_residual: bool = True,
    ):
        super().__init__()
        self.num_layers = num_layers
        self.use_layer_norm = use_layer_norm
        self.use_residual = use_residual

        # Build GNN layers
        self.convs = nn.ModuleList()
        self.norms = nn.ModuleList()

        for i in range(num_layers):
            in_ch = input_dim if i == 0 else hidden_dim
            out_ch = output_dim if i == num_layers - 1 else hidden_dim

            self.convs.append(
                RGCNConv(
                    in_channels=in_ch,
                    out_channels=out_ch,
                    num_relations=num_relations,
                    num_bases=num_bases,
                    dropout=dropout,
                )
            )

            if use_layer_norm:
                self.norms.append(nn.LayerNorm(out_ch))

        # Projection for residual connections when dimensions change
        self.input_proj = (
            nn.Linear(input_dim, hidden_dim)
            if input_dim != hidden_dim and num_layers > 1
            else None
        )
        self.output_proj = (
            nn.Linear(hidden_dim, output_dim)
            if hidden_dim != output_dim and num_layers > 1
            else None
        )

        self.dropout = nn.Dropout(dropout)

        logger.info(
            f"HeterogeneousGNNEncoder: {num_layers} layers, "
            f"{input_dim}→{hidden_dim}→{output_dim}, "
            f"{num_relations} relations"
        )

    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        edge_type: torch.Tensor,
    ) -> torch.Tensor:
        """
        Encode the heterogeneous graph.

        Args:
            x: (num_nodes, input_dim) initial node features
            edge_index: (2, num_edges) edge indices
            edge_type: (num_edges,) edge type indices

        Returns:
            h: (num_nodes, output_dim) node embeddings
        """
        h = x

        for i, conv in enumerate(self.convs):
            h_prev = h

            # Graph convolution
            h = conv(h, edge_index, edge_type)

            # Layer normalization
            if self.use_layer_norm and i < len(self.norms):
                h = self.norms[i](h)

            # Activation (skip on last layer)
            if i < self.num_layers - 1:
                h = F.gelu(h)
                h = self.dropout(h)

            # Residual connection
            if self.use_residual and h.shape == h_prev.shape:
                h = h + h_prev

        return h

    def get_embedding_dim(self) -> int:
        """Return the output embedding dimension."""
        return self.convs[-1].out_channels
