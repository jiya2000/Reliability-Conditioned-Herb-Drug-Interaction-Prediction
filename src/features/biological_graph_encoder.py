"""
Heterogeneous Biological Knowledge Graph Encoder.

Constructs an auditable tripartite pharmacological knowledge graph G = (V, E)
where:
  V = V_herb ∪ V_target ∪ V_drug
  E = E_herb_target ∪ E_drug_target ∪ E_target_target

Initializes node embeddings from human pharmacokinetic target profiles (CYP enzymes,
efflux/uptake transporters, nuclear receptors), and applies a 2-layer Graph Attention
Network (GAT) with residual connections to learn contextualized representations of herbs
and drugs for downstream reliability-conditioned interaction prediction.
"""

from __future__ import annotations

from typing import Dict, List, Tuple, Optional, Any
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from loguru import logger

from src.features.biological_features import (
    CANONICAL_TARGETS,
    TARGET_TO_IDX,
    HERB_BIOLOGICAL_PROFILES,
    DRUG_BIOLOGICAL_PROFILES,
    FEATURE_DIM,
    BiologicalProfileExtractor,
)


# Target-Target functional coupling edges (biological co-regulation / homology)
TARGET_FUNCTIONAL_COUPLINGS = [
    ("NR1I2", "CYP3A4", 0.95),   # PXR master transcription factor inducing CYP3A4
    ("NR1I2", "ABCB1", 0.90),    # PXR co-induction of P-glycoprotein
    ("CYP3A4", "ABCB1", 0.85),   # Canonical coordinated intestinal/hepatic clearance axis
    ("CYP2C9", "CYP2C19", 0.80), # 2C subfamily catalytic overlap
    ("CYP2C9", "VKORC1", 0.90),  # Pharmacokinetic (2C9) - pharmacodynamic (VKORC1) warfarin axis
    ("SLCO1A2", "SLCO1B1", 0.75),# OATP organic anion transporting polypeptide family
    ("PTGS1", "PTGS2", 0.85),    # COX-1 and COX-2 cyclooxygenase isoenzymes
    ("PTGS1", "PTAFR", 0.70),    # Coordinated platelet aggregation pathway
    ("HSD11B2", "CYP3A4", 0.50), # Glucocorticoid clearance network
]


class BiologicalGATLayer(nn.Module):
    """
    Graph Attention Layer operating on biological interaction bipartite/tripartite edges.
    Computes learned attention coefficients with LeakyReLU and softmax,
    applying residual connections and layer normalization.
    """

    def __init__(self, in_features: int, out_features: int, num_heads: int = 2, dropout: float = 0.1):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.num_heads = num_heads
        self.head_dim = out_features // num_heads

        self.lin_src = nn.Linear(in_features, out_features, bias=False)
        self.lin_dst = nn.Linear(in_features, out_features, bias=False)
        self.attn_l = nn.Parameter(torch.empty(1, num_heads, self.head_dim))
        self.attn_r = nn.Parameter(torch.empty(1, num_heads, self.head_dim))

        self.residual = nn.Linear(in_features, out_features) if in_features != out_features else nn.Identity()
        self.norm = nn.LayerNorm(out_features)
        self.dropout = nn.Dropout(dropout)

        self._reset_parameters()

    def _reset_parameters(self):
        nn.init.xavier_uniform_(self.lin_src.weight)
        nn.init.xavier_uniform_(self.lin_dst.weight)
        nn.init.xavier_uniform_(self.attn_l)
        nn.init.xavier_uniform_(self.attn_r)

    def forward(
        self,
        x_src: torch.Tensor,
        x_dst: torch.Tensor,
        edge_index: torch.Tensor,
        edge_weight: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Forward message passing from x_src to x_dst along edge_index [2, E].
        """
        src_nodes, dst_nodes = edge_index[0], edge_index[1]
        h_src = self.lin_src(x_src).view(-1, self.num_heads, self.head_dim)
        h_dst = self.lin_dst(x_dst).view(-1, self.num_heads, self.head_dim)

        # Attention logits on edges
        alpha_src = (h_src[src_nodes] * self.attn_l).sum(dim=-1)  # (E, num_heads)
        alpha_dst = (h_dst[dst_nodes] * self.attn_r).sum(dim=-1)  # (E, num_heads)
        alpha = F.leaky_relu(alpha_src + alpha_dst, negative_slope=0.2)

        if edge_weight is not None:
            alpha = alpha * edge_weight.unsqueeze(-1)

        # Softmax normalization per destination node
        # Using destination-grouped exp normalization
        num_dst = x_dst.size(0)
        exp_alpha = torch.exp(alpha - alpha.max(dim=0, keepdim=True)[0].clamp(min=-10.0))
        sum_exp = torch.zeros(num_dst, self.num_heads, device=x_dst.device)
        sum_exp.index_add_(0, dst_nodes, exp_alpha)
        attn_norm = exp_alpha / (sum_exp[dst_nodes] + 1e-9)
        attn_norm = self.dropout(attn_norm)

        # Aggregate weighted source messages
        weighted_msgs = h_src[src_nodes] * attn_norm.unsqueeze(-1)  # (E, num_heads, head_dim)
        out = torch.zeros(num_dst, self.num_heads, self.head_dim, device=x_dst.device)
        out.index_add_(0, dst_nodes, weighted_msgs)
        out = out.view(num_dst, self.out_features)

        # Residual connection + LayerNorm
        res = self.residual(x_dst)
        out = self.norm(out + res)
        return F.gelu(out)


class HeterogeneousBiologicalGraphEncoder(nn.Module):
    """
    Heterogeneous biological knowledge graph encoder for herb-drug interaction prediction.
    Constructs tripartite topology connecting herbs, canonical human pharmacokinetic targets,
    and drugs. Initializes nodes from curated biological mechanisms and refines representations
    via 2-layer Graph Attention Network with residual preservation.
    """

    def __init__(
        self,
        hidden_dim: int = 16,
        num_heads: int = 2,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.target_dim = len(CANONICAL_TARGETS)

        # Index canonical entities
        self.herb_list = sorted(list(HERB_BIOLOGICAL_PROFILES.keys()))
        self.drug_list = sorted(list(DRUG_BIOLOGICAL_PROFILES.keys()))
        self.target_list = list(CANONICAL_TARGETS)

        self.herb_to_idx = {h: i for i, h in enumerate(self.herb_list)}
        self.drug_to_idx = {d: i for i, d in enumerate(self.drug_list)}
        self.target_to_idx = {t: i for i, t in enumerate(self.target_list)}

        num_herbs = len(self.herb_list)
        num_drugs = len(self.drug_list)
        num_targets = len(self.target_list)

        # Feature projections for initial mechanisms
        self.herb_init_proj = nn.Linear(self.target_dim, hidden_dim)
        self.drug_init_proj = nn.Linear(self.target_dim, hidden_dim)
        self.target_init_proj = nn.Linear(self.target_dim, hidden_dim)

        # 2 GAT Message-passing layers
        self.gat1_t_to_h = BiologicalGATLayer(hidden_dim, hidden_dim, num_heads, dropout)
        self.gat1_t_to_d = BiologicalGATLayer(hidden_dim, hidden_dim, num_heads, dropout)
        self.gat1_t_to_t = BiologicalGATLayer(hidden_dim, hidden_dim, num_heads, dropout)

        self.gat2_t_to_h = BiologicalGATLayer(hidden_dim, hidden_dim, num_heads, dropout)
        self.gat2_t_to_d = BiologicalGATLayer(hidden_dim, hidden_dim, num_heads, dropout)

        # Final projection heads
        self.herb_encoder = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.LayerNorm(hidden_dim),
        )
        self.drug_encoder = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.LayerNorm(hidden_dim),
        )

        self._build_static_tensors()
        logger.info(
            f"Initialized HeterogeneousBiologicalGraphEncoder: {num_herbs} herbs, "
            f"{num_drugs} drugs, {num_targets} canonical targets (hidden_dim={hidden_dim})"
        )

    def _build_static_tensors(self):
        """Build initial pharmacokinetic matrices and edge indices."""
        extractor = BiologicalProfileExtractor(feature_dim=self.target_dim)

        # 1. Herb initial mechanism matrix (num_herbs, 16)
        h_matrix = np.zeros((len(self.herb_list), self.target_dim), dtype=np.float32)
        for h, idx in self.herb_to_idx.items():
            h_matrix[idx] = extractor.get_herb_representation(h)
        self.register_buffer("herb_init_features", torch.from_numpy(h_matrix))

        # 2. Drug initial dependency matrix (num_drugs, 16)
        d_matrix = np.zeros((len(self.drug_list), self.target_dim), dtype=np.float32)
        for d, idx in self.drug_to_idx.items():
            d_matrix[idx] = extractor.get_drug_representation(d)
        self.register_buffer("drug_init_features", torch.from_numpy(d_matrix))

        # 3. Target identity / one-hot matrix (16, 16)
        t_matrix = np.eye(self.target_dim, dtype=np.float32)
        self.register_buffer("target_init_features", torch.from_numpy(t_matrix))

        # 4. Target -> Herb bipartite edges (target_idx -> herb_idx)
        th_edges, th_weights = [], []
        for h, profile in HERB_BIOLOGICAL_PROFILES.items():
            if h in self.herb_to_idx:
                h_idx = self.herb_to_idx[h]
                for tgt, val in profile.items():
                    if tgt in self.target_to_idx and abs(val) > 1e-4:
                        t_idx = self.target_to_idx[tgt]
                        th_edges.append([t_idx, h_idx])
                        th_weights.append(abs(val))

        # Fallback dense connection if empty
        if not th_edges:
            th_edges = [[0, 0]]
            th_weights = [1.0]
        self.register_buffer("th_edge_index", torch.tensor(th_edges, dtype=torch.long).t())
        self.register_buffer("th_edge_weight", torch.tensor(th_weights, dtype=torch.float32))

        # 5. Target -> Drug bipartite edges (target_idx -> drug_idx)
        td_edges, td_weights = [], []
        for d, profile in DRUG_BIOLOGICAL_PROFILES.items():
            if d in self.drug_to_idx:
                d_idx = self.drug_to_idx[d]
                for tgt, val in profile.items():
                    if tgt in self.target_to_idx and abs(val) > 1e-4:
                        t_idx = self.target_to_idx[tgt]
                        td_edges.append([t_idx, d_idx])
                        td_weights.append(abs(val))

        if not td_edges:
            td_edges = [[0, 0]]
            td_weights = [1.0]
        self.register_buffer("td_edge_index", torch.tensor(td_edges, dtype=torch.long).t())
        self.register_buffer("td_edge_weight", torch.tensor(td_weights, dtype=torch.float32))

        # 6. Target -> Target functional coupling edges
        tt_edges, tt_weights = [], []
        for src, dst, w in TARGET_FUNCTIONAL_COUPLINGS:
            if src in self.target_to_idx and dst in self.target_to_idx:
                s_idx = self.target_to_idx[src]
                d_idx = self.target_to_idx[dst]
                tt_edges.append([s_idx, d_idx])
                tt_weights.append(w)
                # Symmetrical edge
                tt_edges.append([d_idx, s_idx])
                tt_weights.append(w)

        if not tt_edges:
            tt_edges = [[0, 0]]
            tt_weights = [1.0]
        self.register_buffer("tt_edge_index", torch.tensor(tt_edges, dtype=torch.long).t())
        self.register_buffer("tt_edge_weight", torch.tensor(tt_weights, dtype=torch.float32))

    def compute_all_node_embeddings(self) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Compute contextualized graph node embeddings for all herbs and drugs.
        Returns:
            herb_embs: (num_herbs, hidden_dim)
            drug_embs: (num_drugs, hidden_dim)
        """
        # Initial projection
        h_emb = self.herb_init_proj(self.herb_init_features)
        d_emb = self.drug_init_proj(self.drug_init_features)
        t_emb = self.target_init_proj(self.target_init_features)

        # Layer 1: Target-Target functional coupling & Target -> Herb/Drug propagation
        t_emb1 = self.gat1_t_to_t(t_emb, t_emb, self.tt_edge_index, self.tt_edge_weight)
        h_emb1 = self.gat1_t_to_h(t_emb1, h_emb, self.th_edge_index, self.th_edge_weight)
        d_emb1 = self.gat1_t_to_d(t_emb1, d_emb, self.td_edge_index, self.td_edge_weight)

        # Layer 2: Refinement propagation
        h_emb2 = self.gat2_t_to_h(t_emb1, h_emb1, self.th_edge_index, self.th_edge_weight)
        d_emb2 = self.gat2_t_to_d(t_emb1, d_emb1, self.td_edge_index, self.td_edge_weight)

        final_herb = self.herb_encoder(h_emb2)
        final_drug = self.drug_encoder(d_emb2)

        # L2 normalize
        final_herb = F.normalize(final_herb, p=2, dim=-1)
        final_drug = F.normalize(final_drug, p=2, dim=-1)

        return final_herb, final_drug

    def encode_entities(self, herb_ids: List[str], drug_ids: List[str]) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Batch lookup/projection for specific herb and drug IDs.
        Returns:
            herb_tensors: (batch, hidden_dim)
            drug_tensors: (batch, hidden_dim)
        """
        all_h, all_d = self.compute_all_node_embeddings()

        h_indices = [self.herb_to_idx.get(h, 0) for h in herb_ids]
        d_indices = [self.drug_to_idx.get(d, 0) for d in drug_ids]

        batch_h = all_h[h_indices]
        batch_d = all_d[d_indices]
        return batch_h, batch_d

    def forward(self, herb_ids: List[str], drug_ids: List[str]) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Forward pass returning (herb_emb, drug_emb, interaction_potential).
        Interaction potential = element-wise product measuring shared mechanism alignment.
        """
        h_emb, d_emb = self.encode_entities(herb_ids, drug_ids)
        interaction_potential = h_emb * d_emb
        return h_emb, d_emb, interaction_potential
