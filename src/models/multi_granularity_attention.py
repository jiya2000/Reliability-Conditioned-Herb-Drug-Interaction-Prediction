"""
Multi-Granularity Cross-Attention

★ NOVEL CONTRIBUTION 4 ★

Extends reliability-conditioned cross-attention to operate at
three granularity levels simultaneously:
1. Token-level: Fine-grained word/subword ↔ molecular substructure
2. Sentence-level: Evidence span relevance weighting
3. Document-level: Source-wide reliability gating (the original R)

These three levels are then hierarchically fused using a learned
weighting mechanism that adapts to the evidence quality.

Patent value: Extends the claim from "a reliability score gates
cross-attention" to "a hierarchical reliability-conditioned attention
mechanism operates at multiple granularities."
"""

from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from loguru import logger


class TokenLevelAttention(nn.Module):
    """
    Fine-grained token-level cross-attention between text tokens
    and molecular embedding dimensions.

    Each text token attends to the molecular embedding independently,
    weighted by a per-token reliability derived from the document-level R.
    """

    def __init__(
        self,
        hidden_dim: int = 128,
        num_heads: int = 4,
        max_tokens: int = 64,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.max_tokens = max_tokens
        self.head_dim = hidden_dim // num_heads

        # Token-level attention projections
        self.W_q_token = nn.Linear(hidden_dim, hidden_dim)
        self.W_k_token = nn.Linear(hidden_dim, hidden_dim)
        self.W_v_token = nn.Linear(hidden_dim, hidden_dim)

        # Token-level reliability modulation
        self.token_reliability_proj = nn.Sequential(
            nn.Linear(hidden_dim + 1, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid(),
        )

        self.norm = nn.LayerNorm(hidden_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        token_embeddings: torch.Tensor,
        molecular_embedding: torch.Tensor,
        document_R: torch.Tensor,
        token_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Args:
            token_embeddings: (batch, max_tokens, hidden_dim)
            molecular_embedding: (batch, hidden_dim)
            document_R: (batch, 1) document-level reliability
            token_mask: (batch, max_tokens) boolean mask

        Returns:
            attended: (batch, hidden_dim) token-level fused embedding
        """
        batch_size = token_embeddings.size(0)

        # Compute per-token reliability by modulating with document R
        R_expanded = document_R.unsqueeze(1).expand(
            -1, token_embeddings.size(1), -1
        )
        token_R_input = torch.cat([token_embeddings, R_expanded], dim=-1)
        token_R = self.token_reliability_proj(token_R_input)  # (batch, tokens, 1)

        # Query from molecule, Key/Value from tokens
        Q = self.W_q_token(molecular_embedding).unsqueeze(1)  # (batch, 1, dim)
        K = self.W_k_token(token_embeddings)  # (batch, tokens, dim)
        V = self.W_v_token(token_embeddings)  # (batch, tokens, dim)

        # Scaled dot-product attention
        scale = self.head_dim ** 0.5
        attn_logits = torch.bmm(Q, K.transpose(1, 2)) / scale  # (batch, 1, tokens)

        # Apply token mask
        if token_mask is not None:
            attn_logits = attn_logits.masked_fill(
                ~token_mask.unsqueeze(1), float("-inf")
            )

        attn_weights = F.softmax(attn_logits, dim=-1)

        # Modulate attention with per-token reliability
        attn_weights = attn_weights * token_R.transpose(1, 2)

        # Re-normalize
        attn_sum = attn_weights.sum(dim=-1, keepdim=True).clamp(min=1e-8)
        attn_weights = attn_weights / attn_sum

        attn_weights = self.dropout(attn_weights)

        # Attend
        attended = torch.bmm(attn_weights, V)  # (batch, 1, dim)
        attended = attended.squeeze(1)  # (batch, dim)

        return self.norm(attended + molecular_embedding)


class SentenceLevelAttention(nn.Module):
    """
    Sentence-level attention that weights multiple evidence spans
    based on their relevance and reliability.

    Each evidence sentence contributes to the fused representation
    proportional to its estimated relevance × reliability.
    """

    def __init__(
        self,
        hidden_dim: int = 128,
        max_sentences: int = 8,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.max_sentences = max_sentences

        # Sentence relevance scorer
        self.relevance_scorer = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, 1),
        )

        # Sentence-level reliability modulation
        self.sentence_reliability_gate = nn.Sequential(
            nn.Linear(hidden_dim + 1, hidden_dim // 2),
            nn.GELU(),
            nn.Linear(hidden_dim // 2, 1),
            nn.Sigmoid(),
        )

        self.output_proj = nn.Linear(hidden_dim, hidden_dim)
        self.norm = nn.LayerNorm(hidden_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        sentence_embeddings: torch.Tensor,
        molecular_embedding: torch.Tensor,
        document_R: torch.Tensor,
        sentence_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Args:
            sentence_embeddings: (batch, max_sentences, hidden_dim)
            molecular_embedding: (batch, hidden_dim)
            document_R: (batch, 1) document-level reliability
            sentence_mask: (batch, max_sentences) boolean mask

        Returns:
            fused: (batch, hidden_dim) sentence-level fused embedding
        """
        batch_size = sentence_embeddings.size(0)
        num_sentences = sentence_embeddings.size(1)

        # Compute relevance: how relevant is each sentence to the molecule
        mol_expanded = molecular_embedding.unsqueeze(1).expand(
            -1, num_sentences, -1
        )
        relevance_input = torch.cat(
            [sentence_embeddings, mol_expanded], dim=-1
        )
        relevance_scores = self.relevance_scorer(relevance_input)  # (batch, sents, 1)

        # Compute per-sentence reliability gate
        R_expanded = document_R.unsqueeze(1).expand(-1, num_sentences, -1)
        gate_input = torch.cat([sentence_embeddings, R_expanded], dim=-1)
        sentence_gates = self.sentence_reliability_gate(gate_input)  # (batch, sents, 1)

        # Combined weight = relevance × reliability gate
        combined_weights = relevance_scores * sentence_gates  # (batch, sents, 1)

        # Apply mask
        if sentence_mask is not None:
            combined_weights = combined_weights.masked_fill(
                ~sentence_mask.unsqueeze(-1), float("-inf")
            )

        # Softmax over sentences
        attention = F.softmax(combined_weights, dim=1)
        attention = self.dropout(attention)

        # Weighted sum
        attended = (attention * sentence_embeddings).sum(dim=1)  # (batch, dim)
        output = self.output_proj(attended)

        return self.norm(output + molecular_embedding)


class MultiGranularityCrossAttention(nn.Module):
    """
    ★ NOVEL: Multi-Granularity Reliability-Conditioned Cross-Attention ★

    Operates at three levels simultaneously:
    1. Token-level: Fine-grained subword ↔ molecular attention
    2. Sentence-level: Evidence span relevance weighting
    3. Document-level: Source-wide reliability gating (original R)

    A learned fusion mechanism combines the three levels,
    adapting the relative importance of each granularity based
    on the evidence characteristics.

    Usage:
        mg_attention = MultiGranularityCrossAttention(hidden_dim=128)
        fused = mg_attention(
            molecular_embedding=mol_emb,
            text_embedding=text_emb,
            reliability_score=R,
            token_embeddings=token_embs,       # optional
            sentence_embeddings=sentence_embs,  # optional
        )
    """

    def __init__(
        self,
        hidden_dim: int = 128,
        num_heads: int = 4,
        max_tokens: int = 64,
        max_sentences: int = 8,
        dropout: float = 0.1,
        temperature: float = 1.0,
    ):
        super().__init__()
        self.hidden_dim = hidden_dim

        # Token-level attention
        self.token_attention = TokenLevelAttention(
            hidden_dim=hidden_dim,
            num_heads=num_heads,
            max_tokens=max_tokens,
            dropout=dropout,
        )

        # Sentence-level attention
        self.sentence_attention = SentenceLevelAttention(
            hidden_dim=hidden_dim,
            max_sentences=max_sentences,
            dropout=dropout,
        )

        # Document-level attention (reuses the base cross-attention pattern)
        self.doc_W_q = nn.Linear(hidden_dim, hidden_dim)
        self.doc_W_k = nn.Linear(hidden_dim, hidden_dim)
        self.doc_W_v = nn.Linear(hidden_dim, hidden_dim)
        self.doc_W_o = nn.Linear(hidden_dim, hidden_dim)
        self.doc_norm = nn.LayerNorm(hidden_dim)

        # Hierarchical fusion: learned weighting of three levels
        self.fusion_weights = nn.Sequential(
            nn.Linear(hidden_dim * 3 + 1, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, 3),
            nn.Softmax(dim=-1),
        )

        # Final projection
        self.output_proj = nn.Linear(hidden_dim, hidden_dim)
        self.output_norm = nn.LayerNorm(hidden_dim)
        self.dropout = nn.Dropout(dropout)
        self.temperature = temperature

        logger.info(
            f"MultiGranularityCrossAttention: dim={hidden_dim}, "
            f"heads={num_heads}, levels=3 (token+sentence+document)"
        )

    def _document_level_attention(
        self,
        molecular_embedding: torch.Tensor,
        text_embedding: torch.Tensor,
        R: torch.Tensor,
    ) -> torch.Tensor:
        """Document-level cross-attention with R gating."""
        Q = self.doc_W_q(molecular_embedding)
        K = self.doc_W_k(text_embedding)
        V = self.doc_W_v(text_embedding)

        scale = self.hidden_dim ** 0.5
        attention = torch.sigmoid(
            (Q * K).sum(dim=-1, keepdim=True) / (scale * self.temperature)
        )

        # Gate with R
        gated_attention = attention * R

        attended = gated_attention * V
        output = self.doc_W_o(attended)
        return self.doc_norm(output + molecular_embedding)

    def forward(
        self,
        molecular_embedding: torch.Tensor,
        text_embedding: torch.Tensor,
        reliability_score: torch.Tensor,
        token_embeddings: Optional[torch.Tensor] = None,
        sentence_embeddings: Optional[torch.Tensor] = None,
        token_mask: Optional[torch.Tensor] = None,
        sentence_mask: Optional[torch.Tensor] = None,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        """
        Multi-granularity forward pass.

        Args:
            molecular_embedding: (batch, hidden_dim)
            text_embedding: (batch, hidden_dim) document-level text
            reliability_score: (batch, 1) R score
            token_embeddings: Optional (batch, max_tokens, hidden_dim)
            sentence_embeddings: Optional (batch, max_sents, hidden_dim)
            token_mask: Optional (batch, max_tokens)
            sentence_mask: Optional (batch, max_sents)

        Returns:
            fused: (batch, hidden_dim) multi-granularity fused embedding
            info: dict with per-level fusion weights and intermediate outputs
        """
        batch_size = molecular_embedding.size(0)

        # Level 1: Document-level (always available)
        doc_fused = self._document_level_attention(
            molecular_embedding, text_embedding, reliability_score
        )

        # Level 2: Token-level (if available)
        if token_embeddings is not None:
            token_fused = self.token_attention(
                token_embeddings, molecular_embedding,
                reliability_score, token_mask,
            )
        else:
            token_fused = doc_fused  # Fallback to document level

        # Level 3: Sentence-level (if available)
        if sentence_embeddings is not None:
            sent_fused = self.sentence_attention(
                sentence_embeddings, molecular_embedding,
                reliability_score, sentence_mask,
            )
        else:
            sent_fused = doc_fused  # Fallback to document level

        # Hierarchical fusion
        concat_levels = torch.cat(
            [token_fused, sent_fused, doc_fused, reliability_score], dim=-1
        )
        level_weights = self.fusion_weights(concat_levels)  # (batch, 3)

        # Weighted combination
        fused = (
            level_weights[:, 0:1] * token_fused
            + level_weights[:, 1:2] * sent_fused
            + level_weights[:, 2:3] * doc_fused
        )

        # Final projection
        output = self.output_proj(fused)
        output = self.dropout(output)
        output = self.output_norm(output + molecular_embedding)

        info = {
            "token_weight": level_weights[:, 0],
            "sentence_weight": level_weights[:, 1],
            "document_weight": level_weights[:, 2],
            "token_fused": token_fused,
            "sentence_fused": sent_fused,
            "document_fused": doc_fused,
        }

        return output, info
