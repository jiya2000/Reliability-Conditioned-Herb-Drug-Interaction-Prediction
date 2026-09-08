"""
Contrastive Learning for Reliability-Aware Embeddings

★ NOVEL CONTRIBUTION 5 ★

InfoNCE-style contrastive loss that structures the embedding space
to be reliability-aware:
- High-R evidence pairs are pulled together in embedding space
- Low-R evidence pairs are pushed apart
- Cross-reliability pairs create a smooth gradient

This goes beyond gating attention weights — it makes the learned
representations themselves encode reliability information, enabling:
1. Better transfer learning across interaction domains
2. More robust predictions on unseen entity pairs
3. A new patent claim: "reliability-conditioned embedding spaces"

Loss formulation:
    L_contrastive = -log(sim(z_i, z_j) / Σ sim(z_i, z_k))

    where similarity is modulated by reliability:
    sim(z_i, z_j) = exp(cos(z_i, z_j) / τ) * w(R_i, R_j)
    w(R_i, R_j) = exp(-|R_i - R_j|)
"""

from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from loguru import logger


class ReliabilityAwareContrastiveLoss(nn.Module):
    """
    InfoNCE-based contrastive loss conditioned on reliability scores.

    Encourages the model to learn an embedding space where:
    - Samples with similar R-scores and similar interactions cluster
    - Samples with different R-scores are proportionally separated
    - The reliability signal is baked into the embedding geometry,
      not just the attention mechanism

    Usage:
        loss_fn = ReliabilityAwareContrastiveLoss(temperature=0.07)
        loss = loss_fn(embeddings, reliability_scores, labels)
    """

    def __init__(
        self,
        temperature: float = 0.07,
        reliability_weight: float = 0.5,
        hard_negative_weight: float = 0.3,
        projection_dim: int = 64,
        hidden_dim: int = 128,
    ):
        """
        Args:
            temperature: InfoNCE temperature parameter
            reliability_weight: How much reliability similarity affects
                                the contrastive loss (0 = ignore R)
            hard_negative_weight: Weight for hard negative mining
            projection_dim: Dimension of the contrastive projection head
            hidden_dim: Input embedding dimension
        """
        super().__init__()
        self.temperature = temperature
        self.reliability_weight = reliability_weight
        self.hard_negative_weight = hard_negative_weight

        # Projection head (projects fused embeddings to contrastive space)
        self.projector = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, projection_dim),
        )

        # Reliability-aware similarity kernel
        self.reliability_kernel = nn.Sequential(
            nn.Linear(2, 16),
            nn.GELU(),
            nn.Linear(16, 1),
            nn.Sigmoid(),
        )

        logger.info(
            f"ReliabilityAwareContrastiveLoss: τ={temperature}, "
            f"R_weight={reliability_weight}, proj_dim={projection_dim}"
        )

    def forward(
        self,
        embeddings: torch.Tensor,
        reliability_scores: torch.Tensor,
        labels: torch.Tensor,
        return_breakdown: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, dict]:
        """
        Compute the reliability-aware contrastive loss.

        Args:
            embeddings: (batch, hidden_dim) fused embeddings
            reliability_scores: (batch, 1) R scores
            labels: (batch,) binary interaction labels

        Returns:
            loss: scalar contrastive loss
            breakdown: (optional) dict with loss components
        """
        batch_size = embeddings.size(0)

        if batch_size < 4:
            # Need minimum batch size for meaningful contrastive learning
            zero_loss = torch.tensor(0.0, device=embeddings.device, requires_grad=True)
            if return_breakdown:
                return zero_loss, {"info_nce": 0.0, "reliability_penalty": 0.0}
            return zero_loss

        # Project to contrastive space
        z = self.projector(embeddings)  # (batch, proj_dim)
        z = F.normalize(z, dim=-1)  # L2 normalize

        R = reliability_scores.squeeze()  # (batch,)

        # Compute pairwise cosine similarity
        sim_matrix = torch.mm(z, z.t()) / self.temperature  # (batch, batch)

        # Compute reliability-aware weights
        R_weights = self._compute_reliability_weights(R)  # (batch, batch)

        # Create positive/negative masks
        pos_mask = self._create_positive_mask(labels)
        neg_mask = ~pos_mask & ~torch.eye(
            batch_size, dtype=torch.bool, device=embeddings.device
        )

        # Modulate similarity with reliability weights
        weighted_sim = sim_matrix + torch.log(R_weights + 1e-8)

        # Vectorized InfoNCE loss
        neg_sims_matrix = weighted_sim.clone()
        neg_sims_matrix[~neg_mask] = -1e9  # Mask out positives

        # Base negative logsumexp for the denominator
        neg_lse = torch.logsumexp(neg_sims_matrix, dim=1)  # (batch,)

        if self.hard_negative_weight > 0:
            # Estimate k based on mean negative count
            k = max(1, int(neg_mask.float().sum(dim=1).mean().item()) // 4)
            if k > 0 and k < neg_sims_matrix.size(1):
                hard_negs, _ = torch.topk(neg_sims_matrix, k, dim=1)
                hard_negs = hard_negs * self.hard_negative_weight
                hard_neg_lse = torch.logsumexp(hard_negs, dim=1)
                neg_lse = torch.logaddexp(neg_lse, hard_neg_lse)

        # Total log denominator: log(exp(pos) + sum(exp(negs)))
        neg_lse_exp = neg_lse.unsqueeze(1).expand_as(weighted_sim)
        log_denom = torch.logaddexp(weighted_sim, neg_lse_exp)

        # InfoNCE = -log(exp(pos) / denom) = -pos + log(denom)
        pair_losses = -(weighted_sim - log_denom)
        
        # We only want the loss for valid positive pairs
        valid_pos_count = max(1, pos_mask.sum().item())
        loss = (pair_losses * pos_mask).sum() / valid_pos_count
        n_valid = valid_pos_count

        # Reliability consistency penalty
        R_penalty = self._reliability_consistency_penalty(z, R)
        total_loss = loss + self.reliability_weight * R_penalty

        if return_breakdown:
            breakdown = {
                "info_nce": loss.item(),
                "reliability_penalty": R_penalty.item(),
                "total_contrastive": total_loss.item(),
                "batch_size": batch_size,
                "valid_anchors": n_valid,
            }
            return total_loss, breakdown

        return total_loss

    def _compute_reliability_weights(
        self, R: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute pairwise reliability similarity weights.

        Pairs with similar R values get higher weight (are considered
        more meaningful comparisons).

        Args:
            R: (batch,) reliability scores

        Returns:
            weights: (batch, batch) pairwise weights
        """
        # Pairwise R features
        R_i = R.unsqueeze(1).expand(-1, len(R))  # (batch, batch)
        R_j = R.unsqueeze(0).expand(len(R), -1)  # (batch, batch)

        # Stack for kernel input
        R_pairs = torch.stack([R_i, R_j], dim=-1)  # (batch, batch, 2)
        flat_pairs = R_pairs.view(-1, 2)

        weights = self.reliability_kernel(flat_pairs)
        weights = weights.view(len(R), len(R))

        return weights

    def _create_positive_mask(
        self, labels: torch.Tensor
    ) -> torch.Tensor:
        """
        Create positive pair mask: samples with same label are positives.

        Args:
            labels: (batch,) binary labels

        Returns:
            mask: (batch, batch) boolean mask
        """
        labels = labels.float()
        # Same label = positive pair
        mask = (labels.unsqueeze(0) == labels.unsqueeze(1))
        # Remove self-pairs
        mask.fill_diagonal_(False)
        return mask

    def _reliability_consistency_penalty(
        self,
        embeddings: torch.Tensor,
        R: torch.Tensor,
    ) -> torch.Tensor:
        """
        Penalty to ensure embedding similarity correlates with R similarity.

        If two samples have similar R, their embeddings should be
        closer than samples with dissimilar R (all else being equal).
        """
        # Pairwise embedding distances
        emb_dists = torch.cdist(embeddings, embeddings, p=2)  # (batch, batch)

        # Pairwise R distances
        R_dists = (R.unsqueeze(0) - R.unsqueeze(1)).abs()  # (batch, batch)

        # We want: small R_dist → small emb_dist (positive correlation)
        # Penalty: cases where R is similar but embeddings are distant
        penalty = (
            (1.0 - R_dists) * emb_dists
        ).mean()

        return penalty


class ReliabilityContrastiveRegularizer(nn.Module):
    """
    A simpler, more efficient regularizer that can be added to any
    training loop without the full contrastive framework.

    Ensures that the embedding space preserves reliability ordering:
    embeddings from high-R evidence should be closer to the ground-truth
    interaction embedding than those from low-R evidence.

    Usage:
        reg = ReliabilityContrastiveRegularizer()
        reg_loss = reg(fused_embeddings, R_scores, labels)
        total_loss = main_loss + 0.1 * reg_loss
    """

    def __init__(self, margin: float = 0.2):
        super().__init__()
        self.margin = margin

    def forward(
        self,
        embeddings: torch.Tensor,
        reliability_scores: torch.Tensor,
        labels: torch.Tensor,
    ) -> torch.Tensor:
        """
        Triplet-style regularization: for each positive sample,
        high-R embeddings should be closer than low-R embeddings.

        Args:
            embeddings: (batch, dim) fused embeddings
            reliability_scores: (batch, 1) R scores
            labels: (batch,) binary labels

        Returns:
            regularization loss
        """
        R = reliability_scores.squeeze()
        pos_mask = labels == 1

        if pos_mask.sum() < 2:
            return torch.tensor(0.0, device=embeddings.device)

        pos_embs = embeddings[pos_mask]
        pos_R = R[pos_mask]

        if len(pos_R) < 2:
            return torch.tensor(0.0, device=embeddings.device)

        # Sort by reliability
        sorted_idx = pos_R.argsort(descending=True)
        sorted_embs = pos_embs[sorted_idx]

        # Adjacent pairs should maintain ordering
        loss = torch.tensor(0.0, device=embeddings.device)
        centroid = pos_embs.mean(dim=0, keepdim=True)

        for i in range(len(sorted_embs) - 1):
            d_high = F.pairwise_distance(
                sorted_embs[i:i+1], centroid
            )
            d_low = F.pairwise_distance(
                sorted_embs[i+1:i+2], centroid
            )
            # High-R should be closer to centroid than low-R
            loss = loss + F.relu(d_high - d_low + self.margin)

        return loss / max(len(sorted_embs) - 1, 1)
