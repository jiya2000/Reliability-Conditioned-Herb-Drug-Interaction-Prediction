"""
Phase 4 Aggregation Model.

Implements: logit = logit_prior + sum_i R_i * delta_i

Where:
- logit_prior: base-rate log-odds (estimated from data or fixed)
- R_i: reliability score for study i (from design + lineage features ONLY)
- delta_i: per-study evidence delta (from study finding ONLY)

R is computed from design and lineage features only (no outcome language).
delta is computed from the study finding only.

This module does NOT train R — it provides the aggregation scaffold.
R training requires consensus_labels.csv to have validated rows.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple, Any

import numpy as np
import torch
import torch.nn as nn


class ReliabilityAggregationModel(nn.Module):
    """
    Aggregation model: logit = logit_prior + sum_i R_i * delta_i

    Parameters
    ----------
    logit_prior : float
        Base-rate log-odds. Default 0.0 (50/50 prior).
    delta_dim : int
        Dimension of study-finding features for computing delta.
    design_dim : int
        Dimension of design + lineage features for computing R.
    """

    def __init__(
        self,
        logit_prior: float = 0.0,
        delta_dim: int = 1,
        design_dim: int = 5,
    ):
        super().__init__()
        self.logit_prior = nn.Parameter(
            torch.tensor(logit_prior, dtype=torch.float32), requires_grad=False
        )

        # delta_i projection: study finding features -> scalar delta
        self.delta_head = nn.Sequential(
            nn.Linear(delta_dim, 16),
            nn.ReLU(),
            nn.Linear(16, 1),
        )

        # R_i projection: design + lineage features -> R in [0, 1]
        self.reliability_head = nn.Sequential(
            nn.Linear(design_dim, 16),
            nn.ReLU(),
            nn.Linear(16, 1),
            nn.Sigmoid(),
        )

    def forward(
        self,
        study_findings: torch.Tensor,
        design_features: torch.Tensor,
        study_mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """
        Aggregate evidence across studies for each pair.

        Parameters
        ----------
        study_findings : (batch, max_studies, delta_dim)
            Per-study finding features.
        design_features : (batch, max_studies, design_dim)
            Per-study design and lineage features.
        study_mask : (batch, max_studies), optional
            Binary mask; 1 = valid study, 0 = padding.

        Returns
        -------
        logits : (batch,)
            Aggregated interaction logit.
        info : dict
            Contains 'R' (batch, max_studies), 'delta' (batch, max_studies),
            'logit_prior' scalar.
        """
        batch_size, max_studies, _ = study_findings.shape

        if max_studies == 0:
            logits = self.logit_prior.expand(batch_size)
            return logits, {
                "R": torch.empty((batch_size, 0)),
                "delta": torch.empty((batch_size, 0)),
                "logit_prior": self.logit_prior.detach().item(),
            }

        # Compute per-study delta and R
        delta = self.delta_head(study_findings).squeeze(-1)  # (batch, max_studies)
        R = self.reliability_head(design_features).squeeze(-1)  # (batch, max_studies)

        # Mask padding studies
        if study_mask is not None:
            delta = delta * study_mask
            R = R * study_mask

        # Aggregation: logit = logit_prior + sum_i R_i * delta_i
        weighted_evidence = (R * delta).sum(dim=-1)  # (batch,)
        logits = self.logit_prior + weighted_evidence

        info = {
            "R": R.detach(),
            "delta": delta.detach(),
            "logit_prior": self.logit_prior.detach().item(),
        }

        return logits, info

    def forward_no_studies(self, batch_size: int = 1) -> torch.Tensor:
        """
        With zero studies, output equals the prior exactly.
        This is a critical property tested in CI.
        """
        return self.logit_prior.expand(batch_size)


class ReliabilityTarget:
    """
    Compute the reliability target for training R.

    agree iff (o=+1 and Y=1) or (o=-1 and Y=0); o=0 excluded.

    Where:
    - o: study finding direction (+1, -1, or 0)
    - Y: consensus ground-truth label (1 = interaction, 0 = no interaction)

    Raises RuntimeError if consensus_labels.csv has no validated rows.
    """

    def __init__(
        self,
        consensus_path: Optional[str] = "data/benchmark/consensus_labels.csv",
        enforce_validated_labels: bool = True,
    ):
        self.consensus_path = consensus_path
        self.enforce_validated_labels = enforce_validated_labels
        if enforce_validated_labels and consensus_path is not None:
            # Must raise clear error if consensus_labels.csv has no validated rows
            self.validated_df = load_consensus_labels(consensus_path)
        else:
            self.validated_df = None

    def compute_target(
        self,
        study_directions: np.ndarray,
        consensus_labels: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Compute reliability targets.

        Parameters
        ----------
        study_directions : (N,) int array
            Per-study direction: +1, -1, or 0.
        consensus_labels : (N,) int array
            Ground-truth consensus labels: 1 or 0.

        Returns
        -------
        targets : (M,) float array
            1.0 if study agrees with truth, 0.0 if disagrees.
        valid_mask : (N,) bool array
            True for studies where o != 0 (included in target).
        """
        study_directions = np.asarray(study_directions, dtype=int)
        consensus_labels = np.asarray(consensus_labels, dtype=int)

        assert len(study_directions) == len(consensus_labels), (
            f"Direction/label length mismatch: {len(study_directions)} vs {len(consensus_labels)}"
        )

        # Exclude o=0 studies
        valid_mask = study_directions != 0

        if not np.any(valid_mask):
            return np.array([], dtype=np.float32), valid_mask

        o = study_directions[valid_mask]
        Y = consensus_labels[valid_mask]

        # agree iff (o=+1 and Y=1) or (o=-1 and Y=0)
        agree = ((o == 1) & (Y == 1)) | ((o == -1) & (Y == 0))
        targets = agree.astype(np.float32)

        return targets, valid_mask


def load_consensus_labels(
    path: str = "data/benchmark/consensus_labels.csv",
) -> "pd.DataFrame":
    """Use the shared consensus contract and block unavailable labels."""
    from src.data.evidence_schema import load_consensus_labels_table

    return load_consensus_labels_table(path, require_labels=True)[0]
