"""
Phase 4: Evidence Aggregation and Reliability Scaffolding (Task B).

Implements:
    logit_fused = logit_prior + sum_i R_i * delta(o_i)

Where:
- logit_prior: base-rate log-odds prior
- delta(o_i): scalar evidence delta derived strictly from study finding o_i
- R_i in [0, 1]: reliability score derived strictly from design fields + lineage features
  (no text, no outcome language, no herb/drug identity)

Rules:
- Never train R on legacy labels (enforced via ConsensusLabelsNotPopulatedError).
- With zero studies (or all-missing studies), output equals logit_prior exactly.
- Task B is not testable on current panel (15 of 120 pairs have >= 2 studies).
"""

from __future__ import annotations

import os
from typing import Dict, List, Optional, Tuple, Any, Union
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.linear_model import LogisticRegression


class ConsensusLabelsNotPopulatedError(RuntimeError):
    """Raised when consensus_labels.csv has no validated rows."""
    pass


def load_consensus_labels(
    path: str = "data/benchmark/consensus_labels.csv",
) -> pd.DataFrame:
    """
    Load validated consensus ground-truth labels.
    
    Raises ConsensusLabelsNotPopulatedError if the file has no validated rows
    (all consensus_label values empty/NaN). R training and Task B evaluation
    cannot proceed until human consensus curation is populated.
    """
    if not os.path.exists(path):
        raise FileNotFoundError(f"Consensus labels file not found: {path}")

    df = pd.read_csv(path)
    required = {"herb_id", "drug_id", "consensus_label"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"{path} missing required columns: {missing}")

    # Check for non-empty validated rows
    valid_mask = df["consensus_label"].notna() & (
        df["consensus_label"].astype(str).str.strip() != ""
    )
    n_valid = int(valid_mask.sum())

    if n_valid == 0:
        raise ConsensusLabelsNotPopulatedError(
            f"{path} has 0 validated rows. "
            "Reliability model training and Task B evaluation are BLOCKED "
            "until external consensus labels are curated in data/benchmark/consensus_labels.csv. "
            "Never invent consensus labels or train on legacy labels."
        )

    return df[valid_mask].copy()


class ReliabilityAggregationModel(nn.Module):
    """
    Aggregation Model:
        logit_fused = logit_prior + sum_i R_i * delta(o_i)

    Parameters
    ----------
    logit_prior : float
        Base-rate log-odds prior (default 0.0, i.e. 50/50 probability).
    delta_dim : int
        Dimension of study finding features (default 1 for directional finding).
    design_dim : int
        Dimension of study design and lineage features (default 7).
    """

    def __init__(
        self,
        logit_prior: float = 0.0,
        delta_dim: int = 1,
        design_dim: int = 7,
    ):
        super().__init__()
        self.logit_prior = nn.Parameter(
            torch.tensor(logit_prior, dtype=torch.float32), requires_grad=False
        )

        # delta(o_i): study finding only -> evidence delta
        self.delta_head = nn.Sequential(
            nn.Linear(delta_dim, 8),
            nn.ReLU(),
            nn.Linear(8, 1),
        )

        # R_i: study design + lineage only -> reliability in [0, 1]
        # Small, regularized network
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
        Compute fused interaction logit for pairs with k studies.

        Parameters
        ----------
        study_findings : (batch_size, max_studies, delta_dim)
        design_features : (batch_size, max_studies, design_dim)
        study_mask : (batch_size, max_studies), optional
            1.0 for valid studies, 0.0 for padding / missing studies.

        Returns
        -------
        logits : (batch_size,)
        info : dict with 'R', 'delta', 'logit_prior'
        """
        batch_size, max_studies, _ = study_findings.shape

        if max_studies == 0:
            logits = self.logit_prior.expand(batch_size)
            return logits, {
                "R": torch.empty((batch_size, 0)),
                "delta": torch.empty((batch_size, 0)),
                "logit_prior": self.logit_prior.detach().item(),
            }

        delta = self.delta_head(study_findings).squeeze(-1)  # (batch, max_studies)
        R = self.reliability_head(design_features).squeeze(-1)  # (batch, max_studies)

        if study_mask is not None:
            # Mask out missing studies
            delta = delta * study_mask
            R = R * study_mask

        # Sum R_i * delta_i over studies
        evidence_sum = (R * delta).sum(dim=-1)  # (batch,)
        logits = self.logit_prior + evidence_sum

        info = {
            "R": R.detach(),
            "delta": delta.detach(),
            "logit_prior": self.logit_prior.detach().item(),
        }
        return logits, info

    def forward_no_studies(self, batch_size: int = 1) -> torch.Tensor:
        """With zero studies, output equals logit_prior exactly."""
        return self.logit_prior.expand(batch_size)


class ReliabilityTarget:
    """
    Computes supervision target for reliability scoring:
        agree iff (o = +1 and Y = 1) or (o = -1 and Y = 0)
        o = 0 excluded from supervision.

    Parameters
    ----------
    enforce_validated_labels : bool
        If True, checks that consensus_labels.csv is populated.
    """

    def __init__(
        self,
        consensus_path: Optional[str] = "data/benchmark/consensus_labels.csv",
        enforce_validated_labels: bool = False,
    ):
        self.consensus_path = consensus_path
        if enforce_validated_labels and consensus_path:
            self.validated_df = load_consensus_labels(consensus_path)
        else:
            self.validated_df = None

    def compute_target(
        self,
        study_directions: np.ndarray,
        consensus_labels: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Compute binary agreement target.

        Returns
        -------
        targets : (M,) float array (1.0 for agree, 0.0 for disagree)
        valid_mask : (N,) bool array (True where study_direction != 0)
        """
        o = np.asarray(study_directions, dtype=int)
        Y = np.asarray(consensus_labels, dtype=int)

        assert len(o) == len(Y), f"Length mismatch: {len(o)} vs {len(Y)}"

        # o = 0 studies are excluded from reliability supervision
        valid_mask = (o != 0)
        if not np.any(valid_mask):
            return np.array([], dtype=np.float32), valid_mask

        o_valid = o[valid_mask]
        Y_valid = Y[valid_mask]

        agree = ((o_valid == 1) & (Y_valid == 1)) | ((o_valid == -1) & (Y_valid == 0))
        targets = agree.astype(np.float32)

        return targets, valid_mask


# ─── Task B Baselines ───────────────────────────────────────────────────

def majority_vote_baseline(study_directions: List[List[int]]) -> np.ndarray:
    """
    Unweighted majority vote over study directions:
    +1 if sum > 0, 0 if sum < 0, 0.5 if tie.
    """
    preds = []
    for dirs in study_directions:
        valid = [d for d in dirs if d != 0]
        if not valid:
            preds.append(0.5)
        else:
            s = sum(valid)
            if s > 0:
                preds.append(1.0)
            elif s < 0:
                preds.append(0.0)
            else:
                preds.append(0.5)
    return np.array(preds, dtype=np.float32)


def quality_weighted_vote_baseline(
    study_directions: List[List[int]],
    quality_weights: List[List[float]],
) -> np.ndarray:
    """
    Vote weighted by study quality scores Q_i.
    """
    preds = []
    for dirs, q_weights in zip(study_directions, quality_weights):
        weighted_sum = sum(d * q for d, q in zip(dirs, q_weights) if d != 0)
        if weighted_sum > 0:
            preds.append(1.0)
        elif weighted_sum < 0:
            preds.append(0.0)
        else:
            preds.append(0.5)
    return np.array(preds, dtype=np.float32)


def echo_discounted_vote_baseline(
    study_directions: List[List[int]],
    quality_weights: List[List[float]],
    echo_ratios: List[List[float]],
) -> np.ndarray:
    """
    Vote discounted by citation echo ratio E_i:
    weight = Q_i / (1 + E_i)
    """
    preds = []
    for dirs, q_weights, e_ratios in zip(study_directions, quality_weights, echo_ratios):
        discounted_sum = sum(
            (d * q) / (1.0 + max(0.0, e))
            for d, q, e in zip(dirs, q_weights, e_ratios)
            if d != 0
        )
        if discounted_sum > 0:
            preds.append(1.0)
        elif discounted_sum < 0:
            preds.append(0.0)
        else:
            preds.append(0.5)
    return np.array(preds, dtype=np.float32)


def train_pair_level_lr_aggregate(
    X_pair_features: np.ndarray,
    y: np.ndarray,
    seed: int = 42,
) -> LogisticRegression:
    """Fit pair-level baseline on aggregated features."""
    clf = LogisticRegression(C=1.0, max_iter=1000, random_state=seed)
    clf.fit(X_pair_features, y)
    return clf


def train_reliability_model_entrypoint(
    consensus_path: str = "data/benchmark/consensus_labels.csv",
    **kwargs,
):
    """
    Official training entrypoint for the reliability model R.
    Must call load_consensus_labels, which raises ConsensusLabelsNotPopulatedError
    while consensus_labels.csv has no validated rows.
    """
    # Guard: will block execution while labels are unpopulated
    validated_df = load_consensus_labels(consensus_path)
    raise NotImplementedError(
        f"Consensus labels populated ({len(validated_df)} rows), but Task B training pipeline "
        "is not yet executed."
    )
