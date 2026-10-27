"""
Reliability-Conditioned Interaction Model with Calibrated Decision and Abstention.

Combines:
1. Biological knowledge graph representations (heterogeneous GNN embeddings)
2. Evidence text representations (BiomedBERT) & lineage feature vectors
3. Reliability scoring (study independence, context compatibility, conflict resolution)
4. Calibrated tri-state prediction: Positive, Negative, or Abstain.

Supports all ablation modes required for experimental verification:
- 'graph_only': Baseline using only graph topology
- 'graph_plus_naive_evidence': Baseline adding naive citation features without lineage
- 'standard_fusion': Cross-attention fusion with unconditioned gating (R = 1.0)
- 'reliability_conditioned': Full proposed architecture
- 'no_study_dependence': Ablation removing study-dependence discounting
- 'no_context_conflict': Ablation removing context and conflict features
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from loguru import logger

from src.models.reliability_scorer import ReliabilityScorer
from src.models.link_predictor import LinkPredictor


@dataclass
class CalibratedDecision:
    """Structured decision output including confidence, abstention, and evidence trail."""
    predicted_label: int  # 1 (interaction), 0 (no interaction), -1 (abstain)
    probability: float
    confidence: float
    epistemic_uncertainty: float
    abstain: bool
    abstention_reason: str
    reliability_score: float
    evidence_trail: Dict[str, Any]


class ReliabilityConditionedInteractionModel(nn.Module):
    """
    Public-data-first, Evidence-lineage-aware Herb-Drug Interaction Model.
    """

    def __init__(
        self,
        graph_dim: int = 128,
        evidence_dim: int = 128,
        fused_dim: int = 128,
        hidden_dim: int = 128,
        mode: str = "reliability_conditioned",
        abstain_uncertainty_threshold: float = 0.35,
        abstain_conflict_threshold: float = 0.50,
        abstain_margin_threshold: float = 0.20,
        dropout: float = 0.1,
        feature_group: Optional[str] = None,
        enable_auxiliary_heads: bool = False,
        num_severity_classes: int = 3,
    ):
        super().__init__()
        valid_modes = {
            "graph_only",
            "evidence_only",
            "graph_plus_naive_evidence",
            "standard_fusion",
            "naive_concat_mlp",
            "reliability_conditioned",
            "reliability_conditioned_quality_only",
            "no_study_dependence",
            "no_context_conflict",
        }
        if mode not in valid_modes:
            raise ValueError(f"Invalid mode '{mode}'. Choose from {sorted(valid_modes)}")

        self.mode = mode
        self.graph_dim = graph_dim
        self.evidence_dim = evidence_dim
        self.fused_dim = fused_dim
        self.abstain_uncertainty_threshold = abstain_uncertainty_threshold
        self.abstain_conflict_threshold = abstain_conflict_threshold
        self.abstain_margin_threshold = abstain_margin_threshold
        self.enable_auxiliary_heads = enable_auxiliary_heads
        self.num_severity_classes = num_severity_classes

        # Default feature group: quality_only is headline default following leakage probe confirmation
        if feature_group is not None:
            self.feature_group = feature_group
        elif mode == "reliability_conditioned_quality_only":
            self.feature_group = "quality_only"
        elif mode == "reliability_conditioned":
            self.feature_group = "quality_only"
        else:
            self.feature_group = "full"

        # Graph pair representation projection: [h_u; h_v; |h_u - h_v|; h_u * h_v]
        pair_dim = graph_dim * 4
        self.graph_pair_proj = nn.Sequential(
            nn.Linear(pair_dim, fused_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )

        # Evidence representation projection (from 8D lineage features or text dim)
        self.evidence_proj = nn.Sequential(
            nn.Linear(evidence_dim, fused_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )

        # Cross-attention / fusion layer
        self.cross_attn = nn.MultiheadAttention(
            embed_dim=fused_dim, num_heads=4, dropout=dropout, batch_first=True
        )

        # Naive concatenation projection: [z_graph; z_ev] -> fused_dim
        self.concat_fusion_proj = nn.Sequential(
            nn.Linear(2 * fused_dim, fused_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )

        # Reliability Scorer
        self.reliability_scorer = ReliabilityScorer(
            hidden_dim=hidden_dim,
            dropout=dropout,
            mc_dropout_rate=0.15,
            mc_samples=10,
            feature_group=self.feature_group,
        )

        # Link Predictor MLP head mapping fused representation to interaction logits
        self.link_predictor = nn.Sequential(
            nn.Linear(fused_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
        )

        # Additive auxiliary task heads (Phase 8)
        if enable_auxiliary_heads:
            # 1. Direction classification: decreased_exposure (0) vs increased_exposure (1)
            self.direction_head = nn.Sequential(
                nn.Linear(fused_dim, max(16, hidden_dim // 2)),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(max(16, hidden_dim // 2), 1),
            )
            # 2. Clinical severity grade: e.g. 0: mild, 1: moderate, 2: severe
            self.severity_head = nn.Sequential(
                nn.Linear(fused_dim, max(16, hidden_dim // 2)),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(max(16, hidden_dim // 2), num_severity_classes),
            )
            # 3. Pharmacokinetic log fold-change regression: ln(AUC_ratio) or ln(Cmax_ratio)
            self.pk_regression_head = nn.Sequential(
                nn.Linear(fused_dim, max(16, hidden_dim // 2)),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(max(16, hidden_dim // 2), 1),
            )
        else:
            self.direction_head = None
            self.severity_head = None
            self.pk_regression_head = None

        logger.info(
            f"Initialized InteractionModel in '{self.mode}' mode (fused_dim={fused_dim}, "
            f"auxiliary_heads={enable_auxiliary_heads})"
        )

    def forward(
        self,
        herb_emb: torch.Tensor,
        drug_emb: torch.Tensor,
        evidence_features: Optional[torch.Tensor] = None,
        metadata: Optional[torch.Tensor] = None,
        return_details: bool = False,
    ) -> Dict[str, Any]:
        """
        Forward pass computing interaction probability.

        Args:
            herb_emb: (batch, graph_dim) Herb representation from biological graph
            drug_emb: (batch, graph_dim) Drug representation from biological graph
            evidence_features: (batch, evidence_dim) Embedded evidence features/text
            metadata: (batch, 8) Lineage metadata or (batch, 5) legacy metadata
            return_details: If True, returns gating and breakdown dictionary

        Returns:
            Dict containing 'logits', 'probs', 'reliability', and optional details.
        """
        batch_size = herb_emb.size(0)

        # 1. Biological Graph Pair representation
        diff = torch.abs(herb_emb - drug_emb)
        prod = herb_emb * drug_emb
        pair_cat = torch.cat([herb_emb, drug_emb, diff, prod], dim=-1)
        z_graph = self.graph_pair_proj(pair_cat)  # (batch, fused_dim)

        # Default fallback / placeholder values
        R = torch.ones(batch_size, 1, device=herb_emb.device)
        uncertainty = torch.zeros(batch_size, 1, device=herb_emb.device)
        breakdown = None

        # 2. Evidence Contextualization & Fusion
        # Supports two explicitly documented pathways:
        # A) Token-/Report-Level Multihead Cross-Attention when evidence_features is 3D: (batch, seq_len, dim)
        # B) Reliability-Gated Representation Fusion when evidence_features is 2D: (batch, dim)
        attn_weights = None

        if self.mode == "graph_only" or evidence_features is None:
            # Baseline (a): Only graph topology
            z_fused = z_graph
            R = torch.zeros(batch_size, 1, device=herb_emb.device)

        elif self.mode == "evidence_only":
            # Baseline: Only literature evidence text / features (zero graph priors)
            if evidence_features is not None:
                if evidence_features.dim() == 3:
                    z_fused = self.evidence_proj(evidence_features).mean(dim=1)
                else:
                    z_fused = self.evidence_proj(evidence_features)
            else:
                z_fused = torch.zeros(batch_size, self.fused_dim, device=herb_emb.device)
            R = torch.ones(batch_size, 1, device=herb_emb.device)

        elif self.mode == "graph_plus_naive_evidence":
            # Baseline (b): Unweighted additive combination without lineage-based reliability gating
            if evidence_features.dim() == 3:
                z_ev = self.evidence_proj(evidence_features).mean(dim=1)
            else:
                z_ev = self.evidence_proj(evidence_features)
            z_fused = 0.5 * z_graph + 0.5 * z_ev
            R = torch.full((batch_size, 1), 0.5, device=herb_emb.device)

        elif self.mode == "standard_fusion":
            # Baseline (c): Cross-attention with fixed unconditioned gating R = 1.0
            if evidence_features.dim() == 3:
                kv = self.evidence_proj(evidence_features)
                q = z_graph.unsqueeze(1)
                attn_out, attn_weights = self.cross_attn(q, kv, kv)
                z_ev = attn_out.squeeze(1)
            else:
                z_ev = self.evidence_proj(evidence_features)
            z_fused = 0.5 * z_graph + 0.5 * z_ev
            R = torch.ones(batch_size, 1, device=herb_emb.device)

        elif self.mode == "naive_concat_mlp":
            # Baseline: Concatenation MLP fusion without cross-attention or reliability gating
            if evidence_features.dim() == 3:
                z_ev = self.evidence_proj(evidence_features).mean(dim=1)
            else:
                z_ev = self.evidence_proj(evidence_features)
            z_fused = self.concat_fusion_proj(torch.cat([z_graph, z_ev], dim=-1))
            R = torch.ones(batch_size, 1, device=herb_emb.device)

        else:
            # Proposed Reliability-Conditioned Architecture (or targeted ablations)
            if evidence_features.dim() == 3:
                # Genuine multi-token evidence cross-attention across sequence of reports/tokens
                kv = self.evidence_proj(evidence_features)  # (batch, seq_len, fused_dim)
                q = z_graph.unsqueeze(1)                    # (batch, 1, fused_dim)
                attn_out, attn_weights = self.cross_attn(q, kv, kv)
                z_contextual = attn_out.squeeze(1)          # (batch, fused_dim)
            else:
                # Reliability-gated representation fusion on pre-aggregated evidence vector
                z_contextual = self.evidence_proj(evidence_features)

            if metadata is not None:
                meta_input = metadata.clone()
                if self.mode == "no_study_dependence":
                    # Zero out echo ratio (index 2 in 12D/8D, index 0 in 7D)
                    if meta_input.size(-1) >= 8:
                        meta_input[:, 2:3] = 0.0
                        if meta_input.size(-1) >= 10:
                            raw_pos = meta_input[:, 8:9]
                            raw_neg = meta_input[:, 9:10]
                            meta_input[:, 0:1] = raw_pos
                            meta_input[:, 1:2] = raw_neg
                            meta_input[:, 6:7] = raw_pos - raw_neg
                    elif meta_input.size(-1) == 7:
                        meta_input[:, 0:1] = 0.0  # echo_ratio
                elif self.mode == "no_context_conflict":
                    # Neutralize context to 1.0, zero contradiction
                    if meta_input.size(-1) >= 8:
                        meta_input[:, 4:5] = 1.0
                        meta_input[:, 5:6] = 0.0
                        meta_input[:, 6:7] = meta_input[:, 0:1]
                    elif meta_input.size(-1) == 7:
                        meta_input[:, 2:3] = 1.0  # context_compat
                        meta_input[:, 3:4] = 0.0  # contradiction

                # Compute reliability and epistemic uncertainty
                R, uncertainty, breakdown = self.reliability_scorer.forward_with_uncertainty(meta_input)
            else:
                R = torch.full((batch_size, 1), 0.5, device=herb_emb.device)
                uncertainty = torch.full((batch_size, 1), 0.2, device=herb_emb.device)

            # Reliability-conditioned gating:
            # When evidence reliability R is high, contextual literature modulates the prediction.
            # When R is low or absent, the model retreats safely to biological graph priors.
            z_fused = (1.0 - R) * z_graph + R * z_contextual

        # 3. Predict interaction probability
        logits = self.link_predictor(z_fused)  # (batch, 1)
        probs = torch.sigmoid(logits)          # (batch, 1)


        result: Dict[str, Any] = {
            "logits": logits,
            "probs": probs,
            "reliability": R,
            "uncertainty": uncertainty,
        }

        if self.enable_auxiliary_heads:
            result["direction_logits"] = self.direction_head(z_fused)
            result["severity_logits"] = self.severity_head(z_fused)
            result["pk_fold_change"] = self.pk_regression_head(z_fused)

        if return_details:
            result["z_fused"] = z_fused
            result["z_graph"] = z_graph
            result["breakdown"] = breakdown
            result["attn_weights"] = attn_weights

        return result

    def compute_multitask_loss(
        self,
        predictions: Dict[str, torch.Tensor],
        target_interaction: torch.Tensor,
        target_direction: Optional[torch.Tensor] = None,
        target_severity: Optional[torch.Tensor] = None,
        target_pk_fold_change: Optional[torch.Tensor] = None,
        weights: Optional[Dict[str, float]] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        Computes multi-task loss with loss masking on unannotated/missing records.

        Args:
            predictions: Model output dict from forward pass.
            target_interaction: (batch, 1) or (batch,) binary interaction label in {0, 1}.
            target_direction: (batch, 1) or (batch,) binary direction (0: decreased, 1: increased, -1: unannotated).
            target_severity: (batch, 1) or (batch,) severity grade (0: mild, 1: moderate, 2: severe, -1: unannotated).
            target_pk_fold_change: (batch, 1) or (batch,) float log fold-change, with NaN for missing values.
            weights: Dict specifying multi-task loss weights (default: interaction=1.0, direction=0.5, severity=0.5, pk=0.25).

        Returns:
            Dict containing loss_total, loss_interaction, loss_direction, loss_severity, and loss_pk.
        """
        import torch.nn.functional as F

        w = {
            "interaction": 1.0,
            "direction": 0.5,
            "severity": 0.5,
            "pk_regression": 0.25,
        }
        if weights is not None:
            w.update(weights)

        device = predictions["logits"].device
        target_int = target_interaction.view(-1, 1).to(device).float()
        loss_int = F.binary_cross_entropy_with_logits(predictions["logits"], target_int)

        loss_dir = torch.tensor(0.0, device=device)
        if self.enable_auxiliary_heads and "direction_logits" in predictions and target_direction is not None:
            t_dir = target_direction.view(-1, 1).to(device).float()
            mask_dir = (t_dir >= 0.0) & (t_dir <= 1.0)
            if mask_dir.sum() > 0:
                loss_dir = F.binary_cross_entropy_with_logits(
                    predictions["direction_logits"][mask_dir],
                    t_dir[mask_dir],
                )

        loss_sev = torch.tensor(0.0, device=device)
        if self.enable_auxiliary_heads and "severity_logits" in predictions and target_severity is not None:
            t_sev = target_severity.view(-1).to(device).long()
            mask_sev = (t_sev >= 0) & (t_sev < self.num_severity_classes)
            if mask_sev.sum() > 0:
                loss_sev = F.cross_entropy(
                    predictions["severity_logits"][mask_sev],
                    t_sev[mask_sev],
                )

        loss_pk = torch.tensor(0.0, device=device)
        if self.enable_auxiliary_heads and "pk_fold_change" in predictions and target_pk_fold_change is not None:
            t_pk = target_pk_fold_change.view(-1, 1).to(device).float()
            mask_pk = ~torch.isnan(t_pk)
            if mask_pk.sum() > 0:
                loss_pk = F.smooth_l1_loss(
                    predictions["pk_fold_change"][mask_pk],
                    t_pk[mask_pk],
                )

        loss_total = (
            w["interaction"] * loss_int
            + w["direction"] * loss_dir
            + w["severity"] * loss_sev
            + w["pk_regression"] * loss_pk
        )

        return {
            "loss_total": loss_total,
            "loss_interaction": loss_int,
            "loss_direction": loss_dir,
            "loss_severity": loss_sev,
            "loss_pk": loss_pk,
        }


    def predict_with_decision(
        self,
        herb_emb: torch.Tensor,
        drug_emb: torch.Tensor,
        evidence_features: Optional[torch.Tensor] = None,
        metadata: Optional[torch.Tensor] = None,
        pair_summaries: Optional[List[Any]] = None,
    ) -> List[CalibratedDecision]:
        """
        Evaluate candidate pairs and return calibrated tri-state decisions:
        Positive (1), Negative (0), or Abstain (-1), along with evidence lineage audit trail.
        """
        self.eval()
        with torch.no_grad():
            output = self.forward(
                herb_emb=herb_emb,
                drug_emb=drug_emb,
                evidence_features=evidence_features,
                metadata=metadata,
                return_details=True,
            )

        probs = output["probs"].cpu().numpy().flatten()
        reliabilities = output["reliability"].cpu().numpy().flatten()
        uncertainties = output["uncertainty"].cpu().numpy().flatten()
        breakdown = output.get("breakdown")

        decisions: List[CalibratedDecision] = []
        batch_size = len(probs)

        for i in range(batch_size):
            p = float(probs[i])
            r = float(reliabilities[i])
            u = float(uncertainties[i])
            confidence = abs(p - 0.5) * 2.0  # 0.0 (maximum uncertainty) to 1.0 (certain)

            # Check for conflict
            conflict = 0.0
            if metadata is not None and metadata.size(-1) == 8:
                conflict = float(metadata[i, 5].item())

            # Abstention policy:
            # Abstain if:
            # 1. Epistemic uncertainty exceeds safety threshold
            # 2. Conflicting literature with equivocal net evidence
            # 3. Prediction is near borderline (confidence < 0.2) with low reliability (R < 0.25)
            abstain = False
            reason = "confident_prediction"

            if u > self.abstain_uncertainty_threshold:
                abstain = True
                reason = "high_epistemic_uncertainty"
            elif conflict > self.abstain_conflict_threshold and confidence < (2.0 * self.abstain_margin_threshold):
                abstain = True
                reason = "active_scientific_conflict"
            elif confidence < self.abstain_margin_threshold and r < 0.25:
                abstain = True
                reason = "insufficient_out_of_distribution_evidence"

            if abstain:
                label = -1
            else:
                label = 1 if p >= 0.5 else 0

            summary_info: Dict[str, Any] = {}
            if pair_summaries and i < len(pair_summaries) and pair_summaries[i]:
                ps = pair_summaries[i]
                summary_info = {
                    "total_records": getattr(ps, "total_records", 0),
                    "pos_clusters": getattr(ps, "num_independent_positive_clusters", 0),
                    "neg_clusters": getattr(ps, "num_independent_negative_clusters", 0),
                    "echo_ratio": getattr(ps, "citation_echo_ratio", 0.0),
                    "audit_sources": getattr(ps, "audit_sources", []),
                }

            decisions.append(
                CalibratedDecision(
                    predicted_label=label,
                    probability=p,
                    confidence=confidence,
                    epistemic_uncertainty=u,
                    abstain=abstain,
                    abstention_reason=reason,
                    reliability_score=r,
                    evidence_trail=summary_info,
                )
            )

        return decisions
