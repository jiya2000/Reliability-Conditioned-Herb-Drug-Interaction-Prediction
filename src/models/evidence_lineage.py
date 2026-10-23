"""
Evidence Lineage Metadata Vector (10-Dimensional) — Novelty Component.

Implements the exact 10-dimensional evidence lineage metadata vector from
the patent specification (Eq. 5), computing all features from the
citation-derivation DAG and pharmacokinetic context scores.

10D Vector Components (Eq. 5 in patent):
[S_pos, S_neg, N_primary, N_echo, kappa_context, C_contradiction,
 q_bar, Delta_time, N_raw_pos, N_raw_neg]

Where:
- S_pos, S_neg: Echo-attenuated cluster strengths for positive/negative claims
- N_primary: Number of independent primary evidence roots
- N_echo: Number of derivative/echo publications
- kappa_context: Mean pharmacokinetic context compatibility score
- C_contradiction: Scientific contradiction index
- q_bar: Mean extraction quality across all evidence
- Delta_time: Temporal recency (normalized years since most recent evidence)
- N_raw_pos, N_raw_neg: Raw publication counts (for transparency/debugging)
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple
import math
import torch
import torch.nn as nn
from loguru import logger

from src.models.citation_dag import CitationDAG, CitationDAGResult, PaperNode
from src.models.context_compatibility import (
    ContextCompatibilityScorer,
    ContextMetadataParser,
    PharmacokineticContext,
)


@dataclass
class EvidenceLineageFeatures:
    """Structured container for the 10D evidence lineage features."""
    S_pos: float = 0.0
    S_neg: float = 0.0
    N_primary: int = 0
    N_echo: int = 0
    kappa_context: float = 0.0
    C_contradiction: float = 0.0
    q_bar: float = 0.0
    Delta_time: float = 0.0
    N_raw_pos: int = 0
    N_raw_neg: int = 0

    def to_tensor(self, device: Optional[torch.device] = None) -> torch.Tensor:
        """Convert to 10D tensor for neural network input."""
        return torch.tensor(
            [
                self.S_pos,
                self.S_neg,
                float(self.N_primary),
                float(self.N_echo),
                self.kappa_context,
                self.C_contradiction,
                self.q_bar,
                self.Delta_time,
                float(self.N_raw_pos),
                float(self.N_raw_neg),
            ],
            dtype=torch.float32,
            device=device,
        ).unsqueeze(0)  # (1, 10)

    @classmethod
    def zeros(cls, device: Optional[torch.device] = None) -> "EvidenceLineageFeatures":
        """Return zero 10D EvidenceLineageFeatures for missing evidence."""
        return cls()

    @staticmethod
    def _compute_contradiction_index(S_pos: float, S_neg: float) -> float:
        """
        Compute scientific contradiction index C_h,d (Eq. 4 in patent).
        C = 2 * min(S_pos, S_neg) / (S_pos + S_neg + epsilon)
        """
        epsilon = 1e-6
        if S_pos + S_neg < epsilon:
            return 0.0
        return (2.0 * min(S_pos, S_neg)) / (S_pos + S_neg + epsilon)


class EvidenceLineageExtractor:
    """
    Extracts the 10-dimensional evidence lineage feature vector from
    raw literature records and a citation DAG.
    
    This is the core novelty component that bridges bibliometric evidence
    analysis with pharmacokinetic context compatibility.
    """

    def __init__(
        self,
        alpha: float = 0.15,
        temporal_half_life_years: float = 5.0,
        context_scorer: Optional[ContextCompatibilityScorer] = None,
    ):
        """
        Args:
            alpha: Echo discount factor for citation DAG (default 0.15)
            temporal_half_life_years: Half-life for temporal decay (default 5 years)
            context_scorer: ContextCompatibilityScorer instance for PK context scoring
        """
        self.alpha = alpha
        self.temporal_half_life = temporal_half_life_years
        self.context_scorer = context_scorer
        self.parser = ContextMetadataParser()

    def extract(
        self,
        records: List[Dict[str, Any]],
        current_year: int = 2024,
    ) -> Tuple[EvidenceLineageFeatures, CitationDAGResult]:
        """
        Full extraction pipeline: build DAG, compute echo-attenuated strengths,
        context compatibility, contradiction index, and temporal features.
        
        Args:
            records: List of evidence record dicts with keys:
                - 'id'/'pmid'/'paper_id': Paper identifier
                - 'year': Publication year
                - 'study_type': 'clinical', 'in_vivo', 'in_vitro', 'review', etc.
                - 'citations'/'references': List of cited paper IDs
                - 'sentiment'/'claim': 'positive' (interaction) or 'negative' (no interaction)
                - 'quality': Extraction quality score [0, 1]
                - PK context fields: 'species', 'preparation', 'route', 'concentration_um'
            current_year: Reference year for temporal recency calculation
            
        Returns:
            Tuple of (EvidenceLineageFeatures, CitationDAGResult)
        """
        if not records:
            return EvidenceLineageFeatures(), CitationDAGResult(
                total_papers=0, num_roots=0, num_derivatives=0,
                effective_independent_count=0.0, echo_chamber_index=0.0,
                max_depth=0, component_depths={}, root_papers=[], derivative_papers=[]
            )

        # 1. Build citation DAG
        dag = CitationDAG(echo_discount_alpha=self.alpha)
        
        # Add all papers
        for rec in records:
            pid = str(rec.get("id", rec.get("pmid", rec.get("paper_id", "unknown"))))
            year = int(rec.get("year", current_year))
            st = str(rec.get("study_type", "in_vivo"))
            is_emp = st not in {"review", "meta_analysis", "guideline"}
            dag.add_paper(paper_id=pid, year=year, is_empirical=is_emp, study_type=st)
        
        # Add citation edges
        for rec in records:
            pid = str(rec.get("id", rec.get("pmid", rec.get("paper_id", "unknown"))))
            cites = rec.get("citations", rec.get("references", rec.get("cites", [])))
            for cited in cites:
                dag.add_citation(citing_id=pid, cited_id=str(cited))
        
        dag_result = dag.analyze()

        # 2. Compute echo-attenuated strengths per claim polarity
        # Partition records by sentiment/claim
        pos_records = [r for r in records if str(r.get("sentiment", r.get("claim", "positive"))).lower() in {"positive", "interaction", "interacts", "1", "yes", "true"}]
        neg_records = [r for r in records if str(r.get("sentiment", r.get("claim", "negative"))).lower() in {"negative", "no_interaction", "no interaction", "does not interact", "0", "no", "false"}]
        
        # For each partition, build sub-DAG and compute S_pos, S_neg
        S_pos, N_primary_pos, N_echo_pos, N_raw_pos, q_bar_pos, kappa_pos, years_pos = self._compute_cluster_strength(dag, pos_records, dag_result)
        S_neg, N_primary_neg, N_echo_neg, N_raw_neg, q_bar_neg, kappa_neg, years_neg = self._compute_cluster_strength(dag, neg_records, dag_result)

        # 3. Aggregate across polarities
        N_primary = N_primary_pos + N_primary_neg
        N_echo = N_echo_pos + N_echo_neg
        N_raw_pos = len(pos_records)
        N_raw_neg = len(neg_records)
        q_bar = (q_bar_pos * N_raw_pos + q_bar_neg * N_raw_neg) / max(N_raw_pos + N_raw_neg, 1)
        kappa_context = (kappa_pos * N_raw_pos + kappa_neg * N_raw_neg) / max(N_raw_pos + N_raw_neg, 1)
        
        all_years = years_pos + years_neg
        Delta_time = self._compute_temporal_recency(all_years, current_year)

        # 4. Contradiction index
        C_contradiction = self._compute_contradiction_index(S_pos, S_neg)

        features = EvidenceLineageFeatures(
            S_pos=S_pos,
            S_neg=S_neg,
            N_primary=N_primary,
            N_echo=N_echo,
            kappa_context=kappa_context,
            C_contradiction=C_contradiction,
            q_bar=q_bar,
            Delta_time=Delta_time,
            N_raw_pos=N_raw_pos,
            N_raw_neg=N_raw_neg,
        )

        return features, dag_result

    def _compute_cluster_strength(
        self,
        dag: CitationDAG,
        records: List[Dict[str, Any]],
        dag_result: CitationDAGResult,
    ) -> Tuple[float, int, int, int, float, float, List[int]]:
        """
        Compute echo-attenuated cluster strength for a polarity partition.
        Returns: (S, N_primary, N_echo, N_raw, q_bar, kappa_context, years)
        """
        if not records:
            return 0.0, 0, 0, 0, 0.0, 0.0, []

        # Map paper_id to record for quality and context lookup
        rec_by_id = {}
        for r in records:
            pid = str(r.get("id", r.get("pmid", r.get("paper_id", "unknown"))))
            rec_by_id[pid] = r

        # Get component depths from DAG result
        depths = dag_result.component_depths
        root_papers = set(dag_result.root_papers)

        total_S = 0.0
        total_q = 0.0
        total_kappa = 0.0
        years = []
        N_primary = 0
        N_echo = 0

        for pid, depth in depths.items():
            rec = rec_by_id.get(pid)
            if rec is None:
                continue
            
            quality = float(rec.get("quality", 0.5))
            year = int(rec.get("year", 2020))
            years.append(year)

            # Context compatibility for this record
            ctx = self.parser.parse_dict(rec)
            kappa = 0.5  # default
            if self.context_scorer is not None:
                try:
                    kappa_tensor = self.context_scorer([ctx])
                    kappa = float(kappa_tensor.squeeze().item())
                except Exception:
                    kappa = 0.5

            # Weight by echo attenuation: w = (1 - alpha)^depth
            weight = math.pow(1.0 - self.alpha, depth)
            total_S += quality * weight
            total_q += quality
            total_kappa += kappa

            if pid in root_papers:
                N_primary += 1
            else:
                N_echo += 1

        N_raw = len(records)
        q_bar = total_q / max(N_raw, 1)
        kappa_context = total_kappa / max(N_raw, 1)

        return total_S, N_primary, N_echo, N_raw, q_bar, kappa_context, years

    def _compute_contradiction_index(self, S_pos: float, S_neg: float) -> float:
        """
        Compute scientific contradiction index C_h,d (Eq. 4 in patent).
        C = 2 * min(S_pos, S_neg) / (S_pos + S_neg + epsilon)
        """
        epsilon = 1e-6
        if S_pos + S_neg < epsilon:
            return 0.0
        return (2.0 * min(S_pos, S_neg)) / (S_pos + S_neg + epsilon)

    def _compute_temporal_recency(self, years: List[int], current_year: int) -> float:
        """
        Compute temporal recency feature Delta_time.
        Normalized exponential decay: exp(-ln(2) * (current_year - max_year) / half_life)
        Returns value in [0, 1] where 1 = most recent evidence, 0 = very old.
        """
        if not years:
            return 0.0
        max_year = max(years)
        age = current_year - max_year
        if age <= 0:
            return 1.0
        return math.exp(-math.log(2) * age / self.temporal_half_life)


class EvidenceLineageEncoder(nn.Module):
    """
    Differentiable neural encoder for the 10D evidence lineage vector.
    
    Maps the 10D structured features to the internal representation dimension
    used by the ReliabilityScorer. This replaces the ad-hoc lineage_proj
    with a properly structured encoder matching the patent's 10D specification.
    """

    def __init__(
        self,
        output_dim: int = 128,
        hidden_dim: int = 64,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.output_dim = output_dim

        # Per-dimension embedding/projection (some dimensions are categorical/count)
        # S_pos, S_neg: continuous strengths -> Linear(1, 16)
        self.s_pos_proj = nn.Sequential(nn.Linear(1, 16), nn.GELU())
        self.s_neg_proj = nn.Sequential(nn.Linear(1, 16), nn.GELU())

        # N_primary, N_echo: counts -> Embedding (up to 50)
        self.n_primary_embed = nn.Embedding(51, 16)
        self.n_echo_embed = nn.Embedding(51, 16)

        # kappa_context, C_contradiction, q_bar, Delta_time: continuous [0,1] -> Linear(1, 16)
        self.kappa_proj = nn.Sequential(nn.Linear(1, 16), nn.GELU())
        self.contr_proj = nn.Sequential(nn.Linear(1, 16), nn.GELU())
        self.q_proj = nn.Sequential(nn.Linear(1, 16), nn.GELU())
        self.time_proj = nn.Sequential(nn.Linear(1, 16), nn.GELU())

        # N_raw_pos, N_raw_neg: counts -> Embedding (up to 100)
        self.n_raw_pos_embed = nn.Embedding(101, 16)
        self.n_raw_neg_embed = nn.Embedding(101, 16)

        # Total embedding dimension = 10 * 16 = 160
        total_embed_dim = 10 * 16

        # MLP to output dimension
        self.mlp = nn.Sequential(
            nn.Linear(total_embed_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, output_dim),
        )

        logger.info(
            f"EvidenceLineageEncoder: 10D structured features -> "
            f"{total_embed_dim}D embeddings -> {output_dim}D output"
        )

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        """
        Args:
            features: (batch, 10) tensor with columns:
                [S_pos, S_neg, N_primary, N_echo, kappa_context,
                 C_contradiction, q_bar, Delta_time, N_raw_pos, N_raw_neg]
        
        Returns:
            encoded: (batch, output_dim)
        """
        batch_size = features.size(0)
        
        # Slice and project each dimension
        s_pos_emb = self.s_pos_proj(features[:, 0:1])
        s_neg_emb = self.s_neg_proj(features[:, 1:2])
        
        n_primary_idx = features[:, 2].long().clamp(0, 50)
        n_echo_idx = features[:, 3].long().clamp(0, 50)
        n_primary_emb = self.n_primary_embed(n_primary_idx)
        n_echo_emb = self.n_echo_embed(n_echo_idx)
        
        kappa_emb = self.kappa_proj(features[:, 4:5])
        contr_emb = self.contr_proj(features[:, 5:6])
        q_emb = self.q_proj(features[:, 6:7])
        time_emb = self.time_proj(features[:, 7:8])
        
        n_raw_pos_idx = features[:, 8].long().clamp(0, 100)
        n_raw_neg_idx = features[:, 9].long().clamp(0, 100)
        n_raw_pos_emb = self.n_raw_pos_embed(n_raw_pos_idx)
        n_raw_neg_emb = self.n_raw_neg_embed(n_raw_neg_idx)

        # Concatenate all embeddings
        combined = torch.cat([
            s_pos_emb, s_neg_emb, n_primary_emb, n_echo_emb,
            kappa_emb, contr_emb, q_emb, time_emb,
            n_raw_pos_emb, n_raw_neg_emb,
        ], dim=-1)

        return self.mlp(combined)


class TemporalEvidenceWeighting(nn.Module):
    """
    Temporal decay weighting for evidence recency — Novelty Component.
    
    Applies exponential decay to evidence contributions based on publication
    age, with a learnable half-life parameter. This implements the Delta_time
    feature and allows the model to learn how quickly evidence becomes stale.
    
    Formula:
        w_temporal = exp(-ln(2) * age / half_life)
    where half_life is a learnable positive parameter.
    """

    def __init__(self, initial_half_life: float = 5.0, learnable: bool = True):
        super().__init__()
        if learnable:
            # Parameterize in log-space for positivity
            init_log = math.log(initial_half_life)
            self.log_half_life = nn.Parameter(torch.tensor([init_log], dtype=torch.float32))
        else:
            self.register_buffer("log_half_life", torch.tensor([math.log(initial_half_life)], dtype=torch.float32))
        self.learnable = learnable

    @property
    def half_life(self) -> torch.Tensor:
        return torch.exp(self.log_half_life)

    def forward(self, years: torch.Tensor, current_year: float = 2024.0) -> torch.Tensor:
        """
        Args:
            years: (batch, num_evidence) tensor of publication years
            current_year: Reference year (float for differentiability)
            
        Returns:
            weights: (batch, num_evidence) temporal weights in (0, 1]
        """
        age = current_year - years  # (batch, num_evidence)
        age = torch.clamp(age, min=0.0)
        # w = exp(-ln(2) * age / half_life)
        weights = torch.exp(-math.log(2) * age / self.half_life)
        return weights


def demo_evidence_lineage():
    """Demonstration of the evidence lineage extraction."""
    # Sample records simulating a real herb-drug evidence set
    records = [
        # Primary clinical trial (root)
        {"id": "PMID_1001", "year": 2015, "study_type": "clinical", "citations": [],
         "sentiment": "positive", "quality": 0.95,
         "species": "human", "preparation": "standardized_extract", "route": "oral"},
        # Review citing the trial (derivative depth 1)
        {"id": "PMID_1002", "year": 2018, "study_type": "review", "citations": ["PMID_1001"],
         "sentiment": "positive", "quality": 0.8,
         "species": "human", "preparation": "standardized_extract", "route": "oral"},
        # Meta-analysis citing the review (derivative depth 2)
        {"id": "PMID_1003", "year": 2021, "study_type": "meta_analysis", "citations": ["PMID_1002"],
         "sentiment": "positive", "quality": 0.85,
         "species": "human", "preparation": "standardized_extract", "route": "oral"},
        # Independent in vitro study (separate root)
        {"id": "PMID_2001", "year": 2019, "study_type": "in_vitro", "citations": [],
         "sentiment": "negative", "quality": 0.7,
         "species": "in_vitro_microsomes", "preparation": "isolated_constituent", "route": "in_vitro_incubation",
         "concentration_um": 100.0},  # supra-physiological
        # Another independent in vitro study (separate root)
        {"id": "PMID_2002", "year": 2022, "study_type": "in_vitro", "citations": [],
         "sentiment": "negative", "quality": 0.75,
         "species": "in_vitro_microsomes", "preparation": "isolated_constituent", "route": "in_vitro_incubation",
         "concentration_um": 25.0},
    ]

    from src.models.context_compatibility import ContextCompatibilityScorer
    context_scorer = ContextCompatibilityScorer(learnable_refinement=False)
    extractor = EvidenceLineageExtractor(alpha=0.15, context_scorer=context_scorer)
    
    features, dag_result = extractor.extract(records, current_year=2024)
    
    print("=== Evidence Lineage Features (10D) ===")
    print(f"S_pos (echo-attenuated positive strength): {features.S_pos:.4f}")
    print(f"S_neg (echo-attenuated negative strength): {features.S_neg:.4f}")
    print(f"N_primary (independent roots): {features.N_primary}")
    print(f"N_echo (derivative publications): {features.N_echo}")
    print(f"kappa_context (mean PK context): {features.kappa_context:.4f}")
    print(f"C_contradiction (contradiction index): {features.C_contradiction:.4f}")
    print(f"q_bar (mean quality): {features.q_bar:.4f}")
    print(f"Delta_time (temporal recency): {features.Delta_time:.4f}")
    print(f"N_raw_pos: {features.N_raw_pos}")
    print(f"N_raw_neg: {features.N_raw_neg}")
    print()
    print("=== DAG Analysis ===")
    print(f"Total papers: {dag_result.total_papers}")
    print(f"Roots: {dag_result.num_roots}")
    print(f"Derivatives: {dag_result.num_derivatives}")
    print(f"Effective independent count: {dag_result.effective_independent_count:.4f}")
    print(f"Echo chamber index: {dag_result.echo_chamber_index:.4f}")
    print(f"Max depth: {dag_result.max_depth}")
    print(f"Component depths: {dag_result.component_depths}")
    
    # Test encoder
    encoder = EvidenceLineageEncoder(output_dim=128)
    feat_tensor = features.to_tensor()
    encoded = encoder(feat_tensor)
    print(f"\nEncoded shape: {encoded.shape}")
    print("Demo complete.")


if __name__ == "__main__":
    demo_evidence_lineage()