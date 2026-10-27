"""
Evidence Lineage, Study Independence, and Conflict Analysis Module.

Detects evidence dependence (the "citation echo chamber"), measures context
compatibility, and represents positive vs. negative contradictions explicitly.

Core Capabilities:
1. EvidenceLineageGraph: Builds citation dependency graphs and clusters dependent studies.
2. Effective Independence: Computes discounted independent study counts instead of raw citation counts.
3. Context Compatibility: Evaluates preparation, dose, species, and route applicability.
4. Contradiction Resolution: Computes independent positive strength, independent negative strength,
   and conflict metrics without lossy scalar cancellation.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Set, Tuple
import math
import networkx as nx
import numpy as np
import pandas as pd
from loguru import logger

from src.data.evidence_schema import EvidenceRecord, EvidenceOutcome, EvidenceType


# Evidence type base reliability prior (source quality proxy prior)
EVIDENCE_TYPE_WEIGHTS: Dict[str, float] = {
    EvidenceType.META_ANALYSIS.value: 1.00,
    EvidenceType.RCT.value: 0.95,
    EvidenceType.CLINICAL.value: 0.90,
    EvidenceType.OBSERVATIONAL.value: 0.80,
    EvidenceType.CASE_REPORT.value: 0.70,
    EvidenceType.ANIMAL.value: 0.50,
    EvidenceType.IN_VITRO.value: 0.30,
    EvidenceType.MECHANISTIC.value: 0.25,
    EvidenceType.IN_SILICO.value: 0.15,
    EvidenceType.INCONCLUSIVE.value: 0.10,
}

# Context species weights for human clinical translation
# Canonical default standard: human 1.00 / in vivo animal 0.50 / in vitro 0.20
SPECIES_TRANSLATION_WEIGHTS: Dict[str, float] = {
    "human": 1.0,
    "homo sapiens": 1.0,
    "patient": 1.0,
    "volunteer": 1.0,
    "rattus norvegicus": 0.50,
    "rat": 0.50,
    "mus musculus": 0.50,
    "mouse": 0.50,
    "rabbit": 0.50,
    "caco-2": 0.20,
    "cell line": 0.20,
    "in vitro": 0.20,
    "unknown": 0.35,
}


# Feature Group Constants for Leakage Control
FEATURE_GROUP_FULL = "full"
FEATURE_GROUP_QUALITY_ONLY = "quality_only"
FEATURE_GROUP_POLARITY_ONLY = "polarity_only"


@dataclass
class PairEvidenceSummary:
    """
    Aggregated evidence summary for a single (herb_id, drug_id) candidate pair.
    Preserves independent positive, independent negative, context, and conflict details.
    """
    herb_id: str
    drug_id: str
    total_records: int
    raw_positive_count: int
    raw_negative_count: int
    raw_conflicting_count: int
    raw_unknown_count: int
    
    # Lineage and Independence
    num_independent_positive_clusters: int
    num_independent_negative_clusters: int
    independent_positive_strength: float
    independent_negative_strength: float
    citation_echo_ratio: float  # Fraction of reports that are redundant derivative citations
    
    # Context and Quality
    mean_source_quality: float
    context_compatibility_score: float
    metadata_missingness: float
    
    # Contradiction / Conflict
    contradiction_strength: float  # High when strong independent positive and negative evidence clash
    net_evidence_margin: float     # S_pos - S_neg
    raw_margin: float = 0.0        # Raw positive count - Raw negative count

    # Lineage audit trail
    primary_study_clusters: List[List[str]] = field(default_factory=list)
    audit_sources: List[str] = field(default_factory=list)

    # Phase 3 PK reachability integration
    rho_reach: float = 0.5
    f_m_CYP: float = 0.5
    pk_branch_used: str = "hepatic"
    missing_dose_or_mw: bool = False

    def to_feature_group(self, feature_group: str = FEATURE_GROUP_QUALITY_ONLY) -> np.ndarray:
        """
        Extract feature subset based on group for rigorous leakage control:
        - 'full': 12D [S_pos, S_neg, echo_ratio, q_mean, W_context, C_hd, margin, missingness, N_raw_pos, N_raw_neg, rho_reach, f_m_CYP]
        - 'quality_only': 7D [echo_ratio, q_mean, W_context, C_hd, missingness, rho_reach, f_m_CYP]
        - 'polarity_only': 5D [S_pos, S_neg, margin, N_raw_pos, N_raw_neg]
        """
        rho_reach = self.rho_reach
        f_m_CYP = self.f_m_CYP

        if feature_group == FEATURE_GROUP_QUALITY_ONLY or feature_group == "quality_only":
            return np.array([
                self.citation_echo_ratio,
                self.mean_source_quality,
                self.context_compatibility_score,
                self.contradiction_strength,
                self.metadata_missingness,
                rho_reach,
                f_m_CYP,
            ], dtype=np.float32)
        elif feature_group == FEATURE_GROUP_POLARITY_ONLY or feature_group == "polarity_only":
            return np.array([
                self.independent_positive_strength,
                self.independent_negative_strength,
                self.net_evidence_margin,
                float(self.raw_positive_count),
                float(self.raw_negative_count),
            ], dtype=np.float32)
        else:  # FEATURE_GROUP_FULL
            return np.array([
                self.independent_positive_strength,
                self.independent_negative_strength,
                self.citation_echo_ratio,
                self.mean_source_quality,
                self.context_compatibility_score,
                self.contradiction_strength,
                self.net_evidence_margin,
                self.metadata_missingness,
                float(self.raw_positive_count),
                float(self.raw_negative_count),
                rho_reach,
                f_m_CYP,
            ], dtype=np.float32)

    def to_feature_vector(self, mode: Optional[str] = None, feature_group: Optional[str] = None) -> np.ndarray:
        """
        Returns an evidence feature vector with rigorous ablation isolation.

        Modes:
        - 'reliability_conditioned': Full proposed model with lineage clustering, context, and conflict.
        - 'reliability_conditioned_quality_only': Quality-only features, strictly leakage-free.
        - 'no_study_dependence': Replaces lineage-clustered strengths with raw counts, zeros echo ratio.
        - 'no_context_conflict': Neutralizes context to 1.0, zeros conflict, and removes conflict subtraction from margin.
        - 'graph_plus_naive_evidence': Baseline using solely raw counts and neutral context/conflict.
        """
        if feature_group == FEATURE_GROUP_QUALITY_ONLY or mode == "reliability_conditioned_quality_only":
            return self.to_feature_group(FEATURE_GROUP_QUALITY_ONLY)
        elif feature_group == FEATURE_GROUP_POLARITY_ONLY:
            return self.to_feature_group(FEATURE_GROUP_POLARITY_ONLY)

        if mode == "no_study_dependence":
            # Ablate study dependence: use un-discounted raw counts, ignore citation echo
            pos = float(self.raw_positive_count)
            neg = float(self.raw_negative_count)
            echo = 0.0
            ctx = self.context_compatibility_score
            conflict = self.contradiction_strength
            margin = float(self.raw_positive_count - self.raw_negative_count)
        elif mode == "no_context_conflict":
            # Ablate context and conflict: neutralize context, zero contradiction, and remove negative deduction from margin
            pos = self.independent_positive_strength
            neg = self.independent_negative_strength
            echo = self.citation_echo_ratio
            ctx = 1.0       # neutral context
            conflict = 0.0  # zero contradiction
            margin = self.independent_positive_strength  # positive evidence only; no conflict subtraction
        elif mode in {"graph_plus_naive_evidence", "raw_naive"}:
            pos = float(self.raw_positive_count)
            neg = float(self.raw_negative_count)
            echo = 0.0
            ctx = 1.0
            conflict = 0.0
            margin = float(self.raw_positive_count - self.raw_negative_count)
        else:
            # Full proposed reliability-conditioned representation
            pos = self.independent_positive_strength
            neg = self.independent_negative_strength
            echo = self.citation_echo_ratio
            ctx = self.context_compatibility_score
            conflict = self.contradiction_strength
            margin = self.net_evidence_margin

        return np.array([
            pos,
            neg,
            echo,
            self.mean_source_quality,
            ctx,
            conflict,
            margin,
            self.metadata_missingness,
        ], dtype=np.float32)

    def to_full_metadata_vector(self, feature_group: str = "full") -> np.ndarray:
        """
        Returns a metadata vector containing lineage-adjusted features and PK reachability parameters.
        - 'full': 12D [0: indep_pos, 1: indep_neg, 2: echo_ratio, 3: source_quality,
                       4: context_compat, 5: contradiction, 6: net_margin, 7: missingness,
                       8: raw_positive_count, 9: raw_negative_count,
                       10: rho_reach, 11: f_m_CYP]
        - 'quality_only': 7D [echo_ratio, source_quality, context_compat, contradiction,
                              missingness, rho_reach, f_m_CYP]
        - 'polarity_only': 5D [indep_pos, indep_neg, net_margin, raw_positive_count, raw_negative_count]
        """
        if feature_group in {FEATURE_GROUP_QUALITY_ONLY, "quality_only"}:
            return self.to_feature_group(FEATURE_GROUP_QUALITY_ONLY)
        elif feature_group in {FEATURE_GROUP_POLARITY_ONLY, "polarity_only"}:
            return self.to_feature_group(FEATURE_GROUP_POLARITY_ONLY)
        return self.to_feature_group(FEATURE_GROUP_FULL)



class EvidenceLineageAnalyzer:
    """
    Builds citation and lineage graphs to uncover study dependency,
    discount redundant echoes, evaluate context, and measure conflict.
    """

    def __init__(
        self,
        echo_discount_factor: float = 0.15,
        feature_group: str = FEATURE_GROUP_QUALITY_ONLY,
        context_scorer: Optional[Any] = None,
    ):
        """
        Args:
            echo_discount_factor: Weight given to each secondary repeating report
                                 within the same study cluster (default 0.15).
            feature_group: Feature group mode ('quality_only', 'full', 'polarity_only').
            context_scorer: Optional ContextCompatibilityScorer or callable for clinical context weighting.
        """
        self.echo_discount_factor = echo_discount_factor
        self.feature_group = feature_group
        self.context_scorer = context_scorer

    def build_lineage_graph(self, records: List[EvidenceRecord]) -> nx.DiGraph:
        """
        Construct a directed citation / provenance graph from evidence records.
        Edges: (citing_study -> cited_study).
        """
        G = nx.DiGraph()

        for rec in records:
            G.add_node(rec.study_id, source_id=rec.source_id, outcome=rec.outcome)
            for cited in rec.cited_study_ids:
                if cited and cited != rec.study_id:
                    G.add_node(cited)
                    G.add_edge(rec.study_id, cited)

        return G

    def cluster_dependent_studies(
        self, records: List[EvidenceRecord]
    ) -> List[List[EvidenceRecord]]:
        """
        Cluster records that belong to the same underlying study or citation chain.
        Two records are in the same cluster if they share study_id or have
        a citation relationship in the lineage graph.
        """
        if not records:
            return []

        # Build undirected view of citation dependency to find connected study components
        lineage_graph = self.build_lineage_graph(records)
        undirected_lineage = lineage_graph.to_undirected()

        # Group records by study_id
        records_by_study: Dict[str, List[EvidenceRecord]] = {}
        for rec in records:
            records_by_study.setdefault(rec.study_id, []).append(rec)

        connected_study_sets = list(nx.connected_components(undirected_lineage))
        clusters: List[List[EvidenceRecord]] = []
        assigned_records: Set[int] = set()

        for study_set in connected_study_sets:
            cluster_recs: List[EvidenceRecord] = []
            for s_id in study_set:
                if s_id in records_by_study:
                    for r in records_by_study[s_id]:
                        cluster_recs.append(r)
                        assigned_records.add(id(r))
            if cluster_recs:
                clusters.append(cluster_recs)

        # Catch any records not in connected components (isolated nodes)
        for r in records:
            if id(r) not in assigned_records:
                clusters.append([r])

        return clusters

    def compute_context_compatibility(self, rec: EvidenceRecord) -> float:
        """
        Score how clinically compatible the contextual parameters of the record are.
        Evaluates species, preparation, and route.
        """
        if self.context_scorer is not None:
            if hasattr(self.context_scorer, "score_record"):
                return float(self.context_scorer.score_record(rec))
            elif callable(self.context_scorer):
                return float(self.context_scorer(rec))

        score = 1.0

        # Species factor
        species_str = (rec.species or "unknown").lower()
        species_match = False
        for sp_key, sp_val in SPECIES_TRANSLATION_WEIGHTS.items():
            if sp_key in species_str:
                score *= sp_val
                species_match = True
                break
        if not species_match:
            score *= 0.50

        # Preparation factor
        prep_str = (rec.preparation or "").lower()
        if any(w in prep_str for w in ["standardized", "clinical formulation"]):
            score *= 1.0
        elif any(w in prep_str for w in ["extract", "decoction", "powder"]):
            score *= 0.85
        elif any(w in prep_str for w in ["isolated", "monomer", "pure"]):
            score *= 0.75
        elif prep_str:
            score *= 0.70
        else:
            score *= 0.60  # unstated preparation

        # Route factor
        route_str = (rec.route or "").lower()
        if "oral" in route_str:
            score *= 1.0
        elif any(w in route_str for w in ["iv", "intravenous", "ip", "intraperitoneal"]):
            score *= 0.75
        elif route_str:
            score *= 0.80
        else:
            score *= 0.70  # unstated route

        return float(np.clip(score, 0.05, 1.0))

    def evaluate_cluster_strength(self, cluster: List[EvidenceRecord]) -> float:
        """
        Compute effective independent evidentiary strength for a cluster of dependent records:
        Strength = Primary_Quality + sum(echo_discount_factor * Secondary_Quality).
        """
        if not cluster:
            return 0.0

        # Sort cluster by quality descending so best record is treated as primary
        def rec_quality(r: EvidenceRecord) -> float:
            base_wt = EVIDENCE_TYPE_WEIGHTS.get(r.evidence_type, 0.20)
            context_wt = self.compute_context_compatibility(r)
            conf = r.extraction_confidence
            return base_wt * context_wt * conf

        sorted_recs = sorted(cluster, key=rec_quality, reverse=True)
        primary = sorted_recs[0]
        primary_score = rec_quality(primary)

        # Secondary derivative reports add diminishing returns
        secondary_score = sum(
            self.echo_discount_factor * rec_quality(r)
            for r in sorted_recs[1:]
        )
        return float(primary_score + secondary_score)

    def aggregate_pair_evidence(
        self, herb_id: str, drug_id: str, records: List[EvidenceRecord]
    ) -> PairEvidenceSummary:
        """
        Aggregate all records for a herb-drug pair into an evidence-lineage summary.
        Separates positive and negative study clusters, computes conflict and echo metrics.
        """
        if not records:
            return PairEvidenceSummary(
                herb_id=herb_id,
                drug_id=drug_id,
                total_records=0,
                raw_positive_count=0,
                raw_negative_count=0,
                raw_conflicting_count=0,
                raw_unknown_count=0,
                num_independent_positive_clusters=0,
                num_independent_negative_clusters=0,
                independent_positive_strength=0.0,
                independent_negative_strength=0.0,
                citation_echo_ratio=0.0,
                mean_source_quality=0.0,
                context_compatibility_score=0.0,
                metadata_missingness=1.0,
                contradiction_strength=0.0,
                net_evidence_margin=0.0,
                primary_study_clusters=[],
                audit_sources=[],
            )

        pos_recs = [
            r for r in records
            if r.outcome in {EvidenceOutcome.POSITIVE.value, EvidenceOutcome.INTERACTION_PRESENT.value}
        ]
        neg_recs = [
            r for r in records
            if r.outcome in {EvidenceOutcome.NEGATIVE.value, EvidenceOutcome.NO_INTERACTION_OBSERVED.value}
        ]
        conf_recs = [r for r in records if r.outcome == EvidenceOutcome.CONFLICTING.value]
        unk_recs = [r for r in records if r.outcome == EvidenceOutcome.UNKNOWN.value]

        # Cluster positive and negative records separately
        pos_clusters = self.cluster_dependent_studies(pos_recs)
        neg_clusters = self.cluster_dependent_studies(neg_recs)

        indep_pos_strength = sum(self.evaluate_cluster_strength(c) for c in pos_clusters)
        indep_neg_strength = sum(self.evaluate_cluster_strength(c) for c in neg_clusters)

        # Add conflicting records as half positive, half negative strength
        for r in conf_recs:
            half_strength = 0.5 * self.evaluate_cluster_strength([r])
            indep_pos_strength += half_strength
            indep_neg_strength += half_strength

        # Citation echo ratio: fraction of reports that are redundant within clusters
        total_clusters = len(pos_clusters) + len(neg_clusters) + len(conf_recs) + len(unk_recs)
        citation_echo_ratio = 0.0
        if len(records) > 1:
            citation_echo_ratio = max(0.0, 1.0 - (total_clusters / float(len(records))))

        # Mean source quality
        qualities = [
            EVIDENCE_TYPE_WEIGHTS.get(r.evidence_type, 0.20) * r.extraction_confidence
            for r in records
        ]
        mean_quality = float(np.mean(qualities)) if qualities else 0.0

        # Mean context compatibility
        contexts = [self.compute_context_compatibility(r) for r in records]
        mean_context = float(np.mean(contexts)) if contexts else 0.0

        # Metadata missingness: check dose, preparation, route, species
        missing_counts = 0
        total_fields = len(records) * 4
        for r in records:
            if not r.dose: missing_counts += 1
            if not r.preparation: missing_counts += 1
            if not r.route: missing_counts += 1
            if not r.species: missing_counts += 1
        missingness = missing_counts / float(total_fields) if total_fields > 0 else 1.0

        # Contradiction strength:
        # High when both independent positive and independent negative evidence are strong
        # Formula: 2 * min(S_pos, S_neg) / (S_pos + S_neg + 1e-5)
        eps = 1e-5
        denom = indep_pos_strength + indep_neg_strength + eps
        contradiction_strength = float((2.0 * min(indep_pos_strength, indep_neg_strength)) / denom)

        # Net evidence margin
        net_evidence_margin = float(indep_pos_strength - indep_neg_strength)

        # Audit sources
        audit_sources = list({r.source_url_or_doi or r.source_id for r in records if r.source_id})
        cluster_study_ids = [
            list({r.study_id for r in c})
            for c in (pos_clusters + neg_clusters)
        ]

        return PairEvidenceSummary(
            herb_id=herb_id,
            drug_id=drug_id,
            total_records=len(records),
            raw_positive_count=len(pos_recs),
            raw_negative_count=len(neg_recs),
            raw_conflicting_count=len(conf_recs),
            raw_unknown_count=len(unk_recs),
            num_independent_positive_clusters=len(pos_clusters),
            num_independent_negative_clusters=len(neg_clusters),
            independent_positive_strength=float(indep_pos_strength),
            independent_negative_strength=float(indep_neg_strength),
            citation_echo_ratio=float(citation_echo_ratio),
            mean_source_quality=float(mean_quality),
            context_compatibility_score=float(mean_context),
            metadata_missingness=float(missingness),
            contradiction_strength=float(contradiction_strength),
            net_evidence_margin=float(net_evidence_margin),
            raw_margin=float(len(pos_recs) - len(neg_recs)),
            primary_study_clusters=cluster_study_ids,
            audit_sources=audit_sources,
        )

    def analyze_dataset(self, df: pd.DataFrame) -> Dict[Tuple[str, str], PairEvidenceSummary]:
        """Aggregate evidence records across all pairs in a dataset using this analyzer instance."""
        return build_pair_summaries_from_dataframe(df, analyzer=self)


def build_pair_summaries_from_dataframe(
    df: pd.DataFrame,
    analyzer: Optional[EvidenceLineageAnalyzer] = None,
) -> Dict[Tuple[str, str], PairEvidenceSummary]:
    """
    Given a canonical evidence DataFrame (1 row per evidence record),
    group by (herb_id, drug_id) and build PairEvidenceSummary objects.
    """
    if analyzer is None:
        analyzer = EvidenceLineageAnalyzer()
    summaries: Dict[Tuple[str, str], PairEvidenceSummary] = {}

    grouped = df.groupby(["herb_id", "drug_id"])
    for (h_id, d_id), group in grouped:
        records: List[EvidenceRecord] = []
        for _, row in group.iterrows():
            records.append(
                EvidenceRecord(
                    herb_id=str(row["herb_id"]),
                    drug_id=str(row["drug_id"]),
                    outcome=str(row.get("outcome", "unknown")),
                    evidence_type=str(row.get("evidence_type", "clinical")),
                    source_id=str(row.get("source_id", "source_unknown")),
                    study_id=str(row.get("study_id", row["source_id"])),
                    cited_study_ids=row.get("cited_study_ids", []),
                    dose=row.get("dose") if pd.notna(row.get("dose")) else None,
                    preparation=row.get("preparation") if pd.notna(row.get("preparation")) else None,
                    route=row.get("route") if pd.notna(row.get("route")) else None,
                    species=row.get("species") if pd.notna(row.get("species")) else None,
                    population=row.get("population") if pd.notna(row.get("population")) else None,
                    publication_date=str(row["publication_date"]) if pd.notna(row.get("publication_date")) else None,
                    extraction_confidence=float(row.get("extraction_confidence", 1.0)),
                    source_url_or_doi=str(row["source_url_or_doi"]) if pd.notna(row.get("source_url_or_doi")) else None,
                    original_text=str(row["original_text"]) if pd.notna(row.get("original_text")) else None,
                    extraction_method=str(row.get("extraction_method", "manual_curation")),
                    raw_label=str(row.get("raw_label", "")) if pd.notna(row.get("raw_label")) else None,
                )
            )
        summaries[(str(h_id), str(d_id))] = analyzer.aggregate_pair_evidence(str(h_id), str(d_id), records)

    logger.info(f"Computed lineage summaries for {len(summaries)} distinct herb-drug pairs")
    return summaries
