"""
Candidate Metadata Builder

Constructs the candidate metadata instances (C, T, B, M, S) that feed
into the reliability scorer. Each extracted interaction claim is
enriched with five metadata dimensions:

C — Corroboration count: number of independent sources confirming
T — Temporal recency: how recent the evidence is
B — Biomedical quality: journal quality / peer-review status
M — Molecular plausibility: structural similarity / known mechanism overlap
S — Source type: categorical (clinical trial, peer-reviewed, forum, etc.)

These metadata instances are the input to the reliability scorer,
which produces the scalar R that gates cross-attention.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Optional

from loguru import logger


@dataclass
class CandidateMetadata:
    """
    A candidate metadata instance for a single interaction claim.

    This is the data structure that flows into the reliability scorer.
    Each interaction extracted from text produces one of these,
    combining all five metadata dimensions.
    """

    # Interaction claim
    entity1_id: str
    entity1_name: str
    entity1_type: str
    entity2_id: str
    entity2_name: str
    entity2_type: str
    relation_type: str
    evidence_text: str = ""

    # Metadata dimensions (C, T, B, M, S)
    corroboration_count: int = 0       # C
    temporal_recency: float = 0.0      # T ∈ [0, 1], 1 = most recent
    biomedical_quality: float = 0.0    # B ∈ [0, 1], 1 = highest quality
    molecular_plausibility: float = 0.0  # M ∈ [0, 1], 1 = highly plausible
    source_type: str = "unknown"       # S — categorical

    # Provenance
    source_documents: list[str] = field(default_factory=list)
    extraction_timestamp: float = 0.0
    is_negated: bool = False
    is_uncertain: bool = False
    confidence: float = 0.0  # NER/RE model confidence

    def to_tensor_features(self) -> list[float]:
        """Convert metadata to a flat feature vector for the reliability scorer."""
        source_type_map = {
            "unknown": 0, "clinical_trial": 1, "peer_reviewed": 2,
            "textbook": 3, "case_report": 4, "health_forum": 5,
            "social_media": 6, "traditional_medicine": 7,
        }
        return [
            float(self.corroboration_count),
            self.temporal_recency,
            self.biomedical_quality,
            self.molecular_plausibility,
            float(source_type_map.get(self.source_type, 0)),
        ]


# Biomedical quality heuristics
SOURCE_QUALITY_SCORES = {
    "clinical_trial": 0.95,
    "systematic_review": 0.92,
    "peer_reviewed": 0.85,
    "case_report": 0.70,
    "textbook": 0.75,
    "preprint": 0.50,
    "health_forum": 0.25,
    "social_media": 0.15,
    "traditional_medicine": 0.40,
    "unknown": 0.10,
}

# Publication venue quality tiers
JOURNAL_QUALITY_TIERS = {
    "tier1": 0.95,  # NEJM, Lancet, JAMA, BMJ, Nature Medicine
    "tier2": 0.85,  # Specialty journals (e.g., Clin Pharmacol Ther)
    "tier3": 0.70,  # Regional journals
    "tier4": 0.50,  # Conference proceedings
    "unranked": 0.30,
}


class CandidateMetadataBuilder:
    """
    Builds candidate metadata instances from extracted relations.

    Aggregates evidence across multiple sources for the same interaction
    claim and computes the five metadata dimensions.

    Usage:
        builder = CandidateMetadataBuilder()

        # Add evidence for an interaction
        builder.add_evidence(
            entity1_id="DB00001", entity1_name="Warfarin", entity1_type="Drug",
            entity2_id="HERB003", entity2_name="St. John's Wort", entity2_type="Herb",
            relation_type="interacts_with",
            evidence_text="Warfarin levels decreased with SJW co-administration",
            source_type="peer_reviewed",
            publication_year=2024,
            journal_tier="tier2",
        )

        # Get finalized metadata instances
        metadata = builder.build()
    """

    def __init__(self):
        # Key: (entity1_id, entity2_id, relation_type) → list of evidence
        self._evidence_store: dict[
            tuple[str, str, str], list[dict]
        ] = {}
        self._built = False

    def add_evidence(
        self,
        entity1_id: str,
        entity1_name: str,
        entity1_type: str,
        entity2_id: str,
        entity2_name: str,
        entity2_type: str,
        relation_type: str,
        evidence_text: str = "",
        source_type: str = "unknown",
        publication_year: Optional[int] = None,
        journal_tier: str = "unranked",
        source_document: str = "",
        is_negated: bool = False,
        is_uncertain: bool = False,
        confidence: float = 0.5,
        molecular_similarity: Optional[float] = None,
    ) -> None:
        """
        Add a piece of evidence for an interaction claim.

        Multiple calls with the same entity pair accumulate evidence
        and increase the corroboration count.
        """
        # Normalize key (order-independent for symmetric relations)
        if entity1_id > entity2_id:
            entity1_id, entity2_id = entity2_id, entity1_id
            entity1_name, entity2_name = entity2_name, entity1_name
            entity1_type, entity2_type = entity2_type, entity1_type

        key = (entity1_id, entity2_id, relation_type)

        evidence = {
            "entity1_id": entity1_id,
            "entity1_name": entity1_name,
            "entity1_type": entity1_type,
            "entity2_id": entity2_id,
            "entity2_name": entity2_name,
            "entity2_type": entity2_type,
            "evidence_text": evidence_text,
            "source_type": source_type,
            "publication_year": publication_year,
            "journal_tier": journal_tier,
            "source_document": source_document,
            "is_negated": is_negated,
            "is_uncertain": is_uncertain,
            "confidence": confidence,
            "molecular_similarity": molecular_similarity,
            "timestamp": time.time(),
        }

        if key not in self._evidence_store:
            self._evidence_store[key] = []
        self._evidence_store[key].append(evidence)

        self._built = False

    def _compute_temporal_recency(
        self, evidence_list: list[dict]
    ) -> float:
        """
        Compute temporal recency score T ∈ [0, 1].

        More recent evidence gets higher scores.
        Uses exponential decay with a half-life of 5 years.
        """
        current_year = 2026
        half_life = 5.0
        max_recency = 0.0

        for ev in evidence_list:
            year = ev.get("publication_year")
            if year is not None:
                age = max(0, current_year - year)
                recency = 2.0 ** (-age / half_life)
                max_recency = max(max_recency, recency)
            else:
                max_recency = max(max_recency, 0.3)  # Unknown year

        return max_recency

    def _compute_biomedical_quality(
        self, evidence_list: list[dict]
    ) -> float:
        """
        Compute biomedical quality score B ∈ [0, 1].

        Takes the maximum quality across all evidence sources.
        """
        max_quality = 0.0

        for ev in evidence_list:
            source_type = ev.get("source_type", "unknown")
            journal_tier = ev.get("journal_tier", "unranked")

            # Source type baseline quality
            source_quality = SOURCE_QUALITY_SCORES.get(source_type, 0.1)

            # Journal tier adjustment
            journal_quality = JOURNAL_QUALITY_TIERS.get(journal_tier, 0.3)

            # Combined quality (weighted average)
            quality = 0.6 * source_quality + 0.4 * journal_quality
            max_quality = max(max_quality, quality)

        return max_quality

    def _compute_molecular_plausibility(
        self, evidence_list: list[dict]
    ) -> float:
        """
        Compute molecular plausibility score M ∈ [0, 1].

        Based on structural similarity scores or known mechanism overlap.
        If no molecular data available, returns a default value.
        """
        similarities = [
            ev.get("molecular_similarity")
            for ev in evidence_list
            if ev.get("molecular_similarity") is not None
        ]

        if similarities:
            return max(similarities)
        return 0.5  # Default: unknown plausibility

    def _determine_source_type(
        self, evidence_list: list[dict]
    ) -> str:
        """
        Determine the best source type for the aggregate.

        Prioritizes higher-quality source types.
        """
        priority = [
            "clinical_trial", "systematic_review", "peer_reviewed",
            "textbook", "case_report", "preprint", "traditional_medicine",
            "health_forum", "social_media", "unknown",
        ]

        source_types = {ev.get("source_type", "unknown") for ev in evidence_list}

        for stype in priority:
            if stype in source_types:
                return stype

        return "unknown"

    def build(self) -> list[CandidateMetadata]:
        """
        Build finalized candidate metadata instances.

        Aggregates all evidence per interaction claim and computes
        the five metadata dimensions (C, T, B, M, S).

        Returns:
            List of CandidateMetadata instances ready for the reliability scorer
        """
        metadata_list = []

        for (e1_id, e2_id, rel_type), evidence_list in self._evidence_store.items():
            # Use first evidence entry for entity names/types
            first = evidence_list[0]

            # C — Corroboration: count of non-negated, independent sources
            non_negated = [e for e in evidence_list if not e.get("is_negated", False)]
            unique_sources = set(
                e.get("source_document", f"src_{i}")
                for i, e in enumerate(non_negated)
            )
            corroboration = len(unique_sources)

            # T — Temporal recency
            temporal = self._compute_temporal_recency(evidence_list)

            # B — Biomedical quality
            quality = self._compute_biomedical_quality(evidence_list)

            # M — Molecular plausibility
            plausibility = self._compute_molecular_plausibility(evidence_list)

            # S — Source type (best available)
            source_type = self._determine_source_type(evidence_list)

            # Aggregate evidence texts
            evidence_texts = [e["evidence_text"] for e in evidence_list if e["evidence_text"]]
            best_evidence = evidence_texts[0] if evidence_texts else ""

            # Check for contradictions (some negated, some not)
            has_negated = any(e.get("is_negated", False) for e in evidence_list)
            has_confirmed = any(not e.get("is_negated", False) for e in evidence_list)
            is_contradicted = has_negated and has_confirmed

            # Average confidence
            avg_confidence = sum(
                e.get("confidence", 0.5) for e in evidence_list
            ) / len(evidence_list)

            metadata = CandidateMetadata(
                entity1_id=e1_id,
                entity1_name=first["entity1_name"],
                entity1_type=first["entity1_type"],
                entity2_id=e2_id,
                entity2_name=first["entity2_name"],
                entity2_type=first["entity2_type"],
                relation_type=rel_type,
                evidence_text=best_evidence,
                corroboration_count=corroboration,
                temporal_recency=temporal,
                biomedical_quality=quality,
                molecular_plausibility=plausibility,
                source_type=source_type,
                source_documents=list(unique_sources),
                extraction_timestamp=time.time(),
                is_negated=has_negated and not has_confirmed,
                is_uncertain=is_contradicted,
                confidence=avg_confidence,
            )

            metadata_list.append(metadata)

        self._built = True
        logger.info(
            f"Built {len(metadata_list)} candidate metadata instances "
            f"from {sum(len(v) for v in self._evidence_store.values())} evidence entries"
        )

        return metadata_list

    def get_statistics(self) -> dict:
        """Get summary statistics of the evidence store."""
        total_evidence = sum(len(v) for v in self._evidence_store.values())
        source_types = {}
        for evidence_list in self._evidence_store.values():
            for ev in evidence_list:
                st = ev.get("source_type", "unknown")
                source_types[st] = source_types.get(st, 0) + 1

        return {
            "unique_interaction_claims": len(self._evidence_store),
            "total_evidence_entries": total_evidence,
            "source_type_distribution": source_types,
            "avg_corroboration": (
                total_evidence / len(self._evidence_store)
                if self._evidence_store
                else 0
            ),
        }
