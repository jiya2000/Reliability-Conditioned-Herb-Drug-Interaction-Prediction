"""
Evidence Surfacer

For every predicted interaction edge, surfaces:
1. R-score breakdown (which metadata dimension drove the trust decision)
2. Evidence spans (the textual evidence supporting the prediction)
3. Confidence indicators and risk assessment

This is the explainability layer that makes predictions interpretable
for pharmacologists and clinicians.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

from loguru import logger


@dataclass
class InteractionExplanation:
    """Complete explanation for a predicted herb-drug interaction."""

    # Entities
    entity1_name: str
    entity1_type: str
    entity2_name: str
    entity2_type: str

    # Prediction
    interaction_probability: float
    risk_level: str  # "high", "moderate", "low", "minimal"

    # Reliability breakdown
    reliability_score: float  # R ∈ [0, 1]
    reliability_breakdown: dict = field(default_factory=dict)
    # Keys: corroboration_weight, temporal_weight, biomedical_weight,
    #        molecular_weight, source_type_weight

    # Evidence
    evidence_spans: list[str] = field(default_factory=list)
    evidence_sources: list[str] = field(default_factory=list)

    # Raw metadata values
    corroboration_count: int = 0
    temporal_recency: float = 0.0
    biomedical_quality: float = 0.0
    molecular_plausibility: float = 0.0
    source_type: str = ""

    # Interpretive text
    explanation_text: str = ""
    recommendations: list[str] = field(default_factory=list)


class EvidenceSurfacer:
    """
    Surfaces human-readable explanations for predicted interactions.

    Takes model outputs (probabilities, R-scores, attention weights,
    metadata) and generates structured explanations.

    Usage:
        surfacer = EvidenceSurfacer()
        explanation = surfacer.explain(
            entity1="Warfarin",
            entity2="St. John's Wort",
            model_output=output,
            metadata=metadata,
        )
        print(explanation.explanation_text)
    """

    RISK_THRESHOLDS = {
        "high": 0.8,
        "moderate": 0.5,
        "low": 0.3,
        "minimal": 0.0,
    }

    def __init__(self):
        logger.info("EvidenceSurfacer initialized")

    def explain(
        self,
        entity1_name: str,
        entity1_type: str,
        entity2_name: str,
        entity2_type: str,
        interaction_probability: float,
        reliability_score: float,
        reliability_breakdown: Optional[dict] = None,
        evidence_texts: Optional[list[str]] = None,
        evidence_sources: Optional[list[str]] = None,
        metadata_values: Optional[dict] = None,
    ) -> InteractionExplanation:
        """
        Generate a complete explanation for a predicted interaction.

        Args:
            entity1_name, entity1_type: First entity info
            entity2_name, entity2_type: Second entity info
            interaction_probability: Model's predicted probability
            reliability_score: R score from reliability scorer
            reliability_breakdown: Per-dimension contribution weights
            evidence_texts: Supporting text evidence
            evidence_sources: Source labels for evidence
            metadata_values: Raw C, T, B, M, S values

        Returns:
            InteractionExplanation with full interpretive details
        """
        risk_level = self._assess_risk(
            interaction_probability, reliability_score
        )

        breakdown = reliability_breakdown or {}
        meta = metadata_values or {}

        explanation_text = self._generate_explanation(
            entity1_name, entity1_type,
            entity2_name, entity2_type,
            interaction_probability, reliability_score,
            risk_level, breakdown, meta,
        )

        recommendations = self._generate_recommendations(
            risk_level, reliability_score,
            entity1_type, entity2_type,
        )

        return InteractionExplanation(
            entity1_name=entity1_name,
            entity1_type=entity1_type,
            entity2_name=entity2_name,
            entity2_type=entity2_type,
            interaction_probability=interaction_probability,
            risk_level=risk_level,
            reliability_score=reliability_score,
            reliability_breakdown=breakdown,
            evidence_spans=evidence_texts or [],
            evidence_sources=evidence_sources or [],
            corroboration_count=int(meta.get("C", 0)),
            temporal_recency=float(meta.get("T", 0)),
            biomedical_quality=float(meta.get("B", 0)),
            molecular_plausibility=float(meta.get("M", 0)),
            source_type=str(meta.get("S", "")),
            explanation_text=explanation_text,
            recommendations=recommendations,
        )

    def _assess_risk(
        self,
        probability: float,
        reliability: float,
    ) -> str:
        """
        Assess risk level from probability and reliability.

        High probability + high reliability = high risk
        High probability + low reliability = moderate (uncertain)
        Low probability = low/minimal regardless of reliability
        """
        # Combined risk score
        risk_score = probability * (0.5 + 0.5 * reliability)

        for level, threshold in self.RISK_THRESHOLDS.items():
            if risk_score >= threshold:
                return level

        return "minimal"

    def _generate_explanation(
        self,
        e1_name: str,
        e1_type: str,
        e2_name: str,
        e2_type: str,
        probability: float,
        reliability: float,
        risk_level: str,
        breakdown: dict,
        metadata: dict,
    ) -> str:
        """Generate human-readable explanation text."""
        lines = []

        # Summary
        lines.append(
            f"**Interaction Prediction: {e1_name} ({e1_type}) ↔ "
            f"{e2_name} ({e2_type})**"
        )
        lines.append("")
        lines.append(
            f"Predicted interaction probability: "
            f"{probability:.1%} (Risk: {risk_level.upper()})"
        )
        lines.append(
            f"Evidence reliability score: {reliability:.2f}/1.00"
        )
        lines.append("")

        # Reliability breakdown
        lines.append("**Reliability Score Breakdown:**")
        if breakdown:
            dimensions = [
                ("Corroboration", "corroboration_weight", "C"),
                ("Temporal Recency", "temporal_weight", "T"),
                ("Biomedical Quality", "biomedical_weight", "B"),
                ("Molecular Plausibility", "molecular_weight", "M"),
                ("Source Type", "source_type_weight", "S"),
            ]
            for dim_name, weight_key, meta_key in dimensions:
                weight = breakdown.get(weight_key, 0)
                raw_value = metadata.get(meta_key, "N/A")

                if isinstance(weight, (int, float)):
                    bar = "█" * int(weight * 10) + "░" * (10 - int(weight * 10))
                    lines.append(
                        f"  {dim_name:<25}: [{bar}] "
                        f"{weight:.2f} (raw: {raw_value})"
                    )
        else:
            lines.append("  No breakdown available")

        lines.append("")

        # Risk interpretation
        if risk_level == "high":
            lines.append(
                "⚠️ **HIGH RISK**: Strong evidence of a clinically "
                "significant interaction. Recommend clinical review."
            )
        elif risk_level == "moderate":
            lines.append(
                "⚡ **MODERATE RISK**: Evidence suggests a possible "
                "interaction. Consider monitoring."
            )
        elif risk_level == "low":
            lines.append(
                "ℹ️ **LOW RISK**: Limited evidence of interaction. "
                "Unlikely to be clinically significant."
            )
        else:
            lines.append(
                "✅ **MINIMAL RISK**: No significant evidence of "
                "interaction found."
            )

        return "\n".join(lines)

    def _generate_recommendations(
        self,
        risk_level: str,
        reliability: float,
        e1_type: str,
        e2_type: str,
    ) -> list[str]:
        """Generate actionable recommendations."""
        recommendations = []

        if risk_level in ("high", "moderate"):
            recommendations.append(
                "Consult with a pharmacist or healthcare provider "
                "before combining these substances."
            )

            if "herb" in (e1_type.lower(), e2_type.lower()):
                recommendations.append(
                    "Inform your doctor about all herbal supplements "
                    "you are currently taking."
                )

            if risk_level == "high":
                recommendations.append(
                    "Consider therapeutic drug monitoring if both "
                    "substances are being used concurrently."
                )

        if reliability < 0.5:
            recommendations.append(
                "Note: Evidence reliability is low. This prediction "
                "should be validated with clinical literature."
            )

        if not recommendations:
            recommendations.append(
                "No specific clinical actions recommended based "
                "on current evidence."
            )

        return recommendations

    def batch_explain(
        self,
        predictions: list[dict],
    ) -> list[InteractionExplanation]:
        """Generate explanations for a batch of predictions."""
        explanations = []
        for pred in predictions:
            explanation = self.explain(
                entity1_name=pred.get("entity1_name", ""),
                entity1_type=pred.get("entity1_type", ""),
                entity2_name=pred.get("entity2_name", ""),
                entity2_type=pred.get("entity2_type", ""),
                interaction_probability=pred.get("probability", 0),
                reliability_score=pred.get("reliability_score", 0),
                reliability_breakdown=pred.get("reliability_breakdown"),
                evidence_texts=pred.get("evidence_texts"),
                metadata_values=pred.get("metadata"),
            )
            explanations.append(explanation)
        return explanations
