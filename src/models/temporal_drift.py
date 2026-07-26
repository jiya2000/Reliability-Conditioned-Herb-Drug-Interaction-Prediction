"""
Temporal Drift Detection for Evolving Interactions

★ NOVEL CONTRIBUTION 3 ★

Detects when the evidence landscape for a herb-drug interaction is
changing over time — e.g., older literature says safe but recent
code-mixed forum corroboration says otherwise.

Uses two complementary approaches:
1. CUSUM (Cumulative Sum) for detecting mean shifts in R-score
   distributions over time
2. KL divergence for detecting distributional shifts in evidence
   metadata patterns

This directly addresses the error analysis case from the implementation
plan and provides a mechanism for continuous model monitoring.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import torch
from loguru import logger


@dataclass
class DriftAlert:
    """Alert for a detected distributional shift in interaction evidence."""

    entity1_id: str
    entity2_id: str
    entity1_name: str
    entity2_name: str
    drift_type: str  # "mean_shift", "distributional", "contradiction"
    severity: str  # "critical", "warning", "info"
    description: str
    old_R_mean: float
    new_R_mean: float
    R_change: float
    timestamp: Optional[str] = None
    supporting_evidence: list[str] = field(default_factory=list)


class TemporalDriftDetector:
    """
    Monitors reliability score distributions over time and detects
    significant changes that could indicate evolving evidence landscapes.

    Particularly useful for detecting:
    1. New evidence contradicting established safety profiles
    2. Emerging case reports of previously unknown interactions
    3. Degradation of evidence quality for known interactions

    Usage:
        detector = TemporalDriftDetector(threshold=3.0)

        # Feed observations over time
        for batch in new_evidence_stream:
            alerts = detector.update(batch)
            for alert in alerts:
                if alert.severity == "critical":
                    notify_pharmacovigilance(alert)
    """

    def __init__(
        self,
        cusum_threshold: float = 3.0,
        kl_threshold: float = 0.5,
        window_size: int = 50,
        min_observations: int = 10,
        n_bins: int = 10,
    ):
        """
        Args:
            cusum_threshold: CUSUM detection threshold (in std devs)
            kl_threshold: KL divergence threshold for distributional shift
            window_size: Size of the sliding window for recent observations
            min_observations: Minimum observations before drift detection
            n_bins: Number of bins for KL divergence histogram comparison
        """
        self.cusum_threshold = cusum_threshold
        self.kl_threshold = kl_threshold
        self.window_size = window_size
        self.min_observations = min_observations
        self.n_bins = n_bins

        # Per-interaction tracking
        self._R_history: dict[tuple[str, str], list[float]] = defaultdict(list)
        self._metadata_history: dict[
            tuple[str, str], list[np.ndarray]
        ] = defaultdict(list)
        self._cusum_state: dict[tuple[str, str], dict] = defaultdict(
            lambda: {"S_pos": 0.0, "S_neg": 0.0, "mean": 0.0, "std": 1.0}
        )
        self._alerts: list[DriftAlert] = []

        logger.info(
            f"TemporalDriftDetector: CUSUM threshold={cusum_threshold}, "
            f"KL threshold={kl_threshold}, window={window_size}"
        )

    def update(
        self,
        entity1_id: str,
        entity2_id: str,
        entity1_name: str,
        entity2_name: str,
        reliability_score: float,
        metadata: Optional[np.ndarray] = None,
    ) -> list[DriftAlert]:
        """
        Update tracking with a new observation and check for drift.

        Args:
            entity1_id, entity2_id: Entity pair identifiers
            entity1_name, entity2_name: Human-readable names
            reliability_score: Current R value for this interaction
            metadata: Optional (5,) array of [C, T, B, M, S]

        Returns:
            List of new drift alerts (empty if no drift detected)
        """
        pair_key = tuple(sorted([entity1_id, entity2_id]))
        new_alerts = []

        # Store observation
        self._R_history[pair_key].append(reliability_score)
        if metadata is not None:
            self._metadata_history[pair_key].append(metadata)

        history = self._R_history[pair_key]

        if len(history) < self.min_observations:
            return new_alerts

        # --- CUSUM Test for Mean Shift ---
        cusum_alert = self._cusum_test(
            pair_key, reliability_score,
            entity1_id, entity2_id, entity1_name, entity2_name,
        )
        if cusum_alert:
            new_alerts.append(cusum_alert)

        # --- KL Divergence Test for Distributional Shift ---
        if len(history) >= self.window_size * 2:
            kl_alert = self._kl_divergence_test(
                pair_key,
                entity1_id, entity2_id, entity1_name, entity2_name,
            )
            if kl_alert:
                new_alerts.append(kl_alert)

        # --- Contradiction Detection ---
        if metadata is not None and len(self._metadata_history[pair_key]) >= 5:
            contra_alert = self._contradiction_test(
                pair_key, metadata,
                entity1_id, entity2_id, entity1_name, entity2_name,
            )
            if contra_alert:
                new_alerts.append(contra_alert)

        self._alerts.extend(new_alerts)
        return new_alerts

    def _cusum_test(
        self,
        pair_key: tuple,
        new_value: float,
        e1_id: str, e2_id: str,
        e1_name: str, e2_name: str,
    ) -> Optional[DriftAlert]:
        """
        CUSUM (Cumulative Sum) test for detecting mean shifts.

        Tracks cumulative deviations from the historical mean.
        When the cumulative sum exceeds the threshold, a shift is detected.
        """
        state = self._cusum_state[pair_key]
        history = self._R_history[pair_key]

        # Update running statistics (use first half as baseline)
        baseline = history[: max(len(history) // 2, self.min_observations)]
        state["mean"] = np.mean(baseline)
        state["std"] = max(np.std(baseline), 1e-6)

        # Normalized deviation
        z = (new_value - state["mean"]) / state["std"]

        # Update CUSUM (two-sided)
        state["S_pos"] = max(0, state["S_pos"] + z - 0.5)
        state["S_neg"] = max(0, state["S_neg"] - z - 0.5)

        # Check for alarm
        if state["S_pos"] > self.cusum_threshold:
            # Upward shift detected
            recent_mean = np.mean(history[-self.min_observations:])
            alert = DriftAlert(
                entity1_id=e1_id, entity2_id=e2_id,
                entity1_name=e1_name, entity2_name=e2_name,
                drift_type="mean_shift",
                severity="warning" if state["S_pos"] < 2 * self.cusum_threshold else "critical",
                description=(
                    f"Upward shift in reliability scores detected for "
                    f"{e1_name} ↔ {e2_name}. Evidence quality may be "
                    f"improving (new clinical data?)."
                ),
                old_R_mean=state["mean"],
                new_R_mean=recent_mean,
                R_change=recent_mean - state["mean"],
            )
            # Reset CUSUM after alarm
            state["S_pos"] = 0
            return alert

        if state["S_neg"] > self.cusum_threshold:
            # Downward shift detected
            recent_mean = np.mean(history[-self.min_observations:])
            alert = DriftAlert(
                entity1_id=e1_id, entity2_id=e2_id,
                entity1_name=e1_name, entity2_name=e2_name,
                drift_type="mean_shift",
                severity="critical",
                description=(
                    f"⚠️ Downward shift in reliability scores for "
                    f"{e1_name} ↔ {e2_name}. Recent evidence may be "
                    f"of lower quality or contradictory."
                ),
                old_R_mean=state["mean"],
                new_R_mean=recent_mean,
                R_change=recent_mean - state["mean"],
            )
            state["S_neg"] = 0
            return alert

        return None

    def _kl_divergence_test(
        self,
        pair_key: tuple,
        e1_id: str, e2_id: str,
        e1_name: str, e2_name: str,
    ) -> Optional[DriftAlert]:
        """
        KL divergence test for distributional shifts.

        Compares the distribution of R-scores in the older window
        vs the recent window.
        """
        history = self._R_history[pair_key]
        old_window = history[-2 * self.window_size:-self.window_size]
        new_window = history[-self.window_size:]

        # Compute histograms
        bins = np.linspace(0, 1, self.n_bins + 1)
        old_hist, _ = np.histogram(old_window, bins=bins, density=True)
        new_hist, _ = np.histogram(new_window, bins=bins, density=True)

        # Add smoothing to prevent division by zero
        eps = 1e-8
        old_hist = old_hist + eps
        new_hist = new_hist + eps

        # Normalize
        old_hist = old_hist / old_hist.sum()
        new_hist = new_hist / new_hist.sum()

        # Symmetric KL divergence (Jensen-Shannon divergence)
        m = 0.5 * (old_hist + new_hist)
        kl_old_m = np.sum(old_hist * np.log(old_hist / m))
        kl_new_m = np.sum(new_hist * np.log(new_hist / m))
        jsd = 0.5 * (kl_old_m + kl_new_m)

        if jsd > self.kl_threshold:
            return DriftAlert(
                entity1_id=e1_id, entity2_id=e2_id,
                entity1_name=e1_name, entity2_name=e2_name,
                drift_type="distributional",
                severity="warning",
                description=(
                    f"Distributional shift detected in R-scores for "
                    f"{e1_name} ↔ {e2_name} (JSD={jsd:.3f}). "
                    f"The evidence landscape is changing."
                ),
                old_R_mean=float(np.mean(old_window)),
                new_R_mean=float(np.mean(new_window)),
                R_change=float(np.mean(new_window) - np.mean(old_window)),
            )

        return None

    def _contradiction_test(
        self,
        pair_key: tuple,
        latest_metadata: np.ndarray,
        e1_id: str, e2_id: str,
        e1_name: str, e2_name: str,
    ) -> Optional[DriftAlert]:
        """
        Detect contradictions in evidence — when high-quality recent
        evidence contradicts established evidence patterns.

        This is the specific case from the implementation plan:
        "older literature says safe, recent code-mixed forum
        corroboration says otherwise."
        """
        meta_history = self._metadata_history[pair_key]
        r_history = self._R_history[pair_key]

        if len(meta_history) < 5:
            return None

        recent_meta = np.array(meta_history[-5:])
        old_meta = np.array(meta_history[:-5])

        if len(old_meta) < 3:
            return None

        # Check: are recent sources from lower-quality channels
        # but showing HIGHER R-scores than older high-quality sources?
        recent_source_quality = recent_meta[:, 2].mean()  # B dimension
        old_source_quality = old_meta[:, 2].mean()

        recent_R = np.mean(r_history[-5:])
        old_R = np.mean(r_history[:-5])

        # Contradiction: recent low-quality sources yielding higher R
        # than established high-quality sources
        if (
            recent_source_quality < old_source_quality - 0.2
            and recent_R > old_R + 0.15
        ):
            return DriftAlert(
                entity1_id=e1_id, entity2_id=e2_id,
                entity1_name=e1_name, entity2_name=e2_name,
                drift_type="contradiction",
                severity="critical",
                description=(
                    f"⚠️ CONTRADICTION: Recent lower-quality evidence for "
                    f"{e1_name} ↔ {e2_name} suggests stronger interaction "
                    f"than established high-quality sources. "
                    f"Old B={old_source_quality:.2f}/R={old_R:.2f} vs "
                    f"Recent B={recent_source_quality:.2f}/R={recent_R:.2f}. "
                    f"Manual review recommended."
                ),
                old_R_mean=old_R,
                new_R_mean=recent_R,
                R_change=recent_R - old_R,
                supporting_evidence=[
                    f"Old evidence quality: {old_source_quality:.2f}",
                    f"Recent evidence quality: {recent_source_quality:.2f}",
                    f"R-score increase: {recent_R - old_R:+.2f}",
                ],
            )

        return None

    def get_interaction_summary(
        self, entity1_id: str, entity2_id: str
    ) -> dict:
        """Get a summary of drift monitoring for a specific interaction."""
        pair_key = tuple(sorted([entity1_id, entity2_id]))
        history = self._R_history.get(pair_key, [])

        if not history:
            return {"status": "no_data", "observations": 0}

        return {
            "status": "monitoring",
            "observations": len(history),
            "current_R_mean": float(np.mean(history[-10:])),
            "overall_R_mean": float(np.mean(history)),
            "R_std": float(np.std(history)),
            "R_trend": float(np.polyfit(range(len(history)), history, 1)[0])
            if len(history) > 2
            else 0.0,
            "alerts_triggered": sum(
                1
                for a in self._alerts
                if sorted([a.entity1_id, a.entity2_id])
                == list(pair_key)
            ),
        }

    def get_all_alerts(
        self, severity: Optional[str] = None
    ) -> list[DriftAlert]:
        """Get all alerts, optionally filtered by severity."""
        if severity:
            return [a for a in self._alerts if a.severity == severity]
        return self._alerts.copy()

    def reset(self, entity1_id: str = None, entity2_id: str = None) -> None:
        """Reset tracking state."""
        if entity1_id and entity2_id:
            pair_key = tuple(sorted([entity1_id, entity2_id]))
            self._R_history.pop(pair_key, None)
            self._metadata_history.pop(pair_key, None)
            self._cusum_state.pop(pair_key, None)
        else:
            self._R_history.clear()
            self._metadata_history.clear()
            self._cusum_state.clear()
            self._alerts.clear()
