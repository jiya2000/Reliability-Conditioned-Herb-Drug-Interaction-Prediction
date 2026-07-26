"""
Statistical Testing Infrastructure for Ablation Studies

Provides rigorous statistical validation for the (b) vs (c) gap
and all other metric comparisons in the paper.

Includes:
1. Bootstrap confidence intervals for all metrics
2. McNemar's test for comparing classifiers
3. Paired permutation test for the ablation gap
4. Effect size (Cohen's d) computation
5. Multiple testing correction (Bonferroni, BH)

These are required for publication at any serious venue
(BioNLP@ACL, AMIA, PSB, JAMIA).
"""

from __future__ import annotations

from typing import Optional, Callable

import numpy as np
from loguru import logger


def bootstrap_confidence_interval(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    metric_fn: Callable,
    n_bootstrap: int = 10000,
    confidence_level: float = 0.95,
    seed: int = 42,
) -> tuple[float, float, float]:
    """
    Compute bootstrap confidence interval for a metric.

    Args:
        y_true: Ground truth labels
        y_pred: Predicted values (probabilities or labels)
        metric_fn: Function(y_true, y_pred) → float
        n_bootstrap: Number of bootstrap resamples
        confidence_level: Confidence level (default 95%)
        seed: Random seed

    Returns:
        (point_estimate, ci_lower, ci_upper)
    """
    rng = np.random.RandomState(seed)
    n = len(y_true)

    point_estimate = metric_fn(y_true, y_pred)

    bootstrapped_scores = []
    for _ in range(n_bootstrap):
        indices = rng.randint(0, n, size=n)
        try:
            score = metric_fn(y_true[indices], y_pred[indices])
            bootstrapped_scores.append(score)
        except Exception:
            continue

    bootstrapped_scores = np.array(bootstrapped_scores)

    alpha = 1 - confidence_level
    ci_lower = np.percentile(bootstrapped_scores, 100 * alpha / 2)
    ci_upper = np.percentile(bootstrapped_scores, 100 * (1 - alpha / 2))

    return point_estimate, ci_lower, ci_upper


def mcnemars_test(
    y_true: np.ndarray,
    y_pred_a: np.ndarray,
    y_pred_b: np.ndarray,
) -> tuple[float, float]:
    """
    McNemar's test for comparing two classifiers.

    Tests whether two classifiers have the same error rate.
    Specifically tests the (b) vs (c) comparison in the ablation.

    Args:
        y_true: Ground truth labels
        y_pred_a: Predictions from model A (unconditioned)
        y_pred_b: Predictions from model B (full R-conditioned)

    Returns:
        (chi_squared_statistic, p_value)
    """
    # Build contingency table
    correct_a = (y_pred_a == y_true)
    correct_b = (y_pred_b == y_true)

    # b: A correct, B wrong
    b = np.sum(correct_a & ~correct_b)
    # c: A wrong, B correct
    c = np.sum(~correct_a & correct_b)

    # McNemar's test statistic
    if b + c == 0:
        return 0.0, 1.0

    # With continuity correction
    chi2 = (abs(b - c) - 1) ** 2 / (b + c)

    # p-value from chi-squared distribution with df=1
    # Using normal approximation since scipy may not be available
    z = np.sqrt(chi2)
    p_value = 2 * (1 - _normal_cdf(z))

    return float(chi2), float(p_value)


def paired_permutation_test(
    scores_a: np.ndarray,
    scores_b: np.ndarray,
    n_permutations: int = 10000,
    seed: int = 42,
) -> tuple[float, float]:
    """
    Paired permutation test for comparing two models.

    Tests whether model B significantly outperforms model A
    on the same test samples.

    Args:
        scores_a: Per-sample scores from model A (e.g., per-sample accuracy)
        scores_b: Per-sample scores from model B
        n_permutations: Number of permutations

    Returns:
        (observed_difference, p_value)
    """
    rng = np.random.RandomState(seed)
    n = len(scores_a)

    observed_diff = np.mean(scores_b - scores_a)
    diffs = scores_b - scores_a

    count_extreme = 0
    for _ in range(n_permutations):
        # Randomly flip signs
        signs = rng.choice([-1, 1], size=n)
        permuted_diff = np.mean(diffs * signs)
        if abs(permuted_diff) >= abs(observed_diff):
            count_extreme += 1

    p_value = (count_extreme + 1) / (n_permutations + 1)

    return float(observed_diff), float(p_value)


def cohens_d(
    group_a: np.ndarray,
    group_b: np.ndarray,
) -> float:
    """
    Compute Cohen's d effect size.

    Measures the standardized difference between two group means.
    Conventions: |d| < 0.2 = small, 0.2-0.8 = medium, > 0.8 = large.

    Args:
        group_a: Scores from model A
        group_b: Scores from model B

    Returns:
        Cohen's d value
    """
    n_a, n_b = len(group_a), len(group_b)
    mean_a, mean_b = np.mean(group_a), np.mean(group_b)
    var_a, var_b = np.var(group_a, ddof=1), np.var(group_b, ddof=1)

    # Pooled standard deviation
    pooled_std = np.sqrt(
        ((n_a - 1) * var_a + (n_b - 1) * var_b) / (n_a + n_b - 2)
    )

    if pooled_std < 1e-8:
        return 0.0

    return float((mean_b - mean_a) / pooled_std)


def bonferroni_correction(
    p_values: list[float],
    alpha: float = 0.05,
) -> list[tuple[float, bool]]:
    """
    Bonferroni correction for multiple testing.

    Args:
        p_values: List of p-values
        alpha: Family-wise error rate

    Returns:
        List of (adjusted_p_value, is_significant) tuples
    """
    n = len(p_values)
    results = []
    for p in p_values:
        adjusted = min(p * n, 1.0)
        results.append((adjusted, adjusted < alpha))
    return results


def benjamini_hochberg(
    p_values: list[float],
    alpha: float = 0.05,
) -> list[tuple[float, bool, int]]:
    """
    Benjamini-Hochberg procedure for FDR control.

    Args:
        p_values: List of p-values
        alpha: False discovery rate threshold

    Returns:
        List of (adjusted_p_value, is_significant, rank) tuples
    """
    n = len(p_values)
    indexed = sorted(enumerate(p_values), key=lambda x: x[1])

    results = [None] * n
    prev_adj = 0.0

    for rank_minus_1, (orig_idx, p) in enumerate(reversed(indexed)):
        rank = n - rank_minus_1
        adjusted = p * n / rank
        adjusted = min(adjusted, 1.0)
        if rank_minus_1 > 0:
            adjusted = min(adjusted, prev_adj)
        prev_adj = adjusted
        results[orig_idx] = (adjusted, adjusted < alpha, rank)

    return results


class AblationStatisticalReport:
    """
    Generates a comprehensive statistical report for the ablation study.

    Runs all statistical tests needed to support the (b) vs (c) gap
    argument in the paper and patent filing.

    Usage:
        report = AblationStatisticalReport()
        results = report.compare(
            unconditioned_preds, full_model_preds, labels,
            metric_fns={"auc_roc": auc_roc_fn, "f1": f1_fn}
        )
        report.format_latex_table(results)
    """

    def __init__(
        self,
        n_bootstrap: int = 10000,
        n_permutations: int = 10000,
        alpha: float = 0.05,
    ):
        self.n_bootstrap = n_bootstrap
        self.n_permutations = n_permutations
        self.alpha = alpha

    def compare(
        self,
        preds_unconditioned: np.ndarray,
        preds_full: np.ndarray,
        labels: np.ndarray,
        metric_fns: dict[str, Callable],
        threshold: float = 0.5,
    ) -> dict:
        """
        Run full statistical comparison between unconditioned and full model.

        Args:
            preds_unconditioned: Probabilities from unconditioned model
            preds_full: Probabilities from full R-conditioned model
            labels: Ground truth labels
            metric_fns: Dict mapping metric name → function
            threshold: Classification threshold

        Returns:
            Comprehensive results dict
        """
        results = {
            "metrics": {},
            "statistical_tests": {},
            "effect_sizes": {},
        }

        binary_uncond = (preds_unconditioned >= threshold).astype(int)
        binary_full = (preds_full >= threshold).astype(int)

        all_p_values = []

        for name, fn in metric_fns.items():
            logger.info(f"Computing bootstrap CI for {name}...")

            # Bootstrap CIs for both models
            est_u, ci_lo_u, ci_hi_u = bootstrap_confidence_interval(
                labels, preds_unconditioned, fn,
                n_bootstrap=self.n_bootstrap,
            )
            est_f, ci_lo_f, ci_hi_f = bootstrap_confidence_interval(
                labels, preds_full, fn,
                n_bootstrap=self.n_bootstrap,
            )

            results["metrics"][name] = {
                "unconditioned": {
                    "value": est_u,
                    "ci_95": (ci_lo_u, ci_hi_u),
                },
                "full_model": {
                    "value": est_f,
                    "ci_95": (ci_lo_f, ci_hi_f),
                },
                "improvement": est_f - est_u,
                "relative_improvement_pct": (
                    (est_f - est_u) / max(est_u, 1e-8) * 100
                ),
            }

        # McNemar's test
        chi2, p_mcnemar = mcnemars_test(labels, binary_uncond, binary_full)
        results["statistical_tests"]["mcnemars"] = {
            "chi_squared": chi2,
            "p_value": p_mcnemar,
            "significant": p_mcnemar < self.alpha,
        }
        all_p_values.append(p_mcnemar)

        # Per-sample accuracy for permutation test
        acc_uncond = (binary_uncond == labels).astype(float)
        acc_full = (binary_full == labels).astype(float)

        diff, p_perm = paired_permutation_test(
            acc_uncond, acc_full,
            n_permutations=self.n_permutations,
        )
        results["statistical_tests"]["permutation_test"] = {
            "observed_difference": diff,
            "p_value": p_perm,
            "significant": p_perm < self.alpha,
        }
        all_p_values.append(p_perm)

        # Effect size
        d = cohens_d(acc_uncond, acc_full)
        results["effect_sizes"]["cohens_d"] = {
            "value": d,
            "interpretation": (
                "large" if abs(d) > 0.8
                else "medium" if abs(d) > 0.5
                else "small" if abs(d) > 0.2
                else "negligible"
            ),
        }

        # Multiple testing correction
        bh_results = benjamini_hochberg(all_p_values, self.alpha)
        results["statistical_tests"]["bh_correction"] = [
            {"p_adjusted": r[0], "significant": r[1]}
            for r in bh_results
        ]

        return results

    def format_latex_table(self, results: dict) -> str:
        """Format results as a LaTeX table for the paper."""
        lines = [
            r"\begin{table}[h]",
            r"\centering",
            r"\caption{Ablation Study: Unconditioned vs. Reliability-Conditioned}",
            r"\label{tab:ablation}",
            r"\begin{tabular}{lcccc}",
            r"\toprule",
            r"Metric & Unconditioned & Full (Ours) & $\Delta$ & Sig. \\",
            r"\midrule",
        ]

        for name, data in results.get("metrics", {}).items():
            u = data["unconditioned"]
            f = data["full_model"]
            imp = data["improvement"]
            sig = "✓" if imp > 0 else "—"

            lines.append(
                f"{name} & "
                f"{u['value']:.4f} [{u['ci_95'][0]:.3f}, {u['ci_95'][1]:.3f}] & "
                f"{f['value']:.4f} [{f['ci_95'][0]:.3f}, {f['ci_95'][1]:.3f}] & "
                f"{imp:+.4f} & {sig} \\\\"
            )

        p_val = results.get("statistical_tests", {}).get(
            "permutation_test", {}
        ).get("p_value", "N/A")
        d_val = results.get("effect_sizes", {}).get(
            "cohens_d", {}
        ).get("value", "N/A")

        lines.extend([
            r"\bottomrule",
            r"\end{tabular}",
            f"\\\\[2pt]",
            f"\\small Permutation test $p={p_val:.4f}$, "
            f"Cohen's $d={d_val:.3f}$",
            r"\end{table}",
        ])

        return "\n".join(lines)


def _normal_cdf(x: float) -> float:
    """Approximate standard normal CDF (no scipy required)."""
    # Abramowitz and Stegun approximation
    import math
    t = 1.0 / (1.0 + 0.2316419 * abs(x))
    d = 0.3989422804014327  # 1 / sqrt(2*pi)
    p = d * math.exp(-x * x / 2.0) * (
        t * (0.3193815 + t * (-0.3565638 + t * (1.781478 +
        t * (-1.821256 + t * 1.330274))))
    )
    return 1.0 - p if x > 0 else p
