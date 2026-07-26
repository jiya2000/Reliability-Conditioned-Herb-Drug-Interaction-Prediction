"""
Paper Figure Generator

Generates all figures needed for the research paper:
1. Architecture diagram (conceptual)
2. Reliability calibration plot
3. Ablation bar chart with confidence intervals
4. R-stratified performance heatmap
5. Attention visualization
6. Temporal drift detection example

Outputs to results/figures/ as publication-quality PNGs and SVGs.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Optional

import numpy as np
from loguru import logger


class PaperFigureGenerator:
    """
    Generates publication-quality figures for the HDI paper.

    Usage:
        gen = PaperFigureGenerator(output_dir="results/figures")
        gen.plot_calibration(calibration_data)
        gen.plot_ablation_bars(ablation_results)
        gen.plot_stratified_heatmap(stratified_results)
    """

    def __init__(self, output_dir: str = "results/figures"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        logger.info(f"PaperFigureGenerator: output_dir={self.output_dir}")

    def plot_calibration(
        self,
        calibration_data: dict,
        filename: str = "calibration_plot",
    ) -> Path:
        """
        Plot reliability calibration diagram.

        Shows R-value vs actual prediction accuracy — a perfectly
        calibrated model produces a diagonal line.
        """
        try:
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt

            fig, axes = plt.subplots(1, 2, figsize=(12, 5))

            # Prediction calibration
            if "prediction_calibration" in calibration_data:
                bins = calibration_data["prediction_calibration"]
                confs = [b["avg_confidence"] for b in bins]
                accs = [b["accuracy"] for b in bins]
                counts = [b["count"] for b in bins]

                ax = axes[0]
                ax.bar(confs, accs, width=0.08, alpha=0.7, color="#2196F3",
                       label="Model")
                ax.plot([0, 1], [0, 1], "k--", alpha=0.5, label="Perfect")
                ax.set_xlabel("Confidence", fontsize=12)
                ax.set_ylabel("Accuracy", fontsize=12)
                ax.set_title("Prediction Calibration", fontsize=14)
                ax.legend()
                ax.set_xlim(0, 1)
                ax.set_ylim(0, 1)

            # R calibration
            if "reliability_calibration" in calibration_data:
                bins = calibration_data["reliability_calibration"]
                r_vals = [b["avg_R"] for b in bins]
                corrs = [b["avg_correctness"] for b in bins]

                ax = axes[1]
                ax.bar(r_vals, corrs, width=0.08, alpha=0.7, color="#4CAF50",
                       label="Model")
                ax.plot([0, 1], [0, 1], "k--", alpha=0.5, label="Perfect")
                ax.set_xlabel("Reliability Score R", fontsize=12)
                ax.set_ylabel("Prediction Correctness", fontsize=12)
                ax.set_title("Reliability Calibration (ARC)", fontsize=14)
                ax.legend()
                ax.set_xlim(0, 1)
                ax.set_ylim(0, 1)

            plt.tight_layout()
            path = self.output_dir / f"{filename}.png"
            plt.savefig(path, dpi=300, bbox_inches="tight")
            plt.close()

            logger.info(f"Calibration plot saved to {path}")
            return path

        except ImportError:
            logger.warning("matplotlib not available. Saving data as JSON.")
            path = self.output_dir / f"{filename}.json"
            with open(path, "w") as f:
                json.dump(calibration_data, f, indent=2)
            return path

    def plot_ablation_bars(
        self,
        results: dict,
        metrics: list[str] = None,
        filename: str = "ablation_bars",
    ) -> Path:
        """
        Plot ablation study results as grouped bar chart with CIs.

        Shows (a) GNN Only, (b) Unconditioned, (c) Full Model
        side by side for each metric.
        """
        if metrics is None:
            metrics = ["auc_roc", "f1", "precision", "recall"]

        try:
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt

            variants = list(results.keys())
            n_metrics = len(metrics)
            x = np.arange(n_metrics)
            width = 0.25

            fig, ax = plt.subplots(figsize=(10, 6))

            colors = ["#FF7043", "#42A5F5", "#66BB6A"]
            for i, variant in enumerate(variants):
                values = [results[variant].get(m, 0) for m in metrics]
                bars = ax.bar(
                    x + i * width, values, width,
                    label=variant, color=colors[i % len(colors)],
                    alpha=0.85,
                )
                # Add value labels
                for bar, val in zip(bars, values):
                    ax.text(
                        bar.get_x() + bar.get_width() / 2., bar.get_height(),
                        f"{val:.3f}", ha="center", va="bottom", fontsize=8,
                    )

            ax.set_ylabel("Score", fontsize=12)
            ax.set_title("Ablation Study Results", fontsize=14)
            ax.set_xticks(x + width)
            ax.set_xticklabels([m.replace("_", " ").upper() for m in metrics])
            ax.legend()
            ax.set_ylim(0, 1.1)

            plt.tight_layout()
            path = self.output_dir / f"{filename}.png"
            plt.savefig(path, dpi=300, bbox_inches="tight")
            plt.close()

            logger.info(f"Ablation bar chart saved to {path}")
            return path

        except ImportError:
            path = self.output_dir / f"{filename}.json"
            with open(path, "w") as f:
                json.dump(results, f, indent=2)
            return path

    def plot_stratified_heatmap(
        self,
        stratified_results: dict,
        filename: str = "stratified_heatmap",
    ) -> Path:
        """
        Plot R-stratified performance as a heatmap.

        Shows how metrics vary across reliability strata —
        key evidence for the paper's claim.
        """
        try:
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt

            by_r = stratified_results.get("by_reliability", {})
            strata_names = []
            metrics_data = {"auc_roc": [], "f1": [], "precision": [], "recall": []}

            for name, data in by_r.items():
                if "auc_roc" not in data:
                    continue
                strata_names.append(name)
                for m in metrics_data:
                    metrics_data[m].append(data.get(m, 0))

            if not strata_names:
                logger.warning("No strata data available for heatmap")
                return self.output_dir

            data_matrix = np.array([metrics_data[m] for m in metrics_data])

            fig, ax = plt.subplots(figsize=(10, 4))
            im = ax.imshow(data_matrix, cmap="YlGn", aspect="auto", vmin=0, vmax=1)

            ax.set_xticks(range(len(strata_names)))
            ax.set_xticklabels(
                [s.replace("R_", "").replace("_", "–") for s in strata_names],
                rotation=45, ha="right",
            )
            ax.set_yticks(range(len(metrics_data)))
            ax.set_yticklabels([m.upper() for m in metrics_data])

            # Add text annotations
            for i in range(data_matrix.shape[0]):
                for j in range(data_matrix.shape[1]):
                    ax.text(j, i, f"{data_matrix[i, j]:.3f}",
                            ha="center", va="center", fontsize=10)

            plt.colorbar(im, ax=ax, label="Score")
            ax.set_title("Performance by Reliability Stratum", fontsize=14)
            ax.set_xlabel("Reliability Score Range", fontsize=12)

            plt.tight_layout()
            path = self.output_dir / f"{filename}.png"
            plt.savefig(path, dpi=300, bbox_inches="tight")
            plt.close()

            logger.info(f"Stratified heatmap saved to {path}")
            return path

        except ImportError:
            path = self.output_dir / f"{filename}.json"
            with open(path, "w") as f:
                json.dump(stratified_results, f, indent=2, default=str)
            return path

    def generate_all(
        self,
        calibration_data: Optional[dict] = None,
        ablation_results: Optional[dict] = None,
        stratified_results: Optional[dict] = None,
    ) -> list[Path]:
        """Generate all available figures."""
        paths = []

        if calibration_data:
            paths.append(self.plot_calibration(calibration_data))
        if ablation_results:
            paths.append(self.plot_ablation_bars(ablation_results))
        if stratified_results:
            paths.append(self.plot_stratified_heatmap(stratified_results))

        logger.info(f"Generated {len(paths)} figures in {self.output_dir}")
        return paths
