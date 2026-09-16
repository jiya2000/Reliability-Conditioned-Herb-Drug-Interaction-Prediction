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


def generate_all_figures(full_results: dict, output_dir: str = "results/figures"):
    """
    Standalone entry point for generating all paper figures from saved results.

    Called by:
        - main.py figures --results-file results/full_results.json
        - run_experiments.py (after training completes)

    Args:
        full_results: Dict loaded from results/full_results.json
        output_dir: Directory to save figures
    """
    gen = PaperFigureGenerator(output_dir=output_dir)

    per_run = full_results.get("per_run", {})

    # 1. Ablation bar chart from evaluation metrics
    ablation_data = {}
    for vname, vdata in per_run.items():
        runs = vdata.get("runs", [])
        if runs and runs[0].get("evaluation"):
            ablation_data[vname] = runs[0]["evaluation"]

    if ablation_data:
        gen.plot_ablation_bars(ablation_data)

    # 2. Calibration plots from full model
    full_runs = per_run.get("full_model", {}).get("runs", [])
    if full_runs:
        calibration = full_runs[0].get("calibration")
        if calibration:
            gen.plot_calibration(calibration)

    # 3. Loss curves
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        colors = {"gnn_only": "#FF7043", "unconditioned": "#42A5F5", "full_model": "#66BB6A"}
        labels = {"gnn_only": "(a) GNN-only", "unconditioned": "(b) Unconditioned", "full_model": "(c) Full (Ours)"}

        for vname, vdata in per_run.items():
            runs = vdata.get("runs", [])
            if not runs:
                continue
            run = runs[0]
            color = colors.get(vname, "gray")
            label = labels.get(vname, vname)

            train_loss = run.get("train_loss_history", [])
            val_loss = run.get("val_loss_history", [])

            if train_loss:
                axes[0].plot(range(1, len(train_loss) + 1), train_loss,
                             label=label, color=color, linewidth=1.5)
            if val_loss:
                axes[1].plot(range(1, len(val_loss) + 1), val_loss,
                             label=label, color=color, linewidth=1.5)

        for ax, title in zip(axes, ["Training Loss", "Validation Loss"]):
            ax.set_xlabel("Epoch")
            ax.set_ylabel("Loss")
            ax.set_title(title)
            ax.legend()
            ax.grid(alpha=0.3)

        fig.tight_layout()
        out_path = Path(output_dir)
        fig.savefig(out_path / "loss_curves.png", dpi=300, bbox_inches="tight")
        fig.savefig(out_path / "loss_curves.svg", bbox_inches="tight")
        plt.close(fig)
        logger.info("Generated loss_curves.png/svg")

    except Exception as e:
        logger.warning(f"Could not generate loss curves: {e}")

    # 4. R-score distribution (if full model has stratified data)
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        full_runs = per_run.get("full_model", {}).get("runs", [])
        if full_runs:
            stratified = full_runs[0].get("stratified", {})
            by_r = stratified.get("by_reliability", {})

            if by_r:
                fig, ax = plt.subplots(figsize=(8, 5))
                strata = []
                n_samples = []
                auc_vals = []

                for sname, sdata in sorted(by_r.items()):
                    if isinstance(sdata, dict) and "auc_roc" in sdata:
                        strata.append(sname.replace("R_", "").replace("_", "–"))
                        n_samples.append(sdata.get("n_samples", 0))
                        auc_vals.append(sdata.get("auc_roc", 0))

                if strata:
                    x = np.arange(len(strata))
                    ax2 = ax.twinx()

                    bars = ax.bar(x, n_samples, alpha=0.6, color="#90CAF9",
                                  label="Sample count")
                    line = ax2.plot(x, auc_vals, "o-", color="#E53935",
                                   linewidth=2, markersize=8,
                                   label="AUC-ROC")

                    ax.set_xlabel("Reliability Score Range", fontsize=12)
                    ax.set_ylabel("Number of Samples", fontsize=12)
                    ax2.set_ylabel("AUC-ROC", fontsize=12)
                    ax.set_xticks(x)
                    ax.set_xticklabels(strata)
                    ax.set_title("Performance by Reliability Stratum", fontsize=14)

                    lines1, labels1 = ax.get_legend_handles_labels()
                    lines2, labels2 = ax2.get_legend_handles_labels()
                    ax.legend(lines1 + lines2, labels1 + labels2, loc="lower right")

                    fig.tight_layout()
                    out_path = Path(output_dir)
                    fig.savefig(out_path / "r_score_distribution.png",
                                dpi=300, bbox_inches="tight")
                    plt.close(fig)
                    logger.info("Generated r_score_distribution.png")

    except Exception as e:
        logger.warning(f"Could not generate R-score distribution: {e}")

    # 5. Source-type performance comparison
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        # Compare source-type performance across variants
        fig, ax = plt.subplots(figsize=(12, 6))
        colors = {"gnn_only": "#FF7043", "unconditioned": "#42A5F5",
                  "full_model": "#66BB6A"}
        labels_map = {"gnn_only": "(a) GNN-only",
                      "unconditioned": "(b) Unconditioned",
                      "full_model": "(c) Full (Ours)"}

        all_sources = set()
        variant_source_data = {}

        for vname, vdata in per_run.items():
            runs = vdata.get("runs", [])
            if not runs:
                continue
            by_source = runs[0].get("stratified", {}).get("by_source", {})
            variant_source_data[vname] = by_source
            all_sources.update(by_source.keys())

        source_names = sorted(all_sources)
        if source_names:
            x = np.arange(len(source_names))
            width = 0.25

            for i, (vname, by_source) in enumerate(variant_source_data.items()):
                vals = [by_source.get(s, {}).get("auc_roc", 0) for s in source_names]
                ax.bar(x + i * width, vals, width,
                       label=labels_map.get(vname, vname),
                       color=colors.get(vname, "gray"), alpha=0.85)

            ax.set_xlabel("Evidence Source Type", fontsize=12)
            ax.set_ylabel("AUC-ROC", fontsize=12)
            ax.set_title("Performance by Evidence Source Type", fontsize=14)
            ax.set_xticks(x + width)
            ax.set_xticklabels([s.replace("_", " ").title() for s in source_names],
                               rotation=45, ha="right")
            ax.legend()
            ax.set_ylim(0, 1.1)
            ax.grid(axis="y", alpha=0.3)

            fig.tight_layout()
            out_path = Path(output_dir)
            fig.savefig(out_path / "source_type_performance.png",
                        dpi=300, bbox_inches="tight")
            fig.savefig(out_path / "source_type_performance.svg",
                        bbox_inches="tight")
            plt.close(fig)
            logger.info("Generated source_type_performance.png/svg")

    except Exception as e:
        logger.warning(f"Could not generate source-type comparison: {e}")

    # 6. KG statistics summary
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        metadata = full_results.get("metadata", {})
        aggregated = full_results.get("aggregated", {})

        fig, axes = plt.subplots(1, 3, figsize=(15, 5))

        # Parameter counts
        param_counts = {}
        for vname, vdata in aggregated.items():
            total = vdata.get("metrics", {}).get("total_parameters", 0)
            if total > 0:
                param_counts[labels_map.get(vname, vname)] = total

        if param_counts:
            ax = axes[0]
            bars = ax.barh(list(param_counts.keys()), list(param_counts.values()),
                           color=["#FF7043", "#42A5F5", "#66BB6A"])
            ax.set_xlabel("Parameters")
            ax.set_title("Model Size Comparison")
            for bar, val in zip(bars, param_counts.values()):
                ax.text(bar.get_width() + 1000, bar.get_y() + bar.get_height()/2,
                        f"{val:,}", va="center", fontsize=9)

        # Training time comparison
        time_data = {}
        for vname, vdata in aggregated.items():
            t = vdata.get("metrics", {}).get("training_time_mean", 0)
            if t > 0:
                time_data[labels_map.get(vname, vname)] = t

        if time_data:
            ax = axes[1]
            ax.barh(list(time_data.keys()), list(time_data.values()),
                    color=["#FF7043", "#42A5F5", "#66BB6A"])
            ax.set_xlabel("Training Time (s)")
            ax.set_title("Training Time Comparison")

        # AUC-ROC comparison (main result)
        auc_data = {}
        auc_std = {}
        for vname, vdata in aggregated.items():
            m = vdata.get("metrics", {}).get("auc_roc", {})
            if isinstance(m, dict):
                auc_data[labels_map.get(vname, vname)] = m.get("mean", 0)
                auc_std[labels_map.get(vname, vname)] = m.get("std", 0)

        if auc_data:
            ax = axes[2]
            names = list(auc_data.keys())
            vals = list(auc_data.values())
            errs = [auc_std.get(n, 0) for n in names]
            bars = ax.barh(names, vals, xerr=errs, capsize=5,
                           color=["#FF7043", "#42A5F5", "#66BB6A"], alpha=0.85)
            ax.set_xlabel("AUC-ROC")
            ax.set_title("AUC-ROC (mean ± std)")
            ax.set_xlim(0, 1)

        fig.tight_layout()
        out_path = Path(output_dir)
        fig.savefig(out_path / "model_comparison.png",
                    dpi=300, bbox_inches="tight")
        plt.close(fig)
        logger.info("Generated model_comparison.png")

    except Exception as e:
        logger.warning(f"Could not generate model comparison: {e}")

    logger.info(f"All figures generated in {output_dir}")

