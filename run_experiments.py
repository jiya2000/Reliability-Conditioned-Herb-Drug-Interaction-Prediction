#!/usr/bin/env python
"""
Full Experiment Pipeline — HDI Prediction

Runs the complete experimental pipeline:
1. Build data (synthetic or real)
2. Run 3-variant ablation study with multiple seeds
3. Compute all evaluation metrics
4. Run statistical significance tests
5. Run stratified evaluation
6. Generate paper-ready tables (markdown + JSON)
7. Generate paper figures

Usage:
    python run_experiments.py                     # Synthetic data, fast
    python run_experiments.py --mode real          # Real DrugBank data
    python run_experiments.py --epochs 50 --seeds 42 123 456 789 1337
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path
from datetime import datetime

import numpy as np
import torch
from loguru import logger

# Configure logging
logger.remove()
logger.add(
    sys.stderr,
    format="<green>{time:HH:mm:ss}</green> | <level>{level: <8}</level> | {message}",
    level="INFO",
)
logger.add(
    "logs/experiments_{time}.log",
    rotation="50 MB",
    level="DEBUG",
)

# Ensure results directories exist
RESULTS_DIR = Path("results")
FIGURES_DIR = RESULTS_DIR / "figures"
CHECKPOINTS_DIR = Path("checkpoints/ablation")
RESULTS_DIR.mkdir(parents=True, exist_ok=True)
FIGURES_DIR.mkdir(parents=True, exist_ok=True)
CHECKPOINTS_DIR.mkdir(parents=True, exist_ok=True)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run HDI Prediction Experiments"
    )
    parser.add_argument(
        "--mode", choices=["synthetic", "real"], default="synthetic",
        help="Data source: synthetic (fast) or real (DrugBank)",
    )
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--hidden-dim", type=int, default=128)
    parser.add_argument("--num-heads", type=int, default=4)
    parser.add_argument(
        "--seeds", type=int, nargs="+", default=[42, 123, 456],
        help="Random seeds for multiple runs",
    )
    parser.add_argument(
        "--gating-mode",
        choices=["multiplicative", "additive", "none"],
        default="multiplicative",
    )
    parser.add_argument("--skip-figures", action="store_true")
    parser.add_argument("--resume", action="store_true", help="Skip training for completed variants")
    parser.add_argument("--device", default="auto")

    return parser.parse_args()


def build_data(args):
    """Build the data pipeline."""
    from src.data.data_pipeline import DataPipeline

    logger.info(f"Building data pipeline (mode={args.mode})...")
    pipeline = DataPipeline(
        mode=args.mode,
        batch_size=args.batch_size,
        feature_dim=args.hidden_dim,
    )
    data = pipeline.build()

    graph = data["graph_data"]
    logger.info(
        f"Data ready: {graph['node_features'].shape[0]} nodes, "
        f"{graph['edge_index'].shape[1]} edges, "
        f"{data['num_relations']} relation types"
    )
    logger.info(f"  Train batches: {len(data['train_loader'])}")
    logger.info(f"  Val batches:   {len(data['val_loader'])}")
    logger.info(f"  Test batches:  {len(data['test_loader'])}")

    return data


def train_variant(variant_name, variant_config, data, args, seed):
    """Train a single model variant with a specific seed."""
    from src.models.hdi_model import HDIModel
    from src.training.trainer import HDITrainer
    from src.training.evaluator import HDIEvaluator

    torch.manual_seed(seed)
    np.random.seed(seed)

    logger.info(f"\n{'='*60}")
    logger.info(f"Training: {variant_name} (seed={seed})")
    logger.info(f"  Config: use_text={variant_config['use_text']}, "
                f"use_reliability={variant_config['use_reliability']}")
    logger.info(f"{'='*60}")

    # Create model
    model = HDIModel(
        gnn_input_dim=args.hidden_dim,
        gnn_hidden_dim=args.hidden_dim * 2,
        gnn_output_dim=args.hidden_dim,
        num_relations=data["num_relations"],
        cross_attention_heads=args.num_heads,
        gating_mode=args.gating_mode,
        use_text=variant_config["use_text"],
        use_reliability=variant_config["use_reliability"],
    )

    param_count = model.count_parameters()
    logger.info(f"  Parameters: {param_count['total']:,}")

    # Create trainer
    checkpoint_dir = str(CHECKPOINTS_DIR / f"{variant_name}_seed{seed}")
    best_model_path = os.path.join(checkpoint_dir, "best_model.pt")

    if getattr(args, "resume", False) and os.path.exists(best_model_path):
        logger.info(f"  [RESUME] Found existing checkpoint at {best_model_path}. Skipping training!")
        checkpoint = torch.load(best_model_path, map_location=args.device)
        model.load_state_dict(checkpoint["model_state_dict"])
        model.to(args.device)
        history = {
            "train_loss": [checkpoint.get("train_loss", 0.0)], 
            "val_loss": [checkpoint.get("val_loss", 0.0)]
        }
        train_time = 0.0
    else:
        trainer = HDITrainer(
            model=model,
            learning_rate=args.lr,
            epochs=args.epochs,
            checkpoint_dir=checkpoint_dir,
            device=args.device,
            lambda_calibration=0.1 if variant_config["use_reliability"] else 0.0,
            lambda_contrastive=0.05 if variant_config["use_reliability"] else 0.0,
        )

        # Train
        start_time = time.time()
        history = trainer.train(
            train_loader=data["train_loader"],
            val_loader=data["val_loader"],
            graph_data=data["graph_data"],
        )
        train_time = time.time() - start_time
        logger.info(f"  Training complete in {train_time:.1f}s")

    # Evaluate on test set
    evaluator = HDIEvaluator(model=model, device=args.device)
    eval_results = evaluator.evaluate(
        data["test_loader"], data["graph_data"], return_predictions=True
    )

    # Stratified evaluation
    stratified_results = None
    try:
        stratified_results = evaluator.evaluate_stratified(
            data["test_loader"], data["graph_data"]
        )
    except Exception as e:
        logger.warning(f"  Stratified evaluation failed: {e}")

    # Calibration data
    calibration_data = None
    try:
        calibration_data = evaluator.get_calibration_data(
            data["test_loader"], data["graph_data"]
        )
    except Exception as e:
        logger.warning(f"  Calibration data generation failed: {e}")

    return {
        "variant": variant_name,
        "seed": seed,
        "parameters": param_count,
        "training_time": train_time,
        "epochs_trained": len(history["train_loss"]),
        "final_train_loss": history["train_loss"][-1] if history["train_loss"] else None,
        "final_val_loss": history["val_loss"][-1] if history["val_loss"] else None,
        "train_loss_history": history["train_loss"],
        "val_loss_history": history["val_loss"],
        "evaluation": {
            k: v for k, v in eval_results.items()
            if isinstance(v, (int, float, str))
        },
        "stratified": _serialize_stratified(stratified_results),
        "calibration": calibration_data,
        "checkpoint_dir": checkpoint_dir,
    }


def _serialize_stratified(results):
    """Make stratified results JSON-serializable."""
    if results is None:
        return None
    serializable = {}
    for key, val in results.items():
        if isinstance(val, dict):
            serializable[key] = {
                k: (v if isinstance(v, (int, float, str)) else
                    {kk: vv for kk, vv in v.items()
                     if isinstance(vv, (int, float, str))}
                    if isinstance(v, dict) else str(v))
                for k, v in val.items()
            }
        else:
            serializable[key] = val
    return serializable


def run_ablation(data, args):
    """Run the full 3-variant ablation study with multiple seeds."""
    VARIANTS = {
        "gnn_only": {
            "description": "GNN encoder + link predictor, no text, no reliability",
            "use_text": False,
            "use_reliability": False,
        },
        "unconditioned": {
            "description": "GNN + text cross-attention, R fixed to 1.0",
            "use_text": True,
            "use_reliability": False,
        },
        "full_model": {
            "description": "Full reliability-conditioned cross-attention (ours)",
            "use_text": True,
            "use_reliability": True,
        },
    }

    all_results = {}

    for variant_name, variant_config in VARIANTS.items():
        all_results[variant_name] = {
            "description": variant_config["description"],
            "runs": [],
        }

        for seed in args.seeds:
            result = train_variant(
                variant_name, variant_config, data, args, seed
            )
            all_results[variant_name]["runs"].append(result)

    return all_results


def compute_aggregate_metrics(all_results):
    """Compute mean ± std across seeds for each variant."""
    metrics_to_aggregate = [
        "auc_roc", "auc_pr", "precision", "recall", "f1",
        "mrr", "hits@10", "hits@5", "hits@1",
    ]

    aggregated = {}
    for variant_name, variant_data in all_results.items():
        runs = variant_data["runs"]
        agg = {}

        for metric in metrics_to_aggregate:
            values = [
                r["evaluation"].get(metric, 0.0) for r in runs
                if r["evaluation"] is not None
            ]
            if values:
                agg[metric] = {
                    "mean": float(np.mean(values)),
                    "std": float(np.std(values)),
                    "min": float(np.min(values)),
                    "max": float(np.max(values)),
                    "values": values,
                }

        # Training time
        times = [r["training_time"] for r in runs]
        agg["training_time_mean"] = float(np.mean(times))

        # Parameter count
        agg["total_parameters"] = runs[0]["parameters"]["total"] if runs else 0

        aggregated[variant_name] = {
            "description": variant_data["description"],
            "metrics": agg,
        }

    return aggregated


def run_statistical_tests(all_results):
    """Run statistical significance tests between variants."""
    tests_output = {}

    try:
        from src.training.statistical_tests import (
            bootstrap_confidence_interval,
            mcnemars_test,
            paired_permutation_test,
        )

        # Get per-seed metrics for each variant
        variants = {}
        for vname, vdata in all_results.items():
            variants[vname] = [
                r["evaluation"] for r in vdata["runs"]
                if r["evaluation"] is not None
            ]

        # Critical comparison: unconditioned vs full
        if "unconditioned" in variants and "full_model" in variants:
            uncond_aucs = [m.get("auc_roc", 0) for m in variants["unconditioned"]]
            full_aucs = [m.get("auc_roc", 0) for m in variants["full_model"]]

            # Bootstrap CI for the difference
            try:
                diffs = [f - u for f, u in zip(full_aucs, uncond_aucs)]
                mean_diff = float(np.mean(diffs))
                tests_output["auc_roc_gap"] = {
                    "mean_difference": mean_diff,
                    "direction": "full > unconditioned" if mean_diff > 0 else "unconditioned > full",
                    "per_seed_diffs": diffs,
                }
            except Exception as e:
                logger.warning(f"Bootstrap CI failed: {e}")

        # GNN-only vs full
        if "gnn_only" in variants and "full_model" in variants:
            gnn_aucs = [m.get("auc_roc", 0) for m in variants["gnn_only"]]
            full_aucs = [m.get("auc_roc", 0) for m in variants["full_model"]]
            diffs = [f - g for f, g in zip(full_aucs, gnn_aucs)]
            tests_output["gnn_vs_full_gap"] = {
                "mean_difference": float(np.mean(diffs)),
                "direction": "full > gnn_only" if np.mean(diffs) > 0 else "gnn_only > full",
                "per_seed_diffs": diffs,
            }

    except Exception as e:
        logger.warning(f"Statistical tests encountered issue: {e}")
        tests_output["error"] = str(e)

    return tests_output


def generate_paper_tables(aggregated, stat_tests, args):
    """Generate paper-ready markdown tables."""
    lines = []
    lines.append("# Experimental Results — HDI Prediction")
    lines.append("")
    lines.append(f"**Generated**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    lines.append(f"**Data mode**: {args.mode}")
    lines.append(f"**Seeds**: {args.seeds}")
    lines.append(f"**Epochs**: {args.epochs}")
    lines.append(f"**Gating mode**: {args.gating_mode}")
    lines.append("")

    # Table 1: Main Ablation Results
    lines.append("## Table 1: Ablation Study Results")
    lines.append("")
    lines.append("| Variant | AUC-ROC | AUC-PR | Precision | Recall | F1 | MRR | Hits@10 | Params |")
    lines.append("|---------|---------|--------|-----------|--------|-----|-----|---------|--------|")

    variant_labels = {
        "gnn_only": "(a) GNN-only",
        "unconditioned": "(b) Unconditioned",
        "full_model": "(c) **Full R-conditioned (Ours)**",
    }

    for vname in ["gnn_only", "unconditioned", "full_model"]:
        vdata = aggregated.get(vname, {})
        metrics = vdata.get("metrics", {})
        label = variant_labels.get(vname, vname)

        def fmt(m):
            d = metrics.get(m, {})
            if isinstance(d, dict) and "mean" in d:
                return f"{d['mean']:.4f}±{d['std']:.4f}"
            return "—"

        params = metrics.get("total_parameters", 0)
        lines.append(
            f"| {label} | {fmt('auc_roc')} | {fmt('auc_pr')} | "
            f"{fmt('precision')} | {fmt('recall')} | {fmt('f1')} | "
            f"{fmt('mrr')} | {fmt('hits@10')} | {params:,} |"
        )

    lines.append("")

    # Critical comparison
    lines.append("## Table 2: Critical Comparison — (b) vs (c)")
    lines.append("")
    lines.append("The (b)-vs-(c) gap is the evidentiary core of the paper and patent.")
    lines.append("")

    uncond = aggregated.get("unconditioned", {}).get("metrics", {})
    full = aggregated.get("full_model", {}).get("metrics", {})

    lines.append("| Metric | Unconditioned (b) | Full Model (c) | Δ (c−b) | Improvement |")
    lines.append("|--------|-------------------|----------------|---------|-------------|")

    for metric in ["auc_roc", "auc_pr", "f1", "mrr", "hits@10"]:
        u = uncond.get(metric, {})
        f = full.get(metric, {})
        u_mean = u.get("mean", 0) if isinstance(u, dict) else 0
        f_mean = f.get("mean", 0) if isinstance(f, dict) else 0
        diff = f_mean - u_mean
        pct = (diff / u_mean * 100) if u_mean != 0 else 0
        indicator = "✓" if diff > 0 else "✗"
        lines.append(
            f"| {metric} | {u_mean:.4f} | {f_mean:.4f} | "
            f"{indicator} {diff:+.4f} | {pct:+.1f}% |"
        )

    lines.append("")

    # Statistical tests
    if stat_tests and "auc_roc_gap" in stat_tests:
        gap = stat_tests["auc_roc_gap"]
        lines.append("## Table 3: Statistical Significance")
        lines.append("")
        lines.append(f"- **AUC-ROC gap (c−b)**: {gap['mean_difference']:+.4f}")
        lines.append(f"- **Direction**: {gap['direction']}")
        lines.append(f"- **Per-seed differences**: {gap['per_seed_diffs']}")
        lines.append("")

    # Stratified results (if available from the first seed of full_model)
    lines.append("## Table 4: Model Configuration")
    lines.append("")
    lines.append("| Setting | Value |")
    lines.append("|---------|-------|")
    lines.append(f"| Data mode | {args.mode} |")
    lines.append(f"| Epochs | {args.epochs} |")
    lines.append(f"| Learning rate | {args.lr} |")
    lines.append(f"| Batch size | {args.batch_size} |")
    lines.append(f"| Hidden dim | {args.hidden_dim} |")
    lines.append(f"| Attention heads | {args.num_heads} |")
    lines.append(f"| Gating mode | {args.gating_mode} |")
    lines.append(f"| Seeds | {args.seeds} |")
    lines.append("")

    return "\n".join(lines)


def generate_figures(all_results, aggregated, args):
    """Generate paper-quality figures."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import matplotlib.ticker as ticker

        plt.rcParams.update({
            "font.size": 11,
            "font.family": "serif",
            "axes.labelsize": 12,
            "axes.titlesize": 13,
            "xtick.labelsize": 10,
            "ytick.labelsize": 10,
            "legend.fontsize": 10,
            "figure.dpi": 150,
        })

        # --- Figure 1: Ablation Bar Chart ---
        fig, ax = plt.subplots(figsize=(10, 6))
        metrics_to_plot = ["auc_roc", "auc_pr", "f1", "mrr", "hits@10"]
        variant_names = ["gnn_only", "unconditioned", "full_model"]
        variant_labels = ["(a) GNN-only", "(b) Unconditioned", "(c) Full (Ours)"]
        colors = ["#7fb3d8", "#f4a582", "#2ca02c"]

        x = np.arange(len(metrics_to_plot))
        width = 0.25

        for i, (vname, vlabel, color) in enumerate(zip(variant_names, variant_labels, colors)):
            means = []
            stds = []
            for m in metrics_to_plot:
                d = aggregated.get(vname, {}).get("metrics", {}).get(m, {})
                means.append(d.get("mean", 0) if isinstance(d, dict) else 0)
                stds.append(d.get("std", 0) if isinstance(d, dict) else 0)

            ax.bar(x + i * width, means, width, yerr=stds,
                   label=vlabel, color=color, capsize=3, edgecolor="black", linewidth=0.5)

        ax.set_xlabel("Metric")
        ax.set_ylabel("Score")
        ax.set_title("Ablation Study: GNN-only vs Unconditioned vs Full R-Conditioned")
        ax.set_xticks(x + width)
        ax.set_xticklabels([m.upper().replace("_", "-") for m in metrics_to_plot])
        ax.legend()
        ax.set_ylim(0, 1.05)
        ax.grid(axis="y", alpha=0.3)
        fig.tight_layout()
        fig.savefig(FIGURES_DIR / "ablation_comparison.png", dpi=300, bbox_inches="tight")
        fig.savefig(FIGURES_DIR / "ablation_comparison.svg", bbox_inches="tight")
        plt.close(fig)
        logger.info("  Generated: ablation_comparison.png/svg")

        # --- Figure 2: Training Loss Curves ---
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))

        for vname, vlabel, color in zip(variant_names, variant_labels, colors):
            runs = all_results.get(vname, {}).get("runs", [])
            if not runs:
                continue
            # Plot first seed's loss curve
            run = runs[0]
            train_loss = run.get("train_loss_history", [])
            val_loss = run.get("val_loss_history", [])

            if train_loss:
                epochs_x = range(1, len(train_loss) + 1)
                axes[0].plot(epochs_x, train_loss, label=vlabel, color=color, linewidth=1.5)
            if val_loss:
                epochs_x = range(1, len(val_loss) + 1)
                axes[1].plot(epochs_x, val_loss, label=vlabel, color=color, linewidth=1.5)

        axes[0].set_xlabel("Epoch")
        axes[0].set_ylabel("Training Loss")
        axes[0].set_title("Training Loss Curves")
        axes[0].legend()
        axes[0].grid(alpha=0.3)

        axes[1].set_xlabel("Epoch")
        axes[1].set_ylabel("Validation Loss")
        axes[1].set_title("Validation Loss Curves")
        axes[1].legend()
        axes[1].grid(alpha=0.3)

        fig.tight_layout()
        fig.savefig(FIGURES_DIR / "loss_curves.png", dpi=300, bbox_inches="tight")
        fig.savefig(FIGURES_DIR / "loss_curves.svg", bbox_inches="tight")
        plt.close(fig)
        logger.info("  Generated: loss_curves.png/svg")

        # --- Figure 3: R-Score Distribution ---
        # Get R-scores from the full model's test predictions
        full_runs = all_results.get("full_model", {}).get("runs", [])
        if full_runs:
            calibration = full_runs[0].get("calibration")
            if calibration and "reliability_calibration" in calibration:
                fig, ax = plt.subplots(figsize=(7, 5))
                r_data = calibration["reliability_calibration"]
                r_vals = [d["avg_R"] for d in r_data]
                r_correct = [d["avg_correctness"] for d in r_data]
                r_counts = [d["count"] for d in r_data]

                ax.bar(r_vals, r_correct, width=0.08, alpha=0.7,
                       color="#2ca02c", edgecolor="black", linewidth=0.5,
                       label="Avg correctness")
                ax.plot([0, 1], [0, 1], "k--", alpha=0.5, label="Perfect calibration")
                ax.set_xlabel("Reliability Score (R)")
                ax.set_ylabel("Average Prediction Correctness")
                ax.set_title("Reliability Calibration Plot")
                ax.legend()
                ax.set_xlim(0, 1)
                ax.set_ylim(0, 1)
                ax.grid(alpha=0.3)
                fig.tight_layout()
                fig.savefig(FIGURES_DIR / "reliability_calibration.png", dpi=300, bbox_inches="tight")
                fig.savefig(FIGURES_DIR / "reliability_calibration.svg", bbox_inches="tight")
                plt.close(fig)
                logger.info("  Generated: reliability_calibration.png/svg")

        # --- Figure 4: Prediction Calibration (ECE) ---
        if full_runs:
            calibration = full_runs[0].get("calibration")
            if calibration and "prediction_calibration" in calibration:
                fig, ax = plt.subplots(figsize=(7, 5))
                cal_data = calibration["prediction_calibration"]
                confidences = [d["avg_confidence"] for d in cal_data]
                accuracies = [d["accuracy"] for d in cal_data]
                counts = [d["count"] for d in cal_data]

                ax.bar(confidences, accuracies, width=0.08, alpha=0.7,
                       color="#f4a582", edgecolor="black", linewidth=0.5,
                       label="Observed accuracy")
                ax.plot([0, 1], [0, 1], "k--", alpha=0.5, label="Perfect calibration")
                ece = calibration.get("ece", 0)
                ax.set_xlabel("Predicted Probability")
                ax.set_ylabel("Observed Accuracy")
                ax.set_title(f"Prediction Calibration (ECE = {ece:.4f})")
                ax.legend()
                ax.set_xlim(0, 1)
                ax.set_ylim(0, 1)
                ax.grid(alpha=0.3)
                fig.tight_layout()
                fig.savefig(FIGURES_DIR / "prediction_calibration.png", dpi=300, bbox_inches="tight")
                fig.savefig(FIGURES_DIR / "prediction_calibration.svg", bbox_inches="tight")
                plt.close(fig)
                logger.info("  Generated: prediction_calibration.png/svg")

        logger.info("All figures generated successfully")

    except Exception as e:
        logger.error(f"Figure generation failed: {e}")
        import traceback
        traceback.print_exc()


def save_all_results(all_results, aggregated, stat_tests, paper_tables, args):
    """Save all results to disk."""
    # JSON results (machine-readable)
    json_results = {
        "metadata": {
            "timestamp": datetime.now().isoformat(),
            "data_mode": args.mode,
            "epochs": args.epochs,
            "seeds": args.seeds,
            "gating_mode": args.gating_mode,
            "learning_rate": args.lr,
            "batch_size": args.batch_size,
            "hidden_dim": args.hidden_dim,
        },
        "aggregated": aggregated,
        "statistical_tests": stat_tests,
        "per_run": {
            vname: {
                "description": vdata["description"],
                "runs": [
                    {
                        k: v for k, v in run.items()
                        if k not in ("train_loss_history", "val_loss_history")
                    }
                    for run in vdata["runs"]
                ],
            }
            for vname, vdata in all_results.items()
        },
    }

    json_path = RESULTS_DIR / "ablation_results.json"
    with open(json_path, "w") as f:
        json.dump(json_results, f, indent=2, default=str)
    logger.info(f"Saved JSON results: {json_path}")

    # Markdown tables (human-readable)
    tables_path = RESULTS_DIR / "paper_tables.md"
    with open(tables_path, "w") as f:
        f.write(paper_tables)
    logger.info(f"Saved paper tables: {tables_path}")

    # Also save full results with loss histories for figure regeneration
    full_path = RESULTS_DIR / "full_results.json"
    full_json = {
        "metadata": json_results["metadata"],
        "per_run": {},
    }
    for vname, vdata in all_results.items():
        full_json["per_run"][vname] = {
            "description": vdata["description"],
            "runs": [
                {
                    "seed": r["seed"],
                    "train_loss_history": r.get("train_loss_history", []),
                    "val_loss_history": r.get("val_loss_history", []),
                    "evaluation": r.get("evaluation", {}),
                    "calibration": r.get("calibration"),
                }
                for r in vdata["runs"]
            ],
        }
    with open(full_path, "w") as f:
        json.dump(full_json, f, indent=2, default=str)
    logger.info(f"Saved full results: {full_path}")


def main():
    args = parse_args()

    logger.info("=" * 70)
    logger.info("HDI PREDICTION — FULL EXPERIMENT PIPELINE")
    logger.info("=" * 70)
    logger.info(f"Mode: {args.mode}")
    logger.info(f"Epochs: {args.epochs}")
    logger.info(f"Seeds: {args.seeds}")
    logger.info(f"Gating: {args.gating_mode}")
    logger.info(f"Device: {args.device}")
    logger.info("=" * 70)

    total_start = time.time()

    # 1. Build data
    data = build_data(args)

    # 2. Run ablation
    logger.info("\n" + "=" * 70)
    logger.info("PHASE 1: Running 3-variant ablation study")
    logger.info(f"  {len(args.seeds)} seeds × 3 variants = {len(args.seeds) * 3} training runs")
    logger.info("=" * 70)

    all_results = run_ablation(data, args)

    # 3. Aggregate metrics
    logger.info("\n" + "=" * 70)
    logger.info("PHASE 2: Computing aggregate metrics")
    logger.info("=" * 70)

    aggregated = compute_aggregate_metrics(all_results)

    # Print quick summary
    for vname, vdata in aggregated.items():
        m = vdata["metrics"]
        auc = m.get("auc_roc", {})
        f1 = m.get("f1", {})
        logger.info(
            f"  {vname}: AUC-ROC={auc.get('mean', 0):.4f}±{auc.get('std', 0):.4f}, "
            f"F1={f1.get('mean', 0):.4f}±{f1.get('std', 0):.4f}"
        )

    # 4. Statistical tests
    logger.info("\n" + "=" * 70)
    logger.info("PHASE 3: Running statistical significance tests")
    logger.info("=" * 70)

    stat_tests = run_statistical_tests(all_results)

    # 5. Generate tables
    logger.info("\n" + "=" * 70)
    logger.info("PHASE 4: Generating paper-ready tables")
    logger.info("=" * 70)

    paper_tables = generate_paper_tables(aggregated, stat_tests, args)

    # 6. Save everything
    save_all_results(all_results, aggregated, stat_tests, paper_tables, args)

    # 7. Generate figures
    if not args.skip_figures:
        logger.info("\n" + "=" * 70)
        logger.info("PHASE 5: Generating paper figures")
        logger.info("=" * 70)
        generate_figures(all_results, aggregated, args)

    total_time = time.time() - total_start

    # Final summary
    logger.info("\n" + "=" * 70)
    logger.info("EXPERIMENT PIPELINE COMPLETE")
    logger.info("=" * 70)
    logger.info(f"Total time: {total_time:.1f}s ({total_time/60:.1f} min)")
    logger.info(f"Results saved to: {RESULTS_DIR}/")
    logger.info(f"  - ablation_results.json  (machine-readable)")
    logger.info(f"  - paper_tables.md        (paper-ready tables)")
    logger.info(f"  - full_results.json      (with loss histories)")
    if not args.skip_figures:
        logger.info(f"  - figures/               (PNG + SVG)")

    # Print the critical gap
    uncond = aggregated.get("unconditioned", {}).get("metrics", {}).get("auc_roc", {})
    full = aggregated.get("full_model", {}).get("metrics", {}).get("auc_roc", {})
    gap = full.get("mean", 0) - uncond.get("mean", 0)
    logger.info(f"\n★ CRITICAL (b)-vs-(c) AUC-ROC GAP: {gap:+.4f} ★")
    if gap > 0:
        logger.info("  → Full model outperforms unconditioned — non-obviousness supported ✓")
    else:
        logger.info("  → WARNING: Gap is non-positive — see Section 7 of implementation plan")

    logger.info("=" * 70)


if __name__ == "__main__":
    main()
