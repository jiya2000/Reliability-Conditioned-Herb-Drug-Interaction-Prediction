"""
Ablation Study Runner

Runs the critical Week 12 ablation comparison:
(a) GNN-only baseline — no text, no reliability
(b) Unconditioned cross-attention — text but R=1.0
(c) Full reliability-conditioned model

The (b)-vs-(c) gap is the evidentiary core of both the paper
and the patent disclosure.
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Optional

import torch
from loguru import logger

from src.models.hdi_model import HDIModel
from src.training.trainer import HDITrainer
from src.training.evaluator import HDIEvaluator


class AblationRunner:
    """
    Runs the three-variant ablation study.

    This is the highest-priority experiment in the project.
    The (b)-vs-(c) gap — unconditioned vs. reliability-conditioned
    cross-attention — provides the evidentiary backing for the
    non-obviousness argument in both the paper and patent filing.

    Usage:
        runner = AblationRunner(config)
        results = runner.run(train_loader, val_loader, test_loader, graph_data)
        runner.save_results("ablation_results.json")
    """

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
            "description": "Full reliability-conditioned cross-attention",
            "use_text": True,
            "use_reliability": True,
        },
    }

    def __init__(
        self,
        gnn_input_dim: int = 128,
        gnn_hidden_dim: int = 256,
        gnn_output_dim: int = 128,
        num_relations: int = 6,
        learning_rate: float = 1e-4,
        epochs: int = 50,
        checkpoint_dir: str = "checkpoints/ablation",
        results_dir: str = "outputs/ablation",
        device: str = "auto",
        seed: int = 42,
    ):
        self.model_kwargs = {
            "gnn_input_dim": gnn_input_dim,
            "gnn_hidden_dim": gnn_hidden_dim,
            "gnn_output_dim": gnn_output_dim,
            "num_relations": num_relations,
        }
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.checkpoint_dir = Path(checkpoint_dir)
        self.results_dir = Path(results_dir)
        self.device = device
        self.seed = seed

        self.results: dict[str, dict] = {}

        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.results_dir.mkdir(parents=True, exist_ok=True)

    def run(
        self,
        train_loader,
        val_loader,
        test_loader,
        graph_data: dict,
    ) -> dict:
        """
        Run all three ablation variants.

        Returns:
            Dict mapping variant name → evaluation results
        """
        logger.info("=" * 60)
        logger.info("ABLATION STUDY — Week 12 Critical Experiment")
        logger.info("=" * 60)

        for variant_name, variant_config in self.VARIANTS.items():
            logger.info(f"\n{'='*60}")
            logger.info(f"Variant: {variant_name}")
            logger.info(f"Description: {variant_config['description']}")
            logger.info(f"{'='*60}")

            # Set seed for reproducibility
            torch.manual_seed(self.seed)

            # Create model variant
            model = HDIModel(
                use_text=variant_config["use_text"],
                use_reliability=variant_config["use_reliability"],
                **self.model_kwargs,
            )

            # Count parameters
            param_count = model.count_parameters()
            logger.info(f"Parameters: {param_count}")

            # Train
            trainer = HDITrainer(
                model=model,
                learning_rate=self.learning_rate,
                epochs=self.epochs,
                checkpoint_dir=str(
                    self.checkpoint_dir / variant_name
                ),
                device=self.device,
            )

            start_time = time.time()
            history = trainer.train(train_loader, val_loader, graph_data)
            train_time = time.time() - start_time

            # Evaluate
            evaluator = HDIEvaluator(model=model, device=self.device)
            eval_results = evaluator.evaluate(
                test_loader, graph_data, return_predictions=False
            )

            # Store results
            self.results[variant_name] = {
                "description": variant_config["description"],
                "parameters": param_count,
                "training_time": train_time,
                "training_history": {
                    "final_train_loss": history["train_loss"][-1]
                    if history["train_loss"] else None,
                    "final_val_loss": history["val_loss"][-1]
                    if history["val_loss"] else None,
                    "epochs_trained": len(history["train_loss"]),
                },
                "evaluation": eval_results,
            }

        # Compare variants
        self._print_comparison()
        self.save_results()

        return self.results

    def _print_comparison(self) -> None:
        """Print a comparison table of all variants."""
        logger.info("\n" + "=" * 80)
        logger.info("ABLATION RESULTS COMPARISON")
        logger.info("=" * 80)

        metrics_to_compare = [
            "auc_roc", "auc_pr", "precision", "recall", "f1", "mrr", "hits@10",
        ]

        # Header
        header = f"{'Metric':<20}"
        for variant in self.VARIANTS:
            header += f"  {variant:<20}"
        logger.info(header)
        logger.info("-" * 80)

        # Metrics rows
        for metric in metrics_to_compare:
            row = f"{metric:<20}"
            for variant in self.VARIANTS:
                value = self.results.get(variant, {}).get(
                    "evaluation", {}
                ).get(metric, 0)
                row += f"  {value:<20.4f}"
            logger.info(row)

        # Critical comparison: unconditioned vs full
        logger.info("\n" + "=" * 80)
        logger.info("CRITICAL COMPARISON: Unconditioned vs. Full Model")
        logger.info("=" * 80)

        uncond = self.results.get("unconditioned", {}).get("evaluation", {})
        full = self.results.get("full_model", {}).get("evaluation", {})

        for metric in metrics_to_compare:
            uncond_val = uncond.get(metric, 0)
            full_val = full.get(metric, 0)
            diff = full_val - uncond_val
            pct = (diff / uncond_val * 100) if uncond_val != 0 else 0

            indicator = "✓" if diff > 0 else "✗"
            logger.info(
                f"  {metric:<20}: {uncond_val:.4f} → {full_val:.4f} "
                f"({indicator} {diff:+.4f}, {pct:+.1f}%)"
            )

    def save_results(
        self, filename: Optional[str] = None
    ) -> None:
        """Save ablation results to JSON."""
        if filename is None:
            filename = "ablation_results.json"

        path = self.results_dir / filename

        # Make results JSON-serializable
        serializable = {}
        for variant, data in self.results.items():
            serializable[variant] = {
                "description": data.get("description", ""),
                "parameters": data.get("parameters", {}),
                "training_time": data.get("training_time", 0),
                "evaluation": {
                    k: v
                    for k, v in data.get("evaluation", {}).items()
                    if isinstance(v, (int, float, str))
                },
            }

        with open(path, "w") as f:
            json.dump(serializable, f, indent=2)

        logger.info(f"Ablation results saved to: {path}")
