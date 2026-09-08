"""
Main Entrypoint for HDI Prediction System

Provides CLI commands for:
- data: Generate or load dataset
- train: Train the model with full pipeline
- ablate: Run the 3-variant ablation study
- demo: Launch the Streamlit and FastAPI demo
- results: Display experiment results
- figures: Generate paper figures from saved results
"""

import argparse
import sys
import subprocess
from pathlib import Path
from loguru import logger

# Configure logging
logger.remove()
logger.add(
    sys.stderr,
    format="<green>{time:HH:mm:ss}</green> | <level>{level: <8}</level> | {message}",
    level="INFO",
)
logger.add(
    "logs/hdi_{time}.log",
    rotation="10 MB",
    level="DEBUG",
)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Reliability-Conditioned Herb-Drug Interaction Prediction"
    )
    subparsers = parser.add_subparsers(dest="command", help="Command to run")

    # Data command
    data_parser = subparsers.add_parser("data", help="Generate/load data")
    data_parser.add_argument(
        "--mode", choices=["synthetic", "real"], default="synthetic",
        help="Data source mode",
    )

    # Train command
    train_parser = subparsers.add_parser("train", help="Train the model")
    train_parser.add_argument("--epochs", type=int, default=50)
    train_parser.add_argument("--lr", type=float, default=1e-4)
    train_parser.add_argument("--batch-size", type=int, default=64)
    train_parser.add_argument("--hidden-dim", type=int, default=128)
    train_parser.add_argument("--num-heads", type=int, default=4)
    train_parser.add_argument("--gating-mode", default="multiplicative",
                              choices=["multiplicative", "additive",
                                       "learned_gate", "uncertainty_aware"])
    train_parser.add_argument("--lambda-cal", type=float, default=0.1,
                              help="ARC calibration loss weight")
    train_parser.add_argument("--lambda-contrastive", type=float, default=0.05,
                              help="Contrastive regularization weight")
    train_parser.add_argument("--device", default="auto")
    train_parser.add_argument("--data-mode", choices=["synthetic", "real"],
                              default="synthetic", help="Data source mode")

    # Ablate command
    ablate_parser = subparsers.add_parser("ablate", help="Run ablation study")
    ablate_parser.add_argument("--epochs", type=int, default=30)
    ablate_parser.add_argument("--batch-size", type=int, default=64)
    ablate_parser.add_argument("--seeds", type=int, nargs="+",
                               default=[42, 123, 456],
                               help="Random seeds for multiple runs")
    ablate_parser.add_argument("--mode", choices=["synthetic", "real"], default="synthetic",
                               help="Data source mode")

    # Demo command
    demo_parser = subparsers.add_parser("demo", help="Run the demo applications")
    demo_parser.add_argument("--backend-only", action="store_true")
    demo_parser.add_argument("--frontend-only", action="store_true")

    # Results command
    results_parser = subparsers.add_parser("results", help="Display experiment results")
    results_parser.add_argument(
        "--file", default="results/paper_tables.md",
        help="Path to results file",
    )

    # Figures command
    figures_parser = subparsers.add_parser("figures", help="Generate paper figures")
    figures_parser.add_argument(
        "--results-file", default="results/full_results.json",
        help="Path to full results JSON",
    )

    return parser.parse_args()


def run_data(args):
    """Generate or load the dataset."""
    from src.data.data_pipeline import DataPipeline

    logger.info(f"Initializing data pipeline (mode={args.mode})...")
    pipeline = DataPipeline(mode=args.mode)
    data = pipeline.build()

    # Print summary
    graph = data["graph_data"]
    logger.info(
        f"✓ Data ready: {graph['node_features'].shape[0]} nodes, "
        f"{graph['edge_index'].shape[1]} edges, "
        f"{data['num_relations']} relation types"
    )
    logger.info(f"  Train batches: {len(data['train_loader'])}")
    logger.info(f"  Val batches: {len(data['val_loader'])}")
    logger.info(f"  Test batches: {len(data['test_loader'])}")

    # Print corpus stats
    try:
        from src.data.expanded_corpus import get_corpus_statistics
        stats = get_corpus_statistics()
        logger.info(
            f"✓ Code-mixed corpus: {stats['total_sentences']} sentences, "
            f"{stats['total_entities']} entities, "
            f"{stats['total_relations']} relations"
        )
    except ImportError:
        pass


def run_train(args):
    """Train the model end-to-end."""
    import torch
    from src.data.data_pipeline import DataPipeline
    from src.models.hdi_model import HDIModel
    from src.training.trainer import HDITrainer

    logger.info(f"Starting training (epochs={args.epochs}, lr={args.lr})...")

    # Build data
    pipeline = DataPipeline(
        mode=args.data_mode,
        batch_size=args.batch_size,
        feature_dim=args.hidden_dim,
    )
    data = pipeline.build()

    # Create model
    model = HDIModel(
        gnn_input_dim=args.hidden_dim,
        gnn_hidden_dim=args.hidden_dim * 2,
        gnn_output_dim=args.hidden_dim,
        num_relations=data["num_relations"],
        cross_attention_heads=args.num_heads,
        gating_mode=args.gating_mode,
    )

    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f"Model: {total_params:,} params ({trainable_params:,} trainable)")

    # Create trainer with novel loss integration
    trainer = HDITrainer(
        model=model,
        learning_rate=args.lr,
        epochs=args.epochs,
        device=args.device,
        lambda_calibration=args.lambda_cal,
        lambda_contrastive=args.lambda_contrastive,
    )

    # Train
    trainer.train(
        train_loader=data["train_loader"],
        val_loader=data["val_loader"],
        graph_data=data["graph_data"],
    )

    logger.info("✓ Training complete!")


def run_ablate(args):
    """Run the 3-variant ablation study with statistical testing."""
    import torch
    from src.data.data_pipeline import DataPipeline
    from src.training.ablation import AblationRunner

    logger.info(f"Starting ablation study (epochs={args.epochs}, seeds={args.seeds})...")

    # Build data
    pipeline = DataPipeline(
        mode=args.mode,
        batch_size=args.batch_size,
    )
    data = pipeline.build()

    # Run ablation
    runner = AblationRunner(
        epochs=args.epochs,
        num_relations=data["num_relations"],
    )

    logger.info("Running 3-variant comparison:")
    logger.info("  (a) GNN Only — no text, no R")
    logger.info("  (b) GNN + Text (unconditioned) — text but R=1")
    logger.info("  (c) GNN + Text + R (full) — reliability-conditioned")

    # For each seed, run the ablation
    all_results = []
    for seed in args.seeds:
        logger.info(f"\n--- Seed {seed} ---")
        torch.manual_seed(seed)

        result = runner.run(
            train_loader=data["train_loader"],
            val_loader=data["val_loader"],
            test_loader=data["test_loader"],
            graph_data=data["graph_data"],
        )
        all_results.append(result)

    # Print summary
    logger.info("\n" + "=" * 60)
    logger.info("ABLATION STUDY COMPLETE")
    logger.info("=" * 60)

    if all_results and all_results[0] is not None:
        for variant_name, metrics in all_results[0].items():
            if isinstance(metrics, dict):
                logger.info(
                    f"  {variant_name}: "
                    + " | ".join(f"{k}={v:.4f}" for k, v in metrics.items()
                                if isinstance(v, float))
                )

    logger.info("\n✓ Ablation complete! Results saved to results/")


def run_demo(args):
    """Launch the demo application."""
    import time

    backend_proc = None
    frontend_proc = None

    try:
        if not args.frontend_only:
            logger.info("Starting FastAPI backend (port 8000)...")
            backend_proc = subprocess.Popen(
                [sys.executable, "-m", "uvicorn",
                 "app.backend:app", "--port", "8000"]
            )
            time.sleep(3)

        if not args.backend_only:
            logger.info("Starting Streamlit frontend (port 8501)...")
            frontend_proc = subprocess.Popen(
                [sys.executable, "-m", "streamlit",
                 "run", "app/frontend.py"]
            )

        logger.info("Demo running. Press Ctrl+C to stop.")

        if backend_proc:
            backend_proc.wait()
        if frontend_proc:
            frontend_proc.wait()

    except KeyboardInterrupt:
        logger.info("Stopping demo...")
    finally:
        if backend_proc:
            backend_proc.terminate()
        if frontend_proc:
            frontend_proc.terminate()


def run_results(args):
    """Display experiment results."""
    results_file = Path(args.file)
    if not results_file.exists():
        logger.error(f"Results file not found: {results_file}")
        logger.info("Run experiments first: python run_experiments.py")
        sys.exit(1)

    with open(results_file, "r") as f:
        content = f.read()

    print(content)

    # Also check for JSON results
    json_path = Path("results/ablation_results.json")
    if json_path.exists():
        import json
        with open(json_path) as f:
            data = json.load(f)
        meta = data.get("metadata", {})
        logger.info(f"\nExperiment metadata:")
        logger.info(f"  Timestamp: {meta.get('timestamp', 'unknown')}")
        logger.info(f"  Data mode: {meta.get('data_mode', 'unknown')}")
        logger.info(f"  Epochs: {meta.get('epochs', 'unknown')}")
        logger.info(f"  Seeds: {meta.get('seeds', 'unknown')}")


def run_figures(args):
    """Generate paper figures from saved results."""
    import json

    results_file = Path(args.results_file)
    if not results_file.exists():
        logger.error(f"Results file not found: {results_file}")
        logger.info("Run experiments first: python run_experiments.py")
        sys.exit(1)

    with open(results_file, "r") as f:
        full_results = json.load(f)

    logger.info("Generating paper figures from saved results...")

    try:
        from src.visualization.paper_figures import generate_all_figures
        generate_all_figures(full_results)
        logger.info("✓ Figures generated in results/figures/")
    except ImportError:
        logger.warning("paper_figures module not available, using run_experiments figures")
        logger.info("Run: python run_experiments.py (figures are generated automatically)")


def main():
    args = parse_args()

    if args.command == "data":
        run_data(args)
    elif args.command == "train":
        run_train(args)
    elif args.command == "ablate":
        run_ablate(args)
    elif args.command == "demo":
        run_demo(args)
    elif args.command == "results":
        run_results(args)
    elif args.command == "figures":
        run_figures(args)
    else:
        print("Please specify a command: data, train, ablate, demo, results, figures")
        print()
        print("Quick start:")
        print("  python run_experiments.py          # Run full experiment pipeline")
        print("  python main.py train --epochs 50   # Train single model")
        print("  python main.py ablate              # Run ablation study")
        print("  python main.py demo                # Launch demo")
        print("  python main.py results             # Show results")
        sys.exit(1)


if __name__ == "__main__":
    main()
