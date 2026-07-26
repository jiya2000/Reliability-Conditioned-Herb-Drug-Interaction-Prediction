"""
Main Entrypoint for HDI Prediction System

Provides CLI commands for:
- data: Generate or load dataset
- train: Train the model with full pipeline
- ablate: Run the 3-variant ablation study
- demo: Launch the Streamlit and FastAPI demo
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

    # Ablate command
    ablate_parser = subparsers.add_parser("ablate", help="Run ablation study")
    ablate_parser.add_argument("--epochs", type=int, default=30)
    ablate_parser.add_argument("--batch-size", type=int, default=64)
    ablate_parser.add_argument("--seeds", type=int, nargs="+",
                               default=[42, 123, 456],
                               help="Random seeds for multiple runs")

    # Demo command
    demo_parser = subparsers.add_parser("demo", help="Run the demo applications")
    demo_parser.add_argument("--backend-only", action="store_true")
    demo_parser.add_argument("--frontend-only", action="store_true")

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
        mode="synthetic",
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
        mode="synthetic",
        batch_size=args.batch_size,
    )
    data = pipeline.build()

    # Run ablation
    runner = AblationRunner(epochs=args.epochs)

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
            num_relations=data["num_relations"],
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
    else:
        print("Please specify a command: data, train, ablate, demo")
        print("Example: python main.py train --epochs 50 --lr 1e-4")
        sys.exit(1)


if __name__ == "__main__":
    main()
