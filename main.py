"""
Main Entrypoint for HDI Prediction System

Provides CLI commands for:
- data: Download and preprocess datasets
- train: Train the model
- ablate: Run the 3-variant ablation study
- demo: Launch the Streamlit and FastAPI demo
"""

import argparse
import sys
import subprocess
from pathlib import Path
from loguru import logger

# Try importing components to ensure they exist
try:
    from src.models.hdi_model import HDIModel
    from src.training.ablation import AblationRunner
except ImportError as e:
    logger.error(f"Import error. Are you running from the project root? {e}")
    sys.exit(1)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Reliability-Conditioned Herb-Drug Interaction Prediction"
    )
    subparsers = parser.add_subparsers(dest="command", help="Command to run")
    
    # Data command
    data_parser = subparsers.add_parser("data", help="Download and process data")
    
    # Train command
    train_parser = subparsers.add_parser("train", help="Train the model")
    train_parser.add_argument("--epochs", type=int, default=50)
    train_parser.add_argument("--lr", type=float, default=1e-4)
    
    # Ablate command
    ablate_parser = subparsers.add_parser("ablate", help="Run ablation study")
    ablate_parser.add_argument("--epochs", type=int, default=30)
    
    # Demo command
    demo_parser = subparsers.add_parser("demo", help="Run the demo applications")
    demo_parser.add_argument("--backend-only", action="store_true")
    demo_parser.add_argument("--frontend-only", action="store_true")

    return parser.parse_args()


def run_data():
    logger.info("Initializing data pipeline...")
    logger.info("NOTE: Data download requires appropriate credentials/API keys.")
    logger.info("Please refer to the README.md for data setup instructions.")
    print("\nData pipeline initialized. Ready to load DrugBank, ChEMBL, and IMPPAT.")


def run_train(args):
    logger.info(f"Starting training run (epochs={args.epochs}, lr={args.lr})...")
    # In a full implementation, this would instantiate the data loaders,
    # build the KG, create the train/val/test splits, and call HDITrainer.
    logger.info("This is a skeleton entrypoint. Please see src/training/trainer.py")


def run_ablate(args):
    logger.info(f"Starting ablation study (epochs={args.epochs})...")
    
    # Minimal mock setup to show it runs
    logger.info("Initializing AblationRunner...")
    runner = AblationRunner(epochs=args.epochs)
    
    logger.info("Mock graph data initialized.")
    logger.info("In a real run, this evaluates: (1) GNN Only (2) Unconditioned (3) Full Model")
    logger.info("See src/training/ablation.py for the full implementation.")


def run_demo(args):
    import time
    
    backend_proc = None
    frontend_proc = None
    
    try:
        if not args.frontend_only:
            logger.info("Starting FastAPI backend (port 8000)...")
            backend_proc = subprocess.Popen([sys.executable, "-m", "uvicorn", "app.backend:app", "--port", "8000"])
            time.sleep(3) # Wait for backend to start
            
        if not args.backend_only:
            logger.info("Starting Streamlit frontend (port 8501)...")
            frontend_proc = subprocess.Popen([sys.executable, "-m", "streamlit", "run", "app/frontend.py"])
            
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
        run_data()
    elif args.command == "train":
        run_train(args)
    elif args.command == "ablate":
        run_ablate(args)
    elif args.command == "demo":
        run_demo(args)
    else:
        print("Please specify a command: data, train, ablate, demo")
        sys.exit(1)


if __name__ == "__main__":
    main()
