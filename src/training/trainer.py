"""
HDI Model Trainer

Handles the full training loop including:
- Optimizer configuration with warmup + cosine decay
- Gradient clipping
- Checkpointing
- Experiment tracking (W&B / MLflow)
- Early stopping
"""

from __future__ import annotations

import os
import time
from pathlib import Path
from typing import Optional

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from loguru import logger

from src.models.hdi_model import HDIModel


class HDITrainer:
    """
    Trainer for the HDI prediction model.

    Supports end-to-end training with:
    - Cosine / step / plateau learning rate scheduling
    - Gradient clipping
    - Early stopping
    - Periodic checkpointing
    - Optional W&B or MLflow logging

    Usage:
        trainer = HDITrainer(model, config)
        trainer.train(train_loader, val_loader, graph_data)
    """

    def __init__(
        self,
        model: HDIModel,
        learning_rate: float = 1e-4,
        weight_decay: float = 1e-5,
        epochs: int = 100,
        warmup_steps: int = 500,
        grad_clip: float = 1.0,
        early_stopping_patience: int = 10,
        checkpoint_dir: str = "checkpoints",
        save_every: int = 5,
        scheduler_type: str = "cosine",
        tracking_backend: Optional[str] = None,
        project_name: str = "hdi-prediction",
        device: str = "auto",
    ):
        self.model = model
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.epochs = epochs
        self.warmup_steps = warmup_steps
        self.grad_clip = grad_clip
        self.patience = early_stopping_patience
        self.checkpoint_dir = Path(checkpoint_dir)
        self.save_every = save_every
        self.scheduler_type = scheduler_type
        self.tracking_backend = tracking_backend
        self.project_name = project_name

        # Device
        if device == "auto":
            self.device = torch.device(
                "cuda" if torch.cuda.is_available() else "cpu"
            )
        else:
            self.device = torch.device(device)

        self.model = self.model.to(self.device)

        # Optimizer
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=learning_rate,
            weight_decay=weight_decay,
        )

        # Scheduler (initialized in train())
        self.scheduler = None

        # Tracking
        self._tracker = None
        self.global_step = 0
        self.best_val_loss = float("inf")
        self.patience_counter = 0

        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)

        logger.info(
            f"HDITrainer: lr={learning_rate}, epochs={epochs}, "
            f"device={self.device}, scheduler={scheduler_type}"
        )

    def _init_scheduler(self, total_steps: int) -> None:
        """Initialize learning rate scheduler."""
        if self.scheduler_type == "cosine":
            from torch.optim.lr_scheduler import CosineAnnealingLR

            self.scheduler = CosineAnnealingLR(
                self.optimizer,
                T_max=total_steps - self.warmup_steps,
            )
        elif self.scheduler_type == "step":
            from torch.optim.lr_scheduler import StepLR

            self.scheduler = StepLR(
                self.optimizer, step_size=total_steps // 3, gamma=0.1
            )
        elif self.scheduler_type == "plateau":
            from torch.optim.lr_scheduler import ReduceLROnPlateau

            self.scheduler = ReduceLROnPlateau(
                self.optimizer, mode="min", factor=0.5, patience=5
            )

    def _warmup_lr(self) -> None:
        """Linear warmup for learning rate."""
        if self.global_step < self.warmup_steps:
            warmup_factor = self.global_step / max(self.warmup_steps, 1)
            for param_group in self.optimizer.param_groups:
                param_group["lr"] = self.learning_rate * warmup_factor

    def _init_tracking(self) -> None:
        """Initialize experiment tracking."""
        if self.tracking_backend == "wandb":
            try:
                import wandb

                wandb.init(project=self.project_name)
                self._tracker = "wandb"
                logger.info("W&B tracking initialized")
            except ImportError:
                logger.warning("wandb not installed. Skipping tracking.")
        elif self.tracking_backend == "mlflow":
            try:
                import mlflow

                mlflow.start_run()
                self._tracker = "mlflow"
                logger.info("MLflow tracking initialized")
            except ImportError:
                logger.warning("mlflow not installed. Skipping tracking.")

    def _log_metrics(self, metrics: dict, step: int) -> None:
        """Log metrics to tracking backend."""
        if self._tracker == "wandb":
            import wandb
            wandb.log(metrics, step=step)
        elif self._tracker == "mlflow":
            import mlflow
            mlflow.log_metrics(metrics, step=step)

    def train(
        self,
        train_loader: DataLoader,
        val_loader: Optional[DataLoader],
        graph_data: dict,
    ) -> dict:
        """
        Full training loop.

        Args:
            train_loader: Training data loader
            val_loader: Validation data loader (optional)
            graph_data: Dict with 'node_features', 'edge_index', 'edge_type'
                        (the graph structure, shared across batches)

        Returns:
            Training history dict
        """
        total_steps = len(train_loader) * self.epochs
        self._init_scheduler(total_steps)
        self._init_tracking()

        # Move graph data to device
        node_features = graph_data["node_features"].to(self.device)
        edge_index = graph_data["edge_index"].to(self.device)
        edge_type = graph_data["edge_type"].to(self.device)

        history = {
            "train_loss": [],
            "val_loss": [],
            "learning_rate": [],
        }

        logger.info(f"Starting training for {self.epochs} epochs")

        for epoch in range(1, self.epochs + 1):
            epoch_start = time.time()

            # Training
            train_loss = self._train_epoch(
                train_loader, node_features, edge_index, edge_type
            )
            history["train_loss"].append(train_loss)

            # Validation
            val_loss = None
            if val_loader is not None:
                val_loss = self._validate(
                    val_loader, node_features, edge_index, edge_type
                )
                history["val_loss"].append(val_loss)

            # Learning rate
            current_lr = self.optimizer.param_groups[0]["lr"]
            history["learning_rate"].append(current_lr)

            # Scheduler step
            if self.scheduler is not None:
                if isinstance(self.scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                    if val_loss is not None:
                        self.scheduler.step(val_loss)
                else:
                    self.scheduler.step()

            epoch_time = time.time() - epoch_start

            # Logging
            metrics = {
                "epoch": epoch,
                "train_loss": train_loss,
                "learning_rate": current_lr,
                "epoch_time": epoch_time,
            }
            if val_loss is not None:
                metrics["val_loss"] = val_loss

            self._log_metrics(metrics, self.global_step)

            logger.info(
                f"Epoch {epoch}/{self.epochs} | "
                f"Train Loss: {train_loss:.4f} | "
                f"Val Loss: {val_loss:.4f if val_loss else 'N/A'} | "
                f"LR: {current_lr:.6f} | "
                f"Time: {epoch_time:.1f}s"
            )

            # Checkpointing
            if epoch % self.save_every == 0:
                self._save_checkpoint(epoch, train_loss, val_loss)

            # Early stopping
            if val_loss is not None:
                if val_loss < self.best_val_loss:
                    self.best_val_loss = val_loss
                    self.patience_counter = 0
                    self._save_checkpoint(epoch, train_loss, val_loss, best=True)
                else:
                    self.patience_counter += 1
                    if self.patience_counter >= self.patience:
                        logger.info(
                            f"Early stopping at epoch {epoch} "
                            f"(patience={self.patience})"
                        )
                        break

        logger.info(
            f"Training complete. Best val loss: {self.best_val_loss:.4f}"
        )
        return history

    def _train_epoch(
        self,
        loader: DataLoader,
        node_features: torch.Tensor,
        edge_index: torch.Tensor,
        edge_type: torch.Tensor,
    ) -> float:
        """Train for one epoch."""
        self.model.train()
        total_loss = 0.0
        num_batches = 0

        for batch in loader:
            self.global_step += 1
            self._warmup_lr()

            # Move batch to device
            source_indices = batch["source_indices"].to(self.device)
            target_indices = batch["target_indices"].to(self.device)
            labels = batch["labels"].to(self.device)
            metadata = batch["metadata"].to(self.device)
            evidence_texts = batch.get("evidence_texts", None)

            # Forward pass
            output = self.model(
                node_features=node_features,
                edge_index=edge_index,
                edge_type=edge_type,
                source_indices=source_indices,
                target_indices=target_indices,
                metadata=metadata,
                evidence_texts=evidence_texts,
                labels=labels,
            )

            loss = output["loss"]

            # Backward pass
            self.optimizer.zero_grad()
            loss.backward()

            # Gradient clipping
            if self.grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(), self.grad_clip
                )

            self.optimizer.step()

            total_loss += loss.item()
            num_batches += 1

        return total_loss / max(num_batches, 1)

    @torch.no_grad()
    def _validate(
        self,
        loader: DataLoader,
        node_features: torch.Tensor,
        edge_index: torch.Tensor,
        edge_type: torch.Tensor,
    ) -> float:
        """Validate on the validation set."""
        self.model.eval()
        total_loss = 0.0
        num_batches = 0

        for batch in loader:
            source_indices = batch["source_indices"].to(self.device)
            target_indices = batch["target_indices"].to(self.device)
            labels = batch["labels"].to(self.device)
            metadata = batch["metadata"].to(self.device)
            evidence_texts = batch.get("evidence_texts", None)

            output = self.model(
                node_features=node_features,
                edge_index=edge_index,
                edge_type=edge_type,
                source_indices=source_indices,
                target_indices=target_indices,
                metadata=metadata,
                evidence_texts=evidence_texts,
                labels=labels,
            )

            total_loss += output["loss"].item()
            num_batches += 1

        return total_loss / max(num_batches, 1)

    def _save_checkpoint(
        self,
        epoch: int,
        train_loss: float,
        val_loss: Optional[float],
        best: bool = False,
    ) -> None:
        """Save model checkpoint."""
        filename = "best_model.pt" if best else f"checkpoint_epoch_{epoch}.pt"
        path = self.checkpoint_dir / filename

        checkpoint = {
            "epoch": epoch,
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "train_loss": train_loss,
            "val_loss": val_loss,
            "global_step": self.global_step,
        }

        torch.save(checkpoint, path)
        logger.info(f"Saved checkpoint: {path}")

    def load_checkpoint(self, path: str | Path) -> dict:
        """Load model from checkpoint."""
        checkpoint = torch.load(path, map_location=self.device)
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        self.global_step = checkpoint.get("global_step", 0)
        logger.info(f"Loaded checkpoint from {path}")
        return checkpoint
