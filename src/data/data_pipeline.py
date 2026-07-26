"""
End-to-End Data Pipeline

Orchestrates the full data pipeline from raw sources to model-ready
tensors. Currently uses synthetic data for development; will be
extended to real DrugBank/ChEMBL data when access is obtained.

Pipeline stages:
1. Data generation / loading
2. Knowledge graph construction
3. Dataset creation with negative sampling
4. DataLoader creation with proper collation
5. Train/val/test split management
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import torch
from torch.utils.data import DataLoader
from loguru import logger

from src.data.synthetic_data import SyntheticDataGenerator
from src.data.dataset import HDIDataset, HDICollator, create_dataloader


class DataPipeline:
    """
    End-to-end data pipeline that produces everything needed for training.

    Usage:
        pipeline = DataPipeline(mode="synthetic")
        data = pipeline.build()

        # data contains:
        # - train_loader, val_loader, test_loader
        # - graph_data (node_features, edge_index, edge_type)
        # - node_to_idx mapping
        # - num_relations
    """

    def __init__(
        self,
        mode: str = "synthetic",
        data_dir: Optional[str] = None,
        batch_size: int = 64,
        negative_ratio: int = 5,
        num_workers: int = 0,
        feature_dim: int = 128,
        seed: int = 42,
    ):
        """
        Args:
            mode: "synthetic" or "real" (real requires data_dir)
            data_dir: Path to real data directory
            batch_size: Batch size for DataLoaders
            negative_ratio: Negative samples per positive edge
            num_workers: DataLoader workers
            feature_dim: Node feature dimensionality
            seed: Random seed
        """
        self.mode = mode
        self.data_dir = Path(data_dir) if data_dir else None
        self.batch_size = batch_size
        self.negative_ratio = negative_ratio
        self.num_workers = num_workers
        self.feature_dim = feature_dim
        self.seed = seed

    def build(self) -> dict:
        """
        Build the complete data pipeline.

        Returns:
            dict with keys:
                train_loader: DataLoader for training
                val_loader: DataLoader for validation
                test_loader: DataLoader for testing
                graph_data: dict with node_features, edge_index, edge_type
                node_to_idx: dict mapping node_id → index
                all_node_ids: list of all node IDs
                node_names: dict mapping node_id → name
                num_relations: number of edge types
        """
        logger.info(f"Building data pipeline (mode={self.mode})...")

        if self.mode == "synthetic":
            return self._build_synthetic()
        elif self.mode == "real":
            return self._build_real()
        else:
            raise ValueError(f"Unknown mode: {self.mode}")

    def _build_synthetic(self) -> dict:
        """Build pipeline from synthetic data."""
        gen = SyntheticDataGenerator(
            feature_dim=self.feature_dim,
            seed=self.seed,
        )
        data = gen.generate()

        # Create datasets
        train_dataset = HDIDataset(
            positive_edges=data["train_edges"],
            all_node_ids=data["all_node_ids"],
            negative_ratio=self.negative_ratio,
            seed=self.seed,
        )
        val_dataset = HDIDataset(
            positive_edges=data["val_edges"],
            all_node_ids=data["all_node_ids"],
            negative_ratio=self.negative_ratio,
            seed=self.seed + 1,
        )
        test_dataset = HDIDataset(
            positive_edges=data["test_edges"],
            all_node_ids=data["all_node_ids"],
            negative_ratio=self.negative_ratio,
            seed=self.seed + 2,
        )

        # Create data loaders
        train_loader = create_dataloader(
            train_dataset,
            data["node_to_idx"],
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
        )
        val_loader = create_dataloader(
            val_dataset,
            data["node_to_idx"],
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
        )
        test_loader = create_dataloader(
            test_dataset,
            data["node_to_idx"],
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
        )

        logger.info(
            f"Pipeline built: "
            f"train={len(train_dataset)} samples, "
            f"val={len(val_dataset)} samples, "
            f"test={len(test_dataset)} samples"
        )

        return {
            "train_loader": train_loader,
            "val_loader": val_loader,
            "test_loader": test_loader,
            "graph_data": data["graph_data"],
            "node_to_idx": data["node_to_idx"],
            "all_node_ids": data["all_node_ids"],
            "node_names": data.get("node_names", {}),
            "num_relations": data["num_relations"],
        }

    def _build_real(self) -> dict:
        """Build pipeline from real data sources."""
        raise NotImplementedError(
            "Real data pipeline requires DrugBank/ChEMBL access. "
            "Use mode='synthetic' for development."
        )
