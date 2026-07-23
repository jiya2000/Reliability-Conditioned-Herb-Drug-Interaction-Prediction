"""
PyTorch Dataset and Collator for HDI Prediction

Wraps the knowledge graph and text data into PyTorch-compatible
Dataset and DataLoader with negative sampling for link prediction.
"""

from __future__ import annotations

import random
from dataclasses import dataclass, field
from typing import Optional

import torch
from torch.utils.data import Dataset, DataLoader
from loguru import logger


@dataclass
class HDISample:
    """A single training/evaluation sample for herb-drug interaction prediction."""

    # Source node (herb or drug)
    source_id: str
    source_name: str
    source_type: str  # "herb", "drug"

    # Target node (drug or herb)
    target_id: str
    target_name: str
    target_type: str  # "drug", "herb"

    # Label
    label: float  # 1.0 = known interaction, 0.0 = no known interaction

    # Text evidence (if available)
    evidence_texts: list[str] = field(default_factory=list)
    evidence_sources: list[str] = field(default_factory=list)

    # Candidate metadata for reliability scoring (C, T, B, M, S)
    corroboration_count: int = 0  # C: number of independent sources
    temporal_recency: float = 0.0  # T: recency score [0, 1]
    biomedical_quality: float = 0.0  # B: journal quality / peer-review
    molecular_plausibility: float = 0.0  # M: structural similarity score
    source_type_code: int = 0  # S: source type (0=unknown, 1=clinical, 2=forum, etc.)


class HDIDataset(Dataset):
    """
    PyTorch Dataset for herb-drug interaction link prediction.

    Provides positive edges (known interactions) and generates negative
    edges (random non-interacting pairs) for training.

    The dataset stores:
    - Graph edge pairs with labels
    - Associated text evidence
    - Candidate metadata for reliability scoring
    """

    SOURCE_TYPE_MAP = {
        "unknown": 0,
        "clinical_trial": 1,
        "peer_reviewed": 2,
        "textbook": 3,
        "case_report": 4,
        "health_forum": 5,
        "social_media": 6,
        "traditional_medicine": 7,
    }

    def __init__(
        self,
        positive_edges: list[dict],
        all_node_ids: list[str],
        negative_ratio: int = 5,
        seed: int = 42,
    ):
        """
        Args:
            positive_edges: List of dicts with keys:
                source_id, source_name, source_type, target_id,
                target_name, target_type, evidence_texts, ...
            all_node_ids: All node IDs in the graph (for negative sampling)
            negative_ratio: Number of negative samples per positive edge
            seed: Random seed for reproducibility
        """
        super().__init__()
        self.positive_edges = positive_edges
        self.all_node_ids = all_node_ids
        self.negative_ratio = negative_ratio
        self.rng = random.Random(seed)

        # Build positive edge set for fast negative validation
        self._positive_set = set()
        for edge in positive_edges:
            pair = (edge["source_id"], edge["target_id"])
            self._positive_set.add(pair)
            self._positive_set.add((pair[1], pair[0]))  # Symmetric

        # Generate samples
        self.samples = self._build_samples()
        logger.info(
            f"HDIDataset: {len(positive_edges)} positive edges, "
            f"{len(self.samples)} total samples (ratio 1:{negative_ratio})"
        )

    def _build_samples(self) -> list[HDISample]:
        """Build positive and negative samples."""
        samples = []

        # Positive samples
        for edge in self.positive_edges:
            samples.append(
                HDISample(
                    source_id=edge["source_id"],
                    source_name=edge.get("source_name", ""),
                    source_type=edge.get("source_type", "drug"),
                    target_id=edge["target_id"],
                    target_name=edge.get("target_name", ""),
                    target_type=edge.get("target_type", "drug"),
                    label=1.0,
                    evidence_texts=edge.get("evidence_texts", []),
                    evidence_sources=edge.get("evidence_sources", []),
                    corroboration_count=edge.get("corroboration_count", 1),
                    temporal_recency=edge.get("temporal_recency", 0.5),
                    biomedical_quality=edge.get("biomedical_quality", 0.5),
                    molecular_plausibility=edge.get(
                        "molecular_plausibility", 0.5
                    ),
                    source_type_code=self.SOURCE_TYPE_MAP.get(
                        edge.get("evidence_source_type", "unknown"), 0
                    ),
                )
            )

        # Negative samples (random non-interacting pairs)
        for edge in self.positive_edges:
            for _ in range(self.negative_ratio):
                neg_target = self.rng.choice(self.all_node_ids)
                # Ensure it's actually negative
                while (edge["source_id"], neg_target) in self._positive_set:
                    neg_target = self.rng.choice(self.all_node_ids)

                samples.append(
                    HDISample(
                        source_id=edge["source_id"],
                        source_name=edge.get("source_name", ""),
                        source_type=edge.get("source_type", "drug"),
                        target_id=neg_target,
                        target_name="",
                        target_type="drug",
                        label=0.0,
                        corroboration_count=0,
                        temporal_recency=0.0,
                        biomedical_quality=0.0,
                        molecular_plausibility=0.0,
                        source_type_code=0,
                    )
                )

        self.rng.shuffle(samples)
        return samples

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> HDISample:
        return self.samples[idx]


class HDICollator:
    """
    Custom collator for batching HDISamples.

    Converts a list of HDISample objects into tensors suitable
    for the HDI model forward pass.
    """

    def __init__(self, node_to_idx: dict[str, int]):
        """
        Args:
            node_to_idx: Mapping from node ID strings to integer indices
                         (matching the graph's node ordering)
        """
        self.node_to_idx = node_to_idx

    def __call__(self, batch: list[HDISample]) -> dict[str, torch.Tensor]:
        """Collate a batch of HDISamples into tensors."""
        source_indices = []
        target_indices = []
        labels = []
        metadata = []  # (C, T, B, M, S) tuples
        evidence_texts = []

        for sample in batch:
            src_idx = self.node_to_idx.get(sample.source_id, 0)
            tgt_idx = self.node_to_idx.get(sample.target_id, 0)

            source_indices.append(src_idx)
            target_indices.append(tgt_idx)
            labels.append(sample.label)

            # Candidate metadata for reliability scorer
            metadata.append(
                [
                    float(sample.corroboration_count),
                    sample.temporal_recency,
                    sample.biomedical_quality,
                    sample.molecular_plausibility,
                    float(sample.source_type_code),
                ]
            )

            # Collect evidence texts (padded to max in batch)
            if sample.evidence_texts:
                evidence_texts.append(sample.evidence_texts[0])
            else:
                evidence_texts.append("")

        return {
            "source_indices": torch.tensor(source_indices, dtype=torch.long),
            "target_indices": torch.tensor(target_indices, dtype=torch.long),
            "labels": torch.tensor(labels, dtype=torch.float),
            "metadata": torch.tensor(metadata, dtype=torch.float),
            "evidence_texts": evidence_texts,  # List of strings (tokenized later)
        }


def create_dataloader(
    dataset: HDIDataset,
    node_to_idx: dict[str, int],
    batch_size: int = 64,
    shuffle: bool = True,
    num_workers: int = 0,
) -> DataLoader:
    """Create a DataLoader with the HDI collator."""
    collator = HDICollator(node_to_idx)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        collate_fn=collator,
    )
