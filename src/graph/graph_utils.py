"""
Graph Utilities

Helper functions for knowledge graph operations:
- Train/val/test edge splitting
- Negative edge sampling
- Graph statistics and validation
- Subgraph extraction
"""

from __future__ import annotations

import random
from typing import Optional

import torch
from loguru import logger


def split_edges(
    edge_index: torch.Tensor,
    train_ratio: float = 0.8,
    val_ratio: float = 0.1,
    test_ratio: float = 0.1,
    seed: int = 42,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Split edge indices into train/val/test sets.

    Args:
        edge_index: (2, num_edges) tensor
        train_ratio: Fraction of edges for training
        val_ratio: Fraction for validation
        test_ratio: Fraction for testing
        seed: Random seed

    Returns:
        (train_edges, val_edges, test_edges) — each a (2, N) tensor
    """
    assert abs(train_ratio + val_ratio + test_ratio - 1.0) < 1e-6

    num_edges = edge_index.size(1)
    rng = random.Random(seed)

    indices = list(range(num_edges))
    rng.shuffle(indices)

    train_end = int(num_edges * train_ratio)
    val_end = train_end + int(num_edges * val_ratio)

    train_idx = indices[:train_end]
    val_idx = indices[train_end:val_end]
    test_idx = indices[val_end:]

    return (
        edge_index[:, train_idx],
        edge_index[:, val_idx],
        edge_index[:, test_idx],
    )


def sample_negative_edges(
    edge_index: torch.Tensor,
    num_nodes: int,
    num_negatives: int,
    seed: int = 42,
) -> torch.Tensor:
    """
    Sample negative edges (non-existing edges) for training.

    Args:
        edge_index: (2, num_edges) positive edge tensor
        num_nodes: Total number of nodes in the graph
        num_negatives: Number of negative edges to sample
        seed: Random seed

    Returns:
        neg_edge_index: (2, num_negatives) tensor of negative edges
    """
    rng = random.Random(seed)

    # Build positive edge set for fast lookup
    positive_set = set()
    for i in range(edge_index.size(1)):
        src, tgt = edge_index[0, i].item(), edge_index[1, i].item()
        positive_set.add((src, tgt))

    neg_sources = []
    neg_targets = []
    attempts = 0
    max_attempts = num_negatives * 10

    while len(neg_sources) < num_negatives and attempts < max_attempts:
        src = rng.randint(0, num_nodes - 1)
        tgt = rng.randint(0, num_nodes - 1)
        if src != tgt and (src, tgt) not in positive_set:
            neg_sources.append(src)
            neg_targets.append(tgt)
            positive_set.add((src, tgt))  # Avoid duplicates
        attempts += 1

    return torch.tensor([neg_sources, neg_targets], dtype=torch.long)


def compute_graph_statistics(
    edge_index: torch.Tensor, num_nodes: int
) -> dict:
    """Compute basic graph statistics."""
    num_edges = edge_index.size(1)

    # Node degrees
    degrees = torch.zeros(num_nodes, dtype=torch.long)
    for i in range(num_edges):
        degrees[edge_index[0, i]] += 1
        degrees[edge_index[1, i]] += 1

    return {
        "num_nodes": num_nodes,
        "num_edges": num_edges,
        "avg_degree": degrees.float().mean().item(),
        "max_degree": degrees.max().item(),
        "min_degree": degrees.min().item(),
        "density": num_edges / (num_nodes * (num_nodes - 1))
        if num_nodes > 1
        else 0,
        "isolated_nodes": (degrees == 0).sum().item(),
    }


def extract_subgraph(
    edge_index: torch.Tensor,
    node_features: torch.Tensor,
    center_node: int,
    num_hops: int = 2,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Extract a k-hop subgraph around a center node.

    Args:
        edge_index: (2, num_edges) edge tensor
        node_features: (num_nodes, feature_dim) feature tensor
        center_node: Index of the center node
        num_hops: Number of hops for neighborhood

    Returns:
        (sub_edge_index, sub_features, node_mapping)
    """
    visited = {center_node}
    frontier = {center_node}

    for _ in range(num_hops):
        new_frontier = set()
        for i in range(edge_index.size(1)):
            src, tgt = edge_index[0, i].item(), edge_index[1, i].item()
            if src in frontier and tgt not in visited:
                new_frontier.add(tgt)
            if tgt in frontier and src not in visited:
                new_frontier.add(src)
        visited.update(new_frontier)
        frontier = new_frontier

    # Create node mapping
    node_list = sorted(visited)
    node_map = {n: i for i, n in enumerate(node_list)}

    # Filter edges
    sub_src, sub_tgt = [], []
    for i in range(edge_index.size(1)):
        src, tgt = edge_index[0, i].item(), edge_index[1, i].item()
        if src in node_map and tgt in node_map:
            sub_src.append(node_map[src])
            sub_tgt.append(node_map[tgt])

    sub_edge_index = torch.tensor([sub_src, sub_tgt], dtype=torch.long)
    sub_features = node_features[node_list]
    node_mapping = torch.tensor(node_list, dtype=torch.long)

    return sub_edge_index, sub_features, node_mapping


class GraphUtils:
    """Container class for graph utility functions."""

    split_edges = staticmethod(split_edges)
    sample_negative_edges = staticmethod(sample_negative_edges)
    compute_graph_statistics = staticmethod(compute_graph_statistics)
    extract_subgraph = staticmethod(extract_subgraph)
