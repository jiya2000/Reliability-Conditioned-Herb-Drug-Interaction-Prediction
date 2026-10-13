"""
Test P0 fixes: verify no graph edge leakage and strict negative sampling.
"""

import pytest
import torch
from src.data.real_data import RealDataGenerator
from src.data.dataset import HDIDataset
from src.data.data_pipeline import DataPipeline


def test_real_data_no_graph_leakage():
    """Verify that message-passing edge_index contains ONLY train edges."""
    generator = RealDataGenerator(
        data_path="data/processed/drugbank_parsed.json",
        feature_dim=64,
        seed=42,
    )
    data = generator.generate()

    edge_index = data["graph_data"]["edge_index"]
    node_to_idx = data["node_to_idx"]

    # Convert graph edges to a set of index pairs
    graph_edges = set()
    for col in range(edge_index.shape[1]):
        u = edge_index[0, col].item()
        v = edge_index[1, col].item()
        graph_edges.add((u, v))

    # Assert that NO validation edge is in the graph (either direction)
    for edge in data["val_edges"]:
        u = node_to_idx[edge["source_id"]]
        v = node_to_idx[edge["target_id"]]
        assert (u, v) not in graph_edges, f"Validation edge ({u}, {v}) leaked into message passing graph!"
        assert (v, u) not in graph_edges, f"Reverse validation edge ({v}, {u}) leaked into message passing graph!"

    # Assert that NO test edge is in the graph (either direction)
    for edge in data["test_edges"]:
        u = node_to_idx[edge["source_id"]]
        v = node_to_idx[edge["target_id"]]
        assert (u, v) not in graph_edges, f"Test edge ({u}, {v}) leaked into message passing graph!"
        assert (v, u) not in graph_edges, f"Reverse test edge ({v}, {u}) leaked into message passing graph!"

    # Assert train edges ARE in the graph
    train_edges_found = 0
    for edge in data["train_edges"]:
        u = node_to_idx[edge["source_id"]]
        v = node_to_idx[edge["target_id"]]
        if (u, v) in graph_edges:
            train_edges_found += 1
    assert train_edges_found == len(data["train_edges"]), "Some train edges are missing from the graph!"


def test_negative_sampling_global_exclusion():
    """Verify negative sampling excludes all known positives across all splits."""
    all_nodes = [f"node_{i}" for i in range(10)]
    train_pos = [{"source_id": "node_0", "target_id": "node_1"}]
    val_pos = [{"source_id": "node_2", "target_id": "node_3"}]
    test_pos = [{"source_id": "node_4", "target_id": "node_5"}]

    all_known_pos = train_pos + val_pos + test_pos

    # Create train dataset passing all_known_pos
    dataset = HDIDataset(
        positive_edges=train_pos,
        all_node_ids=all_nodes,
        negative_ratio=5,
        seed=42,
        known_positive_edges=all_known_pos,
    )

    known_pairs = set()
    for e in all_known_pos:
        known_pairs.add((e["source_id"], e["target_id"]))
        known_pairs.add((e["target_id"], e["source_id"]))

    for sample in dataset.samples:
        if sample.label == 0.0:
            pair = (sample.source_id, sample.target_id)
            assert pair not in known_pairs, f"Known positive {pair} was sampled as a negative!"
            # Negatives should have empty evidence text and zero metadata
            assert sample.evidence_texts == []
            assert sample.corroboration_count == 0
            assert sample.biomedical_quality == 0.0
