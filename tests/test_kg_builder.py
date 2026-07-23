"""
Tests for the Knowledge Graph Builder
"""

import pytest
import torch
import pandas as pd

from src.graph.kg_builder import KnowledgeGraphBuilder


class MockDataLoader:
    def load_drugs(self):
        return pd.DataFrame([
            {"drugbank_id": "DB001", "name": "DrugA"},
            {"drugbank_id": "DB002", "name": "DrugB"}
        ])

    def load_interactions(self):
        return pd.DataFrame([
            {"drug1_id": "DB001", "drug2_id": "DB002"}
        ])
        
    def load_targets(self):
        return pd.DataFrame([
            {"drug_id": "DB001", "uniprot_id": "T001", "target_name": "TargetA"}
        ])


def test_kg_builder_initialization():
    builder = KnowledgeGraphBuilder(feature_dim=16)
    assert builder.feature_dim == 16
    assert len(builder._nodes) == 0


def test_kg_builder_add_node():
    builder = KnowledgeGraphBuilder()
    builder.add_node("n1", "drug", "Aspirin")
    assert "n1" in builder._nodes
    assert builder._nodes["n1"].node_type == "drug"
    assert builder._node_type_counts["drug"] == 1


def test_kg_builder_drugbank_loading():
    builder = KnowledgeGraphBuilder()
    loader = MockDataLoader()
    builder.add_drugs_from_drugbank(loader)
    
    assert "DB001" in builder._nodes
    assert "DB002" in builder._nodes
    assert "T001" in builder._nodes
    assert len(builder._edges) == 2  # 1 DDI, 1 DTI
    
    stats = builder.get_statistics()
    assert stats["total_nodes"] == 3
    assert stats["total_edges"] == 2
