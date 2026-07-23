"""
Heterogeneous Knowledge Graph Builder

Constructs a heterogeneous graph with multiple node and edge types
from DrugBank, ChEMBL, and IMPPAT data sources. Supports both
NetworkX (for inspection) and PyTorch Geometric HeteroData (for GNN).

Node types: drug, herb, target, enzyme, compound, pathway, effect
Edge types: interacts_with, targets, metabolized_by, contains,
            in_pathway, has_effect, etc.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import torch
from loguru import logger

try:
    import networkx as nx
except ImportError:
    nx = None

try:
    from torch_geometric.data import HeteroData
except ImportError:
    HeteroData = None


@dataclass
class KGNode:
    """A node in the knowledge graph."""

    node_id: str
    node_type: str  # drug, herb, target, enzyme, compound, pathway, effect
    name: str
    properties: dict = field(default_factory=dict)


@dataclass
class KGEdge:
    """An edge in the knowledge graph."""

    source_id: str
    target_id: str
    source_type: str
    target_type: str
    edge_type: str  # interacts_with, targets, metabolized_by, contains, etc.
    properties: dict = field(default_factory=dict)
    weight: float = 1.0


class KnowledgeGraphBuilder:
    """
    Builds a heterogeneous knowledge graph from multiple data sources.

    The KG serves as the structural backbone for GNN-based link prediction.
    Ground-truth interactions (from DrugBank) form the anchor set for
    training the reliability scorer end-to-end.

    Usage:
        builder = KnowledgeGraphBuilder()

        # Add nodes and edges from data loaders
        builder.add_drugs_from_drugbank(drugbank_loader)
        builder.add_herbs_from_imppat(imppat_loader)
        builder.add_compounds_from_chembl(chembl_loader)
        builder.add_interactions(interactions_df)

        # Export for GNN
        hetero_data = builder.to_pyg_heterodata()

        # Or for inspection
        nx_graph = builder.to_networkx()
    """

    def __init__(self, feature_dim: int = 128):
        self.feature_dim = feature_dim
        self._nodes: dict[str, KGNode] = {}  # node_id → KGNode
        self._edges: list[KGEdge] = []
        self._node_type_counts: dict[str, int] = {}

        # Index mappings for PyG conversion
        self._type_node_idx: dict[str, dict[str, int]] = {}
        # node_type → {node_id → local_index}

    def add_node(
        self,
        node_id: str,
        node_type: str,
        name: str,
        properties: Optional[dict] = None,
    ) -> None:
        """Add a node to the knowledge graph."""
        if node_id in self._nodes:
            return  # Skip duplicates

        self._nodes[node_id] = KGNode(
            node_id=node_id,
            node_type=node_type,
            name=name,
            properties=properties or {},
        )

        # Update type-specific index
        if node_type not in self._type_node_idx:
            self._type_node_idx[node_type] = {}
        self._type_node_idx[node_type][node_id] = len(
            self._type_node_idx[node_type]
        )

        self._node_type_counts[node_type] = (
            self._node_type_counts.get(node_type, 0) + 1
        )

    def add_edge(
        self,
        source_id: str,
        target_id: str,
        edge_type: str,
        properties: Optional[dict] = None,
        weight: float = 1.0,
    ) -> None:
        """Add an edge to the knowledge graph."""
        if source_id not in self._nodes or target_id not in self._nodes:
            logger.debug(
                f"Skipping edge {source_id} -> {target_id}: "
                "one or both nodes not in graph"
            )
            return

        source_type = self._nodes[source_id].node_type
        target_type = self._nodes[target_id].node_type

        self._edges.append(
            KGEdge(
                source_id=source_id,
                target_id=target_id,
                source_type=source_type,
                target_type=target_type,
                edge_type=edge_type,
                properties=properties or {},
                weight=weight,
            )
        )

    def add_drugs_from_drugbank(self, drugbank_loader) -> None:
        """Add drug nodes and interaction edges from DrugBank."""
        drugs_df = drugbank_loader.load_drugs()
        interactions_df = drugbank_loader.load_interactions()
        targets_df = drugbank_loader.load_targets()

        # Add drug nodes
        for _, row in drugs_df.iterrows():
            self.add_node(
                node_id=row["drugbank_id"],
                node_type="drug",
                name=row["name"],
                properties={
                    "smiles": row.get("smiles", ""),
                    "drug_type": row.get("drug_type", ""),
                    "molecular_weight": row.get("molecular_weight", 0),
                },
            )

        # Add drug-drug interaction edges (ground truth anchor set)
        for _, row in interactions_df.iterrows():
            self.add_edge(
                source_id=row["drug1_id"],
                target_id=row["drug2_id"],
                edge_type="interacts_with",
                properties={"description": row.get("description", "")},
            )

        # Add target nodes and drug-target edges
        for _, row in targets_df.iterrows():
            target_id = row.get("uniprot_id", "") or f"target_{row['target_name']}"
            self.add_node(
                node_id=target_id,
                node_type="target",
                name=row["target_name"],
                properties={
                    "gene_name": row.get("gene_name", ""),
                    "organism": row.get("organism", ""),
                },
            )
            self.add_edge(
                source_id=row["drug_id"],
                target_id=target_id,
                edge_type="targets",
                properties={"actions": row.get("actions", [])},
            )

        logger.info(
            f"Added {len(drugs_df)} drugs, "
            f"{len(interactions_df)} DDIs, "
            f"{len(targets_df)} drug-target edges from DrugBank"
        )

    def add_herbs_from_imppat(self, imppat_loader) -> None:
        """Add herb nodes and herb-compound edges from IMPPAT."""
        herbs_df = imppat_loader.load_herbs()
        phyto_df = imppat_loader.load_phytochemicals()
        edges = imppat_loader.load_herb_compound_edges()

        # Add herb nodes
        for _, row in herbs_df.iterrows():
            self.add_node(
                node_id=row["plant_id"],
                node_type="herb",
                name=row["plant_name"],
                properties={
                    "botanical_name": row.get("botanical_name", ""),
                    "family": row.get("family", ""),
                    "traditional_uses": row.get("traditional_uses", []),
                },
            )

        # Add phytochemical (compound) nodes
        for _, row in phyto_df.iterrows():
            self.add_node(
                node_id=row["compound_id"],
                node_type="compound",
                name=row["compound_name"],
                properties={
                    "smiles": row.get("smiles", ""),
                    "molecular_weight": row.get("molecular_weight", 0),
                },
            )

        # Add herb → compound edges
        for herb_id, compound_id in edges:
            self.add_edge(
                source_id=herb_id,
                target_id=compound_id,
                edge_type="contains",
            )

        logger.info(
            f"Added {len(herbs_df)} herbs, "
            f"{len(phyto_df)} phytochemicals, "
            f"{len(edges)} herb-compound edges from IMPPAT"
        )

    def add_compounds_from_chembl(self, chembl_loader) -> None:
        """Add compound and activity data from ChEMBL."""
        compounds_df = chembl_loader.load_compounds()
        activities_df = chembl_loader.load_activities()
        targets_df = chembl_loader.load_targets()

        # Add compound nodes (merge with existing if possible)
        for _, row in compounds_df.iterrows():
            self.add_node(
                node_id=row["chembl_id"],
                node_type="compound",
                name=row.get("pref_name", row["chembl_id"]),
                properties={
                    "smiles": row.get("smiles", ""),
                    "max_phase": row.get("max_phase", 0),
                    "molecular_weight": row.get("molecular_weight", 0),
                },
            )

        # Add ChEMBL target nodes
        for _, row in targets_df.iterrows():
            self.add_node(
                node_id=row["target_chembl_id"],
                node_type="target",
                name=row.get("pref_name", ""),
                properties={
                    "target_type": row.get("target_type", ""),
                    "organism": row.get("organism", ""),
                },
            )

        # Add compound-target activity edges
        for _, row in activities_df.iterrows():
            self.add_edge(
                source_id=row["chembl_id"],
                target_id=row["target_chembl_id"],
                edge_type="has_activity",
                properties={
                    "type": row.get("standard_type", ""),
                    "value": row.get("standard_value", 0),
                    "pchembl": row.get("pchembl_value", 0),
                },
            )

        logger.info(
            f"Added {len(compounds_df)} compounds, "
            f"{len(targets_df)} targets, "
            f"{len(activities_df)} activity edges from ChEMBL"
        )

    def add_extracted_interactions(
        self,
        metadata_instances: list,
    ) -> None:
        """
        Add interaction edges extracted by the NLP pipeline.

        These are the candidate edges — not yet confirmed.
        The reliability scorer will weight these during fusion.
        """
        added = 0
        for meta in metadata_instances:
            # Ensure both entities exist as nodes
            if meta.entity1_id not in self._nodes:
                self.add_node(
                    node_id=meta.entity1_id,
                    node_type=meta.entity1_type.lower(),
                    name=meta.entity1_name,
                )
            if meta.entity2_id not in self._nodes:
                self.add_node(
                    node_id=meta.entity2_id,
                    node_type=meta.entity2_type.lower(),
                    name=meta.entity2_name,
                )

            self.add_edge(
                source_id=meta.entity1_id,
                target_id=meta.entity2_id,
                edge_type=meta.relation_type,
                properties={
                    "evidence_text": meta.evidence_text,
                    "corroboration": meta.corroboration_count,
                    "confidence": meta.confidence,
                    "source_type": meta.source_type,
                    "is_extracted": True,
                },
            )
            added += 1

        logger.info(f"Added {added} extracted interaction edges to KG")

    def to_pyg_heterodata(self) -> "HeteroData":
        """
        Convert the knowledge graph to PyTorch Geometric HeteroData.

        Returns:
            HeteroData with node features (random init) and edge indices
            for each (source_type, edge_type, target_type) triplet.
        """
        if HeteroData is None:
            raise ImportError(
                "torch_geometric is required. Install with: "
                "pip install torch-geometric"
            )

        data = HeteroData()

        # Node features (random initialization — will be learned)
        for node_type, node_dict in self._type_node_idx.items():
            num_nodes = len(node_dict)
            data[node_type].x = torch.randn(num_nodes, self.feature_dim)
            data[node_type].num_nodes = num_nodes

            # Store node ID mapping
            data[node_type].node_ids = list(node_dict.keys())

        # Edge indices grouped by (src_type, edge_type, tgt_type)
        edge_groups: dict[tuple[str, str, str], tuple[list, list]] = {}

        for edge in self._edges:
            key = (edge.source_type, edge.edge_type, edge.target_type)

            if key not in edge_groups:
                edge_groups[key] = ([], [])

            src_idx = self._type_node_idx.get(
                edge.source_type, {}
            ).get(edge.source_id)
            tgt_idx = self._type_node_idx.get(
                edge.target_type, {}
            ).get(edge.target_id)

            if src_idx is not None and tgt_idx is not None:
                edge_groups[key][0].append(src_idx)
                edge_groups[key][1].append(tgt_idx)

        for (src_type, edge_type, tgt_type), (src_list, tgt_list) in edge_groups.items():
            edge_index = torch.tensor(
                [src_list, tgt_list], dtype=torch.long
            )
            data[src_type, edge_type, tgt_type].edge_index = edge_index

        logger.info(
            f"Built PyG HeteroData: "
            f"{sum(len(v) for v in self._type_node_idx.values())} nodes, "
            f"{len(self._edges)} edges, "
            f"{len(edge_groups)} edge types"
        )

        return data

    def to_networkx(self) -> "nx.MultiDiGraph":
        """Convert to NetworkX for visualization and inspection."""
        if nx is None:
            raise ImportError("networkx is required: pip install networkx")

        G = nx.MultiDiGraph()

        for node_id, node in self._nodes.items():
            G.add_node(
                node_id,
                node_type=node.node_type,
                name=node.name,
                **node.properties,
            )

        for edge in self._edges:
            G.add_edge(
                edge.source_id,
                edge.target_id,
                edge_type=edge.edge_type,
                weight=edge.weight,
                **edge.properties,
            )

        return G

    def get_node_to_idx(self) -> dict[str, int]:
        """Get a flat node_id → global index mapping."""
        mapping = {}
        idx = 0
        for node_type in sorted(self._type_node_idx.keys()):
            for node_id in self._type_node_idx[node_type]:
                mapping[node_id] = idx
                idx += 1
        return mapping

    def get_statistics(self) -> dict:
        """Get graph statistics."""
        edge_type_counts = {}
        for edge in self._edges:
            key = f"{edge.source_type}-{edge.edge_type}-{edge.target_type}"
            edge_type_counts[key] = edge_type_counts.get(key, 0) + 1

        return {
            "total_nodes": len(self._nodes),
            "total_edges": len(self._edges),
            "node_type_counts": dict(self._node_type_counts),
            "edge_type_counts": edge_type_counts,
        }
