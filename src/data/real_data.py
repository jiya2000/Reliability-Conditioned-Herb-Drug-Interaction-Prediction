import json
import torch
import random
import numpy as np
from loguru import logger
import os

class RealDataGenerator:
    def __init__(self, data_path, feature_dim=128, seed=42):
        self.data_path = data_path
        self.feature_dim = feature_dim
        self.seed = seed
        self.rng = random.Random(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)

    def generate(self):
        logger.info(f"Loading real data from {self.data_path}")
        with open(self.data_path, "r") as f:
            data = json.load(f)
            
        drugs = data["drugs"]
        interactions = data["interactions"]
        
        # Sort for determinism
        drug_ids = sorted(list(drugs.keys()))
        
        all_ids = drug_ids
        node_names = drugs
        
        node_to_idx = {nid: i for i, nid in enumerate(all_ids)}
        num_nodes = len(all_ids)
        
        node_features = torch.randn((num_nodes, self.feature_dim))
        
        edge_src, edge_tgt = [], []
        edge_types = []
        all_positive_edges = []
        
        # For simplicity in this demo, let's treat all interactions as "drug_interacts_drug"
        # and randomly assign properties like reliability.
        EDGE_TYPE_MAP = {
            "drug_interacts_drug": 0,
            "herb_interacts_drug": 1,
            "drug_targets_target": 2
        }
        
        for u, v, desc in interactions:
            if u not in node_to_idx or v not in node_to_idx:
                continue
                
            u_idx = node_to_idx[u]
            v_idx = node_to_idx[v]
            
            # Bidirectional graph for RGCN
            edge_src.extend([u_idx, v_idx])
            edge_tgt.extend([v_idx, u_idx])
            edge_types.extend([0, 0])  # drug_interacts_drug
            
            # Add to positive edges for link prediction
            edge_data = {
                "source_id": u,
                "target_id": v,
                "evidence": desc[:200] + "...",  # truncate for memory
                "corroboration_count": max(1, int(np.random.exponential(2))),
                "biomedical_quality": min(1.0, max(0.0, self.rng.gauss(0.8, 0.1))),
                "temporal_recency": min(1.0, max(0.0, self.rng.gauss(0.7, 0.2))),
                "source_type": "peer_reviewed"
            }
            all_positive_edges.append(edge_data)
            
        edge_index = torch.tensor([edge_src, edge_tgt], dtype=torch.long)
        edge_type = torch.tensor(edge_types, dtype=torch.long)
        
        # Splits
        self.rng.shuffle(all_positive_edges)
        n = len(all_positive_edges)
        n_train = int(0.8 * n)
        n_val = int(0.1 * n)
        
        train_edges = all_positive_edges[:n_train]
        val_edges = all_positive_edges[n_train:n_train+n_val]
        test_edges = all_positive_edges[n_train+n_val:]
        
        logger.info(
            f"Real data loaded: {num_nodes} nodes, {edge_index.shape[1]} edges. "
            f"Train={len(train_edges)}, Val={len(val_edges)}, Test={len(test_edges)}"
        )
        
        return {
            "graph_data": {
                "node_features": node_features,
                "edge_index": edge_index,
                "edge_type": edge_type,
            },
            "positive_edges": all_positive_edges,
            "train_edges": train_edges,
            "val_edges": val_edges,
            "test_edges": test_edges,
            "node_to_idx": node_to_idx,
            "all_node_ids": all_ids,
            "node_names": node_names,
            "num_relations": len(EDGE_TYPE_MAP),
        }
