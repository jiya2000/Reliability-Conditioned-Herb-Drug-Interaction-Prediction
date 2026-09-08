"""
Real Data Generator — DrugBank

Loads parsed DrugBank data and constructs a knowledge graph with:
- Deterministic hash-based node features (reproducible per-node)
- Evidence text from DrugBank interaction descriptions
- Metadata derived from the evidence text (not random)
- Edge types parsed from interaction description keywords

Note: This is a drug-drug interaction dataset. Herb-drug interactions
require IMPPAT integration, which is planned as future work.
"""

import json
import hashlib
import re
from collections import Counter

import torch
import random
import numpy as np
from loguru import logger


# ---------------------------------------------------------------------------
# Edge type classification based on interaction description keywords
# ---------------------------------------------------------------------------

EDGE_TYPE_KEYWORDS = {
    "metabolic": {
        "keywords": ["metabolism", "metabolized", "CYP", "enzyme", "hepatic",
                      "clearance", "biotransformation"],
        "type_idx": 0,
    },
    "pharmacokinetic": {
        "keywords": ["serum concentration", "absorption", "bioavailability",
                      "plasma level", "AUC", "Cmax", "half-life",
                      "protein binding", "distribution"],
        "type_idx": 1,
    },
    "pharmacodynamic": {
        "keywords": ["risk or severity", "therapeutic efficacy",
                      "adverse effect", "toxicity", "QT prolongation",
                      "bleeding", "hypotension", "sedation", "serotonin",
                      "CNS depression", "hypoglycemia"],
        "type_idx": 2,
    },
}


def _classify_edge_type(description: str) -> int:
    """Classify an interaction edge type from its DrugBank description."""
    desc_lower = description.lower()
    for etype_info in EDGE_TYPE_KEYWORDS.values():
        for kw in etype_info["keywords"]:
            if kw.lower() in desc_lower:
                return etype_info["type_idx"]
    return 0  # default: metabolic


def _derive_molecular_plausibility(description: str) -> float:
    """Derive a molecular plausibility proxy from description keywords."""
    desc_lower = description.lower()
    # Strong mechanistic descriptions → higher plausibility
    strong_keywords = ["metabolism", "cyp", "enzyme", "inhibit", "induce",
                       "substrate", "transporter", "p-glycoprotein"]
    moderate_keywords = ["serum concentration", "absorption", "clearance",
                         "excretion", "protein binding"]
    weak_keywords = ["risk", "severity", "therapeutic efficacy",
                     "adverse effect"]

    score = 0.5  # baseline
    for kw in strong_keywords:
        if kw in desc_lower:
            score = max(score, 0.85)
            break
    for kw in moderate_keywords:
        if kw in desc_lower:
            score = max(score, 0.70)
            break
    for kw in weak_keywords:
        if kw in desc_lower:
            score = max(score, 0.55)
            break
    return score


def _deterministic_node_features(node_id: str, feature_dim: int) -> torch.Tensor:
    """Generate a deterministic feature vector from a node ID using hashing.

    This replaces torch.randn which gives different features every run.
    The hash ensures the same node always gets the same features.
    """
    # Use the node ID to seed a generator for reproducibility
    hash_bytes = hashlib.sha256(node_id.encode()).digest()
    seed = int.from_bytes(hash_bytes[:4], "big")
    gen = torch.Generator()
    gen.manual_seed(seed)
    return torch.randn(feature_dim, generator=gen)


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

        # ★ FIX: Deterministic hash-based node features instead of random
        node_features = torch.stack([
            _deterministic_node_features(nid, self.feature_dim)
            for nid in all_ids
        ])

        edge_src, edge_tgt = [], []
        edge_types = []
        all_positive_edges = []

        # Count corroboration: how many times each drug pair appears
        pair_counts = Counter()
        for u, v, desc in interactions:
            if u in node_to_idx and v in node_to_idx:
                pair_key = tuple(sorted([u, v]))
                pair_counts[pair_key] += 1

        # Edge type mapping — 3 types derived from descriptions
        NUM_EDGE_TYPES = 3  # metabolic, pharmacokinetic, pharmacodynamic

        seen_pairs = set()  # deduplicate drug pairs
        for u, v, desc in interactions:
            if u not in node_to_idx or v not in node_to_idx:
                continue

            # Deduplicate: keep first occurrence per pair
            pair_key = tuple(sorted([u, v]))
            if pair_key in seen_pairs:
                continue
            seen_pairs.add(pair_key)

            u_idx = node_to_idx[u]
            v_idx = node_to_idx[v]

            # ★ FIX: Classify edge type from description
            etype = _classify_edge_type(desc)

            # Bidirectional graph
            edge_src.extend([u_idx, v_idx])
            edge_tgt.extend([v_idx, u_idx])
            edge_types.extend([etype, etype])

            # ★ FIX: Derive metadata from evidence text and data properties
            corroboration = pair_counts[pair_key]
            mol_plausibility = _derive_molecular_plausibility(desc)

            edge_data = {
                "source_id": u,
                "source_name": drugs.get(u, u),
                "source_type": "drug",
                "target_id": v,
                "target_name": drugs.get(v, v),
                "target_type": "drug",
                # ★ FIX: Use the actual DrugBank description as evidence
                "evidence_texts": [desc],
                "evidence_sources": ["drugbank"],
                "evidence_source_type": "peer_reviewed",
                # ★ FIX: Derived metadata instead of random
                "corroboration_count": corroboration,
                "biomedical_quality": 0.85,  # DrugBank is curated
                "temporal_recency": 0.70,    # DrugBank is maintained
                "molecular_plausibility": mol_plausibility,
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
        val_edges = all_positive_edges[n_train:n_train + n_val]
        test_edges = all_positive_edges[n_train + n_val:]

        logger.info(
            f"Real data loaded: {num_nodes} nodes, {edge_index.shape[1]} edges "
            f"({NUM_EDGE_TYPES} edge types). "
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
            "num_relations": NUM_EDGE_TYPES,
        }
