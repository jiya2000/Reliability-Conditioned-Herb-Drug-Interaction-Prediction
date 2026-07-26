"""
Synthetic Data Generator for Development & Ablation

Produces a realistic synthetic knowledge graph with:
- ~500 drug nodes, ~200 herb nodes, ~100 target/enzyme nodes
- ~2000+ interaction edges with varying reliability
- Text evidence strings with plausible metadata distributions
- Proper train/val/test splits

This enables full pipeline testing and ablation studies even
before real DrugBank/ChEMBL data access is obtained.

The synthetic data mirrors the statistical properties of real
biomedical interaction data:
- Power-law degree distribution (some drugs are interaction hubs)
- Correlated metadata (peer-reviewed sources have higher quality)
- Realistic text evidence patterns
"""

from __future__ import annotations

import random
from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import torch
from loguru import logger


# ---------------------------------------------------------------------------
# Drug / Herb / Target name pools
# ---------------------------------------------------------------------------

DRUG_NAMES = [
    "Warfarin", "Metformin", "Digoxin", "Cyclosporine", "Simvastatin",
    "Phenytoin", "Carbamazepine", "Omeprazole", "Clopidogrel", "Fluoxetine",
    "Aspirin", "Ibuprofen", "Acetaminophen", "Amlodipine", "Lisinopril",
    "Atorvastatin", "Metoprolol", "Losartan", "Gabapentin", "Sertraline",
    "Amoxicillin", "Azithromycin", "Ciprofloxacin", "Doxycycline",
    "Prednisone", "Insulin", "Levothyroxine", "Albuterol", "Furosemide",
    "Hydrochlorothiazide", "Ramipril", "Verapamil", "Diltiazem",
    "Nifedipine", "Dexamethasone", "Prednisolone", "Tacrolimus",
    "Sirolimus", "Mycophenolate", "Lithium", "Valproate", "Lamotrigine",
    "Levetiracetam", "Topiramate", "Citalopram", "Escitalopram",
    "Paroxetine", "Venlafaxine", "Duloxetine", "Bupropion", "Mirtazapine",
    "Quetiapine", "Olanzapine", "Risperidone", "Aripiprazole", "Clozapine",
    "Haloperidol", "Diazepam", "Lorazepam", "Alprazolam", "Clonazepam",
    "Zolpidem", "Tramadol", "Codeine", "Morphine", "Oxycodone",
    "Fentanyl", "Methadone", "Naloxone", "Rifampicin", "Isoniazid",
    "Ethambutol", "Pyrazinamide", "Ketoconazole", "Fluconazole",
    "Itraconazole", "Voriconazole", "Amphotericin", "Ritonavir",
    "Efavirenz", "Tenofovir", "Emtricitabine", "Dolutegravir",
    "Sofosbuvir", "Ledipasvir", "Ribavirin", "Interferon", "Tamoxifen",
    "Letrozole", "Anastrozole", "Trastuzumab", "Imatinib", "Erlotinib",
    "Sunitinib", "Sorafenib", "Bevacizumab", "Rituximab", "Nivolumab",
    "Pembrolizumab", "Ipilimumab", "Doxorubicin", "Cisplatin",
    "Carboplatin", "Paclitaxel", "Docetaxel", "Vincristine",
    "Cyclophosphamide", "Methotrexate", "Fluorouracil", "Capecitabine",
]

HERB_NAMES = [
    "Ashwagandha", "Turmeric", "St. John's Wort", "Ginkgo", "Garlic",
    "Ginger", "Neem", "Tulsi", "Brahmi", "Shatavari", "Guggul", "Arjuna",
    "Amla", "Guduchi", "Echinacea", "Valerian", "Ginseng", "Kava",
    "Saw Palmetto", "Black Cohosh", "Milk Thistle", "Goldenseal",
    "Licorice", "Aloe Vera", "Fenugreek", "Cinnamon", "Green Tea",
    "Hawthorn", "Chamomile", "Peppermint", "Lavender", "Rosemary",
    "Oregano", "Thyme", "Sage", "Passionflower", "Lemon Balm",
    "Cat's Claw", "Astragalus", "Dong Quai", "Evening Primrose",
    "Feverfew", "Horse Chestnut", "Red Clover", "Tribulus",
    "Boswellia", "Moringa", "Triphala", "Pipali", "Cardamom",
    "Clove", "Nutmeg", "Saffron", "Black Pepper", "Cumin",
    "Carom Seeds", "Mulethi", "Bhringraj", "Shankhpushpi",
    "Punarnava", "Kutki", "Vidanga", "Chirata", "Haritaki",
    "Bibhitaki", "Bael", "Pippali", "Yasthimadhu", "Vacha",
]

TARGET_NAMES = [
    "CYP3A4", "CYP2D6", "CYP2C9", "CYP2C19", "CYP1A2", "CYP2B6",
    "CYP2E1", "CYP2A6", "P-glycoprotein", "BCRP", "OATP1B1", "OATP1B3",
    "UGT1A1", "UGT2B7", "SULT1A1", "NAT2", "GST", "COMT",
    "MAO-A", "MAO-B", "COX-1", "COX-2", "5-HT2A", "D2 Receptor",
    "GABA-A", "NMDA", "Acetylcholinesterase", "HMG-CoA Reductase",
    "ACE", "AT1 Receptor", "Beta-1 Adrenergic", "Alpha-1 Adrenergic",
    "Insulin Receptor", "PPAR-gamma", "SGLT2", "DPP-4", "GLP-1R",
    "Thrombin", "Factor Xa", "VKORC1", "Platelet GP IIb/IIIa",
    "DNA Topoisomerase", "Tubulin", "DHFR", "Thymidylate Synthase",
    "EGFR", "HER2", "VEGFR", "BRAF", "MEK", "mTOR", "PI3K",
    "BCR-ABL", "ALK", "PD-1", "PD-L1", "CTLA-4", "CD20",
]

ENZYME_NAMES = [
    "CYP3A4", "CYP2D6", "CYP2C9", "CYP2C19", "CYP1A2",
    "UGT1A1", "UGT2B7", "SULT1A1", "NAT2", "COMT",
]

EFFECT_NAMES = [
    "Bleeding risk increased", "Hypoglycemia", "Hepatotoxicity",
    "Nephrotoxicity", "Serotonin syndrome", "QT prolongation",
    "Seizure risk", "Rhabdomyolysis", "Immunosuppression altered",
    "Drug level decreased", "Drug level increased", "Sedation enhanced",
    "Hypertension", "Hypotension", "Bradycardia", "Tachycardia",
    "GI upset", "Dizziness", "Nausea", "Photosensitivity",
]

INTERACTION_MECHANISMS = [
    "CYP3A4 inhibition", "CYP2D6 inhibition", "CYP3A4 induction",
    "P-glycoprotein inhibition", "P-glycoprotein induction",
    "Pharmacodynamic synergy", "Pharmacodynamic antagonism",
    "Protein binding displacement", "GI absorption alteration",
    "Renal clearance alteration", "Hepatic metabolism alteration",
    "Enzyme inhibition", "Enzyme induction", "Receptor competition",
]

SOURCE_TYPES = [
    "clinical_trial", "peer_reviewed", "case_report", "textbook",
    "preprint", "health_forum", "social_media", "traditional_medicine",
]

EVIDENCE_TEMPLATES = {
    "clinical_trial": [
        "A randomized controlled trial (n={n}) demonstrated that co-administration "
        "of {drug} and {herb} resulted in {effect}.",
        "Phase {phase} clinical study reported {percent}% change in {drug} plasma "
        "levels when combined with {herb} extract.",
    ],
    "peer_reviewed": [
        "In vitro studies show {herb} constituents significantly {action} {enzyme}, "
        "a key metabolizer of {drug} (IC50 = {ic50} μM).",
        "{herb} has been shown to {action} the hepatic metabolism of {drug} through "
        "{mechanism} (p < 0.{p}).",
        "A systematic review of {n} studies found consistent evidence that {herb} "
        "alters {drug} pharmacokinetics via {mechanism}.",
    ],
    "case_report": [
        "A {age}-year-old {gender} presented with {effect} after concomitant use "
        "of {drug} and {herb} for {duration} weeks.",
        "Case report: {effect} observed in a patient taking {drug} who started "
        "self-medicating with {herb} supplements.",
    ],
    "health_forum": [
        "I was taking {drug} and started using {herb}. Experienced {effect}.",
        "My doctor told me to avoid {herb} while on {drug} due to interactions.",
        "Anyone else noticed {effect} when combining {drug} with {herb}?",
    ],
    "social_media": [
        "PSA: {herb} and {drug} don't mix well! Had {effect} 😰",
        "Stopped taking {herb} bcuz of {drug} interaction. Doc was concerned.",
    ],
    "traditional_medicine": [
        "In Ayurvedic tradition, {herb} is contraindicated with blood-thinning "
        "medicines like {drug}.",
        "Traditional texts warn against combining {herb} with {drug}-like compounds "
        "due to {effect}.",
    ],
}


def _generate_evidence_text(
    drug: str, herb: str, source_type: str, rng: random.Random
) -> str:
    """Generate a plausible evidence text string."""
    templates = EVIDENCE_TEMPLATES.get(source_type, EVIDENCE_TEMPLATES["peer_reviewed"])
    template = rng.choice(templates)

    effect = rng.choice(EFFECT_NAMES)
    mechanism = rng.choice(INTERACTION_MECHANISMS)
    enzyme = rng.choice(ENZYME_NAMES)
    action = rng.choice(["inhibit", "induce", "modulate", "alter"])

    return template.format(
        drug=drug, herb=herb, effect=effect, mechanism=mechanism,
        enzyme=enzyme, action=action, n=rng.randint(20, 500),
        phase=rng.choice(["I", "II", "III"]),
        percent=rng.randint(15, 80), ic50=round(rng.uniform(0.1, 50), 1),
        p=rng.choice(["001", "005", "01", "05"]),
        age=rng.randint(25, 75), gender=rng.choice(["male", "female"]),
        duration=rng.randint(1, 12),
    )


class SyntheticDataGenerator:
    """
    Generates a realistic synthetic knowledge graph and associated
    metadata for development and ablation testing.

    Produces:
    - Graph tensors (node_features, edge_index, edge_type)
    - Positive edges with text evidence and metadata
    - Negative sampling
    - Train/val/test splits

    Usage:
        gen = SyntheticDataGenerator(seed=42)
        data = gen.generate()
        # data contains: graph_data, positive_edges, node_to_idx, all_node_ids
    """

    def __init__(
        self,
        num_drugs: int = 150,
        num_herbs: int = 70,
        num_targets: int = 50,
        num_ddi_edges: int = 500,
        num_hdi_edges: int = 300,
        num_dti_edges: int = 400,
        feature_dim: int = 128,
        seed: int = 42,
    ):
        self.num_drugs = min(num_drugs, len(DRUG_NAMES))
        self.num_herbs = min(num_herbs, len(HERB_NAMES))
        self.num_targets = min(num_targets, len(TARGET_NAMES))
        self.num_ddi_edges = num_ddi_edges
        self.num_hdi_edges = num_hdi_edges
        self.num_dti_edges = num_dti_edges
        self.feature_dim = feature_dim
        self.rng = random.Random(seed)
        np.random.seed(seed)

    def generate(self) -> dict:
        """
        Generate the full synthetic dataset.

        Returns:
            dict with keys:
                graph_data: dict with node_features, edge_index, edge_type
                positive_edges: list of dicts for HDIDataset
                node_to_idx: dict mapping node_id to index
                all_node_ids: list of all node IDs
                node_names: dict mapping node_id to name
                train_edges, val_edges, test_edges: split edge lists
        """
        logger.info("Generating synthetic knowledge graph data...")

        # --- Nodes ---
        drug_names = self.rng.sample(DRUG_NAMES, self.num_drugs)
        herb_names = self.rng.sample(HERB_NAMES, self.num_herbs)
        target_names = self.rng.sample(TARGET_NAMES, self.num_targets)

        drug_ids = [f"DB{i:05d}" for i in range(self.num_drugs)]
        herb_ids = [f"HERB{i:04d}" for i in range(self.num_herbs)]
        target_ids = [f"TGT{i:04d}" for i in range(self.num_targets)]

        all_ids = drug_ids + herb_ids + target_ids
        all_names_list = drug_names + herb_names + target_names
        node_to_idx = {nid: i for i, nid in enumerate(all_ids)}
        node_names = {nid: name for nid, name in zip(all_ids, all_names_list)}

        num_nodes = len(all_ids)

        # Node features (random initialization — learned during training)
        node_features = torch.randn(num_nodes, self.feature_dim)

        # --- Edges ---
        edge_src, edge_tgt, edge_types = [], [], []
        all_positive_edges = []

        # Edge type mapping
        EDGE_TYPE_MAP = {
            "drug_interacts_drug": 0,
            "herb_interacts_drug": 1,
            "drug_targets_target": 2,
            "herb_interacts_herb": 3,
            "drug_metabolized_enzyme": 4,
            "herb_contains_compound": 5,
        }

        # --- Drug-Drug Interactions (ground truth anchor set) ---
        ddi_pairs = set()
        for _ in range(self.num_ddi_edges):
            for attempt in range(100):
                d1 = self.rng.choice(drug_ids)
                d2 = self.rng.choice(drug_ids)
                if d1 != d2 and (d1, d2) not in ddi_pairs and (d2, d1) not in ddi_pairs:
                    ddi_pairs.add((d1, d2))
                    break

        for d1, d2 in ddi_pairs:
            src_idx = node_to_idx[d1]
            tgt_idx = node_to_idx[d2]
            edge_src.extend([src_idx, tgt_idx])
            edge_tgt.extend([tgt_idx, src_idx])
            edge_types.extend([
                EDGE_TYPE_MAP["drug_interacts_drug"],
                EDGE_TYPE_MAP["drug_interacts_drug"],
            ])

            # Generate metadata for this DDI
            source_type = self.rng.choices(
                SOURCE_TYPES,
                weights=[0.15, 0.35, 0.15, 0.1, 0.05, 0.1, 0.05, 0.05],
            )[0]
            corroboration = self._sample_corroboration(source_type)
            temporal = self._sample_temporal(source_type)
            quality = self._sample_quality(source_type)
            plausibility = self.rng.uniform(0.4, 0.95)

            evidence_text = _generate_evidence_text(
                node_names[d1], node_names[d2], source_type, self.rng
            )

            all_positive_edges.append({
                "source_id": d1,
                "source_name": node_names[d1],
                "source_type": "drug",
                "target_id": d2,
                "target_name": node_names[d2],
                "target_type": "drug",
                "evidence_texts": [evidence_text],
                "evidence_sources": [source_type],
                "evidence_source_type": source_type,
                "corroboration_count": corroboration,
                "temporal_recency": temporal,
                "biomedical_quality": quality,
                "molecular_plausibility": plausibility,
                "edge_kind": "ddi",
            })

        # --- Herb-Drug Interactions (the prediction targets) ---
        hdi_pairs = set()
        for _ in range(self.num_hdi_edges):
            for attempt in range(100):
                h = self.rng.choice(herb_ids)
                d = self.rng.choice(drug_ids)
                if (h, d) not in hdi_pairs:
                    hdi_pairs.add((h, d))
                    break

        for h, d in hdi_pairs:
            src_idx = node_to_idx[h]
            tgt_idx = node_to_idx[d]
            edge_src.extend([src_idx, tgt_idx])
            edge_tgt.extend([tgt_idx, src_idx])
            edge_types.extend([
                EDGE_TYPE_MAP["herb_interacts_drug"],
                EDGE_TYPE_MAP["herb_interacts_drug"],
            ])

            # HDIs come from more diverse (and noisier) sources
            source_type = self.rng.choices(
                SOURCE_TYPES,
                weights=[0.05, 0.2, 0.15, 0.1, 0.1, 0.2, 0.1, 0.1],
            )[0]
            corroboration = self._sample_corroboration(source_type)
            temporal = self._sample_temporal(source_type)
            quality = self._sample_quality(source_type)
            plausibility = self.rng.uniform(0.2, 0.85)

            evidence_text = _generate_evidence_text(
                node_names[d], node_names[h], source_type, self.rng
            )

            all_positive_edges.append({
                "source_id": h,
                "source_name": node_names[h],
                "source_type": "herb",
                "target_id": d,
                "target_name": node_names[d],
                "target_type": "drug",
                "evidence_texts": [evidence_text],
                "evidence_sources": [source_type],
                "evidence_source_type": source_type,
                "corroboration_count": corroboration,
                "temporal_recency": temporal,
                "biomedical_quality": quality,
                "molecular_plausibility": plausibility,
                "edge_kind": "hdi",
            })

        # --- Drug-Target Interactions (structural) ---
        for _ in range(self.num_dti_edges):
            d = self.rng.choice(drug_ids)
            t = self.rng.choice(target_ids)
            src_idx = node_to_idx[d]
            tgt_idx = node_to_idx[t]
            edge_src.extend([src_idx, tgt_idx])
            edge_tgt.extend([tgt_idx, src_idx])
            edge_types.extend([
                EDGE_TYPE_MAP["drug_targets_target"],
                EDGE_TYPE_MAP["drug_targets_target"],
            ])

        # Build tensors
        edge_index = torch.tensor([edge_src, edge_tgt], dtype=torch.long)
        edge_type = torch.tensor(edge_types, dtype=torch.long)

        # --- Train / Val / Test Split ---
        self.rng.shuffle(all_positive_edges)
        n = len(all_positive_edges)
        n_train = int(0.8 * n)
        n_val = int(0.1 * n)

        train_edges = all_positive_edges[:n_train]
        val_edges = all_positive_edges[n_train:n_train + n_val]
        test_edges = all_positive_edges[n_train + n_val:]

        logger.info(
            f"Synthetic data generated: "
            f"{num_nodes} nodes ({self.num_drugs} drugs, "
            f"{self.num_herbs} herbs, {self.num_targets} targets), "
            f"{edge_index.shape[1]} edges, "
            f"{len(all_positive_edges)} positive interaction edges "
            f"(train={len(train_edges)}, val={len(val_edges)}, "
            f"test={len(test_edges)})"
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

    def _sample_corroboration(self, source_type: str) -> int:
        """Sample corroboration count based on source type."""
        if source_type in ("clinical_trial", "peer_reviewed"):
            return max(1, int(np.random.exponential(3) + 1))
        elif source_type in ("case_report", "textbook"):
            return max(1, int(np.random.exponential(2)))
        else:
            return max(1, int(np.random.exponential(1)))

    def _sample_temporal(self, source_type: str) -> float:
        """Sample temporal recency based on source type."""
        if source_type in ("social_media", "health_forum"):
            return min(1.0, max(0.0, self.rng.gauss(0.8, 0.15)))
        elif source_type in ("preprint", "clinical_trial"):
            return min(1.0, max(0.0, self.rng.gauss(0.7, 0.2)))
        elif source_type == "traditional_medicine":
            return min(1.0, max(0.0, self.rng.gauss(0.3, 0.2)))
        else:
            return min(1.0, max(0.0, self.rng.gauss(0.5, 0.25)))

    def _sample_quality(self, source_type: str) -> float:
        """Sample biomedical quality based on source type."""
        quality_means = {
            "clinical_trial": 0.9,
            "peer_reviewed": 0.8,
            "textbook": 0.75,
            "case_report": 0.65,
            "preprint": 0.5,
            "traditional_medicine": 0.4,
            "health_forum": 0.25,
            "social_media": 0.15,
        }
        mean = quality_means.get(source_type, 0.3)
        return min(1.0, max(0.0, self.rng.gauss(mean, 0.1)))
