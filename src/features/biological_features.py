"""
Biologically Grounded Feature Extraction Module.

Constructs genuine pharmacokinetic and pharmacodynamic target profiles
for herbs and drugs (CYP enzymes, efflux/uptake transporters, nuclear receptors),
and deterministic text evidence representations fit strictly on training data.
Completely replaces random representations (torch.randn) with reproducible biological features.
"""

from __future__ import annotations

import re
from typing import Dict, List, Tuple, Optional
import numpy as np
import pandas as pd
import torch
from loguru import logger


# Canonical pharmacokinetic enzymes, transporters, and receptors involved in HDI
CANONICAL_TARGETS = [
    "CYP3A4",    # Major phase I metabolic enzyme (metabolizes ~50% of prescription drugs)
    "CYP2C9",    # Primary clearance enzyme for warfarin, phenytoin, NSAIDs
    "CYP2D6",    # Phase I metabolism for beta-blockers, antidepressants, opioids
    "CYP1A2",    # Metabolism of theophylline, caffeine, clozapine
    "CYP2C19",   # Metabolism of clopidogrel, omeprazole
    "CYP2E1",    # Metabolism of acetaminophen, volatile anesthetics
    "ABCB1",     # P-glycoprotein (efflux transporter: digoxin, cyclosporine, tacrolimus)
    "SLCO1A2",   # OATP1A2 (uptake transporter: nadolol, fexofenadine, fluoroquinolones)
    "SLCO1B1",   # OATP1B1 (hepatic uptake transporter: statins)
    "HSD11B2",   # 11beta-hydroxysteroid dehydrogenase 2 (cortisol-cortisone conversion)
    "VKORC1",    # Vitamin K epoxide reductase complex subunit 1 (warfarin target)
    "PTGS1",     # Cyclooxygenase-1 (aspirin target, platelet aggregation)
    "PTGS2",     # Cyclooxygenase-2 (inflammatory prostaglandins)
    "GABRA1",    # GABA-A receptor alpha 1 subunit (sedatives, benzodiazepines)
    "PTAFR",     # Platelet-activating factor receptor (platelet activation)
    "NR1I2",     # Pregnane X receptor (PXR - master nuclear receptor inducing CYP3A4 & ABCB1)
]

TARGET_TO_IDX = {t: i for i, t in enumerate(CANONICAL_TARGETS)}
FEATURE_DIM = len(CANONICAL_TARGETS)  # 16

PROFILE_STATUS: str = "legacy, label-adjacent"
PROFILE_VERSION: str = "legacy, label-adjacent"

# Legacy curated biological knowledge profiles for botanical entities.
# CLASSIFICATION: "legacy, label-adjacent" (pending per-value external DOI/PMID citation).
# Values represent modulatory effect: +1.0 = strong induction/activation, -1.0 = strong inhibition,
# +0.5 = mild induction, -0.5 = mild inhibition/antagonism, 0.0 = neutral.
HERB_BIOLOGICAL_PROFILES: Dict[str, Dict[str, float]] = {
    "HERB_HYPERICUM": {
        "NR1I2": 1.0,     # Potent PXR agonist via hyperforin
        "CYP3A4": 1.0,    # Strong in vivo hepatic and intestinal induction
        "ABCB1": 1.0,     # Strong intestinal P-gp induction
        "CYP1A2": 0.8,    # Induction of theophylline clearance
        "CYP2C9": 0.7,    # Induction of warfarin clearance
        "CYP2E1": 0.5,
    },
    "HERB_GINKGO": {
        "PTAFR": -0.9,    # Potent antagonism of PAF receptor by ginkgolides
        "CYP3A4": 0.3,    # Mild intestinal induction
        "CYP2C9": -0.2,   # Mild in vitro inhibition
        "PTGS1": -0.4,    # Mild antiplatelet action
    },
    "HERB_GINSENG": {
        "CYP2C9": 0.7,    # American ginseng (P. quinquefolius) induces CYP2C9
        "PTGS1": -0.3,    # Mild antiplatelet effect
    },
    "HERB_GARLIC": {
        "CYP3A4": 0.8,    # Intestinal CYP3A4 induction by organosulfur compounds
        "ABCB1": 0.6,     # Intestinal P-gp modulation
        "PTGS1": -0.8,    # Potent antiplatelet aggregation by allicin/ajoene
    },
    "HERB_GOLDENSEAL": {
        "CYP3A4": -1.0,   # Potent mechanism-based inhibition by hydrastine & berberine
        "CYP2D6": -0.9,   # Potent inhibition
        "ABCB1": -0.8,    # In vivo intestinal P-gp inhibition
    },
    "HERB_CURCUMA": {
        "CYP3A4": -0.7,   # In vivo CYP3A4 inhibition by curcumin
        "ABCB1": -0.8,    # In vivo P-gp inhibition
        "CYP2C9": -0.6,   # Moderate CYP2C9 inhibition
        "PTGS2": -0.7,    # Anti-inflammatory COX-2 inhibition
    },
    "HERB_GREENTEA": {
        "SLCO1A2": -1.0,  # Potent inhibition of OATP1A2 uptake transporter by EGCG
        "SLCO1B1": -0.7,  # Hepatic OATP inhibition
        "CYP3A4": -0.2,   # Weak inhibition
    },
    "HERB_LICORICE": {
        "HSD11B2": -1.0,  # Potent irreversible inhibition of 11b-HSD2 by glycyrrhetinic acid
        "CYP3A4": -0.3,
    },
    "HERB_MILKTHISTLE": {
        # Negative control: silymarin lacks meaningful in vivo CYP/P-gp modulation at oral clinical doses
        "CYP3A4": 0.05,
        "ABCB1": 0.05,
    },
    "HERB_VALERIAN": {
        # Negative control: valerenic acid acts on GABA-A, neutral on CYP3A4/P-gp
        "GABRA1": 0.8,
        "CYP3A4": 0.0,
        "ABCB1": 0.0,
    },
    "HERB_SAWPALMETTO": {
        # Negative control: liposterolic extract neutral on CYP3A4/P-gp in vivo
        "CYP3A4": 0.0,
        "ABCB1": 0.0,
    },
    "HERB_BLACKCOHOSH": {
        # Negative control: neutral on CYP3A4 and P-gp
        "CYP3A4": 0.0,
        "ABCB1": 0.0,
    },
    "HERB_KAVA": {
        "CYP3A4": -0.9,   # Potent in vivo CYP3A4 inhibition by kavalactones
        "CYP2C9": -0.8,   # Potent CYP2C9 inhibition
        "CYP1A2": -0.8,   # Potent CYP1A2 inhibition
        "GABRA1": 0.7,    # Central GABAergic potentiation
    },
    "HERB_ECHINACEA": {
        "CYP3A4": 0.3,    # Route-dependent: intestinal inhibition vs hepatic induction
        "CYP1A2": -0.6,   # In vivo CYP1A2 inhibition
    },
    "HERB_PEPPERMINT": {
        "CYP3A4": -0.8,   # Intestinal CYP3A4 inhibition by menthol/peppermint oil
        "ABCB1": -0.6,    # Intestinal P-gp modulation
    },
    "HERB_CRANBERRY": {
        "CYP2C9": -0.3,   # Mild in vitro flavonoid modulation (controversial in vivo)
    },
    "HERB_CHAMOMILE": {
        "CYP1A2": -0.7,   # In vitro and clinical CYP1A2 inhibition
        "CYP2C9": -0.5,   # Coumarin-mediated CYP2C9 modulation
        "PTGS1": -0.4,    # Mild antiplatelet action
    },
    "HERB_GRAPEFRUIT": {
        "CYP3A4": -1.0,   # Irreversible mechanism-based intestinal CYP3A4 inactivation
        "ABCB1": -0.5,    # Intestinal P-gp inhibition
    },
    "HERB_KRATOM": {
        "CYP3A4": -0.8,   # Potent mechanism-based/reversible inhibition by mitragynine
        "CYP2D6": -0.7,   # Inhibition of CYP2D6
        "CYP2C9": -0.5,   # Moderate CYP2C9 inhibition
    },
}

# Legacy curated biological pharmacokinetic profiles for prescription drug entities.
# CLASSIFICATION: "legacy, label-adjacent" (selection of 40 drugs was label-adjacent).
# Values represent substrate sensitivity / dependency: 1.0 = primary substrate, 0.5 = secondary substrate,
# 0.0 = not dependent. Negative values represent target receptor site.
DRUG_BIOLOGICAL_PROFILES: Dict[str, Dict[str, float]] = {
    "DRUG_CYCLOSPORINE": {
        "CYP3A4": 1.0,    # Primary clearance via CYP3A4
        "ABCB1": 1.0,     # Primary transport via P-gp (narrow therapeutic index)
    },
    "DRUG_DIGOXIN": {
        "ABCB1": 1.0,     # Critical P-gp substrate (narrow therapeutic index, renally/biliary cleared)
    },
    "DRUG_WARFARIN": {
        "CYP2C9": 1.0,    # S-warfarin cleared almost exclusively by CYP2C9 (narrow therapeutic index)
        "VKORC1": -1.0,   # Direct pharmacodynamic target
    },
    "DRUG_MIDAZOLAM": {
        "CYP3A4": 1.0,    # Gold standard FDA clinical index substrate for CYP3A4
    },
    "DRUG_TACROLIMUS": {
        "CYP3A4": 1.0,    # Sensitive CYP3A4 substrate
        "ABCB1": 0.9,     # P-gp substrate (narrow therapeutic index)
    },
    "DRUG_NADOLOL": {
        "SLCO1A2": 1.0,   # Uptake dependent on intestinal OATP1A2
    },
    "DRUG_ROSUVASTATIN": {
        "SLCO1B1": 1.0,   # Hepatic uptake dependent on OATP1B1
        "CYP2C9": 0.3,
    },
    "DRUG_SAQUINAVIR": {
        "CYP3A4": 1.0,    # CYP3A4 substrate (HIV protease inhibitor)
        "ABCB1": 0.9,     # P-gp substrate
    },
    "DRUG_INDINAVIR": {
        "CYP3A4": 1.0,    # CYP3A4 substrate
    },
    "DRUG_SIMVASTATIN": {
        "CYP3A4": 1.0,    # Extensively metabolized by CYP3A4
        "SLCO1B1": 0.8,
    },
    "DRUG_THEOPHYLLINE": {
        "CYP1A2": 1.0,    # Metabolized primarily by CYP1A2 (narrow therapeutic index)
    },
    "DRUG_ALPRAZOLAM": {
        "CYP3A4": 1.0,    # CYP3A4 substrate
        "GABRA1": -1.0,   # GABA-A receptor target
    },
    "DRUG_ASPIRIN": {
        "PTGS1": -1.0,    # Irreversible COX-1 inhibitor (antiplatelet target)
        "PTGS2": -0.7,
    },
    "DRUG_PREDNISOLONE": {
        "HSD11B2": 1.0,   # Substrate for 11b-HSD2 conversion
        "CYP3A4": 0.5,
    },
    "DRUG_HYDROCHLOROTHIAZIDE": {
        "HSD11B2": 0.5,   # Mineralocorticoid additive risk factor
    },
    "DRUG_RITONAVIR": {
        "CYP3A4": 0.8,
    },
    "DRUG_DOCETAXEL": {
        "CYP3A4": 1.0,    # Chemotherapeutic substrate of CYP3A4
    },
    "DRUG_ETHINYLESTRADIOL": {
        "CYP3A4": 0.9,    # Oral contraceptive steroid metabolized by CYP3A4
    },
    "DRUG_OMEPRAZOLE": {
        "CYP2C19": 1.0,   # Primary clearance via CYP2C19
        "CYP3A4": 0.5,    # Secondary clearance via CYP3A4
    },
    "DRUG_DEXTROMETHORPHAN": {
        "CYP2D6": 1.0,    # Gold standard FDA probe for CYP2D6
    },
    "DRUG_ATORVASTATIN": {
        "CYP3A4": 1.0,    # CYP3A4 substrate
        "SLCO1B1": 0.9,   # OATP1B1 substrate
    },
    "DRUG_METHADONE": {
        "CYP3A4": 0.9,    # CYP3A4 substrate
        "CYP2B6": 0.7,
    },
    "DRUG_OXYCODONE": {
        "CYP3A4": 1.0,    # CYP3A4 substrate
        "CYP2D6": 0.4,
    },
    "DRUG_FEXOFENADINE": {
        "ABCB1": 1.0,     # Intestinal P-gp substrate
        "SLCO1A2": 0.8,   # OATP1A2 substrate
    },
    "DRUG_VERAPAMIL": {
        "CYP3A4": 1.0,    # CYP3A4 substrate
        "ABCB1": 0.9,     # P-gp substrate
    },
    "DRUG_IRINOTECAN": {
        "CYP3A4": 0.9,    # CYP3A4 clearance
        "ABCB1": 0.7,     # P-gp transport
    },
    "DRUG_TOLBUTAMIDE": {
        "CYP2C9": 1.0,    # Gold standard FDA clinical probe for CYP2C9
    },
    "DRUG_NIFEDIPINE": {
        "CYP3A4": 1.0,    # Sensitive CYP3A4 substrate
    },
    "DRUG_METFORMIN": {
        "ABCB1": 0.3,
    },
    "DRUG_PHENELZINE": {
        "CYP2D6": 0.5,
    },
    "DRUG_CHLORZOXAZONE": {
        "CYP2E1": 1.0,    # Gold standard clinical probe for CYP2E1
    },
    "DRUG_TALINOLOL": {
        "ABCB1": 1.0,     # P-glycoprotein transport probe
    },
    "DRUG_PACLITAXEL": {
        "CYP3A4": 0.9,    # CYP3A4 substrate
        "ABCB1": 0.9,     # P-gp substrate
    },
    "DRUG_BORTEZOMIB": {
        "CYP3A4": 0.6,
    },
    "DRUG_CAFFEINE": {
        "CYP1A2": 1.0,    # Gold standard FDA probe for CYP1A2
    },
    "DRUG_SPIRONOLACTONE": {
        "HSD11B2": 0.8,
    },
    "DRUG_FINASTERIDE": {
        "CYP3A4": 0.7,
    },
    "DRUG_FELODIPINE": {
        "CYP3A4": 1.0,    # Sensitive CYP3A4 substrate probe
    },
    "DRUG_LEVODOPA": {
        "GABRA1": 0.2,
    },
    "DRUG_ACETAMINOPHEN": {
        "CYP2E1": 0.8,
        "CYP1A2": 0.3,
    },
}


class BiologicalProfileExtractor:
    """
    Constructs deterministic biological profile tensors for herbs and drugs
    based on human target / enzyme / transporter profiles.

    NOTE: Preserved strictly as a baseline labeled 'legacy, label-adjacent'.
    """

    profile_status: str = "legacy, label-adjacent"

    def __init__(self, feature_dim: int = FEATURE_DIM, feature_set: str = "legacy"):
        self.feature_dim = feature_dim
        self.feature_set = feature_set

    def get_herb_representation(self, herb_id: str) -> np.ndarray:
        """Construct 16D biological mechanism vector for a given herb."""
        vec = np.zeros(self.feature_dim, dtype=np.float32)
        profile = HERB_BIOLOGICAL_PROFILES.get(herb_id, {})
        for target, val in profile.items():
            if target in TARGET_TO_IDX:
                vec[TARGET_TO_IDX[target]] = val
        return vec

    def get_drug_representation(self, drug_id: str) -> np.ndarray:
        """Construct 16D pharmacokinetic dependency vector for a given drug."""
        vec = np.zeros(self.feature_dim, dtype=np.float32)
        profile = DRUG_BIOLOGICAL_PROFILES.get(drug_id, {})
        for target, val in profile.items():
            if target in TARGET_TO_IDX:
                vec[TARGET_TO_IDX[target]] = val
        return vec

    def extract_pair_features(self, herb_id: str, drug_id: str) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Extract herb vector, drug vector, and biological target interaction potential.
        Interaction potential = element-wise product measuring shared target mechanism.
        """
        h_vec = self.get_herb_representation(herb_id)
        d_vec = self.get_drug_representation(drug_id)
        # Shared target interaction potential
        interaction_vec = h_vec * d_vec
        return h_vec, d_vec, interaction_vec


class EvidenceTextFeatureExtractor:
    """
    Reproducible text evidence encoder fit STRICTLY on training split data.
    Extracts bag-of-words / TF-IDF representations of study literature descriptions
    and projects them to a fixed embedding dimension.
    """

    def __init__(self, embedding_dim: int = 16, min_df: int = 1):
        self.embedding_dim = embedding_dim
        self.min_df = min_df
        self.vocabulary_: Dict[str, int] = {}
        self.idf_weights_: np.ndarray = np.array([], dtype=np.float32)
        self.projection_matrix_: np.ndarray = np.array([], dtype=np.float32)
        self._is_fitted = False

    @staticmethod
    def _tokenize(text: str) -> List[str]:
        """Normalize and tokenize pharmacological text into lowercase alphanumeric terms."""
        if not text or not isinstance(text, str):
            return []
        tokens = re.findall(r"[a-z0-9_]{3,}", text.lower())
        # Filter out common stop words
        stopwords = {"the", "and", "was", "for", "with", "that", "this", "from", "after", "were", "human"}
        return [t for t in tokens if t not in stopwords]

    def fit(self, train_texts: List[str]) -> EvidenceTextFeatureExtractor:
        """
        Fit vocabulary, inverse document frequencies, and projection matrix
        STRICTLY on training partition texts.
        """
        doc_tokens = [self._tokenize(t) for t in train_texts]
        doc_count = len(doc_tokens)
        if doc_count == 0:
            raise ValueError("Cannot fit EvidenceTextFeatureExtractor on empty training texts.")

        # Compute term document frequencies
        df_counts: Dict[str, int] = {}
        for tokens in doc_tokens:
            unique_terms = set(tokens)
            for term in unique_terms:
                df_counts[term] = df_counts.get(term, 0) + 1

        # Select vocabulary based on min_df
        vocab = sorted([t for t, count in df_counts.items() if count >= self.min_df])
        if not vocab:
            vocab = ["interaction", "clearance", "plasma", "concentration"]

        self.vocabulary_ = {term: idx for idx, term in enumerate(vocab)}
        vocab_size = len(vocab)

        # Compute smoothed IDF weights: log((1 + N) / (1 + df)) + 1
        idfs = [np.log((1.0 + doc_count) / (1.0 + df_counts.get(t, 0))) + 1.0 for t in vocab]
        self.idf_weights_ = np.array(idfs, dtype=np.float32)

        # Deterministic orthonormal projection matrix from vocab_size to embedding_dim
        # Initialized with a fixed seed so it is completely reproducible
        rng = np.random.RandomState(seed=42)
        rand_proj = rng.randn(vocab_size, self.embedding_dim).astype(np.float32)
        # Orthonormalize columns via QR decomposition
        q, _ = np.linalg.qr(rand_proj)
        self.projection_matrix_ = q[:, :self.embedding_dim].astype(np.float32)

        self._is_fitted = True
        logger.info(
            f"Fitted EvidenceTextFeatureExtractor on {doc_count} training texts "
            f"(vocab size: {vocab_size} -> embedding dim: {self.embedding_dim})"
        )
        return self

    def transform(self, texts: List[str]) -> np.ndarray:
        """Transform texts into normalized embedding representations."""
        if not self._is_fitted:
            raise RuntimeError("EvidenceTextFeatureExtractor must be fit on training data before transforming.")

        vocab_size = len(self.vocabulary_)
        vectors = []

        for text in texts:
            tokens = self._tokenize(text)
            bow = np.zeros(vocab_size, dtype=np.float32)
            for t in tokens:
                if t in self.vocabulary_:
                    bow[self.vocabulary_[t]] += 1.0

            # Apply TF-IDF weighting: tf * idf
            tfidf = bow * self.idf_weights_
            norm = np.linalg.norm(tfidf)
            if norm > 1e-6:
                tfidf /= norm

            # Project to embedding dimension
            dense_embed = tfidf @ self.projection_matrix_
            # L2 normalize final embedding
            embed_norm = np.linalg.norm(dense_embed)
            if embed_norm > 1e-6:
                dense_embed /= embed_norm

            vectors.append(dense_embed)

        return np.array(vectors, dtype=np.float32)
