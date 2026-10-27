"""
Feature-Set Switched Models: Logistic Regression (LR), Gradient Boosted Trees (GBT), and Small GAT.
Phase 3 Implementation.

Supports running models on either:
- 'legacy': Uses legacy label-adjacent profiles (preserved strictly as baseline).
- 'new': Uses blind-curated mechanistic features from data/curation/mechanistic_features.csv.
  STRICT GUARD: Blocks execution and raises UnverifiedCurationError if any rows are UNVERIFIED.
"""

from typing import Dict, List, Optional, Tuple, Any
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import roc_auc_score, average_precision_score, brier_score_loss

from src.features.biological_features import (
    BiologicalProfileExtractor,
    HERB_BIOLOGICAL_PROFILES,
    DRUG_BIOLOGICAL_PROFILES,
    CANONICAL_TARGETS,
)
from src.features.biological_graph_encoder import HeterogeneousBiologicalGraphEncoder
from src.data.mechanistic_curation import (
    load_mechanistic_curation,
    is_curation_verified,
    extract_curated_feature_matrix,
    UnverifiedCurationError,
)
from src.training.evaluation import LeakageSafeSplitter


class SmallGATClassifier(nn.Module):
    """
    Compact Graph Attention Network (GAT) classifier for pairwise biological interaction.
    Uses HeterogeneousBiologicalGraphEncoder to produce contextualized herb and drug
    representations, followed by a lightweight bilinear / cosine interaction head.
    """

    def __init__(self, hidden_dim: int = 16, num_heads: int = 2, dropout: float = 0.1):
        super().__init__()
        self.encoder = HeterogeneousBiologicalGraphEncoder(
            hidden_dim=hidden_dim,
            num_heads=num_heads,
            dropout=dropout,
        )
        self.head = nn.Sequential(
            nn.Linear(hidden_dim * 3, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, herb_ids: List[str], drug_ids: List[str]) -> torch.Tensor:
        """
        Encode herbs and drugs through bipartite GAT layers and predict interaction logit.
        """
        h_emb, d_emb, _ = self.encoder.forward(herb_ids, drug_ids)
        # Pair representation: [h, d, h * d]
        pair_repr = torch.cat([h_emb, d_emb, h_emb * d_emb], dim=-1)
        logits = self.head(pair_repr).squeeze(-1)
        return logits


class FeatureSwitchedBenchmark:
    """
    Executes LR, GBT, and Small GAT models under identical grouped-CV splits
    with a switch between 'legacy' and 'new' mechanistic features.
    """

    def __init__(
        self,
        feature_set: str = "legacy",
        curation_path: str = "data/curation/mechanistic_features.csv",
        seed: int = 42,
    ):
        self.feature_set = feature_set.lower().strip()
        self.curation_path = curation_path
        self.seed = seed

        if self.feature_set not in {"legacy", "new"}:
            raise ValueError(f"Unknown feature_set '{feature_set}'. Must be 'legacy' or 'new'.")

    def get_feature_matrix(self, df: pd.DataFrame) -> Tuple[np.ndarray, str]:
        """
        Extract feature matrix based on the configured feature_set switch.
        
        Strict Guard:
        If feature_set == 'new', requires 100% verified curation table;
        otherwise raises UnverifiedCurationError.
        """
        if self.feature_set == "legacy":
            bio = BiologicalProfileExtractor()
            features = []
            for _, row in df.iterrows():
                _, _, inter = bio.extract_pair_features(row["herb_id"], row["drug_id"])
                overlap_total = float(np.sum(np.abs(inter)))
                feat = np.concatenate([inter, [overlap_total]])
                features.append(feat)
            return np.array(features, dtype=np.float32), "legacy, label-adjacent"

        elif self.feature_set == "new":
            curation_df = load_mechanistic_curation(self.curation_path)
            verified, n_ver, n_unver = is_curation_verified(curation_df)
            if not verified:
                raise UnverifiedCurationError(
                    f"Blocked execution: feature_set='new' requested, but '{self.curation_path}' "
                    f"contains {n_unver} UNVERIFIED entries ({n_ver} verified). "
                    f"Per Phase 3 integrity rules, models will not run on new features until "
                    f"all rows have verified external literature citations."
                )
            feat_matrix = extract_curated_feature_matrix(df, curation_df)
            return feat_matrix, "curated_mechanistic"

    def run_cv(
        self,
        df: pd.DataFrame,
        split_mode: str = "pair_disjoint",
        n_splits: int = 5,
        n_repeats: int = 1,
    ) -> Dict[str, Any]:
        """
        Run LR, GBT, and Small GAT on identical cross-validation splits.
        """
        X, feature_label = self.get_feature_matrix(df)
        y = df["outcome"].astype(str).str.lower().isin(
            ["interaction_present", "positive", "1", "true"]
        ).values.astype(int)

        results: Dict[str, List[float]] = {
            "logistic_regression": [],
            "gradient_boosted_trees": [],
            "small_gat": [],
        }

        herb_ids = df["herb_id"].tolist()
        drug_ids = df["drug_id"].tolist()

        for rep in range(n_repeats):
            splitter = LeakageSafeSplitter(seed=self.seed + rep)
            folds = splitter.grouped_kfold_split(df, mode=split_mode, n_splits=n_splits)

            oof_lr = np.zeros(len(df))
            oof_gbt = np.zeros(len(df))
            oof_gat = np.zeros(len(df))

            for tr_df, te_df in folds:
                tr_idx = tr_df.index.values
                te_idx = te_df.index.values

                # 1. Logistic Regression
                lr = LogisticRegression(max_iter=1000, random_state=self.seed + rep)
                lr.fit(X[tr_idx], y[tr_idx])
                oof_lr[te_idx] = lr.predict_proba(X[te_idx])[:, 1]

                # 2. Gradient Boosted Trees
                gbt = HistGradientBoostingClassifier(random_state=self.seed + rep, max_iter=100)
                gbt.fit(X[tr_idx], y[tr_idx])
                oof_gbt[te_idx] = gbt.predict_proba(X[te_idx])[:, 1]

                # 3. Small GAT (torch)
                torch.manual_seed(self.seed + rep)
                gat = SmallGATClassifier(hidden_dim=16, num_heads=2)
                opt = torch.optim.Adam(gat.parameters(), lr=0.01, weight_decay=1e-4)
                crit = nn.BCEWithLogitsLoss()

                tr_h = [herb_ids[i] for i in tr_idx]
                tr_d = [drug_ids[i] for i in tr_idx]
                tr_y_t = torch.tensor(y[tr_idx], dtype=torch.float32)

                te_h = [herb_ids[i] for i in te_idx]
                te_d = [drug_ids[i] for i in te_idx]

                gat.train()
                for _ in range(20):
                    opt.zero_grad()
                    logits = gat(tr_h, tr_d)
                    loss = crit(logits, tr_y_t)
                    loss.backward()
                    opt.step()

                gat.eval()
                with torch.no_grad():
                    te_logits = gat(te_h, te_d)
                    oof_gat[te_idx] = torch.sigmoid(te_logits).numpy()

            results["logistic_regression"].append(float(roc_auc_score(y, oof_lr)))
            results["gradient_boosted_trees"].append(float(roc_auc_score(y, oof_gbt)))
            results["small_gat"].append(float(roc_auc_score(y, oof_gat)))

        summary = {
            "feature_set": self.feature_set,
            "feature_status": feature_label,
            "split_mode": split_mode,
            "n_splits": n_splits,
            "n_repeats": n_repeats,
            "models": {
                m: {
                    "mean_auc": float(np.mean(scores)),
                    "sd_auc": float(np.std(scores, ddof=1)) if len(scores) > 1 else 0.0,
                }
                for m, scores in results.items()
            },
        }
        return summary
