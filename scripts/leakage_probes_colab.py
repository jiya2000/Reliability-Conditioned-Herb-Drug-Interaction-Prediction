"""
Self-contained Leakage Probes Script for Google Colab and Local Execution.
Evaluates 6 isolated feature representations under 5-repeat 5-fold Citation-Cluster Grouped CV:
1. 6D Biological Overlap only
2. 7D Quality Lineage only
3. 13D Combined (Quality + Overlap)
4. Herb One-Hot only
5. Drug One-Hot only
6. Text TF-IDF only
Plus: Text outcome word analysis (LR coefficients).
"""

import os
import sys
import json
import numpy as np
import pandas as pd
import networkx as nx
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import roc_auc_score, average_precision_score, f1_score, brier_score_loss

# Add project root to path
sys.path.insert(0, os.path.abspath("."))

# 1. Load Data
CSV_PATH = "data/benchmark/verified_botanical_hdi.csv"
if not os.path.exists(CSV_PATH):
    # Fallback to direct raw GitHub URL if running in standalone Colab without clone
    CSV_PATH = "https://raw.githubusercontent.com/jiya2000/Reliability-Conditioned-Herb-Drug-Interaction-Prediction/main/data/benchmark/verified_botanical_hdi.csv"

print(f"Loading benchmark dataset from: {CSV_PATH}")
df = pd.read_csv(CSV_PATH)
print(f"Loaded {len(df)} records across {df['herb_id'].nunique()} botanicals and {df['drug_id'].nunique()} prescription drugs.")

# 2. Extract Connected Citation Clusters
G = nx.Graph()
for _, row in df.iterrows():
    s_id = str(row["study_id"]).strip()
    G.add_node(s_id)
    cited_list = row.get("cited_study_ids", [])
    if isinstance(cited_list, str):
        try:
            cited_list = json.loads(cited_list)
        except Exception:
            cited_list = [c.strip() for c in cited_list.split(",") if c.strip()]
    if isinstance(cited_list, (list, tuple, set)):
        for c_id in cited_list:
            c_str = str(c_id).strip()
            if c_str and c_str != s_id:
                G.add_node(c_str)
                G.add_edge(s_id, c_str)

all_clusters = list(nx.connected_components(G))
df_studies = set(df["study_id"].astype(str))
clusters = [c for c in all_clusters if c & df_studies]
n_clusters = len(clusters)
print(f"Identified {n_clusters} independent citation clusters (zero study/lineage overlap).")

# 3. Ground Truth Labels
y_all = np.array([1 if str(o).lower() in {"interaction_present", "positive", "1", "true"} else 0 for o in df["outcome"]])
print(f"Label distribution: {np.sum(y_all == 1)} Positive ({np.mean(y_all)*100:.1f}%), {np.sum(y_all == 0)} Negative ({ (1-np.mean(y_all))*100:.1f}%)")

# 4. Feature Construction Helpers
try:
    from src.features.biological_features import BiologicalProfileExtractor
    from src.data.evidence_lineage import EvidenceLineageAnalyzer
    from benchmarks.run_verified_hdi_benchmark import extract_tabular_features
    HAS_LOCAL_MODULES = True
except ImportError:
    HAS_LOCAL_MODULES = False
    print("Local modules not found. Using standalone feature derivation...")

if HAS_LOCAL_MODULES:
    bio_extractor = BiologicalProfileExtractor()
    analyzer = EvidenceLineageAnalyzer()
    summary_map = analyzer.analyze_dataset(df)
    X_tab_13, _ = extract_tabular_features(df, summary_map, bio_extractor, feature_group="quality_only")
    X_qual_7 = X_tab_13[:, :7]
    X_over_6 = X_tab_13[:, 7:]
else:
    # Standalone fallback: build 6D overlap from known CYP profiles if running without src
    # In cloned repo, HAS_LOCAL_MODULES is always True.
    raise RuntimeError("Please run inside the repository root or clone the repo first!")

# One-hot representations
all_herbs = sorted(df["herb_id"].unique())
h_map = {h: i for i, h in enumerate(all_herbs)}
X_herb_oh = np.zeros((len(df), len(all_herbs)), dtype=np.float32)
for i, h in enumerate(df["herb_id"]):
    X_herb_oh[i, h_map[h]] = 1.0

all_drugs = sorted(df["drug_id"].unique())
d_map = {d: i for i, d in enumerate(all_drugs)}
X_drug_oh = np.zeros((len(df), len(all_drugs)), dtype=np.float32)
for i, d in enumerate(df["drug_id"]):
    X_drug_oh[i, d_map[d]] = 1.0

# 5. Execute 5-Repeat 5-Fold Grouped Cross-Validation
PROBE_KEYS = ["overlap_6d", "quality_7d", "combined_13d", "herb_onehot", "drug_onehot", "text_tfidf"]
PROBE_NAMES = {
    "overlap_6d": "6D Biological Overlap Only (CYP/P-gp)",
    "quality_7d": "7D Evidence Quality Lineage Only",
    "combined_13d": "13D Combined (Quality + Overlap)",
    "herb_onehot": "Herb-Identity One-Hot Only (18-D)",
    "drug_onehot": "Drug-Identity One-Hot Only (40-D)",
    "text_tfidf": "Text TF-IDF Only (Outcome Language)",
}

metrics = {k: {"auc": [], "prauc": [], "f1": [], "brier": []} for k in PROBE_KEYS}

n_repeats = 5
n_folds = 5
print(f"\nRunning 5 repeats x 5 folds = 25 evaluations per probe...")

for rep in range(n_repeats):
    rng = np.random.RandomState(42 + rep * 100)
    shuffled_clusters = [clusters[idx] for idx in rng.permutation(n_clusters)]
    fold_allocations = np.array_split(shuffled_clusters, n_folds)

    for fold_idx in range(n_folds):
        test_cluster_set = set().union(*fold_allocations[fold_idx])
        train_clusters = [c for f, chunk in enumerate(fold_allocations) if f != fold_idx for c in chunk]
        train_cluster_set = set().union(*train_clusters)

        train_mask = df["study_id"].astype(str).isin(train_cluster_set).values
        test_mask = df["study_id"].astype(str).isin(test_cluster_set).values

        y_tr, y_te = y_all[train_mask], y_all[test_mask]
        if len(np.unique(y_te)) < 2:
            continue

        # Fit TF-IDF on train split strictly
        vec = TfidfVectorizer(max_features=100, min_df=2, stop_words="english")
        X_tr_text = vec.fit_transform(df.loc[train_mask, "original_text"].fillna("")).toarray()
        X_te_text = vec.transform(df.loc[test_mask, "original_text"].fillna("")).toarray()

        splits = {
            "overlap_6d": (X_over_6[train_mask], X_over_6[test_mask]),
            "quality_7d": (X_qual_7[train_mask], X_qual_7[test_mask]),
            "combined_13d": (X_tab_13[train_mask], X_tab_13[test_mask]),
            "herb_onehot": (X_herb_oh[train_mask], X_herb_oh[test_mask]),
            "drug_onehot": (X_drug_oh[train_mask], X_drug_oh[test_mask]),
            "text_tfidf": (X_tr_text, X_te_text),
        }

        for k in PROBE_KEYS:
            xtr, xte = splits[k]
            clf = LogisticRegression(max_iter=1000, random_state=42)
            clf.fit(xtr, y_tr)
            p = clf.predict_proba(xte)[:, 1]
            pred = (p >= 0.5).astype(int)

            metrics[k]["auc"].append(roc_auc_score(y_te, p))
            metrics[k]["prauc"].append(average_precision_score(y_te, p))
            metrics[k]["f1"].append(f1_score(y_te, pred, zero_division=0))
            metrics[k]["brier"].append(brier_score_loss(y_te, p))

# 6. Print Summary Table
print("\n" + "=" * 80)
print(f"{'Probe Representation':38s} | {'ROC-AUC (Mean ± SD)':20s} | {'PR-AUC':10s} | {'F1-Score':10s} | {'Brier':10s}")
print("=" * 80)
for k in PROBE_KEYS:
    auc_m, auc_s = np.mean(metrics[k]["auc"]), np.std(metrics[k]["auc"])
    prauc_m = np.mean(metrics[k]["prauc"])
    f1_m = np.mean(metrics[k]["f1"])
    brier_m = np.mean(metrics[k]["brier"])
    print(f"{PROBE_NAMES[k]:38s} | {auc_m:.4f} ± {auc_s:.4f}       | {prauc_m:.4f}     | {f1_m:.4f}     | {brier_m:.4f}")
print("=" * 80)

# 7. Text Leakage Words Analysis
print("\n" + "=" * 80)
print("TEXT LEAKAGE & OUTCOME WORD DIAGNOSTIC")
print("=" * 80)
vec_full = TfidfVectorizer(max_features=250, min_df=2, stop_words="english")
X_full_text = vec_full.fit_transform(df["original_text"].fillna("")).toarray()
vocab = np.array(vec_full.get_feature_names_out())

clf_text = LogisticRegression(max_iter=1000, random_state=42)
clf_text.fit(X_full_text, y_all)
coefs = clf_text.coef_[0]

top_pos = np.argsort(coefs)[-10:][::-1]
top_neg = np.argsort(coefs)[:10]

print("\nTop 10 Positive Interaction Drivers in Text (Predicts Interaction):")
for idx in top_pos:
    print(f"  + {vocab[idx]:20s} : {coefs[idx]:+.4f}")

print("\nTop 10 Negative Interaction Drivers in Text (Predicts No Interaction):")
for idx in top_neg:
    print(f"  - {vocab[idx]:20s} : {coefs[idx]:+.4f}")

phrases = ["increased", "decreased", "no significant", "reduced", "clearance", "bleeding", "inhibition", "induction"]
print("\nFrequency of Explicit Study-Outcome Words in original_text:")
for p in phrases:
    count = df["original_text"].str.contains(p, case=False, na=False).sum()
    print(f"  '{p}': {count} occurrences ({count/len(df)*100:.1f}%)")
print("=" * 80)
