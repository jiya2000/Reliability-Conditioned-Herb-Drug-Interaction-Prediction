"""Phase 3 diagnostics script:
1. Repeat SD vs fold SD, cluster bootstrap CI resampling herbs and drugs.
2. Single-class test sets analysis in herb-disjoint CV.
3. Level 4 design-only tokens label rates, study-genre proxies, and sanitized design AUC.
4. Stratified performance within study genre (clinical trials vs case reports).
"""

import numpy as np
import pandas as pd
from collections import defaultdict
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.feature_extraction.text import TfidfVectorizer

from benchmarks.run_verified_hdi_benchmark import extract_tabular_features
from src.features.biological_features import BiologicalProfileExtractor
from src.data.evidence_lineage import EvidenceLineageAnalyzer
from src.training.evaluation import LeakageSafeSplitter
from src.features.text_masking import extract_study_design_text

def run_diagnostics():
    df = pd.read_csv("data/benchmark/verified_botanical_hdi.csv")
    y = df["outcome"].astype(str).str.lower().isin(["interaction_present", "positive", "1", "true"]).values.astype(int)

    print(f"Total dataset: N={len(df)}, Positives={y.sum()}, Negatives={len(y)-y.sum()}")

    # --- ITEM 2: Single-class folds in herb-disjoint CV ---
    print("\n==================== ITEM 2: SINGLE-CLASS FOLDS CHECK ====================")
    negative_herbs = ["HERB_BLACKCOHOSH", "HERB_MILKTHISTLE", "HERB_SAWPALMETTO", "HERB_VALERIAN"]
    print(f"100%-Negative Herbs: {negative_herbs}")
    
    # Check stratified vs unstratified across 20 repeats
    for stratify in [True, False]:
        single_class_fold_count = 0
        total_folds = 0
        skipped_folds = []
        for rep in range(20):
            splitter = LeakageSafeSplitter(seed=42 + rep)
            folds = splitter.grouped_kfold_split(df, mode="herb_disjoint", n_splits=5, stratify=stratify)
            for fold_idx, (tr_df, te_df) in enumerate(folds):
                total_folds += 1
                te_y = y[te_df.index]
                if len(np.unique(te_y)) < 2:
                    single_class_fold_count += 1
                    skipped_folds.append((rep, fold_idx, list(te_df["herb_id"].unique()), np.unique(te_y)[0]))
        strat_label = "Stratified (Default)" if stratify else "Unstratified"
        print(f"[{strat_label}] Total folds: {total_folds}, Single-class folds: {single_class_fold_count}")
        if skipped_folds:
            print(f"Sample skipped folds (first 5): {skipped_folds[:5]}")

    # --- ITEM 3: Level 4 Design-Only Tokens & Proxies ---
    print("\n==================== ITEM 3: LEVEL 4 DESIGN TOKENS ====================")
    design_texts = [extract_study_design_text(t) for t in df["original_text"]]
    
    token_counts = defaultdict(int)
    token_pos = defaultdict(int)
    for text, label in zip(design_texts, y):
        for tok in set(text.split()):
            token_counts[tok] += 1
            if label == 1:
                token_pos[tok] += 1

    genre_proxies = {"randomized_trial", "case_observation", "preclinical_in_vivo", "clinical_study"}
    print(f"{'Token':<30} {'Count':<8} {'Pos':<6} {'Neg':<6} {'Label Rate':<12} {'Study-Genre Proxy?'}")
    print("-" * 75)
    for tok, count in sorted(token_counts.items(), key=lambda x: x[1], reverse=True):
        pos = token_pos[tok]
        neg = count - pos
        rate = pos / count
        is_proxy = "FLAGGED PROXY" if tok in genre_proxies else "No (Methodology)"
        print(f"{tok:<30} {count:<8} {pos:<6} {neg:<6} {rate:<12.4f} {is_proxy}")

    # Evaluate full design vs sanitized design (proxies removed)
    def sanitize(text):
        remaining = [t for t in text.split() if t not in genre_proxies]
        return " ".join(remaining) if remaining else "methodology_unspecified"
    
    sanitized_texts = [sanitize(t) for t in design_texts]

    for name, texts in [("Full Level 4 Design", design_texts), ("Sanitized Level 4 (Proxies Removed)", sanitized_texts)]:
        print(f"\n--- {name} Performance ---")
        vec = TfidfVectorizer(token_pattern=r"(?u)\b\w+\b")
        X_tfidf = vec.fit_transform(texts).toarray()
        for mode in ["herb_disjoint", "drug_disjoint", "pair_disjoint"]:
            repeat_aucs = []
            for rep in range(20):
                splitter = LeakageSafeSplitter(seed=42 + rep)
                folds = splitter.grouped_kfold_split(df, mode=mode, n_splits=5)
                oof = np.zeros(len(df))
                for tr_df, te_df in folds:
                    clf = LogisticRegression(max_iter=500, random_state=42 + rep)
                    clf.fit(X_tfidf[tr_df.index], y[tr_df.index])
                    probs = clf.predict_proba(X_tfidf[te_df.index])[:, 1]
                    oof[te_df.index] = probs
                repeat_aucs.append(roc_auc_score(y, oof))
            print(f"{mode:<15}: Mean AUC = {np.mean(repeat_aucs):.4f} ± {np.std(repeat_aucs, ddof=1):.4f}")

    # --- ITEM 4: Stratified Performance Within Study Genre ---
    print("\n==================== ITEM 4: PERFORMANCE WITHIN STUDY GENRE ====================")
    bio = BiologicalProfileExtractor()
    analyzer = EvidenceLineageAnalyzer()
    summary_map = analyzer.analyze_dataset(df)
    X_tab, _ = extract_tabular_features(df, summary_map, bio, "quality_only")
    X_overlap = X_tab[:, 7:] # 6D
    
    # Curation 2D
    from src.features.biological_features import HERB_BIOLOGICAL_PROFILES
    curation_2d = []
    for h in df["herb_id"]:
        prof = HERB_BIOLOGICAL_PROFILES.get(h, {})
        nonzeros = sum(1 for v in prof.values() if abs(v) > 0.05)
        curation_2d.append([1.0 if nonzeros > 0 else 0.0, float(nonzeros)])
    X_curation_2d = np.array(curation_2d)

    vec_design = TfidfVectorizer(token_pattern=r"(?u)\b\w+\b")
    X_design = vec_design.fit_transform(design_texts).toarray()

    # Subsets:
    # 1. Clinical Trials / RCTs (evidence_type in ['clinical', 'rct'])
    # 2. Healthy volunteers only (population contains 'healthy' or 'volunteer')
    # 3. Case reports only (evidence_type == 'case_report')
    is_trial = df["evidence_type"].isin(["clinical", "rct"])
    is_healthy = df["population"].str.lower().str.contains("healthy|volunteer", na=False)
    is_case = df["evidence_type"] == "case_report"

    subsets = [
        ("Clinical Trials Cohort (clinical + rct)", is_trial),
        ("Healthy Volunteers Only Cohort", is_healthy),
        ("Case Reports Only Cohort", is_case),
    ]

    for sub_name, mask in subsets:
        sub_df = df[mask].reset_index(drop=True)
        sub_y = y[mask]
        n_pos = sub_y.sum()
        n_neg = len(sub_y) - n_pos
        print(f"\n--- {sub_name} (N={len(sub_df)}, Pos={n_pos}, Neg={n_neg}) ---")
        if n_pos == 0 or n_neg == 0:
            print("  ROC-AUC is mathematically undefined because the cohort is single-class (all positive or all negative).")
            continue

        for feat_name, X_full in [("Overlap 6D", X_overlap), ("Curation 2D", X_curation_2d), ("Design-Only", X_design)]:
            sub_X = X_full[mask]
            # Evaluate using standard 5-fold CV on this subset
            # We use repeated 5-fold CV on pairs within this genre
            repeat_aucs = []
            for rep in range(20):
                splitter = LeakageSafeSplitter(seed=42 + rep)
                # check if enough groups exist for pair_disjoint
                n_pairs = len(sub_df[["herb_id", "drug_id"]].drop_duplicates())
                mode = "pair_disjoint" if n_pairs >= 5 else "stratified"
                if mode == "pair_disjoint":
                    folds = splitter.grouped_kfold_split(sub_df, mode="pair_disjoint", n_splits=5)
                else:
                    # fallback
                    from sklearn.model_selection import StratifiedKFold
                    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42 + rep)
                    folds = [(sub_df.iloc[tr], sub_df.iloc[te]) for tr, te in skf.split(sub_df, sub_y)]
                
                oof = np.zeros(len(sub_df))
                for tr_df, te_df in folds:
                    clf = LogisticRegression(max_iter=500, random_state=42 + rep)
                    clf.fit(sub_X[tr_df.index], sub_y[tr_df.index])
                    probs = clf.predict_proba(sub_X[te_df.index])[:, 1]
                    oof[te_df.index] = probs
                repeat_aucs.append(roc_auc_score(sub_y, oof))
            print(f"  {feat_name:<15}: Mean AUC = {np.mean(repeat_aucs):.4f} ± {np.std(repeat_aucs, ddof=1):.4f}")

if __name__ == "__main__":
    run_diagnostics()
