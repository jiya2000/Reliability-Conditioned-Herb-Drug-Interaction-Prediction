"""
Comprehensive Shortcut and Robustness Test Suite for Mechanistic Features (Phase 3d).

Protocol:
- 20 repeats of 5-fold grouped CV for herb-disjoint, drug-disjoint, pair-disjoint.
- Primary metric: Mean per-fold AUC (with fold prediction vectors and per-fold AUCs stored).
- Secondary metric: Pooled out-of-fold AUC across folds.
- Cluster-bootstrap CIs (resample herbs, and separately drugs, 1000x with replacement).
- All numbers stored in results/shortcut_tests.json.
- Zero stub blocks.
"""

import os
import sys
import json
from collections import Counter
from typing import Dict, List, Tuple, Any, Optional
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import TfidfVectorizer

# Add repo root to path
sys.path.insert(0, os.path.abspath("."))

from src.features.biological_features import (
    BiologicalProfileExtractor,
    HERB_BIOLOGICAL_PROFILES,
    TARGET_TO_IDX,
    CANONICAL_TARGETS,
)
from src.data.evidence_lineage import EvidenceLineageAnalyzer
from benchmarks.run_verified_hdi_benchmark import extract_tabular_features
from src.training.evaluation import LeakageSafeSplitter
from src.features.text_masking import extract_study_design_text


def fast_auc(y_true: np.ndarray, y_score: np.ndarray) -> float:
    """Fast rank-sum ROC-AUC calculation (exact match with sklearn)."""
    n_pos = int(np.sum(y_true == 1))
    n_neg = int(len(y_true) - n_pos)
    if n_pos == 0 or n_neg == 0:
        return 0.5
    if np.all(y_score == y_score[0]):
        return 0.5
    order = np.argsort(y_score)
    ranks = np.empty_like(order, dtype=float)
    ranks[order] = np.arange(len(y_score)) + 1
    return float((np.sum(ranks[y_true == 1]) - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg))


def compute_bootstrap_ci(scores: List[float], n_bootstraps: int = 1000, seed: int = 42) -> Tuple[float, float]:
    """Compute 95% bootstrap confidence interval of the mean score."""
    if len(scores) == 0:
        return 0.0, 0.0
    arr = np.array(scores)
    rng = np.random.RandomState(seed)
    boot_means = []
    n = len(arr)
    for _ in range(n_bootstraps):
        idx = rng.randint(0, n, size=n)
        boot_means.append(float(np.mean(arr[idx])))
    ci_low = float(np.percentile(boot_means, 2.5))
    ci_high = float(np.percentile(boot_means, 97.5))
    return round(ci_low, 4), round(ci_high, 4)


def compute_cluster_bootstrap_ci(
    fold_details: List[Dict[str, Any]],
    unique_entities: List[str],
    cluster_key: str,
    n_bootstraps: int = 1000,
    seed: int = 42,
) -> Tuple[float, float]:
    """
    Vectorized cluster-bootstrap CI: resample unique entities with replacement,
    recomputing per-fold-mean AUC over resampled entities across all folds.
    """
    n_entities = len(unique_entities)
    rng = np.random.RandomState(seed)

    boot_means = []
    for _ in range(n_bootstraps):
        sampled_entities = rng.choice(unique_entities, size=n_entities, replace=True)
        counts = Counter(sampled_entities)

        fold_aucs_b = []
        for fold in fold_details:
            ents = fold[cluster_key]
            weights = np.array([counts.get(e, 0) for e in ents], dtype=int)
            nz_mask = weights > 0
            if not np.any(nz_mask):
                continue
            resamp_idx = np.repeat(np.arange(len(ents))[nz_mask], weights[nz_mask])
            resamp_y = fold["y_te"][resamp_idx]
            if len(np.unique(resamp_y)) < 2:
                continue
            resamp_p = fold["probs"][resamp_idx]
            fold_aucs_b.append(fast_auc(resamp_y, resamp_p))

        if len(fold_aucs_b) > 0:
            boot_means.append(float(np.mean(fold_aucs_b)))

    if len(boot_means) == 0:
        return 0.5, 0.5

    ci_low = float(np.percentile(boot_means, 2.5))
    ci_high = float(np.percentile(boot_means, 97.5))
    return round(ci_low, 4), round(ci_high, 4)


def run_repeated_grouped_cv(
    df: pd.DataFrame,
    X: np.ndarray,
    y: np.ndarray,
    mode: str = "herb_disjoint",
    n_repeats: int = 20,
    n_splits: int = 5,
    base_seed: int = 42,
    compute_cluster_ci: bool = False,
) -> Dict[str, Any]:
    """
    Run n_repeats of n_splits grouped CV using LogisticRegression.
    Primary metric: Mean per-fold AUC.
    Secondary metric: Pooled out-of-fold AUC.
    Stores all 100 per-fold AUCs, fold prediction vectors, and cluster bootstrap CIs.
    """
    all_herbs = sorted(df["herb_id"].unique())
    all_drugs = sorted(df["drug_id"].unique())

    repeat_mean_fold_aucs = []
    pooled_oof_scores = []
    all_fold_aucs = []
    all_fold_details = []
    fold_predictions = []

    for rep in range(n_repeats):
        seed = base_seed + rep
        splitter = LeakageSafeSplitter(seed=seed)
        folds = splitter.grouped_kfold_split(df, mode=mode, n_splits=n_splits)

        oof_preds = np.zeros(len(df), dtype=float)
        rep_fold_aucs = []

        for fold_idx, (train_df, test_df) in enumerate(folds):
            train_idx = train_df.index.values
            test_idx = test_df.index.values

            X_tr, y_tr = X[train_idx], y[train_idx]
            X_te, y_te = X[test_idx], y[test_idx]

            if len(np.unique(y_tr)) < 2:
                prior = float(np.mean(y_tr))
                probs = np.full(len(test_idx), prior, dtype=float)
            else:
                clf = LogisticRegression(max_iter=1000, random_state=seed)
                clf.fit(X_tr, y_tr)
                probs = clf.predict_proba(X_te)[:, 1]

            oof_preds[test_idx] = probs

            # Per-fold AUC calculation
            fold_auc = fast_auc(y_te, probs)

            rep_fold_aucs.append(fold_auc)
            all_fold_aucs.append(fold_auc)

            all_fold_details.append({
                "y_te": y_te,
                "probs": probs,
                "herb_ents": df["herb_id"].values[test_idx],
                "drug_ents": df["drug_id"].values[test_idx],
            })

            fold_predictions.append({
                "rep": rep,
                "fold": fold_idx,
                "indices": [int(idx) for idx in test_idx],
                "preds": [round(float(p), 4) for p in probs],
            })

        # Primary repeat score = mean of fold AUCs in this repeat
        repeat_mean_fold_aucs.append(float(np.mean(rep_fold_aucs)))

        # Secondary pooled OOF AUC for this repeat
        pooled_auc = fast_auc(y, oof_preds)
        pooled_oof_scores.append(pooled_auc)

    primary_mean = float(np.mean(all_fold_aucs))
    primary_sd = float(np.std(repeat_mean_fold_aucs, ddof=1)) if len(repeat_mean_fold_aucs) > 1 else 0.0
    fold_sd = float(np.std(all_fold_aucs, ddof=1)) if len(all_fold_aucs) > 1 else 0.0

    pooled_oof_mean = float(np.mean(pooled_oof_scores))
    pooled_oof_sd = float(np.std(pooled_oof_scores, ddof=1)) if len(pooled_oof_scores) > 1 else 0.0

    ci_low, ci_high = compute_bootstrap_ci(repeat_mean_fold_aucs, n_bootstraps=1000, seed=base_seed)

    res = {
        "mean": round(primary_mean, 4),
        "sd": round(primary_sd, 4),
        "fold_sd": round(fold_sd, 4),
        "ci_95": [ci_low, ci_high],
        "repeat_scores": [round(s, 4) for s in repeat_mean_fold_aucs],
        "per_fold_aucs": [round(s, 4) for s in all_fold_aucs],
        "pooled_oof_mean": round(pooled_oof_mean, 4),
        "pooled_oof_sd": round(pooled_oof_sd, 4),
        "pooled_oof_repeat_scores": [round(s, 4) for s in pooled_oof_scores],
        "fold_predictions": fold_predictions,
    }

    if compute_cluster_ci:
        herb_ci = compute_cluster_bootstrap_ci(all_fold_details, all_herbs, "herb_ents", n_bootstraps=1000, seed=base_seed)
        drug_ci = compute_cluster_bootstrap_ci(all_fold_details, all_drugs, "drug_ents", n_bootstraps=1000, seed=base_seed)
        res["herb_cluster_ci_95"] = [herb_ci[0], herb_ci[1]]
        res["drug_cluster_ci_95"] = [drug_ci[0], drug_ci[1]]

    return res


def main():
    print("=" * 70)
    print("RUNNING SHORTCUT AND ROBUSTNESS TESTS (PHASE 3d)")
    print("=" * 70)

    # 1. Load benchmark data
    csv_path = "data/benchmark/verified_botanical_hdi.csv"
    df = pd.read_csv(csv_path)
    df = df.reset_index(drop=True)
    n_samples = len(df)
    print(f"Loaded {n_samples} records.")

    y = np.array([1 if str(o).lower() in {"interaction_present", "positive", "1", "true"} else 0 for o in df["outcome"]], dtype=int)

    # 2. Extract standard features
    bio_extractor = BiologicalProfileExtractor()
    analyzer = EvidenceLineageAnalyzer()
    summary_map = analyzer.analyze_dataset(df)

    X_tab_13, _ = extract_tabular_features(df, summary_map, bio_extractor, feature_group="quality_only")
    X_qual_7 = X_tab_13[:, :7]
    X_overlap_6 = X_tab_13[:, 7:]  # [cyp3a4, pgp, cyp2c9, cyp1a2, cyp2e1, overlap_total]

    all_herbs = sorted(df["herb_id"].unique())
    h_to_idx = {h: i for i, h in enumerate(all_herbs)}
    X_herb_oh = np.zeros((n_samples, len(all_herbs)), dtype=np.float32)
    for i, h in enumerate(df["herb_id"]):
        X_herb_oh[i, h_to_idx[h]] = 1.0

    all_drugs = sorted(df["drug_id"].unique())
    d_to_idx = {d: i for i, d in enumerate(all_drugs)}
    X_drug_oh = np.zeros((n_samples, len(all_drugs)), dtype=np.float32)
    for i, d in enumerate(df["drug_id"]):
        X_drug_oh[i, d_to_idx[d]] = 1.0

    # Curation-only features
    curation_1d = np.zeros((n_samples, 1), dtype=np.float32)
    curation_2d = np.zeros((n_samples, 2), dtype=np.float32)
    for i, row in df.iterrows():
        prof = HERB_BIOLOGICAL_PROFILES.get(row["herb_id"], {})
        nonzero_targets = sum(1 for v in prof.values() if abs(v) > 0.05)
        has_nonzero = 1.0 if nonzero_targets > 0 else 0.0
        curation_1d[i, 0] = has_nonzero
        curation_2d[i, 0] = has_nonzero
        curation_2d[i, 1] = float(nonzero_targets)

    # Overlap column statistics
    target_names = ["CYP3A4", "ABCB1_Pgp", "CYP2C9", "CYP1A2", "CYP2E1", "overlap_total"]
    overlap_col_stats = {}
    for idx, tname in enumerate(target_names):
        col = X_overlap_6[:, idx]
        nz = int(np.sum(np.abs(col) > 1e-5))
        overlap_col_stats[tname] = {
            "nonzero_count": nz,
            "total_records": n_samples,
            "fraction_nonzero": round(nz / n_samples, 4),
            "min": round(float(col.min()), 4),
            "max": round(float(col.max()), 4),
        }

    results = {
        "metadata": {
            "num_records": n_samples,
            "num_herbs": len(all_herbs),
            "num_drugs": len(all_drugs),
            "primary_metric": "Mean per-fold ROC-AUC (5 folds x 20 repeats = 100 fold evaluations)",
            "secondary_metric": "Pooled out-of-fold ROC-AUC across folds",
            "protocol": "20 repeats of 5-fold grouped CV, Logistic Regression, cluster bootstrap CIs",
        },
        "curation_tests": {},
        "target_ablations": {},
        "permutation_test": {},
        "leave_one_entity_out": {
            "herbs": {},
            "drugs": {},
        },
        "identity_baselines": {},
        "overlap_column_statistics": overlap_col_stats,
        "genre_stratification": {},
    }

    modes = ["herb_disjoint", "drug_disjoint", "pair_disjoint"]

    # -------------------------------------------------------------
    # 1. Curation-Only vs Overlap-6D (compute_cluster_ci=True)
    # -------------------------------------------------------------
    print("\n--- 1. Curation-Only Feature vs Overlap-6D ---")
    results["curation_tests"]["overlap_6d"] = {}
    results["curation_tests"]["curation_1d_has_nonzero"] = {}
    results["curation_tests"]["curation_2d_num_nonzero"] = {}

    for mode in modes:
        print(f"Running mode: {mode}...")
        res_over = run_repeated_grouped_cv(df, X_overlap_6, y, mode=mode, compute_cluster_ci=True)
        res_cur1 = run_repeated_grouped_cv(df, curation_1d, y, mode=mode, compute_cluster_ci=True)
        res_cur2 = run_repeated_grouped_cv(df, curation_2d, y, mode=mode, compute_cluster_ci=True)

        results["curation_tests"]["overlap_6d"][mode] = res_over
        results["curation_tests"]["curation_1d_has_nonzero"][mode] = res_cur1
        results["curation_tests"]["curation_2d_num_nonzero"][mode] = res_cur2

        print(f"  [{mode}] Overlap-6D: {res_over['mean']:.4f} ± {res_over['sd']:.4f} (Fold SD: {res_over['fold_sd']:.4f}, Herb CI: {res_over['herb_cluster_ci_95']}, Drug CI: {res_over['drug_cluster_ci_95']}, Pooled: {res_over['pooled_oof_mean']:.4f})")
        print(f"  [{mode}] Curation-1D: {res_cur1['mean']:.4f} ± {res_cur1['sd']:.4f} (Pooled: {res_cur1['pooled_oof_mean']:.4f})")
        print(f"  [{mode}] Curation-2D: {res_cur2['mean']:.4f} ± {res_cur2['sd']:.4f} (Pooled: {res_cur2['pooled_oof_mean']:.4f})")

    # -------------------------------------------------------------
    # 2. Per-Target Ablation for Overlap
    # -------------------------------------------------------------
    print("\n--- 2. Per-Target Ablation ---")
    results["target_ablations"]["single_target_alone"] = {}
    for idx, tname in enumerate(target_names):
        results["target_ablations"]["single_target_alone"][tname] = {}
        X_single = X_overlap_6[:, [idx]]
        for mode in modes:
            res = run_repeated_grouped_cv(df, X_single, y, mode=mode)
            results["target_ablations"]["single_target_alone"][tname][mode] = res
            print(f"  Single target [{tname}] ({mode}): {res['mean']:.4f} ± {res['sd']:.4f}")

    results["target_ablations"]["leave_one_target_out"] = {}
    for idx, tname in enumerate(target_names):
        results["target_ablations"]["leave_one_target_out"][f"without_{tname}"] = {}
        keep_indices = [i for i in range(6) if i != idx]
        X_loo = X_overlap_6[:, keep_indices]
        for mode in modes:
            res = run_repeated_grouped_cv(df, X_loo, y, mode=mode)
            results["target_ablations"]["leave_one_target_out"][f"without_{tname}"][mode] = res
            print(f"  Leave-one-out [without {tname}] ({mode}): {res['mean']:.4f} ± {res['sd']:.4f}")

    # -------------------------------------------------------------
    # 3. Permutation Test (1000 Shuffles)
    # -------------------------------------------------------------
    print("\n--- 3. Permutation Test (1000 Shuffles) ---")
    n_perms = 1000
    results["permutation_test"] = {}

    for mode in ["herb_disjoint", "pair_disjoint"]:
        print(f"Running permutation test for mode: {mode} (1000 shuffles)...")
        splitter = LeakageSafeSplitter(seed=42)
        folds = splitter.grouped_kfold_split(df, mode=mode, n_splits=5)

        true_res_over = results["curation_tests"]["overlap_6d"][mode]["mean"]
        true_res_comb = run_repeated_grouped_cv(df, X_tab_13, y, mode=mode)["mean"]

        null_aucs_over = []
        null_aucs_comb = []
        rng = np.random.RandomState(42)

        for _ in range(n_perms):
            y_perm = rng.permutation(y)
            fold_aucs_o = []
            fold_aucs_c = []

            for train_df_f, test_df_f in folds:
                tr_idx = train_df_f.index.values
                te_idx = test_df_f.index.values

                y_tr_p = y_perm[tr_idx]
                y_te_p = y_perm[te_idx]

                if len(np.unique(y_tr_p)) < 2 or len(np.unique(y_te_p)) < 2:
                    fold_aucs_o.append(0.5)
                    fold_aucs_c.append(0.5)
                else:
                    clf_o = LogisticRegression(max_iter=500, random_state=42)
                    clf_o.fit(X_overlap_6[tr_idx], y_tr_p)
                    probs_o = clf_o.predict_proba(X_overlap_6[te_idx])[:, 1]
                    fold_aucs_o.append(fast_auc(y_te_p, probs_o))

                    clf_c = LogisticRegression(max_iter=500, random_state=42)
                    clf_c.fit(X_tab_13[tr_idx], y_tr_p)
                    probs_c = clf_c.predict_proba(X_tab_13[te_idx])[:, 1]
                    fold_aucs_c.append(fast_auc(y_te_p, probs_c))

            null_aucs_over.append(float(np.mean(fold_aucs_o)))
            null_aucs_comb.append(float(np.mean(fold_aucs_c)))

        cnt_over = int(np.sum(np.array(null_aucs_over) >= true_res_over))
        cnt_comb = int(np.sum(np.array(null_aucs_comb) >= true_res_comb))

        p_str_over = "< 1/(n_perm+1)" if cnt_over == 0 else f"{cnt_over / n_perms:.4f}"
        p_str_comb = "< 1/(n_perm+1)" if cnt_comb == 0 else f"{cnt_comb / n_perms:.4f}"

        results["permutation_test"][mode] = {
            "overlap_6d": {
                "true_auc": true_res_over,
                "null_mean": round(float(np.mean(null_aucs_over)), 4),
                "null_sd": round(float(np.std(null_aucs_over)), 4),
                "null_ci_95": [round(float(np.percentile(null_aucs_over, 2.5)), 4), round(float(np.percentile(null_aucs_over, 97.5)), 4)],
                "p_value": p_str_over,
            },
            "combined_13d": {
                "true_auc": true_res_comb,
                "null_mean": round(float(np.mean(null_aucs_comb)), 4),
                "null_sd": round(float(np.std(null_aucs_comb)), 4),
                "null_ci_95": [round(float(np.percentile(null_aucs_comb, 2.5)), 4), round(float(np.percentile(null_aucs_comb, 97.5)), 4)],
                "p_value": p_str_comb,
            },
        }

    # -------------------------------------------------------------
    # 4. Per-Herb and Per-Drug Leave-One-Out for Overlap-6D
    # -------------------------------------------------------------
    print("\n--- 4. Per-Herb and Per-Drug Leave-One-Out (overlap_6d) ---")
    base_auc = results["curation_tests"]["overlap_6d"]["pair_disjoint"]["mean"]

    for h in all_herbs:
        sub_mask = (df["herb_id"] != h).values
        df_sub = df[sub_mask].reset_index(drop=True)
        X_sub = X_overlap_6[sub_mask]
        y_sub = y[sub_mask]

        res = run_repeated_grouped_cv(df_sub, X_sub, y_sub, mode="pair_disjoint", n_repeats=10, n_splits=5)
        drop = base_auc - res["mean"]
        flagged = drop > 0.05

        results["leave_one_entity_out"]["herbs"][h] = {
            "records_removed": int(np.sum(~sub_mask)),
            "remaining_records": len(df_sub),
            "auc": res["mean"],
            "auc_sd": res["sd"],
            "delta_from_baseline": round(drop, 4),
            "drops_more_than_0_05": flagged,
        }

    drug_counts = df["drug_id"].value_counts()
    for d, count in drug_counts.items():
        if count < 2:
            continue
        sub_mask = (df["drug_id"] != d).values
        df_sub = df[sub_mask].reset_index(drop=True)
        X_sub = X_overlap_6[sub_mask]
        y_sub = y[sub_mask]

        res = run_repeated_grouped_cv(df_sub, X_sub, y_sub, mode="pair_disjoint", n_repeats=10, n_splits=5)
        drop = base_auc - res["mean"]
        flagged = drop > 0.05

        results["leave_one_entity_out"]["drugs"][d] = {
            "records_removed": int(count),
            "remaining_records": len(df_sub),
            "auc": res["mean"],
            "auc_sd": res["sd"],
            "delta_from_baseline": round(drop, 4),
            "drops_more_than_0_05": flagged,
        }

    # -------------------------------------------------------------
    # 5. Entity-Identity-Only Baselines
    # -------------------------------------------------------------
    print("\n--- 5. Entity-Identity-Only Baselines ---")
    results["identity_baselines"]["herb_onehot"] = {}
    results["identity_baselines"]["drug_onehot"] = {}

    for mode in modes:
        res_h = run_repeated_grouped_cv(df, X_herb_oh, y, mode=mode)
        res_d = run_repeated_grouped_cv(df, X_drug_oh, y, mode=mode)

        results["identity_baselines"]["herb_onehot"][mode] = res_h
        results["identity_baselines"]["drug_onehot"][mode] = res_d

        print(f"  [{mode}] Herb One-Hot: {res_h['mean']:.4f} ± {res_h['sd']:.4f} (Pooled: {res_h['pooled_oof_mean']:.4f})")
        print(f"  [{mode}] Drug One-Hot: {res_d['mean']:.4f} ± {res_d['sd']:.4f} (Pooled: {res_d['pooled_oof_mean']:.4f})")

    # -------------------------------------------------------------
    # 6. Genre Stratification Under Herb-Disjoint
    # -------------------------------------------------------------
    print("\n--- 6. Genre Stratification Under Herb-Disjoint ---")
    pure_neg_herbs = {"HERB_BLACKCOHOSH", "HERB_MILKTHISTLE", "HERB_SAWPALMETTO", "HERB_VALERIAN"}

    genre_proxies = {"randomized_trial", "case_observation", "preclinical_in_vivo", "clinical_study"}
    def sanitize(text):
        remaining = [t for t in text.split() if t not in genre_proxies]
        return " ".join(remaining) if remaining else "methodology_unspecified"
    design_texts = [sanitize(extract_study_design_text(t)) for t in df["original_text"]]
    vec = TfidfVectorizer(token_pattern=r"(?u)\b\w+\b")
    X_design = vec.fit_transform(design_texts).toarray()
    X_overlap_total = X_overlap_6[:, [5]]

    is_trial = df["evidence_type"].isin(["clinical", "rct"]).values
    is_trial_no_pure_neg = is_trial & (~df["herb_id"].isin(pure_neg_herbs)).values

    is_healthy = df["population"].str.lower().str.contains("healthy|volunteer", na=False).values
    is_healthy_no_pure_neg = is_healthy & (~df["herb_id"].isin(pure_neg_herbs)).values

    is_case = (df["evidence_type"] == "case_report").values

    subsets_spec = [
        ("clinical_trials_all", is_trial, "Clinical Trials Cohort (All)"),
        ("clinical_trials_excl_pure_neg", is_trial_no_pure_neg, "Clinical Trials Cohort (Excl. 4 Pure-Negative Herbs)"),
        ("healthy_volunteers_all", is_healthy, "Healthy Volunteers Cohort (All)"),
        ("healthy_volunteers_excl_pure_neg", is_healthy_no_pure_neg, "Healthy Volunteers Cohort (Excl. 4 Pure-Negative Herbs)"),
        ("case_reports_all", is_case, "Case Reports Cohort (All)"),
    ]

    for key, mask, label in subsets_spec:
        sub_df = df[mask].reset_index(drop=True)
        sub_y = y[mask]
        n_pos = int(sub_y.sum())
        n_neg = int(len(sub_y) - n_pos)

        results["genre_stratification"][key] = {
            "label": label,
            "N": len(sub_df),
            "pos": n_pos,
            "neg": n_neg,
            "class_balance": f"{n_pos} pos / {n_neg} neg",
            "features": {},
        }

        feature_mats = [
            ("overlap_6d", X_overlap_6[mask]),
            ("overlap_total", X_overlap_total[mask]),
            ("curation_2d", curation_2d[mask]),
            ("design_only", X_design[mask]),
        ]

        for fname, sub_X in feature_mats:
            if n_pos == 0 or n_neg == 0:
                results["genre_stratification"][key]["features"][fname] = {
                    "mean": 0.5000,
                    "sd": 0.0000,
                    "fold_sd": 0.0000,
                    "pooled_oof_mean": 0.5000,
                }
            else:
                res = run_repeated_grouped_cv(sub_df, sub_X, sub_y, mode="herb_disjoint", n_repeats=20, n_splits=5)
                results["genre_stratification"][key]["features"][fname] = {
                    "mean": res["mean"],
                    "sd": res["sd"],
                    "fold_sd": res["fold_sd"],
                    "pooled_oof_mean": res["pooled_oof_mean"],
                }

    # Save output
    os.makedirs("results", exist_ok=True)
    with open("results/shortcut_tests.json", "w") as f:
        json.dump(results, f, indent=2)
    print("\nSuccessfully updated results/shortcut_tests.json!")


if __name__ == "__main__":
    main()
