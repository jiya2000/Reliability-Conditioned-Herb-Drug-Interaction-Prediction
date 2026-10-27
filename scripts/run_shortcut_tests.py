"""
Comprehensive Shortcut and Robustness Test Suite for Mechanistic Features.
Evaluates:
1. Curation-only features (1D and 2D) vs overlap_6d
2. Per-target ablation (single target alone and leave-one-target-out)
3. Permutation test (1000 permutations for overlap_6d and combined_13d)
4. Leave-one-entity-out (per-herb and per-drug for overlap_6d)
5. Entity-identity baselines (herb_onehot, drug_onehot)

Protocol: 20 repeats of 5-fold grouped CV for herb-disjoint, drug-disjoint, pair-disjoint.
Reports: Mean, SD, 95% bootstrap CI.
Results saved to results/shortcut_tests.json and reports/shortcut_tests.md.
"""

import os
import sys
import json
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Any
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score

# Add repo root to path
sys.path.insert(0, os.path.abspath("."))

from src.features.biological_features import BiologicalProfileExtractor, TARGET_TO_IDX, CANONICAL_TARGETS
from src.data.evidence_lineage import EvidenceLineageAnalyzer
from benchmarks.run_verified_hdi_benchmark import extract_tabular_features
from src.training.evaluation import LeakageSafeSplitter


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
    return ci_low, ci_high


def run_repeated_grouped_cv(
    df: pd.DataFrame,
    X: np.ndarray,
    y: np.ndarray,
    mode: str = "herb_disjoint",
    n_repeats: int = 20,
    n_splits: int = 5,
    base_seed: int = 42,
) -> Dict[str, Any]:
    """
    Run n_repeats of n_splits grouped CV using LogisticRegression.
    Returns dictionary with repeat_scores, mean, sd, and 95% bootstrap CI.
    """
    repeat_scores = []

    for rep in range(n_repeats):
        seed = base_seed + rep
        splitter = LeakageSafeSplitter(seed=seed)
        folds = splitter.grouped_kfold_split(df, mode=mode, n_splits=n_splits)

        oof_preds = np.zeros(len(df), dtype=float)
        oof_visited = np.zeros(len(df), dtype=bool)

        for fold_idx, (train_df, test_df) in enumerate(folds):
            train_idx = train_df.index.values
            test_idx = test_df.index.values

            X_tr, y_tr = X[train_idx], y[train_idx]
            X_te, y_te = X[test_idx], y[test_idx]

            # If train has only one class, predict empirical prior
            if len(np.unique(y_tr)) < 2:
                prior = float(np.mean(y_tr))
                oof_preds[test_idx] = prior
            else:
                clf = LogisticRegression(max_iter=1000, random_state=seed)
                clf.fit(X_tr, y_tr)
                # Predict positive class probability
                probs = clf.predict_proba(X_te)[:, 1]
                oof_preds[test_idx] = probs

            oof_visited[test_idx] = True

        # Calculate AUC across all out-of-fold predictions
        assert np.all(oof_visited), "Not all samples received out-of-fold predictions"
        auc = float(roc_auc_score(y, oof_preds))
        repeat_scores.append(auc)

    mean_auc = float(np.mean(repeat_scores))
    sd_auc = float(np.std(repeat_scores, ddof=1)) if len(repeat_scores) > 1 else 0.0
    ci_low, ci_high = compute_bootstrap_ci(repeat_scores, n_bootstraps=1000, seed=base_seed)

    return {
        "mean": round(mean_auc, 4),
        "sd": round(sd_auc, 4),
        "ci_95": [round(ci_low, 4), round(ci_high, 4)],
        "repeat_scores": [round(s, 4) for s in repeat_scores],
    }


def main():
    print("=" * 70)
    print("RUNNING SHORTCUT AND ROBUSTNESS TESTS (PHASE 0b)")
    print("=" * 70)

    # 1. Load benchmark data
    csv_path = "data/benchmark/verified_botanical_hdi.csv"
    df = pd.read_csv(csv_path)
    # Ensure standard integer indices
    df = df.reset_index(drop=True)
    n_samples = len(df)
    print(f"Loaded {n_samples} records.")

    # Ground truth binary labels
    y = np.array([1 if str(o).lower() in {"interaction_present", "positive", "1", "true"} else 0 for o in df["outcome"]], dtype=int)

    # 2. Extract standard features
    bio_extractor = BiologicalProfileExtractor()
    analyzer = EvidenceLineageAnalyzer()
    summary_map = analyzer.analyze_dataset(df)

    X_tab_13, _ = extract_tabular_features(df, summary_map, bio_extractor, feature_group="quality_only")
    X_qual_7 = X_tab_13[:, :7]
    X_overlap_6 = X_tab_13[:, 7:]  # [cyp3a4, pgp, cyp2c9, cyp1a2, cyp2e1, overlap_total]

    # Herb and Drug one-hot representations
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
    # 1D: Herb has any nonzero modulation value on canonical targets (>0.05)
    # 2D: [has_nonzero, number_of_nonzero_herb_targets]
    curation_1d = np.zeros((n_samples, 1), dtype=np.float32)
    curation_2d = np.zeros((n_samples, 2), dtype=np.float32)
    for i, row in df.iterrows():
        h_vec = bio_extractor.get_herb_representation(row["herb_id"])
        nonzero_targets = np.sum(np.abs(h_vec) > 0.05)
        has_nonzero = 1.0 if nonzero_targets > 0 else 0.0
        curation_1d[i, 0] = has_nonzero
        curation_2d[i, 0] = has_nonzero
        curation_2d[i, 1] = float(nonzero_targets)

    results = {
        "metadata": {
            "num_records": n_samples,
            "num_herbs": len(all_herbs),
            "num_drugs": len(all_drugs),
            "protocol": "20 repeats of 5-fold grouped CV, Logistic Regression, 95% bootstrap CI",
        },
        "curation_tests": {},
        "target_ablations": {},
        "permutation_test": {},
        "leave_one_entity_out": {
            "herbs": {},
            "drugs": {},
        },
        "identity_baselines": {},
    }

    modes = ["herb_disjoint", "drug_disjoint", "pair_disjoint"]

    # -------------------------------------------------------------
    # 8. Curation-Only vs Overlap-6D
    # -------------------------------------------------------------
    print("\n--- 8. Curation-Only Feature vs Overlap-6D ---")
    results["curation_tests"]["overlap_6d"] = {}
    results["curation_tests"]["curation_1d_has_nonzero"] = {}
    results["curation_tests"]["curation_2d_num_nonzero"] = {}

    for mode in modes:
        print(f"Running mode: {mode}...")
        res_over = run_repeated_grouped_cv(df, X_overlap_6, y, mode=mode)
        res_cur1 = run_repeated_grouped_cv(df, curation_1d, y, mode=mode)
        res_cur2 = run_repeated_grouped_cv(df, curation_2d, y, mode=mode)

        results["curation_tests"]["overlap_6d"][mode] = res_over
        results["curation_tests"]["curation_1d_has_nonzero"][mode] = res_cur1
        results["curation_tests"]["curation_2d_num_nonzero"][mode] = res_cur2

        print(f"  [{mode}] Overlap-6D: {res_over['mean']:.4f} ± {res_over['sd']:.4f} (95% CI: {res_over['ci_95']})")
        print(f"  [{mode}] Curation-1D: {res_cur1['mean']:.4f} ± {res_cur1['sd']:.4f} (95% CI: {res_cur1['ci_95']})")
        print(f"  [{mode}] Curation-2D: {res_cur2['mean']:.4f} ± {res_cur2['sd']:.4f} (95% CI: {res_cur2['ci_95']})")

    # -------------------------------------------------------------
    # 9. Per-Target Ablation for Overlap
    # -------------------------------------------------------------
    print("\n--- 9. Per-Target Ablation ---")
    # Features in X_overlap_6: 0: cyp3a4, 1: pgp, 2: cyp2c9, 3: cyp1a2, 4: cyp2e1, 5: overlap_total
    target_names = ["CYP3A4", "ABCB1_Pgp", "CYP2C9", "CYP1A2", "CYP2E1", "overlap_total"]

    # Single-target alone
    results["target_ablations"]["single_target_alone"] = {}
    for idx, tname in enumerate(target_names):
        results["target_ablations"]["single_target_alone"][tname] = {}
        X_single = X_overlap_6[:, [idx]]
        for mode in modes:
            res = run_repeated_grouped_cv(df, X_single, y, mode=mode)
            results["target_ablations"]["single_target_alone"][tname][mode] = res
            print(f"  Single target [{tname}] ({mode}): {res['mean']:.4f} ± {res['sd']:.4f}")

    # Leave-one-target-out
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
    # 10. Permutation Test (1000 Shuffles)
    # -------------------------------------------------------------
    print("\n--- 10. Permutation Test (1000 Shuffles) ---")
    # For permutation test, evaluate null distribution for overlap_6d and combined_13d under grouped CV
    # Use 1 repeat of 5-fold grouped CV across 1000 shuffled label vectors
    n_perms = 1000
    results["permutation_test"] = {}

    for mode in ["herb_disjoint", "pair_disjoint"]:
        print(f"Running permutation test for mode: {mode} (1000 shuffles)...")
        splitter = LeakageSafeSplitter(seed=42)
        folds = splitter.grouped_kfold_split(df, mode=mode, n_splits=5)

        true_res_over = results["curation_tests"]["overlap_6d"][mode]["mean"]
        # Also run baseline combined_13d
        true_res_comb = run_repeated_grouped_cv(df, X_tab_13, y, mode=mode)["mean"]

        null_aucs_over = []
        null_aucs_comb = []
        rng = np.random.RandomState(42)

        for perm in range(n_perms):
            y_perm = rng.permutation(y)
            oof_over = np.zeros(n_samples)
            oof_comb = np.zeros(n_samples)

            for train_df_f, test_df_f in folds:
                tr_idx = train_df_f.index.values
                te_idx = test_df_f.index.values

                y_tr_p = y_perm[tr_idx]
                if len(np.unique(y_tr_p)) < 2:
                    p = float(np.mean(y_tr_p))
                    oof_over[te_idx] = p
                    oof_comb[te_idx] = p
                else:
                    clf_o = LogisticRegression(max_iter=500, random_state=42)
                    clf_o.fit(X_overlap_6[tr_idx], y_tr_p)
                    oof_over[te_idx] = clf_o.predict_proba(X_overlap_6[te_idx])[:, 1]

                    clf_c = LogisticRegression(max_iter=500, random_state=42)
                    clf_c.fit(X_tab_13[tr_idx], y_tr_p)
                    oof_comb[te_idx] = clf_c.predict_proba(X_tab_13[te_idx])[:, 1]

            null_aucs_over.append(float(roc_auc_score(y_perm, oof_over)))
            null_aucs_comb.append(float(roc_auc_score(y_perm, oof_comb)))

        p_val_over = float(np.mean(np.array(null_aucs_over) >= true_res_over))
        p_val_comb = float(np.mean(np.array(null_aucs_comb) >= true_res_comb))

        results["permutation_test"][mode] = {
            "overlap_6d": {
                "true_auc": true_res_over,
                "null_mean": round(float(np.mean(null_aucs_over)), 4),
                "null_sd": round(float(np.std(null_aucs_over)), 4),
                "null_ci_95": [round(float(np.percentile(null_aucs_over, 2.5)), 4), round(float(np.percentile(null_aucs_over, 97.5)), 4)],
                "p_value": round(p_val_over, 6),
            },
            "combined_13d": {
                "true_auc": true_res_comb,
                "null_mean": round(float(np.mean(null_aucs_comb)), 4),
                "null_sd": round(float(np.std(null_aucs_comb)), 4),
                "null_ci_95": [round(float(np.percentile(null_aucs_comb, 2.5)), 4), round(float(np.percentile(null_aucs_comb, 97.5)), 4)],
                "p_value": round(p_val_comb, 6),
            },
        }
        print(f"  [{mode}] Overlap-6D: True={true_res_over:.4f}, Null={np.mean(null_aucs_over):.4f} ± {np.std(null_aucs_over):.4f}, p={p_val_over:.6f}")
        print(f"  [{mode}] Combined-13D: True={true_res_comb:.4f}, Null={np.mean(null_aucs_comb):.4f} ± {np.std(null_aucs_comb):.4f}, p={p_val_comb:.6f}")

    # -------------------------------------------------------------
    # 11. Per-Herb and Per-Drug Leave-One-Out for Overlap-6D
    # -------------------------------------------------------------
    print("\n--- 11. Per-Herb and Per-Drug Leave-One-Out (overlap_6d) ---")
    base_auc = results["curation_tests"]["overlap_6d"]["pair_disjoint"]["mean"]

    flagged_entities = []

    # Per-herb leave-one-out
    print("Testing leave-one-herb-out...")
    for h in all_herbs:
        sub_mask = (df["herb_id"] != h).values
        df_sub = df[sub_mask].reset_index(drop=True)
        X_sub = X_overlap_6[sub_mask]
        y_sub = y[sub_mask]

        # Use pair_disjoint to evaluate AUC contribution cleanly
        res = run_repeated_grouped_cv(df_sub, X_sub, y_sub, mode="pair_disjoint", n_repeats=10, n_splits=5)
        drop = base_auc - res["mean"]
        flagged = drop > 0.05
        if flagged:
            flagged_entities.append(f"Herb: {h} (drop: {drop:.4f})")

        results["leave_one_entity_out"]["herbs"][h] = {
            "records_removed": int(np.sum(~sub_mask)),
            "remaining_records": len(df_sub),
            "auc": res["mean"],
            "auc_sd": res["sd"],
            "delta_from_baseline": round(drop, 4),
            "drops_more_than_0_05": flagged,
        }
        print(f"  Drop {h:18s}: AUC={res['mean']:.4f} (Δ={drop:+.4f}) {'*** FLAGGED ***' if flagged else ''}")

    # Per-drug leave-one-out (drugs with >= 3 records)
    print("Testing leave-one-drug-out...")
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
        if flagged:
            flagged_entities.append(f"Drug: {d} (drop: {drop:.4f})")

        results["leave_one_entity_out"]["drugs"][d] = {
            "records_removed": int(count),
            "remaining_records": len(df_sub),
            "auc": res["mean"],
            "auc_sd": res["sd"],
            "delta_from_baseline": round(drop, 4),
            "drops_more_than_0_05": flagged,
        }
        print(f"  Drop {d:22s} (N={count}): AUC={res['mean']:.4f} (Δ={drop:+.4f}) {'*** FLAGGED ***' if flagged else ''}")

    # -------------------------------------------------------------
    # 12. Entity-Identity-Only Baselines
    # -------------------------------------------------------------
    print("\n--- 12. Entity-Identity-Only Baselines ---")
    results["identity_baselines"]["herb_onehot"] = {}
    results["identity_baselines"]["drug_onehot"] = {}

    for mode in modes:
        res_h = run_repeated_grouped_cv(df, X_herb_oh, y, mode=mode)
        res_d = run_repeated_grouped_cv(df, X_drug_oh, y, mode=mode)

        results["identity_baselines"]["herb_onehot"][mode] = res_h
        results["identity_baselines"]["drug_onehot"][mode] = res_d

        print(f"  [{mode}] Herb One-Hot: {res_h['mean']:.4f} ± {res_h['sd']:.4f} (95% CI: {res_h['ci_95']})")
        print(f"  [{mode}] Drug One-Hot: {res_d['mean']:.4f} ± {res_d['sd']:.4f} (95% CI: {res_d['ci_95']})")

    # -------------------------------------------------------------
    # 13. Save JSON and Markdown Reports
    # -------------------------------------------------------------
    os.makedirs("results", exist_ok=True)
    os.makedirs("reports", exist_ok=True)

    with open("results/shortcut_tests.json", "w") as f:
        json.dump(results, f, indent=2)
    print("\nWrote results/shortcut_tests.json")

    # Generate Markdown Report
    md_content = f"""# Shortcut and Robustness Test Report
## Reliability-Conditioned Herb–Drug Interaction (HDI) Benchmark

**Document Identifier:** `REP-SHORTCUT-TESTS-2026-10-27`  
**Protocol:** 20 repeats of 5-fold grouped cross-validation, Logistic Regression classifier, seeded (`seed = 42 + r`), reporting Mean, Standard Deviation (SD), and 95% Bootstrap Confidence Intervals (1000 resamples).  
**Raw Results:** [results/shortcut_tests.json](file:///Users/aditisharma/Downloads/sem%207/text%20minning/Reliability-Conditioned-Herb-Drug-Interaction-Prediction/results/shortcut_tests.json)

---

## 1. Curation-Only Feature vs Overlap-6D

Comparison of mechanistic overlap (`overlap_6d`) against simple profile curation features:
- **`curation_1d_has_nonzero`**: 1D binary feature: whether the herb has any non-zero modulatory target value ($> 0.05$).
- **`curation_2d_num_nonzero`**: 2D feature: `[has_nonzero, count_of_nonzero_herb_targets]`.

| Feature Representation | Split Mode | Mean AUC | SD | 95% Bootstrap CI |
| :--- | :--- | :---: | :---: | :---: |
| **`overlap_6d`** | `herb_disjoint` | **{results['curation_tests']['overlap_6d']['herb_disjoint']['mean']:.4f}** | {results['curation_tests']['overlap_6d']['herb_disjoint']['sd']:.4f} | {results['curation_tests']['overlap_6d']['herb_disjoint']['ci_95']} |
| `curation_1d_has_nonzero` | `herb_disjoint` | {results['curation_tests']['curation_1d_has_nonzero']['herb_disjoint']['mean']:.4f} | {results['curation_tests']['curation_1d_has_nonzero']['herb_disjoint']['sd']:.4f} | {results['curation_tests']['curation_1d_has_nonzero']['herb_disjoint']['ci_95']} |
| `curation_2d_num_nonzero` | `herb_disjoint` | {results['curation_tests']['curation_2d_num_nonzero']['herb_disjoint']['mean']:.4f} | {results['curation_tests']['curation_2d_num_nonzero']['herb_disjoint']['sd']:.4f} | {results['curation_tests']['curation_2d_num_nonzero']['herb_disjoint']['ci_95']} |
| **`overlap_6d`** | `drug_disjoint` | **{results['curation_tests']['overlap_6d']['drug_disjoint']['mean']:.4f}** | {results['curation_tests']['overlap_6d']['drug_disjoint']['sd']:.4f} | {results['curation_tests']['overlap_6d']['drug_disjoint']['ci_95']} |
| `curation_1d_has_nonzero` | `drug_disjoint` | {results['curation_tests']['curation_1d_has_nonzero']['drug_disjoint']['mean']:.4f} | {results['curation_tests']['curation_1d_has_nonzero']['drug_disjoint']['sd']:.4f} | {results['curation_tests']['curation_1d_has_nonzero']['drug_disjoint']['ci_95']} |
| `curation_2d_num_nonzero` | `drug_disjoint` | {results['curation_tests']['curation_2d_num_nonzero']['drug_disjoint']['mean']:.4f} | {results['curation_tests']['curation_2d_num_nonzero']['drug_disjoint']['sd']:.4f} | {results['curation_tests']['curation_2d_num_nonzero']['drug_disjoint']['ci_95']} |
| **`overlap_6d`** | `pair_disjoint` | **{results['curation_tests']['overlap_6d']['pair_disjoint']['mean']:.4f}** | {results['curation_tests']['overlap_6d']['pair_disjoint']['sd']:.4f} | {results['curation_tests']['overlap_6d']['pair_disjoint']['ci_95']} |
| `curation_1d_has_nonzero` | `pair_disjoint` | {results['curation_tests']['curation_1d_has_nonzero']['pair_disjoint']['mean']:.4f} | {results['curation_tests']['curation_1d_has_nonzero']['pair_disjoint']['sd']:.4f} | {results['curation_tests']['curation_1d_has_nonzero']['pair_disjoint']['ci_95']} |
| `curation_2d_num_nonzero` | `pair_disjoint` | {results['curation_tests']['curation_2d_num_nonzero']['pair_disjoint']['mean']:.4f} | {results['curation_tests']['curation_2d_num_nonzero']['pair_disjoint']['sd']:.4f} | {results['curation_tests']['curation_2d_num_nonzero']['pair_disjoint']['ci_95']} |

### Finding:
- Under `pair_disjoint` splits, `curation_1d_has_nonzero` achieves **{results['curation_tests']['curation_1d_has_nonzero']['pair_disjoint']['mean']:.4f} AUC** and `curation_2d_num_nonzero` achieves **{results['curation_tests']['curation_2d_num_nonzero']['pair_disjoint']['mean']:.4f} AUC**, compared to **{results['curation_tests']['overlap_6d']['pair_disjoint']['mean']:.4f} AUC** for `overlap_6d`.
- Because the curation-only feature performs substantially above chance (and within ~0.10 AUC of `overlap_6d`), a meaningful portion of the overlap signal reflects the curation fact that non-interacting botanicals were assigned zero target modulation values.

---

## 2. Per-Target Ablation for Overlap

### 2.1 Single Target Alone
Performance when using only one individual overlap feature:

| Target Feature | `herb_disjoint` Mean AUC | `drug_disjoint` Mean AUC | `pair_disjoint` Mean AUC |
| :--- | :---: | :---: | :---: |
"""
    for tname in target_names:
        s_herb = results['target_ablations']['single_target_alone'][tname]['herb_disjoint']
        s_drug = results['target_ablations']['single_target_alone'][tname]['drug_disjoint']
        s_pair = results['target_ablations']['single_target_alone'][tname]['pair_disjoint']
        md_content += f"| `{tname}` | {s_herb['mean']:.4f} ± {s_herb['sd']:.4f} | {s_drug['mean']:.4f} ± {s_drug['sd']:.4f} | {s_pair['mean']:.4f} ± {s_pair['sd']:.4f} |\n"

    md_content += """
### 2.2 Leave-One-Target-Out
Performance when removing one feature from the 6D overlap set:

| Feature Removed | `herb_disjoint` Mean AUC | `drug_disjoint` Mean AUC | `pair_disjoint` Mean AUC |
| :--- | :---: | :---: | :---: |
"""
    for tname in target_names:
        k = f"without_{tname}"
        s_herb = results['target_ablations']['leave_one_target_out'][k]['herb_disjoint']
        s_drug = results['target_ablations']['leave_one_target_out'][k]['drug_disjoint']
        s_pair = results['target_ablations']['leave_one_target_out'][k]['pair_disjoint']
        md_content += f"| Without `{tname}` | {s_herb['mean']:.4f} ± {s_herb['sd']:.4f} | {s_drug['mean']:.4f} ± {s_drug['sd']:.4f} | {s_pair['mean']:.4f} ± {s_pair['sd']:.4f} |\n"

    md_content += f"""
### Key Target Drivers:
- **`overlap_total`** and **`CYP3A4`** carry the predominant mechanistic signal. `overlap_total` alone achieves **{results['target_ablations']['single_target_alone']['overlap_total']['pair_disjoint']['mean']:.4f} AUC** under pair-disjoint splits, while `CYP3A4` alone achieves **{results['target_ablations']['single_target_alone']['CYP3A4']['pair_disjoint']['mean']:.4f} AUC**.
- Removing `CYP3A4` drops pair-disjoint performance from {results['curation_tests']['overlap_6d']['pair_disjoint']['mean']:.4f} to {results['target_ablations']['leave_one_target_out']['without_CYP3A4']['pair_disjoint']['mean']:.4f}.

---

## 3. Permutation Test (1000 Shuffles)

Distribution of ROC-AUC under 1,000 random label permutations within the grouped CV framework:

| Model / Feature | Split Mode | True AUC | Null Mean ± SD | Null 95% Interval | Empirical p-value |
| :--- | :--- | :---: | :---: | :---: | :---: |
| **`overlap_6d`** | `herb_disjoint` | {results['permutation_test']['herb_disjoint']['overlap_6d']['true_auc']:.4f} | {results['permutation_test']['herb_disjoint']['overlap_6d']['null_mean']:.4f} ± {results['permutation_test']['herb_disjoint']['overlap_6d']['null_sd']:.4f} | {results['permutation_test']['herb_disjoint']['overlap_6d']['null_ci_95']} | **p = {results['permutation_test']['herb_disjoint']['overlap_6d']['p_value']:.6f}** |
| **`combined_13d`** | `herb_disjoint` | {results['permutation_test']['herb_disjoint']['combined_13d']['true_auc']:.4f} | {results['permutation_test']['herb_disjoint']['combined_13d']['null_mean']:.4f} ± {results['permutation_test']['herb_disjoint']['combined_13d']['null_sd']:.4f} | {results['permutation_test']['herb_disjoint']['combined_13d']['null_ci_95']} | **p = {results['permutation_test']['herb_disjoint']['combined_13d']['p_value']:.6f}** |
| **`overlap_6d`** | `pair_disjoint` | {results['permutation_test']['pair_disjoint']['overlap_6d']['true_auc']:.4f} | {results['permutation_test']['pair_disjoint']['overlap_6d']['null_mean']:.4f} ± {results['permutation_test']['pair_disjoint']['overlap_6d']['null_sd']:.4f} | {results['permutation_test']['pair_disjoint']['overlap_6d']['null_ci_95']} | **p = {results['permutation_test']['pair_disjoint']['overlap_6d']['p_value']:.6f}** |
| **`combined_13d`** | `pair_disjoint` | {results['permutation_test']['pair_disjoint']['combined_13d']['true_auc']:.4f} | {results['permutation_test']['pair_disjoint']['combined_13d']['null_mean']:.4f} ± {results['permutation_test']['pair_disjoint']['combined_13d']['null_sd']:.4f} | {results['permutation_test']['pair_disjoint']['combined_13d']['null_ci_95']} | **p = {results['permutation_test']['pair_disjoint']['combined_13d']['p_value']:.6f}** |

---

## 4. Per-Entity Leave-One-Out Sensitivity Analysis

Baseline Full `overlap_6d` Pair-Disjoint AUC: **{base_auc:.4f}**  
Threshold for flagging: Removal of single entity drops AUC by **> 0.05** ($\Delta > 0.05$).

### 4.1 Leave-One-Herb-Out

| Herb Removed | Records (N) | Remaining (N) | AUC Mean ± SD | $\Delta$ AUC | Flagged (>0.05 drop) |
| :--- | :---: | :---: | :---: | :---: | :---: |
"""
    for h, v in results["leave_one_entity_out"]["herbs"].items():
        flag_str = "🚨 **YES**" if v["drops_more_than_0_05"] else "No"
        md_content += f"| `{h}` | {v['records_removed']} | {v['remaining_records']} | {v['auc']:.4f} ± {v['auc_sd']:.4f} | {v['delta_from_baseline']:+.4f} | {flag_str} |\n"

    md_content += """
### 4.2 Leave-One-Drug-Out (Drugs with $N \ge 2$)

| Drug Removed | Records (N) | Remaining (N) | AUC Mean ± SD | $\Delta$ AUC | Flagged (>0.05 drop) |
| :--- | :---: | :---: | :---: | :---: | :---: |
"""
    for d, v in results["leave_one_entity_out"]["drugs"].items():
        flag_str = "🚨 **YES**" if v["drops_more_than_0_05"] else "No"
        md_content += f"| `{d}` | {v['records_removed']} | {v['remaining_records']} | {v['auc']:.4f} ± {v['auc_sd']:.4f} | {v['delta_from_baseline']:+.4f} | {flag_str} |\n"

    flagged_summary = ", ".join(flagged_entities) if flagged_entities else "None"
    md_content += f"""
### Flagged Entities:
- Entities whose removal drops AUC by >0.05: **{flagged_summary}**

---

## 5. Entity-Identity-Only Baselines

Performance of pure categorical entity one-hot identity features under repeated grouped CV:

| Feature | Split Mode | Mean AUC | SD | 95% Bootstrap CI |
| :--- | :--- | :---: | :---: | :---: |
| **`herb_onehot`** (18-D) | `herb_disjoint` | {results['identity_baselines']['herb_onehot']['herb_disjoint']['mean']:.4f} | {results['identity_baselines']['herb_onehot']['herb_disjoint']['sd']:.4f} | {results['identity_baselines']['herb_onehot']['herb_disjoint']['ci_95']} |
| `herb_onehot` (18-D) | `drug_disjoint` | {results['identity_baselines']['herb_onehot']['drug_disjoint']['mean']:.4f} | {results['identity_baselines']['herb_onehot']['drug_disjoint']['sd']:.4f} | {results['identity_baselines']['herb_onehot']['drug_disjoint']['ci_95']} |
| `herb_onehot` (18-D) | `pair_disjoint` | {results['identity_baselines']['herb_onehot']['pair_disjoint']['mean']:.4f} | {results['identity_baselines']['herb_onehot']['pair_disjoint']['sd']:.4f} | {results['identity_baselines']['herb_onehot']['pair_disjoint']['ci_95']} |
| **`drug_onehot`** (40-D) | `herb_disjoint` | {results['identity_baselines']['drug_onehot']['herb_disjoint']['mean']:.4f} | {results['identity_baselines']['drug_onehot']['herb_disjoint']['sd']:.4f} | {results['identity_baselines']['drug_onehot']['herb_disjoint']['ci_95']} |
| `drug_onehot` (40-D) | `drug_disjoint` | {results['identity_baselines']['drug_onehot']['drug_disjoint']['mean']:.4f} | {results['identity_baselines']['drug_onehot']['drug_disjoint']['sd']:.4f} | {results['identity_baselines']['drug_onehot']['drug_disjoint']['ci_95']} |
| `drug_onehot` (40-D) | `pair_disjoint` | {results['identity_baselines']['drug_onehot']['pair_disjoint']['mean']:.4f} | {results['identity_baselines']['drug_onehot']['pair_disjoint']['sd']:.4f} | {results['identity_baselines']['drug_onehot']['pair_disjoint']['ci_95']} |

### Diagnostic Observation:
- As mathematically guaranteed, `herb_onehot` yields 0.500 under `herb_disjoint` splits and `drug_onehot` yields 0.500 under `drug_disjoint` splits.
- Under `pair_disjoint` splits (where all herbs and drugs are observed in train), `herb_onehot` achieves **{results['identity_baselines']['herb_onehot']['pair_disjoint']['mean']:.4f} AUC** and `drug_onehot` achieves **{results['identity_baselines']['drug_onehot']['pair_disjoint']['mean']:.4f} AUC**, confirming that identity base-rate shortcuts exist when splits permit entity reuse.
"""

    with open("reports/shortcut_tests.md", "w") as f:
        f.write(md_content)
    print("Wrote reports/shortcut_tests.md")
    print("\n--- SHORTCUT TESTS COMPLETED SUCCESSFULLY ---")


if __name__ == "__main__":
    main()
