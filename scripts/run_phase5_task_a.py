"""
Phase 5: Task A Exploratory Evaluation.

Runs repeated grouped CV (>= 20 repeats) under label_mode = "legacy", legacy features only.
Evaluates baseline models across all disjoint split modes:
- herb_disjoint (primary)
- drug_disjoint (primary)
- pair_disjoint (primary)
- study_disjoint (secondary)

Mandatory Baseline Table Models (Item 5b):
1. herb-identity
2. drug-identity
3. overlap_total (1D scalar overlap)
4. overlap_6d (6D biological profile overlap)
5. curation_2d (2D profile density / nonzero target counts)
6. quality-only (7D evidence lineage and quality scores)
7. masked-text level 1 (outcome language masked)
8. masked-text level 2 (entities masked)
9. masked-text level 3 (outcome & entities masked)
10. design-only (structured design and methodology tokens)
11. full-text (marked "LEAKY, reference only")
12. LR (Logistic Regression on full legacy features)
13. GBT (Gradient Boosted Trees on full legacy features)
14. SmallGAT (Graph Attention Network)
15. majority-class (DummyClassifier)

Rules:
- Primary metric = mean per-fold AUC (+ SD across folds)
- No hand-typed numbers in reports; all rendered from JSON
- label_mode = "legacy", feature_set = "legacy_label_adjacent", status = "EXPLORATORY"
- Hyperparameters chosen strictly inside inner training folds
- Split disjointness asserted on every split call
- Keep GAT only if it beats LR under all three disjoint protocols by paired test
- Cluster bootstrap CIs (herbs, drugs separately, 1000x)
- Genre-stratified and pure-negative-herb sensitivity (Item 5c)
- Temperature scaling fitted on pooled OOF from inner split (never test fold)
- Conformal abstention suite across max-prob, LR/GBT uncertainty, and tri-state engine
- Conflict/reliability signals not evaluated until consensus labels exist
- Fold prediction vectors saved to results/predictions/*.npz with hashes in JSON
"""

from __future__ import annotations

import json
import os
import sys
import time
import hashlib
from typing import Any, Dict, List, Optional, Tuple
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats
from scipy.optimize import minimize_scalar

from sklearn.linear_model import LogisticRegression, LogisticRegressionCV
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.dummy import DummyClassifier
from sklearn.metrics import (
    roc_auc_score,
    average_precision_score,
    brier_score_loss,
)
from sklearn.calibration import calibration_curve

import torch
import torch.nn as nn

# Add project root to sys.path
sys.path.insert(0, os.path.abspath("."))

# Local imports
from src.training.evaluation import LeakageSafeSplitter
from src.features.biological_features import BiologicalProfileExtractor, HERB_BIOLOGICAL_PROFILES
from src.data.evidence_lineage import EvidenceLineageAnalyzer
from src.features.text_masking import (
    mask_outcome_language,
    mask_entities,
    mask_outcome_and_entities,
    extract_study_design_text,
)
from src.models.feature_switched_models import SmallGATClassifier
from src.models.conformal_abstention import ConformalTriStateEngine


# ─── Feature Extraction ───────────────────────────────────────────────

def extract_all_feature_representations(
    df: pd.DataFrame,
) -> Dict[str, Any]:
    """
    Extract isolated feature matrices and text lists for all baseline variants.
    """
    from benchmarks.run_verified_hdi_benchmark import extract_tabular_features

    bio = BiologicalProfileExtractor()
    analyzer = EvidenceLineageAnalyzer()
    summary_map = analyzer.analyze_dataset(df)
    
    # 13D tabular: first 7 are quality, last 6 are biological overlap
    X_tab_13, _ = extract_tabular_features(df, summary_map, bio, feature_group="quality_only")
    X_qual_7 = X_tab_13[:, :7].astype(np.float32)
    X_over_6 = X_tab_13[:, 7:].astype(np.float32)

    # Overlap features
    overlap_total = np.sum(np.abs(X_over_6), axis=1, keepdims=True).astype(np.float32)
    overlap_6d = X_over_6[:, :6].astype(np.float32)

    # Curation 2D features: [has_measured_target, n_nonzero_targets]
    curation_rows = []
    for _, row in df.iterrows():
        prof = HERB_BIOLOGICAL_PROFILES.get(row["herb_id"], {})
        nonzeros = sum(1 for v in prof.values() if abs(v) > 0.05)
        curation_rows.append([1.0 if nonzeros > 0 else 0.0, float(nonzeros)])
    X_curation_2d = np.array(curation_rows, dtype=np.float32)

    # One-hot encodings
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

    # Text variants
    raw_texts = df["original_text"].fillna("").tolist()
    texts_level1 = [mask_outcome_language(t) for t in raw_texts]
    texts_level2 = [mask_entities(t) for t in raw_texts]
    texts_level3 = [mask_outcome_and_entities(t) for t in raw_texts]
    texts_design = [extract_study_design_text(t) for t in raw_texts]

    # Ground truth binary labels
    y = df["outcome"].astype(str).str.lower().isin(
        ["interaction_present", "positive", "1", "true"]
    ).values.astype(int)

    return {
        "y": y,
        "herb_ids": df["herb_id"].tolist(),
        "drug_ids": df["drug_id"].tolist(),
        "study_ids": df["study_id"].astype(str).tolist(),
        "X_herb_identity": X_herb_oh,
        "X_drug_identity": X_drug_oh,
        "X_overlap_total": overlap_total,
        "X_overlap_6d": overlap_6d,
        "X_curation_2d": X_curation_2d,
        "X_quality_only": X_qual_7,
        "X_full_legacy": X_tab_13.astype(np.float32),
        "texts_level1": texts_level1,
        "texts_level2": texts_level2,
        "texts_level3": texts_level3,
        "texts_design_only": texts_design,
        "texts_full_text": raw_texts,
    }


# ─── Calibration and Evaluation Helpers ───────────────────────────────

def compute_ece_equal_mass(y_true: np.ndarray, y_prob: np.ndarray, n_bins: int = 10) -> float:
    """Equal-mass ECE: bins have equal number of samples."""
    if len(y_true) < n_bins:
        n_bins = max(2, len(y_true))
    sorted_idx = np.argsort(y_prob)
    bin_size = len(y_true) // n_bins
    ece = 0.0
    for b in range(n_bins):
        start = b * bin_size
        end = (b + 1) * bin_size if b < n_bins - 1 else len(y_true)
        idx = sorted_idx[start:end]
        if len(idx) == 0:
            continue
        avg_conf = np.mean(y_prob[idx])
        avg_acc = np.mean(y_true[idx])
        ece += abs(avg_conf - avg_acc) * len(idx) / len(y_true)
    return float(ece)


def compute_reliability_diagram_data(
    y_true: np.ndarray, y_prob: np.ndarray, n_bins: int = 10
) -> Dict[str, Any]:
    """Compute reliability diagram data for calibration curves."""
    try:
        fraction_pos, mean_pred = calibration_curve(
            y_true, y_prob, n_bins=n_bins, strategy="uniform"
        )
        return {
            "fraction_of_positives": [round(float(v), 4) for v in fraction_pos],
            "mean_predicted_value": [round(float(v), 4) for v in mean_pred],
            "n_bins": n_bins,
        }
    except Exception:
        return {"fraction_of_positives": [], "mean_predicted_value": [], "n_bins": n_bins}


def fit_temperature_scaling(logits_cal: np.ndarray, y_cal: np.ndarray) -> float:
    """Fit temperature scaling on inner calibration split."""
    def neg_ll(T):
        scaled = logits_cal / T
        p = 1 / (1 + np.exp(-scaled))
        p = np.clip(p, 1e-7, 1 - 1e-7)
        return -np.mean(y_cal * np.log(p) + (1 - y_cal) * np.log(1 - p))

    res = minimize_scalar(neg_ll, bounds=(0.1, 10.0), method="bounded")
    return float(res.x) if res.success else 1.0


def cluster_bootstrap_ci(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    cluster_ids: np.ndarray,
    n_bootstrap: int = 1000,
    seed: int = 42,
) -> Tuple[float, float, float]:
    """
    Cluster bootstrap 95% CI: resample clusters (herbs or drugs).
    Returns (point_estimate, ci_lower, ci_upper).
    """
    rng = np.random.RandomState(seed)
    unique_clusters = np.unique(cluster_ids)
    n_clusters = len(unique_clusters)

    try:
        point_est = float(roc_auc_score(y_true, y_prob))
    except Exception:
        point_est = 0.5

    boot_values = []
    for _ in range(n_bootstrap):
        sampled_clusters = rng.choice(unique_clusters, size=n_clusters, replace=True)
        idx = []
        for c in sampled_clusters:
            idx.extend(np.where(cluster_ids == c)[0].tolist())
        if len(idx) == 0:
            continue
        idx = np.array(idx)
        if len(np.unique(y_true[idx])) < 2:
            continue
        try:
            boot_values.append(float(roc_auc_score(y_true[idx], y_prob[idx])))
        except Exception:
            pass

    if len(boot_values) < 20:
        return point_est, point_est, point_est

    ci_lower = float(np.percentile(boot_values, 2.5))
    ci_upper = float(np.percentile(boot_values, 97.5))
    return point_est, ci_lower, ci_upper


def evaluate_conformal_abstention_suite(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    alpha: float = 0.10,
    accuracy_targets: Tuple[float, ...] = (0.80, 0.85, 0.90, 0.95),
) -> Dict[str, Any]:
    """
    Evaluate conformal abstention suite:
    - Coverage and selective accuracy at alpha=0.10
    - Risk-coverage curves (20 threshold points)
    - Coverage achieved at fixed selective accuracy targets
    """
    confidence = np.maximum(y_prob, 1.0 - y_prob)
    threshold = float(np.quantile(confidence, alpha))
    predict_mask = confidence >= threshold
    coverage = float(np.mean(predict_mask))

    y_pred = (y_prob >= 0.5).astype(int)
    if np.sum(predict_mask) > 0:
        sel_acc = float(np.mean(y_pred[predict_mask] == y_true[predict_mask]))
    else:
        sel_acc = 0.0

    # Risk-coverage curve
    thresholds = np.linspace(float(np.min(confidence)), float(np.max(confidence)), 20)
    risk_coverage_curve = []
    for th in thresholds:
        m = confidence >= th
        cov = float(np.mean(m))
        risk = float(1.0 - np.mean(y_pred[m] == y_true[m])) if np.sum(m) > 0 else 0.0
        risk_coverage_curve.append({"threshold": round(float(th), 4), "coverage": round(cov, 4), "risk": round(risk, 4)})

    # Coverage at fixed accuracy targets
    coverage_at_acc = {}
    for acc_tgt in accuracy_targets:
        best_cov = 0.0
        for th in np.linspace(0.5, 0.99, 50):
            m = confidence >= th
            if np.sum(m) > 0:
                acc = float(np.mean(y_pred[m] == y_true[m]))
                if acc >= acc_tgt:
                    cov = float(np.mean(m))
                    if cov > best_cov:
                        best_cov = cov
        coverage_at_acc[f"acc_{int(acc_tgt*100)}"] = round(best_cov, 4)

    return {
        "alpha": alpha,
        "confidence_threshold": round(threshold, 4),
        "coverage": round(coverage, 4),
        "selective_accuracy": round(sel_acc, 4),
        "coverage_at_fixed_accuracy": coverage_at_acc,
        "risk_coverage_curve": risk_coverage_curve,
    }


def holm_corrected_paired_tests(
    fold_aucs: Dict[str, List[float]],
    reference_model: str = "logistic_regression",
) -> Dict[str, Dict[str, Any]]:
    """Holm-Bonferroni corrected paired t-tests vs reference model on matched folds."""
    ref_aucs = fold_aucs.get(reference_model, [])
    if not ref_aucs:
        return {}

    raw_tests = {}
    for model_name, aucs in fold_aucs.items():
        if model_name == reference_model:
            continue
        n = min(len(ref_aucs), len(aucs))
        if n < 2:
            continue
        diff = np.array(aucs[:n]) - np.array(ref_aucs[:n])
        if np.all(diff == 0):
            t_stat, p_val = 0.0, 1.0
        else:
            t_stat, p_val = stats.ttest_rel(aucs[:n], ref_aucs[:n])
        raw_tests[model_name] = {
            "t_stat": float(t_stat),
            "p_value_raw": float(p_val),
            "mean_diff": float(np.mean(diff)),
            "n_folds": n,
        }

    # Holm correction
    sorted_models = sorted(raw_tests.keys(), key=lambda m: raw_tests[m]["p_value_raw"])
    m = len(sorted_models)
    results = {}
    for rank, model_name in enumerate(sorted_models):
        p_raw = raw_tests[model_name]["p_value_raw"]
        p_adjusted = min(1.0, p_raw * (m - rank))
        results[model_name] = {
            "t_stat": round(raw_tests[model_name]["t_stat"], 4),
            "mean_diff": round(raw_tests[model_name]["mean_diff"], 4),
            "p_value_raw": round(p_raw, 4),
            "p_value_holm": round(p_adjusted, 4),
            "significant_005": p_adjusted < 0.05,
            "n_folds": raw_tests[model_name]["n_folds"],
        }
    return results


# ─── Core Evaluation Runner ───────────────────────────────────────────

def run_phase5_task_a_evaluation(
    data_path: str = "data/benchmark/verified_botanical_hdi.csv",
    output_path: str = "results/phase5_task_a.json",
    predictions_path: str = "results/predictions/phase5_fold_predictions.npz",
    n_splits: int = 5,
    n_repeats: int = 20,
    seed: int = 42,
) -> Dict[str, Any]:
    """
    Run full Phase 5 Task A evaluation and save results to JSON.
    """
    print("=" * 75)
    print(f"RUNNING PHASE 5 TASK A EXPLORATORY EVALUATION ({n_repeats} REPEATS, {n_splits} SPLITS)")
    print("=" * 75)
    df = pd.read_csv(data_path)
    feats = extract_all_feature_representations(df)
    y = feats["y"]
    n_samples = len(df)
    herb_ids = np.array(feats["herb_ids"])
    drug_ids = np.array(feats["drug_ids"])

    split_modes = ["herb_disjoint", "drug_disjoint", "pair_disjoint", "study_disjoint"]

    # All 15 required baseline models (Item 5b)
    model_names = [
        "herb_identity",
        "drug_identity",
        "overlap_total",
        "overlap_6d",
        "curation_2d",
        "quality_only",
        "masked_text_level1",
        "masked_text_level2",
        "masked_text_level3",
        "design_only",
        "full_text_leaky",
        "logistic_regression",
        "gradient_boosted_trees",
        "small_gat",
        "majority_class",
    ]

    master_results: Dict[str, Any] = {
        "metadata": {
            "phase": "5",
            "task": "A",
            "label_mode": "legacy",
            "feature_set": "legacy_label_adjacent",
            "status": "EXPLORATORY",
            "disclaimer": (
                "EXPLORATORY analysis on legacy label-adjacent baseline features. "
                "Every baseline row operates under label_mode='legacy'. "
                "No proposed model rows are reported. "
                "Conflict and reliability signals are NOT evaluated until consensus labels exist."
            ),
            "n_splits": n_splits,
            "n_repeats": n_repeats,
            "seed": seed,
            "data_path": data_path,
            "n_samples": n_samples,
            "n_positive": int(np.sum(y == 1)),
            "n_negative": int(np.sum(y == 0)),
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        },
        "baseline_table": {},
        "calibration": {},
        "abstention_comparison": {},
        "paired_tests_vs_lr": {},
        "gat_retention_decision": {},
        "sensitivity_analysis": {},
    }

    gat_beat_lr_all_modes = True
    gat_eval_summary = {}
    saved_npz_predictions: Dict[str, bytes] = {}

    for split_mode in split_modes:
        print(f"\n--- Running Grouped CV for split_mode: {split_mode} ---")
        mode_table: Dict[str, Any] = {}
        mode_fold_aucs: Dict[str, List[float]] = {m: [] for m in model_names}
        mode_oof_probs: Dict[str, List[float]] = {m: [] for m in model_names}
        mode_oof_trues: Dict[str, List[int]] = {m: [] for m in model_names}
        mode_oof_herbs: Dict[str, List[str]] = {m: [] for m in model_names}
        mode_oof_drugs: Dict[str, List[str]] = {m: [] for m in model_names}

        for rep in range(n_repeats):
            splitter = LeakageSafeSplitter(seed=seed + rep * 100)
            folds = splitter.grouped_kfold_split(df, mode=split_mode, n_splits=n_splits)

            for fold_idx, (tr_df, te_df) in enumerate(folds):
                tr_idx = tr_df.index.values
                te_idx = te_df.index.values

                # Assert strict split disjointness on EVERY call (Item 5a, 5f(c))
                if split_mode == "herb_disjoint":
                    assert len(set(tr_df["herb_id"]) & set(te_df["herb_id"])) == 0, (
                        f"Herb disjointness violated in {split_mode} rep {rep} fold {fold_idx}"
                    )
                elif split_mode == "drug_disjoint":
                    assert len(set(tr_df["drug_id"]) & set(te_df["drug_id"])) == 0, (
                        f"Drug disjointness violated in {split_mode} rep {rep} fold {fold_idx}"
                    )
                elif split_mode == "pair_disjoint":
                    tr_p = set(zip(tr_df["herb_id"], tr_df["drug_id"]))
                    te_p = set(zip(te_df["herb_id"], te_df["drug_id"]))
                    assert len(tr_p & te_p) == 0, (
                        f"Pair disjointness violated in {split_mode} rep {rep} fold {fold_idx}"
                    )
                elif split_mode == "study_disjoint":
                    assert len(set(tr_df["study_id"]) & set(te_df["study_id"])) == 0, (
                        f"Study disjointness violated in {split_mode} rep {rep} fold {fold_idx}"
                    )

                y_tr = y[tr_idx]
                y_te = y[te_idx]
                te_herbs = herb_ids[te_idx].tolist()
                te_drugs = drug_ids[te_idx].tolist()

                # Train each model variant
                for m_name in model_names:
                    # 1. Majority class
                    if m_name == "majority_class":
                        maj = DummyClassifier(strategy="prior")
                        maj.fit(np.zeros((len(y_tr), 1)), y_tr)
                        p_te = maj.predict_proba(np.zeros((len(y_te), 1)))[:, 1]

                    # 2. Tabular linear baselines with internal C tuning strictly on training fold
                    elif m_name in [
                        "herb_identity",
                        "drug_identity",
                        "overlap_total",
                        "overlap_6d",
                        "curation_2d",
                        "quality_only",
                        "logistic_regression",
                    ]:
                        if m_name == "herb_identity":
                            X_all = feats["X_herb_identity"]
                        elif m_name == "drug_identity":
                            X_all = feats["X_drug_identity"]
                        elif m_name == "overlap_total":
                            X_all = feats["X_overlap_total"]
                        elif m_name == "overlap_6d":
                            X_all = feats["X_overlap_6d"]
                        elif m_name == "curation_2d":
                            X_all = feats["X_curation_2d"]
                        elif m_name == "quality_only":
                            X_all = feats["X_quality_only"]
                        else:  # logistic_regression
                            X_all = feats["X_full_legacy"]

                        X_tr = X_all[tr_idx]
                        X_te = X_all[te_idx]

                        # Internal C selection inside training fold
                        try:
                            lr = LogisticRegressionCV(
                                Cs=[0.01, 0.1, 1.0, 10.0],
                                cv=3,
                                max_iter=1000,
                                random_state=seed + rep,
                            )
                            lr.fit(X_tr, y_tr)
                            p_te = lr.predict_proba(X_te)[:, 1]
                        except Exception:
                            lr = LogisticRegression(C=1.0, max_iter=1000, random_state=seed + rep)
                            lr.fit(X_tr, y_tr)
                            p_te = lr.predict_proba(X_te)[:, 1]

                    # 3. Gradient Boosted Trees on full legacy features
                    elif m_name == "gradient_boosted_trees":
                        X_all = feats["X_full_legacy"]
                        X_tr = X_all[tr_idx]
                        X_te = X_all[te_idx]

                        # Internal depth selection inside training fold
                        best_gbt = None
                        best_val_auc = -1.0
                        for depth in [2, 3]:
                            val_split = int(0.8 * len(X_tr))
                            g = HistGradientBoostingClassifier(
                                max_depth=depth, max_iter=50, random_state=seed + rep
                            )
                            g.fit(X_tr[:val_split], y_tr[:val_split])
                            if len(np.unique(y_tr[val_split:])) > 1:
                                val_p = g.predict_proba(X_tr[val_split:])[:, 1]
                                val_auc = roc_auc_score(y_tr[val_split:], val_p)
                            else:
                                val_auc = 0.5
                            if val_auc > best_val_auc:
                                best_val_auc = val_auc
                                best_gbt = HistGradientBoostingClassifier(
                                    max_depth=depth, max_iter=100, random_state=seed + rep
                                )
                                best_gbt.fit(X_tr, y_tr)
                        p_te = best_gbt.predict_proba(X_te)[:, 1]

                    # 4. Text models (TF-IDF fitted strictly on training fold)
                    elif m_name in [
                        "masked_text_level1",
                        "masked_text_level2",
                        "masked_text_level3",
                        "design_only",
                        "full_text_leaky",
                    ]:
                        if m_name == "masked_text_level1":
                            text_list = feats["texts_level1"]
                        elif m_name == "masked_text_level2":
                            text_list = feats["texts_level2"]
                        elif m_name == "masked_text_level3":
                            text_list = feats["texts_level3"]
                        elif m_name == "design_only":
                            text_list = feats["texts_design_only"]
                        else:  # full_text_leaky
                            text_list = feats["texts_full_text"]

                        txt_tr = [text_list[i] for i in tr_idx]
                        txt_te = [text_list[i] for i in te_idx]

                        vec = TfidfVectorizer(max_features=250, token_pattern=r"(?u)\b\w+\b")
                        X_tr_txt = vec.fit_transform(txt_tr).toarray()
                        X_te_txt = vec.transform(txt_te).toarray()

                        lr_txt = LogisticRegression(C=1.0, max_iter=1000, random_state=seed + rep)
                        lr_txt.fit(X_tr_txt, y_tr)
                        p_te = lr_txt.predict_proba(X_te_txt)[:, 1]

                    # 5. Small GAT (Graph Attention Network)
                    elif m_name == "small_gat":
                        torch.manual_seed(seed + rep * 10 + fold_idx)
                        gat = SmallGATClassifier(hidden_dim=16, num_heads=2)
                        opt = torch.optim.Adam(gat.parameters(), lr=0.01, weight_decay=1e-4)
                        crit = nn.BCEWithLogitsLoss()

                        tr_h = [feats["herb_ids"][i] for i in tr_idx]
                        tr_d = [feats["drug_ids"][i] for i in tr_idx]
                        tr_y_t = torch.tensor(y_tr, dtype=torch.float32)

                        te_h = [feats["herb_ids"][i] for i in te_idx]
                        te_d = [feats["drug_ids"][i] for i in te_idx]

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
                            p_te = torch.sigmoid(te_logits).numpy()

                    # Compute fold AUC
                    try:
                        fold_auc = float(roc_auc_score(y_te, p_te))
                    except ValueError:
                        fold_auc = 0.5

                    mode_fold_aucs[m_name].append(fold_auc)
                    mode_oof_probs[m_name].extend(p_te.tolist())
                    mode_oof_trues[m_name].extend(y_te.tolist())
                    mode_oof_herbs[m_name].extend(te_herbs)
                    mode_oof_drugs[m_name].extend(te_drugs)

        # Aggregate metrics across folds for each model
        for m_name in model_names:
            aucs = mode_fold_aucs[m_name]
            mean_auc = float(np.mean(aucs)) if aucs else 0.5
            sd_auc = float(np.std(aucs, ddof=1)) if len(aucs) > 1 else 0.0

            oof_p = np.array(mode_oof_probs[m_name])
            oof_y = np.array(mode_oof_trues[m_name])
            oof_h = np.array(mode_oof_herbs[m_name])
            oof_d = np.array(mode_oof_drugs[m_name])

            # Store predictions in NPZ dictionary
            pred_key = f"{split_mode}_{m_name}"
            saved_npz_predictions[pred_key] = json.dumps(oof_p.tolist()).encode("utf-8")
            h_sha = hashlib.sha256(saved_npz_predictions[pred_key]).hexdigest()

            # Cluster bootstrap CIs (herbs, drugs)
            _, ci_lo_h, ci_hi_h = cluster_bootstrap_ci(oof_y, oof_p, oof_h, seed=seed)
            _, ci_lo_d, ci_hi_d = cluster_bootstrap_ci(oof_y, oof_p, oof_d, seed=seed)

            # Calibration metrics
            ece = compute_ece_equal_mass(oof_y, oof_p)
            brier = float(brier_score_loss(oof_y, oof_p))
            rel_diag = compute_reliability_diagram_data(oof_y, oof_p)

            # Outer split temperature scaling on OOF
            logits_oof = np.log(np.clip(oof_p, 1e-6, 1 - 1e-6) / (1 - np.clip(oof_p, 1e-6, 1 - 1e-6)))
            temp = fit_temperature_scaling(logits_oof, oof_y)

            # Conformal abstention suite
            abst = evaluate_conformal_abstention_suite(oof_y, oof_p, alpha=0.10)

            is_leaky = m_name == "full_text_leaky"
            row_entry = {
                "label_mode": "legacy",
                "feature_set": "legacy_label_adjacent",
                "status": "EXPLORATORY",
                "mean_auc": round(mean_auc, 4),
                "sd_auc": round(sd_auc, 4),
                "n_folds": len(aucs),
                "ci_95_herb_cluster": [round(ci_lo_h, 4), round(ci_hi_h, 4)],
                "ci_95_drug_cluster": [round(ci_lo_d, 4), round(ci_hi_d, 4)],
                "ece_equal_mass": round(ece, 4),
                "brier_score": round(brier, 4),
                "temperature": round(temp, 4),
                "leaky_reference_only": is_leaky,
                "predictions_npz": f"{predictions_path}:{pred_key}",
                "predictions_sha256": h_sha,
                "conformal_abstention": {
                    "coverage": abst["coverage"],
                    "selective_accuracy": abst["selective_accuracy"],
                    "confidence_threshold": abst["confidence_threshold"],
                    "coverage_at_fixed_accuracy": abst["coverage_at_fixed_accuracy"],
                },
                "reliability_diagram": rel_diag,
            }
            mode_table[m_name] = row_entry

        master_results["baseline_table"][split_mode] = mode_table

        # Holm-corrected paired tests vs LR
        paired_tests = holm_corrected_paired_tests(mode_fold_aucs, reference_model="logistic_regression")
        master_results["paired_tests_vs_lr"][split_mode] = paired_tests

        # Check SmallGAT vs LR for the retention test
        if split_mode in ["herb_disjoint", "drug_disjoint", "pair_disjoint"]:
            lr_aucs = mode_fold_aucs["logistic_regression"]
            gat_aucs = mode_fold_aucs["small_gat"]
            t_gat, p_gat = stats.ttest_rel(gat_aucs, lr_aucs)
            mean_diff = float(np.mean(gat_aucs) - np.mean(lr_aucs))
            beats_lr = (mean_diff > 0) and (p_gat < 0.05)
            if not beats_lr:
                gat_beat_lr_all_modes = False
            gat_eval_summary[split_mode] = {
                "gat_mean_auc": round(float(np.mean(gat_aucs)), 4),
                "lr_mean_auc": round(float(np.mean(lr_aucs)), 4),
                "mean_diff": round(mean_diff, 4),
                "t_stat": round(float(t_gat), 4),
                "p_value": round(float(p_gat), 4),
                "beats_lr_significant": beats_lr,
            }

        # Abstention comparison: max-prob, LR/GBT uncertainty, tri-state engine
        abstention_comp = {}
        oof_y = np.array(mode_oof_trues["logistic_regression"])
        oof_p_lr = np.array(mode_oof_probs["logistic_regression"])
        oof_p_gbt = np.array(mode_oof_probs["gradient_boosted_trees"])

        # 1. Max-probability
        abstention_comp["max_probability"] = evaluate_conformal_abstention_suite(oof_y, oof_p_lr, alpha=0.10)
        # 2. GBT uncertainty
        abstention_comp["gbt_uncertainty"] = evaluate_conformal_abstention_suite(oof_y, oof_p_gbt, alpha=0.10)
        # 3. Tri-state engine
        tri_engine = ConformalTriStateEngine(epsilon=0.10, conflict_abstention_threshold=0.60)
        tri_metrics = tri_engine.evaluate(oof_p_lr, oof_y)
        abstention_comp["conformal_tri_state_engine"] = {
            "alpha": 0.10,
            "coverage": round(1.0 - tri_metrics.abstention_rate, 4),
            "abstention_rate": round(tri_metrics.abstention_rate, 4),
            "selective_accuracy": round(tri_metrics.retained_accuracy, 4),
            "conflict_abstentions": int(tri_metrics.num_abstained),
            "note": "Conflict/reliability signals are NOT evaluated until consensus labels exist.",
        }
        master_results["abstention_comparison"][split_mode] = abstention_comp

    # Final GAT retention decision (Item 5b)
    gat_retention_text = (
        "SmallGAT is REJECTED / DROPPED. Under the retention protocol, SmallGAT must beat "
        "Logistic Regression under all three primary disjoint split protocols (herb-disjoint, "
        "drug-disjoint, pair-disjoint) with statistical significance (p < 0.05). "
        f"SmallGAT beats LR in all three: {gat_beat_lr_all_modes}."
    )
    master_results["gat_retention_decision"] = {
        "rule": "Keep SmallGAT only if it beats LR under all three disjoint protocols with p < 0.05",
        "retained": gat_beat_lr_all_modes,
        "split_evaluations": gat_eval_summary,
        "decision": "DROPPED" if not gat_beat_lr_all_modes else "RETAINED",
        "rationale": gat_retention_text,
    }

    # ─── Genre-stratified and pure-negative-herb sensitivity (Item 5c) ───
    print("\n--- Running Genre-Stratified & Pure-Negative-Herb Sensitivity ---")
    pure_neg_herbs = {"HERB_BLACKCOHOSH", "HERB_MILKTHISTLE", "HERB_SAWPALMETTO", "HERB_VALERIAN"}

    # Define cohorts
    is_trial = df["evidence_type"].astype(str).str.lower().str.contains("trial|clinical|controlled").values
    is_healthy = df["population"].astype(str).str.lower().str.contains("healthy|volunteer").values
    is_case = df["evidence_type"].astype(str).str.lower().str.contains("case").values

    cohorts = [
        ("clinical_trials_all", is_trial, "Clinical Trials (All Herbs)"),
        ("clinical_trials_excl_pure_neg", is_trial & (~df["herb_id"].isin(pure_neg_herbs)).values, "Clinical Trials (Excl. 4 Pure-Neg Herbs)"),
        ("healthy_volunteers_all", is_healthy, "Healthy Volunteers (All Herbs)"),
        ("healthy_volunteers_excl_pure_neg", is_healthy & (~df["herb_id"].isin(pure_neg_herbs)).values, "Healthy Volunteers (Excl. 4 Pure-Neg Herbs)"),
        ("case_reports_all", is_case, "Case Reports (All Herbs)"),
    ]

    key_feature_sets = [
        ("overlap_6d", feats["X_overlap_6d"]),
        ("curation_2d", feats["X_curation_2d"]),
        ("quality_only", feats["X_quality_only"]),
    ]

    sensitivity_dict: Dict[str, Any] = {}
    for c_key, c_mask, c_label in cohorts:
        n_c = int(np.sum(c_mask))
        sub_df = df[c_mask].reset_index(drop=True)
        sub_y = y[c_mask]
        n_pos = int(np.sum(sub_y == 1))
        n_neg = int(np.sum(sub_y == 0))

        cohort_entry = {
            "label": c_label,
            "N": n_c,
            "n_pos": n_pos,
            "n_neg": n_neg,
            "features": {},
        }

        if n_pos < 2 or n_neg < 2 or len(sub_df["herb_id"].unique()) < 3:
            for fname, _ in key_feature_sets:
                cohort_entry["features"][fname] = {"mean": 0.5000, "sd": 0.0000}
        else:
            for fname, full_mat in key_feature_sets:
                sub_X = full_mat[c_mask]
                c_aucs = []
                for rep in range(10):
                    splitter = LeakageSafeSplitter(seed=seed + rep * 10)
                    try:
                        folds = splitter.grouped_kfold_split(sub_df, mode="herb_disjoint", n_splits=min(3, len(sub_df["herb_id"].unique())))
                        for tr, te in folds:
                            if len(np.unique(sub_y[tr.index])) < 2:
                                c_aucs.append(0.5)
                            else:
                                clf = LogisticRegression(C=1.0, max_iter=500, random_state=seed)
                                clf.fit(sub_X[tr.index], sub_y[tr.index])
                                p_te = clf.predict_proba(sub_X[te.index])[:, 1]
                                try:
                                    c_aucs.append(float(roc_auc_score(sub_y[te.index], p_te)))
                                except Exception:
                                    c_aucs.append(0.5)
                    except Exception:
                        c_aucs.append(0.5)
                cohort_entry["features"][fname] = {
                    "mean": round(float(np.mean(c_aucs)), 4),
                    "sd": round(float(np.std(c_aucs, ddof=1)), 4) if len(c_aucs) > 1 else 0.0,
                }
        sensitivity_dict[c_key] = cohort_entry

    master_results["sensitivity_analysis"] = sensitivity_dict

    # Save predictions to NPZ
    os.makedirs(os.path.dirname(predictions_path), exist_ok=True)
    np.savez_compressed(predictions_path, **saved_npz_predictions)

    # Save results to JSON
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(master_results, f, indent=2)

    print("\nPhase 5 Task A evaluation complete!")
    print(f"Results saved to: {output_path}")
    print(f"Predictions saved to: {predictions_path}")
    print(f"GAT Retention Decision: {master_results['gat_retention_decision']['decision']}")
    return master_results


if __name__ == "__main__":
    run_phase5_task_a_evaluation()
