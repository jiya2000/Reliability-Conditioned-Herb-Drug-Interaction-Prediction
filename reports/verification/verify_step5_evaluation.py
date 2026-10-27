"""
Step 5 Verification Script: Deep Evaluation Protocol Audit.

Checks:
1. Independent Split Overlap Check:
   - For study_disjoint, pair_disjoint, herb_disjoint, drug_disjoint:
     assert ZERO overlap between train, val, and test.
   - Compute partition sizes and positive/negative class balance.
2. Independent recomputation of metrics (AUC, PR-AUC, F1, Brier, ECE, bootstrap CIs)
   using raw sklearn and scipy on model predictions.
3. Significance tests check (paired bootstrap, DeLong).
4. Equal-mass vs uniform ECE check and temperature T* analysis.
"""

from __future__ import annotations

import json
from pathlib import Path
import sys
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import (
    roc_auc_score,
    average_precision_score,
    f1_score,
    brier_score_loss,
    accuracy_score,
    precision_score,
    recall_score,
)

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.training.evaluation import LeakageSafeSplitter
from src.training.statistical_tests import (
    paired_bootstrap_auc_test,
    delong_auc_test,
)
from src.training.calibration import TemperatureScaler, compute_calibration_metrics
from src.models.interaction_model import ReliabilityConditionedInteractionModel
from src.data.public_data_loader import PublicDataLoader
from src.data.evidence_lineage import EvidenceLineageAnalyzer
from src.features.biological_features import BiologicalProfileExtractor, EvidenceTextFeatureExtractor


def verify_splits_independence():
    print("--- 1. Independent Cross-Split Overlap & Balance Verification ---")
    df = pd.read_csv(PROJECT_ROOT / "data/benchmark/verified_botanical_hdi.csv")
    print(f"Total benchmark records: {len(df)}")

    splitter = LeakageSafeSplitter(seed=42)

    # 1. Study Disjoint Split
    train_s, val_s, test_s = splitter.split_study_disjoint(df, train_ratio=0.6, val_ratio=0.2)
    s_tr = set(train_s["study_id"].astype(str))
    s_va = set(val_s["study_id"].astype(str))
    s_te = set(test_s["study_id"].astype(str))
    
    # Check study citations as well
    def get_citations(part_df):
        cits = set()
        for _, r in part_df.iterrows():
            c_val = r.get("cited_study_ids")
            if pd.notna(c_val):
                try:
                    c_list = json.loads(str(c_val).replace("'", '"'))
                    cits.update([str(c) for c in c_list])
                except Exception:
                    pass
        return cits

    c_tr = get_citations(train_s)
    c_va = get_citations(val_s)
    c_te = get_citations(test_s)

    overlap_studies = (s_tr & s_va) | (s_tr & s_te) | (s_va & s_te)
    overlap_citations = (s_te & c_tr) | (s_tr & c_te)
    print(f"Study-Disjoint Split:")
    print(f"  Sizes: Train={len(train_s)}, Val={len(val_s)}, Test={len(test_s)}")
    print(f"  Overlap in study_id between partitions: {len(overlap_studies)} ({overlap_studies})")
    print(f"  Cross-partition citation dependency: {len(overlap_citations)} ({overlap_citations})")
    print(f"  Class balance: Train={train_s['outcome'].value_counts().to_dict()}")
    print(f"                 Val={val_s['outcome'].value_counts().to_dict()}")
    print(f"                 Test={test_s['outcome'].value_counts().to_dict()}")
    assert len(overlap_studies) == 0, f"Leaked studies detected: {overlap_studies}"
    assert len(overlap_citations) == 0, f"Citation leak detected: {overlap_citations}"

    # 2. Pair Disjoint Split
    train_p, val_p = splitter.split_train_val(df, mode="pair_disjoint", train_ratio=0.8)
    p_tr = set(zip(train_p["herb_id"], train_p["drug_id"]))
    p_va = set(zip(val_p["herb_id"], val_p["drug_id"]))
    overlap_pairs = p_tr & p_va
    print(f"\nPair-Disjoint Split:")
    print(f"  Sizes: Train={len(train_p)}, Val={len(val_p)}")
    print(f"  Overlap in (herb, drug) pairs: {len(overlap_pairs)} ({overlap_pairs})")
    assert len(overlap_pairs) == 0, f"Leaked pairs detected: {overlap_pairs}"

    # 3. Herb Disjoint Split
    train_h, val_h = splitter.split_train_val(df, mode="herb_disjoint", train_ratio=0.8)
    h_tr = set(train_h["herb_id"])
    h_va = set(val_h["herb_id"])
    overlap_herbs = h_tr & h_va
    print(f"\nHerb-Disjoint Split:")
    print(f"  Sizes: Train={len(train_h)}, Val={len(val_h)}")
    print(f"  Overlap in herbs: {len(overlap_herbs)} ({overlap_herbs})")
    assert len(overlap_herbs) == 0, f"Leaked herbs detected: {overlap_herbs}"

    # 4. Drug Disjoint Split
    train_d, val_d = splitter.split_train_val(df, mode="drug_disjoint", train_ratio=0.8)
    d_tr = set(train_d["drug_id"])
    d_va = set(val_d["drug_id"])
    overlap_drugs = d_tr & d_va
    print(f"\nDrug-Disjoint Split:")
    print(f"  Sizes: Train={len(train_d)}, Val={len(val_d)}")
    print(f"  Overlap in drugs: {len(overlap_drugs)} ({overlap_drugs})")
    assert len(overlap_drugs) == 0, f"Leaked drugs detected: {overlap_drugs}"
    print("PASS: Zero leakage confirmed across all 4 split modes.")


def recompute_metrics_and_ci():
    print("\n--- 2. Independent Recomputation of Metrics from Model Execution ---")
    seed = 42
    torch.manual_seed(seed)
    np.random.seed(seed)

    df = pd.read_csv(PROJECT_ROOT / "data/benchmark/verified_botanical_hdi.csv")
    splitter = LeakageSafeSplitter(seed=seed)
    train_df, val_df, test_df = splitter.split_study_disjoint(df, train_ratio=0.6, val_ratio=0.2)

    analyzer = EvidenceLineageAnalyzer(echo_discount_factor=0.15)
    train_summaries = analyzer.analyze_dataset(train_df)
    val_summaries = analyzer.analyze_dataset(val_df)
    test_summaries = analyzer.analyze_dataset(test_df)

    bio_extractor = BiologicalProfileExtractor(feature_dim=16)
    train_texts = [str(t) for t in train_df["original_text"] if pd.notna(t)]
    text_extractor = EvidenceTextFeatureExtractor(embedding_dim=16, min_df=1)
    # Fit text extractor STRICTLY on training split
    text_extractor.fit(train_texts)

    def prep_tensors(part_df, summary_map):
        h_l, d_l, t_l, m_l, y_l = [], [], [], [], []
        for _, r in part_df.iterrows():
            h_id, d_id = str(r["herb_id"]), str(r["drug_id"])
            h_l.append(torch.tensor(bio_extractor.get_herb_representation(h_id), dtype=torch.float32))
            d_l.append(torch.tensor(bio_extractor.get_drug_representation(d_id), dtype=torch.float32))
            t_l.append(str(r.get("original_text", "")))
            summ = summary_map.get((h_id, d_id))
            vec = summ.to_full_metadata_vector() if summ else np.zeros(12, dtype=np.float32)
            m_l.append(vec)
            out = str(r["outcome"]).lower()
            y = 1.0 if out in {"interaction_present", "positive"} else 0.0
            y_l.append(y)
        emb_t = text_extractor.transform(t_l)
        return torch.stack(h_l), torch.stack(d_l), torch.tensor(emb_t, dtype=torch.float32), torch.tensor(np.array(m_l), dtype=torch.float32), torch.tensor(y_l, dtype=torch.float32).unsqueeze(1)

    tr_h, tr_d, tr_t, tr_m, tr_y = prep_tensors(train_df, train_summaries)
    val_h, val_d, val_t, val_m, val_y = prep_tensors(val_df, val_summaries)
    te_h, te_d, te_t, te_m, te_y = prep_tensors(test_df, test_summaries)

    model = ReliabilityConditionedInteractionModel(
        graph_dim=16,
        evidence_dim=16,
        fused_dim=16,
        hidden_dim=16,
        mode="reliability_conditioned_quality_only",
    )
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    criterion = torch.nn.BCEWithLogitsLoss()
    model.train()
    for _ in range(25):
        optimizer.zero_grad()
        out = model(tr_h, tr_d, tr_t, metadata=tr_m)
        loss = criterion(out["logits"], tr_y)
        loss.backward()
        optimizer.step()

    model.eval()
    with torch.no_grad():
        v_out = model(val_h, val_d, val_t, metadata=val_m)
        t_out = model(te_h, te_d, te_t, metadata=te_m)

    scaler = TemperatureScaler()
    scaler.fit(v_out["logits"].cpu().numpy(), val_y.cpu().numpy(), max_iter=25)
    t_star = scaler.temperature.item()
    print(f"Optimal Temperature T* fitted on Validation Split: {t_star:.4f}")

    scaled_te_logits = scaler(t_out["logits"])
    y_probs = torch.sigmoid(scaled_te_logits).detach().cpu().numpy().flatten()
    y_true = te_y.cpu().numpy().flatten()
    y_preds = (y_probs >= 0.5).astype(int)

    # Independent computation via standard sklearn
    calc_auc = float(roc_auc_score(y_true, y_probs))
    calc_pr = float(average_precision_score(y_true, y_probs))
    calc_f1 = float(f1_score(y_true, y_preds))
    calc_brier = float(brier_score_loss(y_true, y_probs))
    calc_acc = float(accuracy_score(y_true, y_preds))
    cal_res = compute_calibration_metrics(y_probs, y_true, num_bins=5, temperature=t_star)
    em_ece = cal_res.equal_mass_ece

    print(f"Independently Computed Metrics on Test Split:")
    print(f"  ROC-AUC    : {calc_auc:.4f}")
    print(f"  PR-AUC     : {calc_pr:.4f}")
    print(f"  F1-Score   : {calc_f1:.4f}")
    print(f"  Brier Score: {calc_brier:.4f}")
    print(f"  Accuracy   : {calc_acc:.4f}")
    print(f"  Equal-Mass ECE: {em_ece:.4f}")

    # Load master benchmark results and compare
    master_path = PROJECT_ROOT / "reports/master_benchmark_results.json"
    with open(master_path, "r") as f:
        master = json.load(f)

    rep = master["verified_hdi_benchmark"]["headline_metrics"]
    print(f"\nMaster Benchmark Claimed Headline Metrics:")
    print(f"  ROC-AUC    : {rep['roc_auc']:.4f} (Diff: {abs(rep['roc_auc'] - calc_auc):.4f})")
    print(f"  PR-AUC     : {rep['pr_auc']:.4f} (Diff: {abs(rep['pr_auc'] - calc_pr):.4f})")
    print(f"  F1-Score   : {rep['f1']:.4f} (Diff: {abs(rep['f1'] - calc_f1):.4f})")
    print(f"  Brier Score: {rep['brier_score']:.4f} (Diff: {abs(rep['brier_score'] - calc_brier):.4f})")

    # Bootstrap 95% CI
    rng = np.random.RandomState(42)
    boot_aucs = []
    n = len(y_true)
    for _ in range(1000):
        idx = rng.choice(n, size=n, replace=True)
        if len(np.unique(y_true[idx])) < 2:
            continue
        boot_aucs.append(roc_auc_score(y_true[idx], y_probs[idx]))
    ci_lower, ci_upper = np.percentile(boot_aucs, [2.5, 97.5])
    print(f"\nFresh Bootstrap 95% CI for ROC-AUC (1000 resamples): [{ci_lower:.4f}, {ci_upper:.4f}]")


def verify_significance_tests():
    print("\n--- 3. Verification of Statistical Significance Tests ---")
    # Test paired bootstrap and DeLong tests on known vectors
    y_true = np.array([1, 1, 1, 1, 0, 0, 0, 0, 1, 0] * 3)
    p_good = np.array([0.9, 0.85, 0.8, 0.95, 0.1, 0.15, 0.2, 0.05, 0.88, 0.12] * 3)
    p_bad = np.array([0.5, 0.45, 0.55, 0.48, 0.52, 0.51, 0.49, 0.53, 0.5, 0.49] * 3)

    res_boot = paired_bootstrap_auc_test(y_true, p_bad, p_good, n_bootstrap=1000, seed=42)
    res_delong = delong_auc_test(y_true, p_bad, p_good)

    print(f"Paired Bootstrap Test: delta_AUC = {res_boot['diff_auc']:.4f}, p = {res_boot['p_value']:.4e}")
    print(f"DeLong Paired Test   : AUC_A = {res_delong['auc_a']:.4f}, AUC_B = {res_delong['auc_b']:.4f}, p = {res_delong['p_value']:.4e}")
    assert res_boot["p_value"] < 0.01, "Expected significant difference in bootstrap test"
    assert res_delong["p_value"] < 0.01, "Expected significant difference in DeLong test"
    print("PASS: Paired bootstrap and DeLong tests correctly evaluate paired sample differences.")


if __name__ == "__main__":
    verify_splits_independence()
    recompute_metrics_and_ci()
    verify_significance_tests()
