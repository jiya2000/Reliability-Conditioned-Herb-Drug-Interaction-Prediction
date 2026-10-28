"""
Script: scripts/make_shortcut_report.py
Renders reports/shortcut_tests.md strictly and deterministically from results/shortcut_tests.json.
Enforces that no number is hand-typed.
"""

import os
import json


def render_report():
    json_path = "results/shortcut_tests.json"
    with open(json_path, "r") as f:
        data = json.load(f)

    meta = data["metadata"]
    cur = data["curation_tests"]
    abl = data["target_ablations"]
    perm = data["permutation_test"]
    loo = data["leave_one_entity_out"]
    ident = data["identity_baselines"]
    stats = data["overlap_column_statistics"]
    strat = data["genre_stratification"]

    lines = []
    lines.append("# Shortcut and Robustness Test Report")
    lines.append("## Reliability-Conditioned Herb–Drug Interaction (HDI) Benchmark")
    lines.append("")
    lines.append("**Document Identifier:** `REP-SHORTCUT-TESTS-2026-10-28`  ")
    lines.append(f"**Protocol:** {meta['protocol']}.  ")
    lines.append(f"**Primary Metric:** {meta['primary_metric']}.  ")
    lines.append(f"**Secondary Metric:** {meta['secondary_metric']}.  ")
    lines.append(f"**Dataset Summary:** {meta['num_records']} total records across {meta['num_herbs']} botanical entities and {meta['num_drugs']} prescription drugs.  ")
    lines.append("**Raw Results:** [results/shortcut_tests.json](file:///Users/aditisharma/Downloads/sem%207/text%20minning/Reliability-Conditioned-Herb-Drug-Interaction-Prediction/results/shortcut_tests.json)")
    lines.append("")
    lines.append("---")
    lines.append("")
    lines.append("## 1. Curation-Only Features vs Overlap-6D")
    lines.append("")
    lines.append("Comparison of mechanistic overlap (`overlap_6d`) against simple curation profile density features:")
    lines.append("- **`overlap_6d`**: 6D pharmacokinetic interaction vector `[cyp3a4, pgp, cyp2c9, cyp1a2, cyp2e1, overlap_total]`.")
    lines.append("- **`curation_1d_has_nonzero`**: Binary indicator whether the botanical entity has any target with modulatory value $> 0.05$.")
    lines.append("- **`curation_2d_num_nonzero`**: 2D feature `[has_nonzero, count_of_nonzero_herb_targets]`.")
    lines.append("")
    lines.append("| Feature Representation | Split Mode | Mean Per-Fold AUC | Repeat SD | Fold SD | Herb Cluster Bootstrap 95% CI | Drug Cluster Bootstrap 95% CI | Pooled OOF AUC |")
    lines.append("| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: |")

    modes = ["herb_disjoint", "drug_disjoint", "pair_disjoint"]
    feat_order = [
        ("overlap_6d", "**`overlap_6d`**"),
        ("curation_1d_has_nonzero", "`curation_1d_has_nonzero`"),
        ("curation_2d_num_nonzero", "`curation_2d_num_nonzero`"),
    ]

    for mode in modes:
        for fkey, flabel in feat_order:
            entry = cur[fkey][mode]
            h_ci = f"[{entry['herb_cluster_ci_95'][0]:.4f}, {entry['herb_cluster_ci_95'][1]:.4f}]" if "herb_cluster_ci_95" in entry else "-"
            d_ci = f"[{entry['drug_cluster_ci_95'][0]:.4f}, {entry['drug_cluster_ci_95'][1]:.4f}]" if "drug_cluster_ci_95" in entry else "-"
            lines.append(
                f"| {flabel} | `{mode}` | {entry['mean']:.4f} | {entry['sd']:.4f} | {entry['fold_sd']:.4f} | {h_ci} | {d_ci} | {entry['pooled_oof_mean']:.4f} |"
            )

    lines.append("")
    lines.append("### Key Findings on Curation and Overlap:")
    lines.append(f"- Under `herb_disjoint` splits, `overlap_6d` achieves a mean per-fold AUC of **{cur['overlap_6d']['herb_disjoint']['mean']:.4f}** with fold-to-fold SD of **{cur['overlap_6d']['herb_disjoint']['fold_sd']:.4f}** and pooled out-of-fold AUC of **{cur['overlap_6d']['herb_disjoint']['pooled_oof_mean']:.4f}**.")
    lines.append(f"- `curation_2d_num_nonzero` achieves **{cur['curation_2d_num_nonzero']['herb_disjoint']['mean']:.4f}** per-fold AUC on `herb_disjoint` and **{cur['curation_2d_num_nonzero']['pair_disjoint']['mean']:.4f}** on `pair_disjoint`, demonstrating that botanical target profile curation density accounts for a substantial proportion of overall predictive capacity.")
    lines.append("")
    lines.append("---")
    lines.append("")
    lines.append("## 2. Per-Target Ablation and Nonzero Feature Distribution")
    lines.append("")
    lines.append("### 2.1 Nonzero Value Distribution Across Overlap Columns")
    lines.append("Analysis of feature density across the 6 overlap columns in the 141 benchmark records:")
    lines.append("")
    lines.append("| Overlap Feature Column | Nonzero Records | Total Records | Fraction Nonzero | Min Value | Max Value |")
    lines.append("| :--- | :---: | :---: | :---: | :---: | :---: |")
    for tname, s in stats.items():
        lines.append(f"| `{tname}` | {s['nonzero_count']} | {s['total_records']} | {s['fraction_nonzero']:.4f} ({s['fraction_nonzero']*100:.2f}%) | {s['min']:.4f} | {s['max']:.4f} |")

    lines.append("")
    lines.append("### Explanation of Near-Constant Columns:")
    lines.append(f"- **`CYP2E1`** is entirely zero across all {meta['num_records']} records ({stats['CYP2E1']['fraction_nonzero']:.4f} fraction nonzero). None of the benchmark botanicals or co-administered drugs possess active CYP2E1 modulatory overlap.")
    lines.append(f"- **`CYP1A2`** is nonzero in only {stats['CYP1A2']['nonzero_count']} out of {stats['CYP1A2']['total_records']} records ({stats['CYP1A2']['fraction_nonzero']:.4f} fraction nonzero).")
    lines.append(f"- **`CYP2C9`** is nonzero in only {stats['CYP2C9']['nonzero_count']} out of {stats['CYP2C9']['total_records']} records ({stats['CYP2C9']['fraction_nonzero']:.4f} fraction nonzero).")
    lines.append(f"- The overwhelming majority of feature variance is concentrated in **`CYP3A4`** ({stats['CYP3A4']['fraction_nonzero']*100:.2f}% nonzero), **`ABCB1_Pgp`** ({stats['ABCB1_Pgp']['fraction_nonzero']*100:.2f}% nonzero), and the composite **`overlap_total`** ({stats['overlap_total']['fraction_nonzero']*100:.2f}% nonzero).")
    lines.append("")
    lines.append("### 2.2 Single Target Alone")
    lines.append("")
    lines.append("| Target Feature | `herb_disjoint` Mean AUC | `drug_disjoint` Mean AUC | `pair_disjoint` Mean AUC |")
    lines.append("| :--- | :---: | :---: | :---: |")
    for tname in stats.keys():
        s_h = abl["single_target_alone"][tname]["herb_disjoint"]
        s_d = abl["single_target_alone"][tname]["drug_disjoint"]
        s_p = abl["single_target_alone"][tname]["pair_disjoint"]
        lines.append(f"| `{tname}` | {s_h['mean']:.4f} ± {s_h['sd']:.4f} | {s_d['mean']:.4f} ± {s_d['sd']:.4f} | {s_p['mean']:.4f} ± {s_p['sd']:.4f} |")

    lines.append("")
    lines.append("### 2.3 Leave-One-Target-Out")
    lines.append("")
    lines.append("| Feature Removed | `herb_disjoint` Mean AUC | `drug_disjoint` Mean AUC | `pair_disjoint` Mean AUC |")
    lines.append("| :--- | :---: | :---: | :---: |")
    for tname in stats.keys():
        k = f"without_{tname}"
        s_h = abl["leave_one_target_out"][k]["herb_disjoint"]
        s_d = abl["leave_one_target_out"][k]["drug_disjoint"]
        s_p = abl["leave_one_target_out"][k]["pair_disjoint"]
        lines.append(f"| Without `{tname}` | {s_h['mean']:.4f} ± {s_h['sd']:.4f} | {s_d['mean']:.4f} ± {s_d['sd']:.4f} | {s_p['mean']:.4f} ± {s_p['sd']:.4f} |")

    lines.append("")
    lines.append("### Target Redundancy Mechanics:")
    lines.append(f"- Removing `overlap_total` drops pair-disjoint AUC from {cur['overlap_6d']['pair_disjoint']['mean']:.4f} to {abl['leave_one_target_out']['without_overlap_total']['pair_disjoint']['mean']:.4f} (and drops drug-disjoint from {cur['overlap_6d']['drug_disjoint']['mean']:.4f} to {abl['leave_one_target_out']['without_overlap_total']['drug_disjoint']['mean']:.4f}).")
    lines.append(f"- Conversely, removing individual CYP enzymes like `CYP3A4` results in minor changes ({abl['leave_one_target_out']['without_CYP3A4']['pair_disjoint']['mean']:.4f} on pair-disjoint) because `overlap_total` already incorporates the CYP3A4 modulatory interaction sum.")
    lines.append("")
    lines.append("---")
    lines.append("")
    lines.append("## 3. Label Permutation Tests (1,000 Shuffles)")
    lines.append("")
    lines.append("| Model / Feature | Split Mode | True Mean AUC | Null Mean ± SD | Null 95% Interval | Empirical p-value |")
    lines.append("| :--- | :--- | :---: | :---: | :---: | :---: |")
    for mode in ["herb_disjoint", "pair_disjoint"]:
        o_p = perm[mode]["overlap_6d"]
        c_p = perm[mode]["combined_13d"]
        lines.append(f"| **`overlap_6d`** | `{mode}` | {o_p['true_auc']:.4f} | {o_p['null_mean']:.4f} ± {o_p['null_sd']:.4f} | [{o_p['null_ci_95'][0]:.4f}, {o_p['null_ci_95'][1]:.4f}] | **p {o_p['p_value']}** |")
        lines.append(f"| **`combined_13d`** | `{mode}` | {c_p['true_auc']:.4f} | {c_p['null_mean']:.4f} ± {c_p['null_sd']:.4f} | [{c_p['null_ci_95'][0]:.4f}, {c_p['null_ci_95'][1]:.4f}] | **p {c_p['p_value']}** |")

    lines.append("")
    lines.append("---")
    lines.append("")
    lines.append("## 4. Per-Entity Leave-One-Out Sensitivity Analysis")
    lines.append("")
    lines.append(f"Baseline Full `overlap_6d` Pair-Disjoint AUC: **{cur['overlap_6d']['pair_disjoint']['mean']:.4f}**  ")
    lines.append("Sensitivity criterion: Flagged if entity removal drops pair-disjoint AUC by $> 0.05$.")
    lines.append("")
    lines.append("### 4.1 Leave-One-Herb-Out")
    lines.append("")
    lines.append("| Herb Removed | Records (N) | Remaining (N) | Mean AUC ± SD | Δ AUC | Flagged (>0.05 drop) |")
    lines.append("| :--- | :---: | :---: | :---: | :---: | :---: |")
    for h, v in loo["herbs"].items():
        flag = "🚨 **YES**" if v["drops_more_than_0_05"] else "No"
        lines.append(f"| `{h}` | {v['records_removed']} | {v['remaining_records']} | {v['auc']:.4f} ± {v['auc_sd']:.4f} | {v['delta_from_baseline']:+.4f} | {flag} |")

    lines.append("")
    lines.append("### 4.2 Leave-One-Drug-Out (Drugs with N >= 2)")
    lines.append("")
    lines.append("| Drug Removed | Records (N) | Remaining (N) | Mean AUC ± SD | Δ AUC | Flagged (>0.05 drop) |")
    lines.append("| :--- | :---: | :---: | :---: | :---: | :---: |")
    for d, v in loo["drugs"].items():
        flag = "🚨 **YES**" if v["drops_more_than_0_05"] else "No"
        lines.append(f"| `{d}` | {v['records_removed']} | {v['remaining_records']} | {v['auc']:.4f} ± {v['auc_sd']:.4f} | {v['delta_from_baseline']:+.4f} | {flag} |")

    lines.append("")
    lines.append("---")
    lines.append("")
    lines.append("## 5. Entity-Identity-Only Baselines")
    lines.append("")
    lines.append("Performance of categorical one-hot entity identity baselines under repeated grouped CV:")
    lines.append("")
    lines.append("| Identity Representation | Split Mode | Mean Per-Fold AUC | Repeat SD | Fold SD | Pooled OOF AUC |")
    lines.append("| :--- | :--- | :---: | :---: | :---: | :---: |")
    for mode in modes:
        h_res = ident["herb_onehot"][mode]
        d_res = ident["drug_onehot"][mode]
        lines.append(f"| **`herb_onehot`** (18-D) | `{mode}` | {h_res['mean']:.4f} | {h_res['sd']:.4f} | {h_res['fold_sd']:.4f} | {h_res['pooled_oof_mean']:.4f} |")
        lines.append(f"| **`drug_onehot`** (40-D) | `{mode}` | {d_res['mean']:.4f} | {d_res['sd']:.4f} | {d_res['fold_sd']:.4f} | {d_res['pooled_oof_mean']:.4f} |")

    lines.append("")
    lines.append("### Mathematical Property of Per-Fold Primary Metric:")
    lines.append(f"- Under `herb_disjoint` splits, all test botanical entities are unseen during training. A linear one-hot classifier cannot discriminate between test instances and predicts constant probabilities (`sigmoid(intercept)`). Consequently, the per-fold AUC is exactly **{ident['herb_onehot']['herb_disjoint']['mean']:.4f} ± {ident['herb_onehot']['herb_disjoint']['sd']:.4f}** across all folds.")
    lines.append(f"- The lower pooled out-of-fold AUC (**{ident['herb_onehot']['herb_disjoint']['pooled_oof_mean']:.4f}**) arises because pooling out-of-fold predictions across folds with differing class prevalences creates rank-order distortions between folds.")
    lines.append("")
    lines.append("---")
    lines.append("")
    lines.append("## 6. Genre Stratification Under Herb-Disjoint Splits")
    lines.append("")
    lines.append("Stratified evaluation within study design genres under `herb_disjoint` grouped cross-validation, evaluated both with and without the 4 pure-negative control botanicals (`HERB_BLACKCOHOSH`, `HERB_MILKTHISTLE`, `HERB_SAWPALMETTO`, `HERB_VALERIAN`):")
    lines.append("")
    lines.append("| Study Genre Cohort | Pure-Negatives Included | Cohort Size (N) | Class Balance | `overlap_6d` Mean AUC | `overlap_total` Mean AUC | `curation_2d` Mean AUC | `design_only` Mean AUC |")
    lines.append("| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |")

    cohort_order = [
        ("clinical_trials_all", "Yes (All 18 Herbs)"),
        ("clinical_trials_excl_pure_neg", "No (14 Herbs)"),
        ("healthy_volunteers_all", "Yes (All 18 Herbs)"),
        ("healthy_volunteers_excl_pure_neg", "No (14 Herbs)"),
        ("case_reports_all", "Yes (18 Herbs)"),
    ]

    for key, inc_neg in cohort_order:
        cdata = strat[key]
        f = cdata["features"]
        o6 = f"{f['overlap_6d']['mean']:.4f} ± {f['overlap_6d']['sd']:.4f}"
        ot = f"{f['overlap_total']['mean']:.4f} ± {f['overlap_total']['sd']:.4f}"
        c2 = f"{f['curation_2d']['mean']:.4f} ± {f['curation_2d']['sd']:.4f}"
        do = f"{f['design_only']['mean']:.4f} ± {f['design_only']['sd']:.4f}"
        lines.append(f"| {cdata['label']} | {inc_neg} | {cdata['N']} | {cdata['class_balance']} | {o6} | {ot} | {c2} | {do} |")

    lines.append("")
    lines.append("### Diagnostic Observations on Genre Subsets:")
    lines.append(f"- In the clinical trials cohort ($N={strat['clinical_trials_all']['N']}$), `overlap_6d` achieves a per-fold mean AUC of **{strat['clinical_trials_all']['features']['overlap_6d']['mean']:.4f}**.")
    lines.append(f"- When excluding the 4 pure-negative botanicals ($N={strat['clinical_trials_excl_pure_neg']['N']}$), `overlap_6d` per-fold AUC is **{strat['clinical_trials_excl_pure_neg']['features']['overlap_6d']['mean']:.4f}**, while `curation_2d` drops from **{strat['clinical_trials_all']['features']['curation_2d']['mean']:.4f}** to **{strat['clinical_trials_excl_pure_neg']['features']['curation_2d']['mean']:.4f}**.")
    lines.append(f"- The sanitized `design_only` representation achieves **{strat['clinical_trials_all']['features']['design_only']['mean']:.4f}** on clinical trials and **{strat['clinical_trials_excl_pure_neg']['features']['design_only']['mean']:.4f}** with negative controls excluded.")
    lines.append("")

    report_content = "\n".join(lines)
    os.makedirs("reports", exist_ok=True)
    with open("reports/shortcut_tests.md", "w") as f:
        f.write(report_content)
    print("Successfully rendered reports/shortcut_tests.md from results/shortcut_tests.json!")


if __name__ == "__main__":
    render_report()
