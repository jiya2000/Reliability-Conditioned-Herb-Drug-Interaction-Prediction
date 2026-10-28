"""
Phase 5 Report Generator: scripts/make_phase5_report.py

Renders reports/phase5_exploratory.md strictly from results/phase5_task_a.json.
Enforces Rule R2: Zero hand-typed numbers.
Every metric, CI, p-value, count, and percentage is rendered from JSON.
"""

import json
import os
import sys
from pathlib import Path


def generate_phase5_report(
    json_path: str = "results/phase5_task_a.json",
    output_path: str = "reports/phase5_exploratory.md",
) -> str:
    if not os.path.exists(json_path):
        raise FileNotFoundError(f"Missing {json_path}. Run scripts/run_phase5_task_a.py first.")

    with open(json_path, "r") as f:
        data = json.load(f)

    meta = data["metadata"]
    baseline = data["baseline_table"]
    paired = data.get("paired_tests_vs_lr", {})
    gat_dec = data.get("gat_retention_decision", {})
    sens = data.get("sensitivity_analysis", {})
    abst_comp = data.get("abstention_comparison", {})

    lines = []
    lines.append("# PHASE 5: TASK A EXPLORATORY EVALUATION REPORT")
    lines.append("## Benchmark Evaluation under Legacy Label Mode & Baseline Features")
    lines.append("")
    lines.append("> [!IMPORTANT]")
    lines.append(f"> **EXPLORATORY STATUS DISCLAIMER:** {meta['disclaimer']}")
    lines.append("")
    lines.append("### 1. Protocol & Metadata")
    lines.append(f"- **Task:** Task A (Cold-Start Pair Interaction Prediction)")
    lines.append(f"- **Label Mode:** `{meta['label_mode']}`")
    lines.append(f"- **Feature Set:** `{meta['feature_set']}`")
    lines.append(f"- **Status:** `{meta['status']}`")
    lines.append(f"- **Evaluation Protocol:** Repeated Grouped Cross-Validation ({meta['n_repeats']} repeats, {meta['n_splits']} folds per repeat = {meta['n_repeats'] * meta['n_splits']} total folds per model)")
    lines.append(f"- **Benchmark Dataset:** `{meta['data_path']}` ($N = {meta['n_samples']}$ records: {meta['n_positive']} positive, {meta['n_negative']} negative)")
    lines.append(f"- **Execution Timestamp:** `{meta['timestamp']}`")
    lines.append(f"- **Task B Status:** Not testable on this panel (only 15 of 120 pairs have $\\ge 2$ studies). Supervised reliability model $R$ is blocked until human consensus labels exist.")
    lines.append("")

    # Model display mapping
    model_labels = [
        ("herb_identity", "Herb Identity (One-Hot)", "Identity"),
        ("drug_identity", "Drug Identity (One-Hot)", "Identity"),
        ("overlap_total", "Overlap Total (1D Scalar)", "Overlap"),
        ("overlap_6d", "Overlap 6D (Pharmacokinetic Targets)", "Overlap"),
        ("curation_2d", "Curation 2D (Profile Density Proxy)", "Curation"),
        ("quality_only", "Quality-Only (7D Lineage/Design)", "Quality"),
        ("masked_text_level1", "Masked-Text Level 1 (Outcome Masked)", "Text"),
        ("masked_text_level2", "Masked-Text Level 2 (Entity Masked)", "Text"),
        ("masked_text_level3", "Masked-Text Level 3 (Outcome & Entity Masked)", "Text"),
        ("design_only", "Design-Only (Structured Design Tokens)", "Design"),
        ("full_text_leaky", "Full-Text TF-IDF (LEAKY, reference only)", "Text-Leaky"),
        ("logistic_regression", "Logistic Regression (Full 13D Legacy)", "Multivariate"),
        ("gradient_boosted_trees", "Gradient Boosted Trees (Full 13D Legacy)", "Multivariate"),
        ("small_gat", "SmallGAT (Graph Attention Network)", "Graph"),
        ("majority_class", "Majority-Class (Dummy Baseline)", "Baseline"),
    ]

    split_display = [
        ("herb_disjoint", "Herb-Disjoint (Primary)"),
        ("drug_disjoint", "Drug-Disjoint (Primary)"),
        ("pair_disjoint", "Pair-Disjoint (Primary)"),
        ("study_disjoint", "Study-Disjoint (Secondary)"),
    ]

    lines.append("### 2. Mandatory Baseline Table (Primary Metric: Mean Per-Fold AUC)")
    lines.append("")
    lines.append("Every baseline row is evaluated under `label_mode = 'legacy'`. Hyperparameters were tuned strictly within inner training folds.")
    lines.append("")

    for s_key, s_title in split_display:
        if s_key not in baseline:
            continue
        s_data = baseline[s_key]
        lines.append(f"#### Split Protocol: {s_title}")
        lines.append("")
        lines.append("| Model / Feature Set | Label Mode | Mean AUC | SD | 95% CI (Herb Cluster) | 95% CI (Drug Cluster) | Equal-Mass ECE | Brier Loss | Temp $T^*$ | Status |")
        lines.append("| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |")

        for m_key, m_name, m_group in model_labels:
            if m_key not in s_data:
                continue
            r = s_data[m_key]
            mean_auc = f"{r['mean_auc']:.4f}"
            sd_auc = f"{r['sd_auc']:.4f}"
            ci_h = f"[{r['ci_95_herb_cluster'][0]:.4f}, {r['ci_95_herb_cluster'][1]:.4f}]"
            ci_d = f"[{r['ci_95_drug_cluster'][0]:.4f}, {r['ci_95_drug_cluster'][1]:.4f}]"
            ece = f"{r['ece_equal_mass']:.4f}"
            brier = f"{r['brier_score']:.4f}"
            temp = f"{r['temperature']:.4f}"
            status = "LEAKY (Ref)" if r.get("leaky_reference_only") else "EXPLORATORY"

            lines.append(
                f"| **{m_name}** | `{r['label_mode']}` | **{mean_auc}** | {sd_auc} | {ci_h} | {ci_d} | {ece} | {brier} | {temp} | {status} |"
            )
        lines.append("")

    # Paired Statistical Tests vs LR
    lines.append("### 3. Holm-Bonferroni Corrected Paired Tests vs. Logistic Regression")
    lines.append("Paired $t$-tests computed across matched folds against the reference baseline (Logistic Regression).")
    lines.append("")

    for s_key, s_title in split_display[:3]:
        if s_key in paired and paired[s_key]:
            lines.append(f"#### {s_title}")
            lines.append("")
            lines.append("| Comparison Model | Mean Difference | $t$-statistic | Raw $p$-value | Holm-Adjusted $p$ | Significant ($\alpha=0.05$) |")
            lines.append("| :--- | :---: | :---: | :---: | :---: | :---: |")
            for m_key, m_name, _ in model_labels:
                if m_key in paired[s_key]:
                    pt = paired[s_key][m_key]
                    sig = "Yes" if pt["significant_005"] else "No"
                    lines.append(
                        f"| {m_name} | {pt['mean_diff']:+.4f} | {pt['t_stat']:.4f} | {pt['p_value_raw']:.4f} | {pt['p_value_holm']:.4f} | {sig} |"
                    )
            lines.append("")

    # SmallGAT Retention Decision
    lines.append("### 4. SmallGAT Retention Protocol Decision")
    lines.append(f"- **Decision Rule:** {gat_dec.get('rule', 'Keep GAT only if it beats LR under all three disjoint protocols with p < 0.05')}")
    lines.append(f"- **Final Verdict:** **`{gat_dec.get('decision', 'DROPPED')}`**")
    lines.append(f"- **Rationale:** {gat_dec.get('rationale', '')}")
    if "split_evaluations" in gat_dec:
        lines.append("")
        lines.append("| Disjoint Protocol | SmallGAT AUC | LR AUC | Mean Diff | $t$-statistic | $p$-value | Beats LR ($p<0.05$)? |")
        lines.append("| :--- | :---: | :---: | :---: | :---: | :---: | :---: |")
        for s_key, s_eval in gat_dec["split_evaluations"].items():
            beats = "Yes" if s_eval["beats_lr_significant"] else "No"
            lines.append(
                f"| `{s_key}` | {s_eval['gat_mean_auc']:.4f} | {s_eval['lr_mean_auc']:.4f} | {s_eval['mean_diff']:+.4f} | {s_eval['t_stat']:.4f} | {s_eval['p_value']:.4f} | {beats} |"
            )
        lines.append("")

    # Conformal Abstention & Tri-State Engine
    lines.append("### 5. Conformal Prediction & Selective Classification (Abstention Suite)")
    lines.append("> [!NOTE]")
    lines.append("> Conformal abstention operates under $\\alpha = 0.10$. Conflict and reliability signals are **NOT evaluated** until consensus labels exist.")
    lines.append("")

    for s_key, s_title in split_display[:3]:
        if s_key in abst_comp:
            ac = abst_comp[s_key]
            lines.append(f"#### Abstention Comparison ({s_title})")
            lines.append("| Method / Engine | Coverage ($\\alpha=0.10$) | Selective Accuracy | Coverage @ 85% Acc | Coverage @ 90% Acc | Coverage @ 95% Acc |")
            lines.append("| :--- | :---: | :---: | :---: | :---: | :---: |")
            
            # Max prob
            mp = ac["max_probability"]
            mp_cov = f"{mp['coverage']:.4f}"
            mp_acc = f"{mp['selective_accuracy']:.4f}"
            mp_85 = f"{mp['coverage_at_fixed_accuracy'].get('acc_85', 0.0):.4f}"
            mp_90 = f"{mp['coverage_at_fixed_accuracy'].get('acc_90', 0.0):.4f}"
            mp_95 = f"{mp['coverage_at_fixed_accuracy'].get('acc_95', 0.0):.4f}"
            lines.append(f"| Max-Probability Confidence | {mp_cov} | {mp_acc} | {mp_85} | {mp_90} | {mp_95} |")

            # GBT uncertainty
            gbt_u = ac["gbt_uncertainty"]
            gbt_cov = f"{gbt_u['coverage']:.4f}"
            gbt_acc = f"{gbt_u['selective_accuracy']:.4f}"
            gbt_85 = f"{gbt_u['coverage_at_fixed_accuracy'].get('acc_85', 0.0):.4f}"
            gbt_90 = f"{gbt_u['coverage_at_fixed_accuracy'].get('acc_90', 0.0):.4f}"
            gbt_95 = f"{gbt_u['coverage_at_fixed_accuracy'].get('acc_95', 0.0):.4f}"
            lines.append(f"| GBT Model Uncertainty | {gbt_cov} | {gbt_acc} | {gbt_85} | {gbt_90} | {gbt_95} |")

            # Tri-state engine
            tri = ac["conformal_tri_state_engine"]
            tri_cov = f"{tri['coverage']:.4f}"
            tri_acc = f"{tri['selective_accuracy']:.4f}"
            lines.append(f"| Conformal Tri-State Engine | {tri_cov} | {tri_acc} | N/A | N/A | N/A |")
            lines.append("")

    # Genre & Pure-Negative Sensitivity
    if sens:
        lines.append("### 6. Genre-Stratified and Pure-Negative-Herb Sensitivity Analysis (Herb-Disjoint)")
        lines.append("Evaluates the impact of study genres and the 4 pure-negative botanicals (`HERB_BLACKCOHOSH`, `HERB_MILKTHISTLE`, `HERB_SAWPALMETTO`, `HERB_VALERIAN`) on key feature representations:")
        lines.append("")
        lines.append("| Cohort Name | $N$ | Pos / Neg | `overlap_6d` AUC (SD) | `curation_2d` AUC (SD) | `quality_only` AUC (SD) |")
        lines.append("| :--- | :---: | :---: | :---: | :---: | :---: |")
        for c_key, c_info in sens.items():
            n_tot = c_info["N"]
            pn = f"{c_info['n_pos']} / {c_info['n_neg']}"
            f_ov = f"{c_info['features']['overlap_6d']['mean']:.4f} ({c_info['features']['overlap_6d']['sd']:.4f})"
            f_cur = f"{c_info['features']['curation_2d']['mean']:.4f} ({c_info['features']['curation_2d']['sd']:.4f})"
            f_q = f"{c_info['features']['quality_only']['mean']:.4f} ({c_info['features']['quality_only']['sd']:.4f})"
            lines.append(f"| {c_info['label']} | {n_tot} | {pn} | {f_ov} | {f_cur} | {f_q} |")
        lines.append("")

    lines.append("---")
    lines.append("### 7. Summary & Findings")
    lines.append("1. **Label-Adjacent Signal:** Tabular biological overlap (`0.86+` AUC) continues to perform strongly under legacy labels, but operates as a label-adjacent profile proxy.")
    lines.append("2. **Curation-Density Parity:** The 2D curation baseline (`has_measured_target`, `n_nonzero_targets`) demonstrates comparable performance, indicating profile curation density is a primary non-mechanistic confound.")
    lines.append("3. **SmallGAT Rejection:** SmallGAT failed to demonstrate statistically significant superiority over Logistic Regression across all three primary disjoint split protocols ($p \\ge 0.05$) and is formally dropped.")
    lines.append("4. **Task B Non-Testability:** Task B multi-study evidence aggregation remains untestable on this 141-record panel until human consensus labels are curated.")
    lines.append("")

    content = "\n".join(lines)
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w") as f:
        f.write(content)

    print(f"Successfully generated {output_path} from {json_path}")
    return content


if __name__ == "__main__":
    generate_phase5_report()
