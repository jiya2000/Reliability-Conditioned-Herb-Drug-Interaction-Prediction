# Ablation Feature Audit: Lineage Columns & Leakage Control

This audit documents every model mode in `src/models/interaction_model.py` and `src/data/evidence_lineage.py`, detailing the exact subset of the 12 evidence lineage columns passed to the reliability scorer and the classifier.

---

## 1. The 12 Canonical Evidence Lineage Columns

In `PairEvidenceSummary.to_full_metadata_vector()`, candidate pairs are described by 12 metadata dimensions:

| Index | Symbol | Attribute Name | Type | Description | Leakage Status |
| :---: | :---: | :--- | :--- | :--- | :--- |
| **0** | $S_{\text{pos}}$ | `independent_positive_strength` | Polarity | Effective independent positive study strength | 🚨 **Polarity Leakage** |
| **1** | $S_{\text{neg}}$ | `independent_negative_strength` | Polarity | Effective independent negative study strength | 🚨 **Polarity Leakage** |
| **2** | $\text{echo}$ | `citation_echo_ratio` | Quality | Redundant citation fraction within study clusters | ✅ Quality-Only |
| **3** | $q_{\text{mean}}$ | `mean_source_quality` | Quality | Mean publication grade $\times$ extraction confidence | ✅ Quality-Only |
| **4** | $W_{\text{context}}$ | `context_compatibility_score` | Quality | Clinical compatibility (human in vivo vs in vitro) | ✅ Quality-Only |
| **5** | $C_{\text{hd}}$ | `contradiction_strength` | Quality | Degree of conflicting independent evidence | ✅ Quality-Only |
| **6** | $\text{margin}$ | `net_evidence_margin` | Polarity | $S_{\text{pos}} - S_{\text{neg}}$ (net evidentiary margin) | 🚨 **Polarity Leakage (Perfect Separator)** |
| **7** | $\text{missing}$ | `metadata_missingness` | Quality | Fraction of unstated context fields | ✅ Quality-Only |
| **8** | $N_{\text{raw\_pos}}$ | `raw_positive_count` | Polarity | Raw unweighted count of positive reports | 🚨 **Polarity Leakage** |
| **9** | $N_{\text{raw\_neg}}$ | `raw_negative_count` | Polarity | Raw unweighted count of negative reports | 🚨 **Polarity Leakage** |
| **10** | $\rho_{\text{reach}}$ | `rho_reach` | Quality | Pharmacokinetic reachability score | ✅ Quality-Only |
| **11** | $f_{m,\text{CYP}}$ | `f_m_CYP` | Quality | Drug hepatic CYP clearance fraction | ✅ Quality-Only |

### Core Root Cause of the `no_context_conflict` Leakage Anomaly
Under benchmark evaluation, labels correspond to published clinical study outcomes. In the original implementation of `no_context_conflict`, the model received the full 12D vector with context set to $1.0$, contradiction set to $0.0$, and margin computed as $S_{\text{pos}}$ alone (removing the conflict subtraction). Because $S_{\text{pos}} > 0 \iff y=1$ and $S_{\text{neg}} > 0 \iff y=0$, the margin feature and polarity strengths enabled the end-to-end reliability scorer to perfectly separate positive and negative pairs, yielding an artificial test ROC-AUC of **1.0000 [1.00, 1.00]**.

---

## 2. Feature Audit Matrix: All Modes in `interaction_model.py` and `evidence_lineage.py`

Below is the exhaustive audit of features passed to the **Reliability Scorer** and to the **Classifier** across all modes, comparing the prior state against the leakage-free redesigned state:

| Model Mode | Reliability Scorer Columns (Prior) | Reliability Scorer Columns (Clean Redesign) | Classifier Input Columns | Polarity Columns Present? | Leakage Status |
| :--- | :--- | :--- | :--- | :---: | :--- |
| `reliability_conditioned` | Indices [2, 3, 4, 5, 7, 10, 11] (7D quality) | **Indices [2, 3, 4, 5, 7, 10, 11] (7D quality)** | $z_{\text{fused}} = (1-R)z_{\text{graph}} + R z_{\text{ev}}$ | **NO** | ✅ Clean (Headline Model) |
| `reliability_conditioned_quality_only` | Indices [2, 3, 4, 5, 7, 10, 11] (7D quality) | **Indices [2, 3, 4, 5, 7, 10, 11] (7D quality)** | $z_{\text{fused}} = (1-R)z_{\text{graph}} + R z_{\text{ev}}$ | **NO** | ✅ Clean (Identical to proposed) |
| `no_study_dependence` | Indices [0..11] (12D: col 2 zeroed, cols 0, 1, 6 replaced with raw counts) 🚨 | **7D Quality-Only with undiscounted quantities ($\alpha=1$): `echo_ratio=0.0`, undiscounted $C_{\text{hd}}$, $q_{\text{mean}}, W_{\text{ctx}}, \text{missing}, \rho, f_m$** | $z_{\text{fused}} = (1-R)z_{\text{graph}} + R z_{\text{ev}}$ | **NO** | ✅ Clean (Ablation) |
| `no_context_conflict` | Indices [0..11] (12D: col 4=1.0, col 5=0.0, col 6=$S_{\text{pos}}$) 🚨 | **5D Quality-Only: drops $W_{\text{context}}$ and $C_{\text{hd}}$ leaving `[echo_ratio, q_mean, missingness, rho_reach, f_m_CYP]`** | $z_{\text{fused}} = (1-R)z_{\text{graph}} + R z_{\text{ev}}$ | **NO** | ✅ Clean (Ablation) |
| `standard_fusion` | None ($R=1.0$ unconditioned) | **None ($R=1.0$ unconditioned)** | $z_{\text{fused}} = 0.5 z_{\text{graph}} + 0.5 z_{\text{ev}}$ | **NO** | ✅ Clean (Baseline) |
| `graph_plus_naive_evidence` | None ($R=0.5$ unweighted) | **None ($R=0.5$ unweighted)** | $z_{\text{fused}} = 0.5 z_{\text{graph}} + 0.5 z_{\text{ev}}$ | **NO** | ✅ Clean (Baseline) |
| `naive_concat_mlp` | None ($R=1.0$) | **None ($R=1.0$)** | $z_{\text{fused}} = \text{MLP}([z_{\text{graph}}; z_{\text{ev}}])$ | **NO** | ✅ Clean (Baseline) |
| `evidence_only` | None ($R=1.0$) | **None ($R=1.0$)** | $z_{\text{fused}} = z_{\text{ev}}$ | **NO** | ✅ Clean (Baseline) |
| `graph_only` | None ($R=0.0$) | **None ($R=0.0$)** | $z_{\text{fused}} = z_{\text{graph}}$ | **NO** | ✅ Clean (Baseline) |
| `gradient_boosted_trees` | N/A | N/A | 7D Quality + 6D CYP Overlap = 13D | **NO** | ✅ Clean (Tabular Baseline) |
| `logistic_regression` | N/A | N/A | 7D Quality + 6D CYP Overlap = 13D | **NO** | ✅ Clean (Tabular Baseline) |
| `mechanistic_rule_baseline` | N/A | N/A | Pharmacokinetic clearance overlap | **NO** | ✅ Clean (Heuristic) |

---

## 3. Leaky Reference Models (Kept ONLY in Separate Reference Table)

The following models incorporate polarity columns ($S_{\text{pos}}, S_{\text{neg}}, \text{margin}, N_{\text{raw\_pos}}, N_{\text{raw\_neg}}$) and are retained strictly for leakage demonstration in a separate reference section:

| Reference Model Variant | Scorer Columns | Classifier Input Columns | Polarity Columns | Status |
| :--- | :--- | :--- | :---: | :--- |
| `leaky_polarity_probe` | N/A | Indices [0, 1, 6, 8, 9] (5D polarity) | $S_{\text{pos}}, S_{\text{neg}}, \text{margin}, N_{\text{pos}}, N_{\text{neg}}$ | ⚠️ **LEAKY REFERENCE (not a valid model)** |
| `leaky_full_vector_no_context_conflict` | Indices [0..11] (12D with $\text{margin}=S_{\text{pos}}$) | $z_{\text{fused}}$ modulated by leaky $R$ | $S_{\text{pos}}, S_{\text{neg}}, \text{margin}, N_{\text{pos}}, N_{\text{neg}}$ | ⚠️ **LEAKY REFERENCE (not a valid model)** |
| `leaky_full_vector_reliability_conditioned` | Indices [0..11] (12D full lineage vector) | $z_{\text{fused}}$ modulated by leaky $R$ | $S_{\text{pos}}, S_{\text{neg}}, \text{margin}, N_{\text{pos}}, N_{\text{neg}}$ | ⚠️ **LEAKY REFERENCE (not a valid model)** |

---

## 4. Policy Guarantees
1. **Zero Polarity Transmission:** No polarity column ($S_{\text{pos}}, S_{\text{neg}}, \text{margin}, N_{\text{raw\_pos}}, N_{\text{raw\_neg}}$) may reach any model in the main comparison table, either directly or via the reliability scorer.
2. **5D Quality Ablation for `no_context_conflict`:** Exactly 5 quality features remain (`echo_ratio, q_mean, missingness, rho_reach, f_m_CYP`). Both context compatibility ($W_{\text{context}}$) and contradiction strength ($C_{\text{hd}}$) are removed.
3. **Undiscounted Quality Features for `no_study_dependence`:** Evaluated strictly using $\alpha=1.0$ (setting `echo_ratio = 0.0` and contradiction strength to undiscounted $C_{\text{hd}}$) within the 7D quality feature group.
4. **Segregation:** Leaky reference models are permanently quarantined in a distinct sub-table clearly marked `LEAKY REFERENCE (not a valid model)` and never appear in the headline comparison.
