# Reliability-Conditioned Herb–Drug Interaction Prediction

> **A Defensible, Clinically Calibrated Machine Learning Framework for Predicting Botanical–Pharmaceutical Interactions via Reliability-Conditioned Graph Neural Networks, Citation-Derivation DAGs with Echo Attenuation, Pharmacokinetic Context Compatibility, and Conformal Tri-State Abstention.**

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.1+-ee4c2c.svg)](https://pytorch.org/)
[![Tests](https://img.shields.io/badge/pytest-116%20passed-brightgreen.svg)](tests/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

---

## 📖 Invention & Framework Overview

This repository implements an end-to-end, scientifically defensible system for **reliability-conditioned prediction of botanical–pharmaceutical interactions (HDIs)**. The architecture directly addresses core methodological flaws present in conventional multi-modal interaction models: circular literature citation inflation, ungrounded in vitro concentrations, epistemic uncertainty neglect, and forced error rates on ambiguous or contradictory scientific literature.

The framework integrates five patented/peer-defensible technological pillars:

1. **Citation-Derivation DAG with Multi-Factor Echo Attenuation** — Parses literature citation networks into primary empirical root studies and applies depth-discounted attenuation ($\alpha = 0.15$) scaled by study methodology (`review: 1.80`, `meta_analysis: 1.50`, `rct: 0.30`) and Jaccard text-reuse similarity. Reiterations that refute primary findings are preserved with full independent weight ($w = 1.0$).
2. **Pharmacokinetic Context Compatibility & IVIVE Screening** — Multi-factor scoring weighting biological species ($w_{\text{human}} > w_{\text{animal}} > w_{\text{in\_vitro}}$), botanical extract standardization, and administration route. Preclinical in vitro findings are evaluated via site-of-action pharmacokinetic branching:
   - *Hepatic / Systemic:* $\rho_{\text{reach}} = \min(1.0, [I]_{u,\max} / \text{IC}_{50})$ using unbound human plasma concentration.
   - *Intestinal CYP3A4 / P-gp:* $I_{\text{gut}} = \frac{\text{dose\_mg} \times 4000.0}{\text{MW}}\,\mu\text{M}$ with auditable fallback flags for unannotated doses.
3. **Dynamic Neural Reliability Gating** — Balances biological mechanism priors ($\mathbf{z}_{\text{graph}}$) against literature contextual representations ($\mathbf{z}_{\text{contextual}}$) via continuous coefficient $R \in [0, 1]$. In unstudied pairs or when metadata is missing ($M=1.0$), the system strictly enforces $R \to 0.0$, gracefully retreating to biological graph priors without hallucinated confidence.
4. **Validation-Fit Temperature Scaling & Conformal Calibration** — Temperature parameter $T^*$ and split-conformal quantile $\hat{q}$ are fit strictly on validation partitions (zero test contamination), yielding calibrated posterior probabilities and distribution-free coverage guarantees ($P(Y \in C(X)) \ge 1 - \epsilon$).
5. **Conformal Tri-State Decision Engine (Non-Device CDS)** — Routes predictions into three clinically safe pathways:
   - `INTERACTION_PRESENT` ($P(Y=1) \ge 0.5$, non-conformal singleton $\{1\}$): Automated EHR clinical alert.
   - `NO_INTERACTION` ($P(Y=1) < 0.5$, non-conformal singleton $\{0\}$): Automated safe co-administration clearance.
   - `ABSTAIN_CLINICAL_REVIEW` (conformal set $\{0, 1\}$, scientific contradiction $C_{h,d} > \tau_c$, or low reliability with borderline margin): Asynchronous clinical pharmacist review queue.

---

## 🛡️ Leakage Audit & Default Headline Policy

In Phase 1 of our formal hardening audit (`benchmarks/leakage_probe.py`), we identified that using label-adjacent polarity counts ($S_{\text{pos}}, S_{\text{neg}}, \text{margin}$) computed on test rows yielded an artificial **1.0000 ROC-AUC** for simple linear models, proving direct target leakage.

To ensure peer defensibility and journal rigor:
- **`quality_only` (7D) is the MANDATORY DEFAULT headline feature configuration**:  
  $\mathbf{f}_{\text{quality}} = [\text{echo\_ratio}, \bar{q}_{\text{source}}, W_{\text{context}}, C_{h,d}, \text{missingness}, \rho_{\text{reach}}, f_{m,\text{CYP}}]$.
  It evaluates evidentiary robustness and PK plausibility without encoding study findings.
- **Strict Study-Disjoint Partitioning**: Zero citation ancestors, descendants, or empirical cohorts overlap across train, validation, and test splits.
- **Strict Zero-Synthetic Data Policy**: Zero fabricated or synthesized records are permitted. All 141 records in the canonical benchmark (`data/benchmark/verified_botanical_hdi.csv`) are verified against PubMed PMIDs and CrossRef DOIs in the `AuditableSourceRegistry`.

---

## 📊 Benchmark Results & Empirical Defensibility

Evaluated on the audited benchmark of **141 peer-reviewed publications across 18 botanicals and 40 prescription medications** encompassing **123 independent study clusters**, partitioned into 90 train (73 clusters), 25 validation (25 clusters), and 26 test (25 clusters) records:

| Model Architecture & Input Modalities | ROC-AUC (95% CI) | PR-AUC | F1-Score | Brier Score | ECE | Test Accuracy |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Proposed Reliability-Conditioned (Quality-Only, Leakage-Free)** | **0.8512** [0.66, 0.99] | **0.8794** | **0.7826** | **0.1554** | **0.1621** | **80.77% (21/26)** |
| Naive Concat MLP (No Reliability Gating) | 0.9048 [0.77, 1.00] | 0.9207 | 0.8000 | 0.1329 | 0.1089 | 80.77% (21/26) |
| Logistic Regression (13D Tabular) | 0.8274 [0.64, 0.98] | 0.8616 | 0.8182 | 0.1467 | 0.1561 | 84.62% (22/26) |
| Standard Multimodal Fusion (Concat + MLP) | 0.8274 [0.65, 0.96] | 0.8504 | 0.6087 | 0.1795 | 0.2553 | 65.38% (17/26) |
| Gradient Boosted Trees (13D Tabular) | 0.8214 [0.64, 0.97] | 0.8309 | 0.7826 | 0.1620 | 0.1543 | 80.77% (21/26) |
| Mechanistic Rule Baseline (Heuristic) | 0.8423 [0.67, 0.98] | 0.8475 | 0.7407 | 0.1799 | 0.1644 | 73.08% (19/26) |
| **External Baseline: DeepDDI** (Ryu et al. 2018) | 0.8452 [0.68, 0.97] | 0.8414 | 0.7500 | 0.1687 | 0.1149 | 76.92% (20/26) |
| **External Baseline: CompGCN** (Vashishth et al. 2020) | 0.7262 [0.51, 0.92] | 0.7774 | 0.7273 | 0.1918 | 0.2246 | 76.92% (20/26) |
| **External Baseline: KGNN** (Wang et al. 2019) | 0.7202 [0.49, 0.93] | 0.7805 | 0.6667 | 0.2194 | 0.2390 | 69.23% (18/26) |
| Biological Graph Only (GAT Mechanism Graph Only) | 0.7202 [0.49, 0.92] | 0.6858 | 0.6667 | 0.2108 | 0.2320 | 69.23% (18/26) |
| Evidence Text Only (Unstructured Tokens Only) | 0.8095 [0.61, 0.98] | 0.7647 | 0.7692 | 0.1934 | 0.1917 | 76.92% (20/26) |
| Graph + Naive Evidence (No Lineage Attenuation) | 0.8512 [0.69, 0.97] | 0.8657 | 0.7273 | 0.1652 | 0.1273 | 76.92% (20/26) |
| *Ablation: Full 12D Lineage (With Polarity Features)* | *0.7976* [0.61, 0.95] | *0.8278* | *0.6667* | *0.1878* | *0.1483* | *69.23% (18/26)* |

### Explicit Metric Denominator Definitions
- **Test Partition Size ($N_{\text{test}}$)**: $26$ held-out study-disjoint evaluation records across 25 independent study clusters ($N_{\text{pos}} = 12, N_{\text{neg}} = 14$).
- **Raw Test Accuracy Denominator**:  
  $$\text{Accuracy} = \frac{N_{\text{correct, total}}}{N_{\text{total}}} = \frac{21}{26} = 80.77\% \quad (\text{decision threshold } \tau = 0.50)$$
- **Pairwise Ranking Denominator**:  
  $$\text{ROC-AUC Pair Comparisons} = N_{\text{pos}} \times N_{\text{neg}} = 12 \times 14 = 168 \text{ pairs}$$

---

## ⚡ Out-of-Distribution External Validation

The headline model was evaluated **without refitting or fine-tuning** on an external clinical validation cohort constructed from the Northwest Center of Excellence for Natural Product-Drug Interaction Research (**NaPDI**) template (`data/external/napdi_template.csv`):
- **Cohort**: Curcumin + Midazolam (negative control RCT), Goldenseal + Metformin (OCT/MATE transporter clinical trial), Kratom + Oxycodone (CYP3A4 opioid toxicity trial), Goldenseal + Midazolam (CYP3A probe clinical study), and Curcumin + Talinolol (P-gp efflux study).
- **Zero-Refitting Generalization**: Freezes source weights and calibrators.
- **Safety Abstention**: Correctly detects bounded conformal uncertainty on novel pairs without literature history ($R \to 0.0$), successfully routing them to clinical review without making unverified forced errors.
- **Raw Binary Accuracy**: $80.0\%$ (4/5 correctly predicted under forced binary threshold).

---

## 🚀 Reproduction & Verification

To reproduce all benchmarks, audits, and external validations in a single command:

```bash
# 1. Clone repository & install dependencies
git clone https://github.com/jiya2000/Reliability-Conditioned-Herb-Drug-Interaction-Prediction.git
cd Reliability-Conditioned-Herb-Drug-Interaction-Prediction
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
pip install -e .

# 2. Run master reproduction pipeline (Single command)
make reproduce
# Alternatively: bash benchmarks/run_all.sh

# 3. Run full unit test suite (116 tests passing)
make test
# Or fast quiet mode: make test-fast
```

### Launch Interactive Decision Support UI
```bash
make app
# Or: streamlit run app/interactive_app.py
```

---

## 🏗️ Codebase Organization

```
├── app/
│   └── interactive_app.py          # Clinical Decision Support UI (Non-device CDS)
├── benchmarks/
│   ├── run_all.sh                  # Master reproduction script
│   ├── aggregate_master_results.py # Aggregator for master_benchmark_results.json
│   ├── run_verified_hdi_benchmark.py # Verified 141-study benchmark & 8 ablations
│   ├── external_baselines.py       # DeepDDI, KGNN, CompGCN implementations
│   ├── external_validation.py      # Zero-refitting NaPDI external evaluation
│   ├── leakage_probe.py            # Phase 1 polarity feature leakage probe
│   └── sensitivity_analysis.py     # Alpha & threshold sensitivity sweep
├── data/
│   ├── benchmark/                  # Verified 141-study canonical benchmark CSV
│   ├── external/                   # NaPDI external validation template CSV
│   └── audit/                      # Citation DAG audited component records
├── docs/
│   ├── evidence_schema.md          # Canonical evidence data dictionary & schema
│   └── data_extension.md           # Dataset scaling protocol (>1,000 pairs)
├── reports/
│   ├── revision_log.md             # Formal engineering & hardening log (Phases 0-11)
│   ├── master_benchmark_results.json # Consolidated master metrics record
│   ├── final_summary.md            # Comprehensive before/after audit dossier
│   ├── audit.md                    # Phase 0 architectural audit report
│   ├── leakage_probe_report.md     # Phase 1 leakage probe findings
│   ├── sensitivity_analysis.md     # Phase 4 hyperparameter sensitivity report
│   ├── citation_dag_audit.md       # Phase 9 citation DAG audit report
│   └── external_validation_report.md # Phase 10 NaPDI generalization report
├── src/
│   ├── data/
│   │   ├── evidence_lineage.py     # 7D quality-only & 12D full lineage processor
│   │   ├── evidence_schema.md      # Schema validators & Pydantic typing
│   │   ├── source_registry.py      # Auditable source registry (PMID/DOI matching)
│   │   └── public_data_loader.py   # Zero-synthetic data ingestion pipeline
│   ├── features/
│   │   ├── biological_features.py  # 16D biological PK profiles & TF-IDF encoder
│   │   └── biological_graph_encoder.py # Heterogeneous biological GAT encoder
│   ├── models/
│   │   ├── interaction_model.py    # Headline ReliabilityConditionedInteractionModel
│   │   ├── citation_dag.py         # Citation DAG & study-type echo attenuator
│   │   ├── context_compatibility.py # Context weights & learnable multipliers
│   │   ├── pk_reachability.py      # Physiological site-of-action PK branching
│   │   ├── conformal_abstention.py # Split-conformal tri-state decision engine
│   │   └── reliability_scorer.py   # Continuous R gating with zero fallback
│   └── training/
│       ├── calibration.py          # TemperatureScaler, ECE, & Brier score
│       ├── evaluation.py           # LeakageSafeSplitter & repeated nested CV
│       └── statistical_tests.py    # DeLong test & paired bootstrap comparisons
└── tests/                          # 116 comprehensive unit & regression tests
```

---

## ⚖️ Regulatory & Patent Disclosure Notice

This software is designed as **Non-Device Clinical Decision Support (CDS)** software under Section 520(o)(1)(E) of the U.S. Federal Food, Drug, and Cosmetic Act and the FDA Clinical Decision Support Guidance (2022). It provides transparent evidentiary rationales, intermediate reliability scores, and audited primary literature citations so that healthcare professionals can independently review and interpret all predictions. It does not replace clinical judgment and is intended for research and investigational decision support.

Licensed under the [MIT License](LICENSE).