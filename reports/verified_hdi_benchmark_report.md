# Verified Botanical HDI Benchmark Report

| Model Variant | Split | ROC-AUC (95% CI) | PR-AUC | F1 | ECE (Calibration) | Brier Score | Coverage | Selective Acc |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| `reliability_conditioned_quality_only` | test_study_disjoint | 0.8512 [0.66, 0.99] | 0.8794 | 0.7826 | 0.1621 | 0.1554 | 1.000 | 0.808 |
| `reliability_conditioned` | test_study_disjoint | 0.7976 [0.61, 0.95] | 0.8278 | 0.6667 | 0.1483 | 0.1878 | 1.000 | 0.692 |
| `standard_fusion` | test_study_disjoint | 0.8274 [0.65, 0.96] | 0.8504 | 0.6087 | 0.2553 | 0.1795 | 1.000 | 0.654 |
| `naive_concat_mlp` | test_study_disjoint | 0.9048 [0.77, 1.00] | 0.9207 | 0.8000 | 0.1089 | 0.1329 | 1.000 | 0.808 |
| `graph_only` | test_study_disjoint | 0.7202 [0.49, 0.92] | 0.6858 | 0.6667 | 0.2320 | 0.2108 | 1.000 | 0.692 |
| `evidence_only` | test_study_disjoint | 0.8095 [0.61, 0.98] | 0.7647 | 0.7692 | 0.1917 | 0.1934 | 1.000 | 0.769 |
| `graph_plus_naive_evidence` | test_study_disjoint | 0.8512 [0.69, 0.97] | 0.8657 | 0.7273 | 0.1273 | 0.1652 | 1.000 | 0.769 |
| `no_study_dependence` | test_study_disjoint | 0.7321 [0.50, 0.93] | 0.6973 | 0.6667 | 0.1421 | 0.1952 | 1.000 | 0.692 |
| `no_context_conflict` | test_study_disjoint | 1.0000 [1.00, 1.00] | 1.0000 | 0.9600 | 0.1344 | 0.0395 | 1.000 | 0.962 |
| `gradient_boosted_trees` | test_study_disjoint | 0.8214 [0.64, 0.97] | 0.8309 | 0.7826 | 0.1543 | 0.1620 | 1.000 | 0.808 |
| `logistic_regression` | test_study_disjoint | 0.8274 [0.64, 0.98] | 0.8616 | 0.8182 | 0.1561 | 0.1467 | 1.000 | 0.846 |
| `mechanistic_rule_baseline` | test_study_disjoint | 0.8423 [0.67, 0.98] | 0.8475 | 0.7407 | 0.1644 | 0.1799 | 1.000 | 0.731 |
| `DeepDDI (Ryu et al. 2018)` | test_study_disjoint | 0.8452 [0.68, 0.97] | 0.8414 | 0.7500 | 0.1149 | 0.1687 | 1.000 | 0.769 |
| `KGNN (Wang et al. 2019)` | test_study_disjoint | 0.7202 [0.49, 0.93] | 0.7805 | 0.6667 | 0.2390 | 0.2194 | 1.000 | 0.692 |
| `CompGCN (Vashishth et al. 2020)` | test_study_disjoint | 0.7262 [0.51, 0.92] | 0.7774 | 0.7273 | 0.2246 | 0.1918 | 1.000 | 0.769 |

Note on Published DDI Baselines (DeepDDI, KGNN, CompGCN): These models were originally engineered for synthetic small-molecule drug-drug interactions (DDIs) using structural fingerprints or multi-relational knowledge graphs. When adapted to botanical HDI, they are provided with canonical target enzyme and transporter modulation profiles (16D) as input representations. They evaluate identical candidate pairs under the same study-disjoint split, but have no access to unstructured literature text, citation lineage DAGs, or context compatibility adjustments.
