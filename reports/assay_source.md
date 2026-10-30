# Retrospective ChEMBL compound-assay experiment

Endpoint: pIC50. These are held-out assay predictions, not clinical herb–drug predictions.

Split: source; seed: 20261030. Model selection uses validation MAE.

| Target | Selected model | Test n | MAE | RMSE | R² | Interval coverage |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| ABCB1 | median | 166 | 0.587 | 0.773 | -0.051 | 0.958 |
| CYP1A2 | consistency_weighted_extra_trees | 180 | 0.751 | 0.992 | 0.123 | 0.794 |
| CYP2C19 | ridge | 158 | 0.644 | 0.843 | -0.060 | 0.949 |
| CYP2C9 | consistency_weighted_extra_trees | 347 | 0.514 | 0.720 | -0.026 | 0.939 |
| CYP2D6 | extra_trees | 378 | 0.637 | 0.841 | 0.124 | 0.926 |
| CYP3A4 | consistency_weighted_extra_trees | 847 | 0.731 | 1.044 | -0.222 | 0.839 |

Complete baseline results, confidence intervals and role counts are in [`assay_source.json`](../results/assay_source.json).

Cross-role publication/scaffold overlaps and all source-split purges are in [`assay_source_split_audit.json`](../results/assay_source_split_audit.json).

- Compound-target assay potency only; no whole-herb or clinical interaction prediction.
- Text screening is conservative and incomplete; assay protocols and substrates remain heterogeneous.
- Repeat consistency weights measure dispersion, not study truthfulness or clinical reliability.
- Scaffold/compound holdouts may share publications; consult the explicit overlap audit.
- Source holdouts purge cross-role structures; purging changes the evaluated population.
- Conformal intervals assume exchangeable calibration/test errors; grouped distribution shift can violate coverage.
- Bootstrap intervals describe held-out compounds conditional on this fitted split, not split-to-split uncertainty.
- No tautomer normalization, external validation, induction, exposure or drug-clearance inference.

Saved prediction rows: [test predictions](assay_source_test_predictions.csv).

Fitted bundles and complete row-level provenance remain in the local ignored experiment directory. Re-run the documented CLI to reproduce them in another checkout.
