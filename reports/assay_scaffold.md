# Retrospective ChEMBL compound-assay experiment

Endpoint: pIC50. These are held-out assay predictions, not clinical herb–drug predictions.

Split: scaffold; seed: 20261030. Model selection uses validation MAE.

| Target | Selected model | Test n | MAE | RMSE | R² | Interval coverage |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| ABCB1 | extra_trees | 155 | 0.495 | 0.658 | 0.507 | 0.955 |
| CYP1A2 | extra_trees | 251 | 0.584 | 0.756 | 0.221 | 0.936 |
| CYP2C19 | extra_trees | 217 | 0.474 | 0.600 | 0.129 | 0.935 |
| CYP2C9 | extra_trees | 358 | 0.468 | 0.607 | 0.242 | 0.874 |
| CYP2D6 | consistency_weighted_extra_trees | 375 | 0.513 | 0.666 | 0.322 | 0.944 |
| CYP3A4 | consistency_weighted_extra_trees | 760 | 0.482 | 0.641 | 0.350 | 0.880 |

Complete baseline results, confidence intervals and role counts are in [`assay_scaffold.json`](../results/assay_scaffold.json).

Cross-role publication/scaffold overlaps and all source-split purges are in [`assay_scaffold_split_audit.json`](../results/assay_scaffold_split_audit.json).

- Compound-target assay potency only; no whole-herb or clinical interaction prediction.
- Text screening is conservative and incomplete; assay protocols and substrates remain heterogeneous.
- Repeat consistency weights measure dispersion, not study truthfulness or clinical reliability.
- Scaffold/compound holdouts may share publications; consult the explicit overlap audit.
- Source holdouts purge cross-role structures; purging changes the evaluated population.
- Conformal intervals assume exchangeable calibration/test errors; grouped distribution shift can violate coverage.
- Bootstrap intervals describe held-out compounds conditional on this fitted split, not split-to-split uncertainty.
- No tautomer normalization, external validation, induction, exposure or drug-clearance inference.

Saved prediction rows: [test predictions](assay_scaffold_test_predictions.csv).

Fitted bundles and complete row-level provenance remain in the local ignored experiment directory. Re-run the documented CLI to reproduce them in another checkout.
