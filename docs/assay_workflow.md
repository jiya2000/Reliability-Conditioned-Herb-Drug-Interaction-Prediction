# Real-data compound-assay implementation

The official ChEMBL download enables a retrospective molecular assay experiment.
It does not complete the clinical herb–drug task. The new `src/assay/` package is
separate from the existing Task A/B release gates and historical results.

## Run from a fresh environment

```sh
python3 -m venv venv
venv/bin/python -m pip install -r requirements_assay.txt
venv/bin/python -m src.assay.experiment --dataset data/external/chembl_assays_20261030 --output artifacts/assay/scaffold --split scaffold
venv/bin/python -m src.assay.experiment --dataset data/external/chembl_assays_20261030 --output artifacts/assay/source --split source
venv/bin/python -m streamlit run app/interactive_app.py
```

Choose **Compound assay experiment** in the website. The assay view displays
measured test performance, baseline results and a compound-SMILES form. Training
runs through the CLI, never on website startup. The default presentation and
evidence explorer still work without a fitted assay model.

Output directories must be new. A run writes its completion report last; an
interrupted directory is not a completed experiment. Fitted bundles and row-level
outputs remain under ignored `artifacts/assay/`; summary reports can be retained
under `results/` and `reports/` without overwriting historical benchmarks.

## Scientific endpoint and screening

The regression target is **pIC50 = 9 − log10(IC50 in nM)**. A larger value means
greater potency in the reported assay. The model does not infer induction,
exposure, clinical severity, whole-herb composition, drug clearance or clinical
interaction probabilities.

Before processing, the workflow verifies the source manifest, captured-response
hashes and CSV hashes. It starts from the structurally screened candidate table.
It additionally requires an exact positive human-target IC50 in nM, assay type A
or B and inhibition wording in the assay description. Named nonhuman species,
unparseable structures and pChEMBL values inconsistent with the source quantity
are excluded with activity-level reasons. This deterministic text screen is an
operational assay definition, not an expert review of every assay. It can both
miss valid experiments and retain ambiguous ones. Source-assigned human targets
can have conflicting descriptions, which is why the extra screen exists.

The largest carbon-containing fragment is canonicalized without stereochemistry.
Equivalent SMILES, salt parents and stereoisomers share a structure group.
Tautomers and protonation states are not standardized. Fingerprints use RDKit
Morgan radius two and a fixed bit vector. Neither study text nor publication IDs,
compound names, observed pChEMBL, assay IDs or potency measurements enter the
molecular feature vector. Models are fitted separately for each target.

For each retained structure–target pair, repeated values are summarized first by
the median within each publication, then the median of publication medians.
All contributing activity/document/molecule IDs are saved. The range and median
absolute deviation expose conflict. Conflicting measurements remain visible;
their disagreement is not silently relabeled as a clinical consensus.

## Roles and separation

Groups are assigned with a fixed seed before aggregation. The prespecified roles
are train, validation, interval calibration and test. Their group fractions are
60%, 15%, 10% and 15%; sample proportions can differ because group sizes differ.
Targets with insufficient samples in any role are explicitly skipped.

- **Scaffold split** (primary): all compounds sharing a Bemis–Murcko scaffold
  stay in one role. Acyclic compounds share a single group. Both structures and
  scaffolds are disjoint. Publications may overlap and are counted in the audit.
- **Source split** (sensitivity): publications are assigned first. Structures
  spanning roles are retained only in the highest-priority role: test, then
  calibration, then validation, then train. Earlier-role measurements for that
  structure are purged. Thus publications and structures are both disjoint.
  All purges are logged. The resulting population differs from the scaffold run;
  the two experiments are not paired statistical replicates.
- **Compound split** (optional): canonical structures are disjoint; scaffolds
  and publications can overlap. Use `--split compound` for this less stringent
  diagnostic.

`record_roles.csv`, `samples.csv`, `purged_records.csv` and `split_audit.json`
make these claims inspectable. Role-overlap assertions fail the run if a required
separation is violated. Activity IDs and labels do not decide role assignment.

## Models, consistency weighting and uncertainty

Four fixed comparisons use identical molecular inputs and role assignments:
training-median baseline, ridge regression, ExtraTrees, and ExtraTrees with
repeat-consistency weights. The weighted variant uses inverse one-plus-MAD from
the **training pair's** measurements. It is an exploratory assay-consistency
ablation. It is not the proposed clinical agreement-supervised reliability model,
and no benefit is assumed. Repeats that are consistently biased can still receive
large weights. A single measurement receives unit weight without establishing
its scientific credibility.

The lowest validation MAE selects one fitted model per target; the fixed model
order resolves ties. There is no test-driven tuning, refit on calibration data or
test-driven interval selection. All model test scores remain in the report, even
when the selected model performs poorly. Regression metrics are MAE, RMSE and R²;
an accuracy percentage or clinical AUC is not manufactured for this endpoint.

The selected model's absolute residuals on its separate calibration role produce
a finite-sample split-conformal interval. Nominal coverage defaults to 90%.
Actual test coverage and width are reported. Distribution shift and correlated
assays can violate exchangeability, so nominal coverage is not a guarantee for
unseen scaffolds, publications or prospective molecules. The bootstrap MAE
interval resamples held-out canonical structures and retains their multiple
target predictions. It describes this fitted split, ignores residual correlation
between different structures in the same scaffold/publication, and does not
estimate variation from repeated retraining or prove statistical superiority.

Inference includes nearest-training Morgan Tanimoto similarity and whether the
exact canonical parent appeared in training. These are descriptive domain checks;
no validated similarity threshold is claimed. Predictions on training compounds
are labeled illustrations. Log-potency intervals are not probabilities and
are not converted to an interaction/no-interaction decision.

```sh
venv/bin/python -m src.assay.predict --experiment artifacts/assay/scaffold --smiles 'CC(=O)Oc1ccccc1C(=O)O'
```

The model is joblib/pickle based. Only load bundles generated locally or received
from a trusted publisher. Artifact hashes detect changes, not publisher identity.
There is no web upload path for model bundles.

## Outputs and verification

Each completed run saves the input-manifest hash, source-output hashes, exact
protocol, dependency versions, role assignments, exclusions, aggregates, baseline
and selected-model predictions, model bundle and report. Prediction rows include
the observed endpoint, model output and selected-model interval. Summaries are
derived from those rows. The report hashes every required experiment artifact.

The independent verifier recomputes metrics from saved predictions, checks their
correspondence to held-out samples, validates model selection and intervals, and
rechecks required group separation without fitting:

```sh
venv/bin/python -m src.assay.verify --experiment artifacts/assay/scaffold
venv/bin/python -m src.assay.verify --experiment artifacts/assay/source
```

The executed results are retained in `results/assay_scaffold.json` and
`results/assay_source.json`, with readable summaries in `reports/assay_scaffold.md`
and `reports/assay_source.md`. Publication-holdout performance is weaker and
several target-specific R² values are negative. Some observed interval coverages
fall below nominal. These results do not establish clinical predictive validity
or a benefit from consistency weighting. The website uses the full locally fitted
experiment directories.

```sh
PYTHONDONTWRITEBYTECODE=1 venv/bin/python -m pytest tests/test_assay_workflow.py tests/test_chembl_dataset.py tests/test_chembl_assay_review.py tests/test_project_presentation.py -q -p no:cacheprovider
```

## Remaining clinical requirements

The original aim still requires authenticated clinical study mappings, reviewed
herb constituent/composition/exposure data, independently curated clinical
reference labels, drug pharmacokinetic fractions and feasible independent role
groups. The original consensus file is empty and the original mechanism grid
contains no measured assay values. Those datasets remain unchanged. A downloaded
compound IC50 cannot supply the missing values or authorize clinical Task A/B.
See `docs/DATASET_SOURCE_AUDIT.md` and `docs/dataset_release.md`.
