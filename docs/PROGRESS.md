# Progress

## 29 October 2026 — First 24-pair annotation batch prepared

Prepared `annotation_batch_24_seed20261029/` without overwriting existing files.
Inspected the existing candidate CSV (120 unique pairs, 18 herbs, 40 drugs) and
the existing annotation CLI. Preserved all preceding working-tree changes and
historical artifacts. No application code or scientific decisions were changed.

Added `src/data/annotation_batch.py`: identifier-only coverage selection, seed
20261029, size 24. It reads only herb_id/drug_id, normalizes/deduplicates/sorts,
then greedily prioritizes unseen herbs, unseen drugs, and lowest combined selected
herb/drug counts, in that lexicographic order. SHA-256 of JSON [seed, herb, drug]
breaks ties. This is a reproducible coverage pilot, not proportional random
sampling, globally optimal joint coverage, class stratification or a validation
cohort. No study findings, labels, scores, predictions or legacy profiles are
consulted. The separate selection_manifest.json records the method/version,
seed, all selected identifiers, per-entity counts, pool/selection coverage and
identifier-only pool/output hashes (not hashes of outcome-bearing columns).

Selected coverage: **24 unique pairs, all 18 herbs and 23 of 40 drugs**.
`selected_pairs.csv` contains identifiers only. The existing CLI initialized
`annotation_batch_24_seed20261029/review_assignments/`, including its own frozen
pairs/manifest plus reviewer_a.csv, reviewer_b.csv and adjudication.csv.
All 24 decisions in each workbook remain unresolved; all reviewer, source and
attestation fields remain blank. No consensus export or canonical promotion.

Exact commands run from the repository root (new directories required):

```sh
venv/bin/python -m pytest tests/test_annotation_batch.py -q -p no:cacheprovider
venv/bin/python -m src.data.annotation_batch --pairs data/benchmark/consensus_labels_pairs_to_label.csv --output annotation_batch_24_seed20261029 --size 24 --seed 20261029
venv/bin/python -m src.data.consensus_annotation init --pairs annotation_batch_24_seed20261029/selected_pairs.csv --output annotation_batch_24_seed20261029/review_assignments
```

Focused tests: **6 passed**, exit 0. Cover seeded reproducibility, row-order
invariance, identifier membership/coverage, duplicate and outcome/score/prediction
perturbation invariance (including the manifest), overwrite refusal, invalid
sizes, and real-pool identifier-only initialization with blank unresolved reviews.
Post-generation read-only checks confirmed the actual frozen assignment matches
the selected 24 pairs and all three workbooks remain unresolved with blank metadata.
`git diff --check` passed. No full suite, Colab, training or model evaluation run.

Human next steps: coordinator retains both manifests and distributes only each
reviewer's own workbook and independent external-reference access. Two qualified
reviewers work independently, blinded to project findings, predictions and one
another; document actual compendium/version/section/citation/access date, clinical
scope, rationale, primary-source dependencies and circularity. Lock original
reviews before a separate third reviewer adjudicates every resolved decision.
Missing evidence remains unresolved, never an inferred negative; reviewer
disagreement alone does not establish disputed status. All 24 pairs are assigned
double review (20% of the original 120 only once actually completed). Preserve
pre-adjudication decisions, prespecify unresolved handling for agreement analysis,
and require human source authentication/data-steward review before any promotion.
No scientific validity is established. Stop after batch preparation.

## 28 October 2026 — Mechanistic schema-to-feature boundary

Completed only the scoped extraction repair. No model, evaluation, Streamlit, real dataset, or existing test files were changed.

Changed files:
- `src/data/mechanistic_curation.py`: removed access to nonexistent `value`; validates actual measurement fields, ranges, statuses, entity roles and duplicate rows; maps inhibition/induction polarity and drug weights; preserves unknown pathways as NaN; retains ten-column float32 output and blocks unverified data even with the legacy bypass flag.
- `tests/test_mechanistic_boundary.py`: added 36 synthetic-fixture cases covering extraction, ordering, summary, polarity, missingness, required columns, invalid values/statuses/roles, duplicates, empty inputs, and provenance guards against the unchanged real table.
- `docs/mechanistic_features.md`: documented mappings, target-activity sign convention, missingness, provenance limits and unsupported clinical interpretations.
- `docs/PROGRESS.md`: this phase record.

Verification:
- Command: `venv/bin/python -m pytest tests/test_mechanistic_boundary.py tests/test_phase3_and_ci_checks.py -q -p no:cacheprovider`
- Result: **46 passed, zero failed, one warning** (joblib could not determine physical core count and used logical cores). Includes all ten existing Phase 3 checks without editing them.
- `git diff --check`: passed.
- Full suite was not rerun; prior audit failures are not claimed resolved.
- Historical audit run (before this repair): **195 passed, five failed, 22 warnings**. Two failing checks concerned a stored non-reference benchmark score above the permitted threshold and an external conformal-set expectation (four observed versus five expected). Three failures were sandbox restrictions on repository artifact writes in external validation, novelty ablation, and benchmark execution. These remain unresolved/unretested by this phase.
- Summary follow-up: reviewed the phase record and working-tree scope; `git diff --check` passed again. No tests were rerun during this documentation-only follow-up, and no next-phase work was started.

Mapping limits and remaining blockers:
- Signed output describes target activity: inhibition negative, induction positive. Potency establishes direction only; no unsupported strength or clinical exposure conversion is inferred.
- Unknown pathways and their absolute-sum summary remain NaN, except a known zero factor establishes zero contribution. Downstream missingness handling remains a separate phase; no imputation or consumer changes were introduced.
- All real mechanistic rows remain subject to provenance blocking. Source/citation strings are checked structurally, not independently authenticated. Synthetic fixture markers are test-only and never inserted into scientific data.
- Verified per-value citations, clinical context, and actual consensus labels are still required before scientific prediction/evaluation. The schema lacks enough compartment/free-exposure information for reachability and has no NTI field.
- The duplicate aggregation scaffolds, clean Task A/B training, calibration isolation, broader evaluation failures and application integration remain outside this phase.

Stopped after the boundary repair.

## 28 October 2026 — Consensus-label contract and safeguards

Completed only the consensus data-contract phase. The pre-existing mechanistic extraction changes were preserved. No model architecture, calibration, Streamlit, clinical dataset, label, or citation was changed.

Implemented:
- `src/data/evidence_schema.py`: one shared consensus validator requires role-correct `HERB_`/`DRUG_` identifiers, one of `interaction`/`no_interaction`/`disputed`, source name, citation and ISO retrieval date. Rejects missing/null/placeholder required values, duplicate columns and duplicate pairs (including whitespace-equivalent IDs). Optional severity, direction and notes may remain missing.
- Citation checks require non-placeholder text, validate supplied HTTP(S) URLs and explicit DOI/PMID syntax, and permit bibliographic references for compendia. These structural checks do not authenticate sources, establish external independence, or prove blinded annotation; those remain human curation requirements.
- A schema-bearing header-only table is valid for inspection. `load_consensus_labels_table(..., require_labels=True)` raises `ConsensusLabelsNotPopulatedError` for zero rows. Missing/headerless files and partially labeled/invalid tables raise rather than triggering fallback or silently dropping rows. The retained tuple boolean means labels are unavailable, never permission to substitute publication outcomes.
- Both existing aggregation-module label loaders now use the shared validator with `require_labels=True`; the existing training entrypoint therefore remains blocked on the actual empty consensus table. This is a loader safeguard, not implementation of Task A/B training or evaluation.
- `tests/test_consensus_contract.py`: 74 synthetic cases cover allowed verdicts, invalid identifiers/labels/citations/dates, missing values/provenance/columns, duplicate pairs/columns, normalization, empty-table inspection, empty-label blocking in both loaders and the training entrypoint, and rejection of unlabeled candidate rows. Fixture provenance is explicitly synthetic and is never written into real data. Existing schema test now supplies the required retrieval date and describes empty-table behavior without implying legacy fallback.

Verification:
- Initial focused run: **92 passed**.
- Final command: `venv/bin/python -m pytest tests/test_consensus_contract.py tests/test_evidence_schema.py tests/test_phase4_scaffolding.py tests/test_task_extensions.py tests/test_mechanistic_boundary.py tests/test_phase3_and_ci_checks.py -q -p no:cacheprovider`
- Final result: **144 passed, zero failed, one warning** (joblib physical-core detection fallback).
- `git diff --check`: passed; working-tree scope inspected. No dataset files changed. Full suite was not rerun; the audit's two unrelated failures are not claimed resolved.

Real consensus curation remains empty and blocked for consensus-dependent training/evaluation. No unstudied pair was assigned a negative label. Stopped after this phase.

## 28 October 2026 — Evaluation-integrity phase

Scope: investigated the two audit failures before editing implementation, inspected the primary and legacy feature routes, repaired verified evaluation defects, and stopped at this phase. No Streamlit, model architecture, clinical data, consensus target, citation, or historical result was changed. No applicable `AGENTS.md` exists in this checkout or its parent repository search; read `PROJECT_SPEC.md`, `docs/IMPLEMENTATION_AUDIT.md`, and this progress log. Initial Git status contained the preceding mechanistic/consensus work; those changes were preserved.

### Reproduction and root causes

- Before edits, `venv/bin/python -m pytest tests/test_ablation_leakage.py tests/test_external_validation_overlap.py -q -p no:cacheprovider`: **23 passed, two failed, two warnings**. The exact failures were `test_no_actual_benchmark_model_exceeds_threshold` (stored `no_context_conflict` AUC 1.0) and `test_external_validation_execution_and_artifacts` (four two-class sets versus a fixed expectation of five).
- High-AUC guard: independently recomputed ROC-AUC from the 26 saved prediction rows in `results/verified_hdi_benchmark_results.json`: **1.0**, matching the stored metric. Maximum negative probability was `0.5630831122398376`; minimum positive was `0.8317841291427612`. Thus this is not an AUC arithmetic defect. The guard correctly rejects a historical result presented as non-reference despite a legacy study-outcome evaluation using raw outcome-bearing abstracts and entity-specific, explicitly label-adjacent profiles. `no_context_conflict` removes metadata context/conflict, but retains those text/profile paths. The artifact has no complete code/data/config checkpoint manifest; a unique causal attribution of the perfect ranking to one input is **unresolved**, and is not inferred from the guard alone. Historical scores and the threshold/assertion remain untouched; no relabeling of the stored model to evade the guard.
- External mismatch: the initial seeded execution fitted `q_hat=0.7765`, not the narrative's fixed `0.8148`, and produced four two-class sets. The test encoded an old observed count and the implementation asserted clinical appropriateness independent of actual predictions. Verified a separate layout defect: a 7D quality vector was padded to 12D, then interpreted by the model as full metadata, moving source-quality/context/missingness/PK inputs to incorrect positions. Diagnostics also used rounded probabilities and a fabricated constant epistemic value. No threshold was tuned to recover five sets.
- Additional verified outcome leakage: contradiction is computed from positive/negative finding masses, yet appeared in `quality_only`. Echo ratio also depended on separately clustered positive/negative findings, so changing a finding could change an allegedly structural feature. These defects are fixed and tested independently of the historical high-AUC result; they are not claimed to explain that variant's perfect ranking.
- Calibration contamination: repeated nested CV first calibrated each fold on inner validation data, then fitted another temperature on pooled **outer-test labels** before scoring those same rows. Removed the second fit; scoring now pools the already fold-calibrated probabilities. Calibration fits remain wholly inside each outer training partition. This uses existing inner holdouts, not a newly implemented pooled inner-OOF scheme.

### Feature-path inspection

| Path | Verified inputs and remaining interpretation |
|---|---|
| Curated Task A extraction (`mechanistic_curation.py`, `feature_switched_models.py`) | Entity IDs are lookup keys, not emitted one-hot/string features. Signed per-target numeric values are emitted without abstract text/outcome columns. Actual curated rows remain blocked as unverified. `FeatureSwitchedEvaluator.run_cv` still uses publication outcomes, so it is not completed consensus evaluation. |
| Task B scaffolds (`aggregation.py`, `aggregation_model.py`, `extract_study_design_fields`) | Reliability head accepts separate design tensors; finding tensors enter the evidence delta, as permitted for Task B. Design extraction selects species, assay, route, preparation, dose, population and standardization rather than outcome/text/IDs. Generic tensor APIs cannot authenticate caller-supplied feature semantics; no complete primary training path exists. Consensus loaders continue to block empty labels. |
| Shared lineage/quality inputs (`evidence_lineage.py`, `reliability_scorer.py`) | Removed outcome-dependent conflict from quality vectors and deterministic/MC reliability input layouts. Kept its slot as reserved zero for tensor compatibility. Citation echo now clusters all records independently of findings. Quality/context/missingness use design metadata; current reachability/fraction fields in lineage summaries default to 0.5 and do not establish sourced PK values. Findings/conflict remain available separately for Task B aggregation, abstention, and explicit full-vector reference diagnostics. |
| Legacy neural evaluation (`run_verified_hdi_benchmark.py`, `evaluation.py`, `external_validation.py`, `interaction_model.py`) | TF-IDF/BiomedBERT can consume raw abstracts containing findings and entity names even when metadata is called quality-only. Training-only vocabulary/projection fitting does not remove that shortcut. Profile/GAT branches consume entity-specific legacy profiles; graph-only bypasses text but retains label-adjacent profiles. Concat/fusion/evidence modes retain raw text. New benchmark/external results explicitly identify these runs as legacy exploratory and invalid for Task A/B; nested CV docstring states the same. No clean replacement model was introduced. |
| Legacy tabular, graph and shortcut comparisons (`extract_tabular_features`, `biological_features.py`, `biological_graph_encoder.py`, `scripts/run_phase5_task_a.py`) | Tabular inputs use numeric overlap plus quality metadata; profiles contain outcome-adjacent choices such as negative-control annotations, and cannot establish independently curated mechanisms. Entity-specific profiles may permit memorization without one-hot IDs. Phase 5 script explicitly includes identity/raw-text/masked-text diagnostics under legacy exploratory status. Masking does not by itself certify removal of all proxies. No architecture or diagnostic scores were modified. |

Entity/citation separation remains dependent on the selected splitter: study-disjoint does not establish herb/drug/pair disjointness, and pair-disjoint does not jointly group citation components. External template studies have zero study/source overlap but four training-pair overlaps. These legacy runs do not establish cold-start performance or clean primary-model validity.

### Changed files in this phase

- `src/data/evidence_lineage.py`: outcome-independent echo ratio; reserved zero conflict slot in quality vectors, preserving separately available conflict/evidence summaries.
- `src/models/reliability_scorer.py`: defense against conflict entering quality-only reliability through 7D/8D/10D/12D tensors in deterministic and MC paths.
- `src/training/evaluation.py`: removed outer-label temperature refit and obsolete pooled logits; pools fold-calibrated probabilities and retains per-fold temperature summaries, without a misleading `pooled_t` field.
- `benchmarks/external_validation.py`: correct 7D layout; separate contradiction input for abstention; full-precision records; fitted-calibrator requirement for diagnostics; missing epistemic estimate represented as null; measured narrative without clinical safety/coverage claims; explicit legacy evaluation status.
- `benchmarks/run_verified_hdi_benchmark.py`: explicit legacy exploratory evaluation status on newly generated output.
- `tests/test_evaluation_integrity.py`: seven regression cases covering finding/text perturbation invariance, citation clustering, four reliability tensor layouts (deterministic and seeded MC paths), actual conformal sets/missing uncertainty, and instrumentation proving exactly the inner-validation calibration fits and unchanged pooled fold probabilities.
- `tests/test_external_validation_overlap.py`: assert every actual metadata tensor is 7D; independently derive each expected conformal set from fitted threshold/full-precision probabilities and verify aggregate counts, rather than requiring five or changing thresholds.
- `tests/test_ablation_leakage.py`, `tests/test_phase1_leakage_control.py`: now require exclusion of outcome-derived conflict, including the former undiscounted-conflict path. High-AUC guard is unchanged.
- `tests/test_evaluation_overhaul.py`: require absence of assessment-label `pooled_t` refit metadata.
- `tests/test_end_to_end_and_uncertainty.py`: retained all legacy contradiction-heuristic assertions under explicit full-vector reference inputs. The previous test treated a manually added conflict penalty as primary epistemic validation; quality-only outcome invariance is now tested separately with identical MC randomness.
- `tests/test_real_hdi_benchmark.py`: redirect benchmark artifact root to `tmp_path`; execution assertions are unchanged. The first full run exposed writes to historical reports/results; those test-generated changes were restored byte-for-byte from their initially clean Git versions before final verification.
- `docs/PROGRESS.md`: this phase record.

### Verification and blockers

- Intermediate full run: **315 passed, two failed, 22 warnings**. Besides the preserved high-AUC guard, the legacy uncertainty test required outcome-dependent conflict inflation under default quality-only inputs; corrected its reference-only scope as described above.
- Focused follow-up across evaluation-integrity, end-to-end/uncertainty, ablation leakage, external overlap, evaluation overhaul, quality partitioning and interaction/calibration tests: **53 passed, one failed, two warnings**. Only the unchanged historical high-AUC guard failed.
- Final full-suite command: `venv/bin/python -m pytest -q -p no:cacheprovider`: **316 passed, one failed, 22 warnings**, 317 collected, 36.91 seconds. Sole failure: unchanged `tests/test_ablation_leakage.py::test_no_actual_benchmark_model_exceeds_threshold`, historical `no_context_conflict` AUC 1.0. External overlap/abstention checks and all seven new integrity cases pass. Warnings: 18 TorchScript deprecations, two intentional suspicious-AUC fixture warnings, one tokenizer deprecation, one joblib core-detection fallback.
- Final full run exposed that benchmark `output_json`/`output_md` are relative to cwd even when the prediction/mirror root is patched. Explicit temporary output arguments were then added to the benchmark test, and its two report writes were restored from initially clean Git versions. Reran that execution test file separately (result recorded below); implementation was unchanged after the full suite.

Remaining blockers: the historical high-AUC alert is intentionally unresolved, not skipped or weakened. A fresh valid scientific replacement requires populated independent consensus labels and verified mechanisms; the current empty/unverified tables remain blocked. Raw text and entity-specific legacy profiles still disqualify legacy neural evaluations as primary results. Joint pair/citation grouping, pooled inner-OOF calibration, separate temperature/conformal calibration partitions, dependence-aware intervals, complete clean Task A/B training, and external clinical validation remain unimplemented. Reusing source validation data for temperature and conformal fitting does not establish formal split-conformal coverage; no such guarantee is claimed by this repair. No new scientific performance result is asserted. Stopped after this phase.

Final artifact-isolation follow-up: `venv/bin/python -m pytest tests/test_real_hdi_benchmark.py -q -p no:cacheprovider`: **two passed, zero failed, one joblib warning**, 16.12 seconds. `git diff --check` passed. Final Git inspection confirms no historical `reports/` or `results/` artifacts or scientific data changed; pre-existing consensus/mechanistic edits are preserved. No full-suite rerun after this test-output-only adjustment.

## 28 October 2026 — Historical AUC guard forensic investigation

Investigated only the remaining historical AUC=1.0 failure. Read this log and inspected Git status before work; all preceding uncommitted changes were preserved. **Only this progress document changed in this investigation.** No implementation, test, historical score, scientific data, or architecture was changed.

### Conclusion

**Confirmed genuine held-out outcome-feature leakage in the historical model.** This is neither the synthetic sentinel fixture nor an invalid expectation that every possible model must score below 0.98. The guard is an integrity alert, and this particular alert is justified. The source path below establishes outcome leakage. The earlier numerical replay/intervention claims were subsequently withdrawn where their diagnostic artifacts could not be recovered; see the 29 October review. No test-expectation change is proposed.

### Exact test, artifact and provenance

- `tests/test_ablation_leakage.py::test_no_actual_benchmark_model_exceeds_threshold` has no fixture arguments. It reads `results/verified_hdi_benchmark_results.json` (falling back to the report only if absent), iterates stored reports, and fails on `no_context_conflict`, AUC 1.0. It does not instantiate a model or consume the nearby `sentinel_lineage_summary` synthetic fixture. That sentinel and the separate perfect-separation warning fixtures are unrelated to this failure.
- The artifact was introduced by commit **`1daf3d39831882f6fbb6038ecaac3e05a1be7391`**, identified with `git log -1 -- results/verified_hdi_benchmark_results.json`. Artifact SHA-256: `67e072a5fc06c4fa2b138a51522b300bb53685b6aba24acd3f9cc35f546c2f60`.
- The earlier investigation reported an exact training replay and frozen-model interventions, but its original machine-readable trace and frozen weights are not available in the repository or the searched temporary diagnostic locations. The retained instrumentation script alone cannot authenticate those outputs. The numerical replay claims are withdrawn; this review does not rerun training to recreate them.
- A read-only check on 29 October recomputed ROC-AUC **1.0** from the artifact's **26 saved predictions**. Historical source inspection independently establishes the outcome-feature path below. These checks validate the alert without claiming to have reproduced historical model weights.

### Direct feature-to-label path

Inspection of the original commit, rather than inference from the present implementation, establishes this chain:

1. `benchmarks/run_verified_hdi_benchmark.py` separately calls `analyze_dataset(test_df)` and builds each held-out row's metadata from the resulting pair summary via `to_full_metadata_vector()`. Its targets come from the same rows' `outcome` field.
2. `src/data/evidence_lineage.py` partitions records by `outcome` to compute positive/negative evidence masses and raw finding counts. The full 12D vector contains `S_pos` at column 0, `S_neg` at 1, net margin at 6, raw positive count at 8, and raw negative count at 9. Thus held-out findings feed a model that is scored for predicting those findings, rather than independent consensus truth.
3. Historical `ReliabilityConditionedInteractionModel.__init__` selects `quality_only` only for its two named reliability modes; the `no_context_conflict` mode falls through to **`feature_group='full'`**. Its ablation sets context column 4 to 1, conflict column 5 to 0, and **copies positive evidence strength column 0 into column 6**. It never removes columns 0/1/8/9.
4. Historical `ReliabilityScorer._embed_metadata` sends all 12 columns to `full_lineage_proj`, and `forward_with_uncertainty` produces the gate R. That gate controls `z_fused = (1-R)*z_graph + R*z_contextual`, which reaches the prediction head. Consequently label leakage can travel through the gate even though the final classifier does not concatenate raw polarity columns directly.

Evidence on the actual 26 held-out rows (12 scored positive, 14 scored negative):

- Raw positive count: every scored negative has **0**; every positive has **1 or 2**. This single leaked feature has **AUC 1.0**.
- Positive evidence strength is explicitly derived from outcome-partitioned studies in the historical source. The previously quoted numerical ranges and feature-level score lack a retained diagnostic artifact and are withdrawn.
- One scored negative is `STUDY_ANDERSON_2003`, whose publication outcome is `conflicting`: the legacy benchmark creates target 0.5, then historical metrics cast it to integer 0. This is another limitation of this historical binary assessment, not evidence of an external consensus verdict. The replay deliberately retained this casting to match the saved result; no label or metric was corrected during this investigation.

### Withdrawn numerical intervention claims

The earlier table of frozen-weight intervention scores, reliability means and maximum logit changes has been removed because no retained machine-readable diagnostic output or frozen-weight artifact can substantiate it. Some quoted values happened to match unrelated numbers in result JSON; numerical membership is not provenance. No diagnostic result has been inserted into historical JSON to make the documentation test pass. The source-level outcome leakage and the recomputed saved ranking above remain independently checkable; a unique quantitative causal decomposition is not claimed.

### Current implementation and decision

The current `no_context_conflict` default already selects `quality_only`, constructs a five-column vector `[echo_ratio, mean_source_quality, missingness, rho_reach, f_m_CYP]`, and strips polarity from legacy full tensors. These safeguards predate this investigation (present in checkout commit `daf2093`); the prior evaluation-integrity phase further removed outcome-derived conflict and fixed structural echo computation. There is no reason to reintroduce or redundantly patch the historical defect in current code.

Checked the current model separately with actual held-out full metadata and fixed random representation tensors: swapping the same positive/negative columns with identical MC randomness produces **maximum absolute logit difference 0.0**; `feature_group` is `quality_only` and the mode's summary vector width is **5**. This is an input-isolation check, not a fresh clinical performance evaluation. The saved historical file was not regenerated by these fixes, so its guard correctly continues to fail. Raw text/profile validity remains outside this narrow follow-up and is not claimed resolved.

### Actual verification

- Initial targeted reproduction: `venv/bin/python -m pytest tests/test_ablation_leakage.py::test_no_actual_benchmark_model_exceeds_threshold -q -p no:cacheprovider`: **one failed**, 2.05 seconds; expected historical `no_context_conflict` AUC=1.0 alert.
- After investigation: `venv/bin/python -m pytest tests/test_ablation_leakage.py -q -p no:cacheprovider`: **22 passed, one failed, two warnings**, 1.45 seconds. Sole failure is the same unchanged historical guard; warnings are the existing intentional perfect-ranking warning fixtures.
- Archived replay and frozen interventions ran in temporary storage only; exact replay and current-isolation results are recorded above. No full suite was run for this documentation-only investigation.

Disposition: retain the historical artifact and alert unchanged. Replacing the artifact requires a separately authorized valid evaluation with independent targets and clean inputs; deleting it, renaming its variant to bypass the test, or raising the threshold would conceal the confirmed defect. No such change was made. Stopped after this investigation.

## 28 October 2026 — Primary Task A feature-contract audit and safeguards

Read `PROJECT_SPEC.md`, `docs/IMPLEMENTATION_AUDIT.md` and this progress log; inspected Git status. Preserved all preceding uncommitted work, the historical AUC result and its guard. Implemented only feature-contract safeguards; no new model, scientific dataset, source/citation, Streamlit or historical report/result changed.

Audit finding: there is still **no complete primary consensus-supervised Task A trainer**. The nearest candidate, `FeatureSwitchedBenchmark(feature_set='new')`, previously emitted an unnamed ten-column curated matrix, then would fit publication outcomes and also run a GAT that ignores that matrix and uses legacy entity profiles. The ordinary legacy/Phase 5 diagnostic paths combine profiles, quality/design/lineage, identity or raw/masked text and are not approved primary models. The earlier provenance guard prevented actual unverified data from reaching this latent fallthrough, but that guard alone was insufficient once curation becomes populated.

Implemented contract:

- Explicit primary allowlist: `signed_effect_CYP3A4`, `signed_effect_CYP2C9`, `signed_effect_CYP2C19`, `signed_effect_CYP2D6`, `signed_effect_CYP1A2`, `signed_effect_ABCB1`, `sum_abs_signed_effect`. Every admitted column is **mechanistic**, following the six pathways in the spec's pair-feature equation. Signs retain the existing target-activity convention; no clinical exposure mapping or strength is invented.
- Raw-to-feature builder projects only `herb_id`/`drug_id` lookup keys into the existing provenance-checked extractor, selects those six target effects and recomputes their absolute sum. ID strings, raw study/text columns and caller-supplied feature values are not predictors. Identically named raw columns cannot override extracted measurements. Consistent key renaming without measurement changes leaves X unchanged. The general nine-target/ten-column extractor remains unchanged; CYP2E1/SLCO1B1/NR1I2 cannot enter indirectly through its general summary.
- Named DataFrame checks run before scaling and immediately before estimator fitting in a reusable pipeline around the **existing** LR/HGB classes. Extras are rejected rather than selected by dtype or silently dropped. Checks reject encoded identity/text/outcome/design/lineage columns, unknown aliases, duplicate or missing names, unnamed arrays, non-real/non-numeric values, infinities and missing mechanisms. Fixed name-preserving scaling fits only the supplied training partition. No imputation, automatic indicators, text/categorical encoder or remainder passthrough is introduced.
- Curated `get_feature_matrix` now returns the named allowlisted matrix. Curated `run_cv` retains the unverified-data failure first, then explicitly blocks before publication-target access or legacy LR/HGB/GAT fitting, even with structurally verified synthetic measurements. The legacy exploratory branch is preserved. Completing consensus-supervised training is outside this phase; the guarded pipeline does not itself load/validate target provenance.

Changed files in this phase:

- `src/features/task_a_contract.py` (new): explicit target/feature tuples, raw-to-named-feature builder, strict sklearn-compatible transformer and guarded preprocessing pipeline factory for existing LR/HGB estimators.
- `src/models/feature_switched_models.py`: use the contract for curated features; block curated-to-legacy target/GAT fallthrough; document the restricted route. No legacy model architecture or calculation changed.
- `tests/test_task_a_feature_contract.py` (new): **69 synthetic regression cases**, including 48 prohibited raw/encoded/derived column names; pre-scaler rejection; appended/unnamed preprocessor-output rejection before estimator fit; upstream `ColumnTransformer(remainder='passthrough')` rejection; actual LR/HGB `.fit` inspection; prediction-time checks; raw-column perturbation/approved-name spoofing immunity in the builder; identity-key renaming invariance; canonical ordering and six-target sum; train-only scaling; real-data provenance/missingness blocking; and refusal to train curated X on legacy publication targets/GAT.
- `docs/task_a_feature_contract.md` (new): complete raw measurement → extraction → projection → scaling → fitting trace, per-feature classification, exclusion inventory for mechanistic/study-design/lineage/identity/outcome-text families, and explicit limits. OATP1B1 is in the spec's knowledge table but not its six-pathway pair equation; it and other future sourced features require a later explicit contract extension, not automatic admission. No new max/NTI/reachability/dose feature was invented.
- `docs/PROGRESS.md`: this phase record.

Verification:

- Initial new-test development run: **65 passed, four failed, one warning**. Four new fault-injection tests initially patched an internal scaler method whose result sklearn automatically wraps/renames; the tests now inject at the public preprocessing output boundary (`fit_transform`) seen by the following contract. This tests actual boundary admission, including unnamed arrays, rather than sklearn's internal representation. No existing test was relaxed to accommodate this change.
- `venv/bin/python -m pytest tests/test_task_a_feature_contract.py tests/test_mechanistic_boundary.py tests/test_phase3_and_ci_checks.py -q -p no:cacheprovider`: **115 passed, zero failed, one joblib warning**, 4.07 seconds. All 69 new contract cases pass; all existing mechanistic/Phase 3 checks pass unchanged.
- Final focused command: `venv/bin/python -m pytest tests/test_task_a_feature_contract.py tests/test_mechanistic_boundary.py tests/test_phase3_and_ci_checks.py tests/test_evaluation_integrity.py tests/test_ablation_leakage.py -q -p no:cacheprovider`: **144 passed, one failed, three warnings**, 145 collected, 5.61 seconds. Sole failure is the unchanged `test_no_actual_benchmark_model_exceeds_threshold` for historical `no_context_conflict` AUC=1.0. Warnings are one joblib physical-core fallback and two intentional high-AUC warning fixtures. The earlier forensic conclusion and stored result are preserved; this phase does not claim a green historical guard.
- `git diff --check` passed. Scientific data and `reports/`/`results/` remain unchanged. No full-suite run or new scientific performance evaluation was performed.

Remaining blockers and limits:

- Actual consensus targets are empty and actual mechanisms remain unverified; neither is fabricated or bypassed. The allowlist certifies feature categories/shape, not source authenticity, blinded annotation, or clinical validity.
- There is no complete consensus-supervised pair trainer, approved missingness policy, or validated primary performance result. Unknown mechanisms remain NaN for inspection and fail training explicitly. Primary training with independent labels, complete PK/NTI/tissue features and grouped assessment remains later work.
- Column checks cannot detect deliberate replacement of a measurement with a label vector under the same approved name, or authenticate arbitrary third-party preprocessing. The supported raw builder reconstructs values from the guarded curation table; the supported pipeline uses fixed scaling. Removing the guards, invoking bare estimators directly or deliberately falsifying provenance is outside the contract, not a supported bypass.
- All study-design/lineage inputs, including otherwise valid quality descriptors, remain excluded from primary Task A X. They may belong to separately audited Task B reliability/evidence inputs. Existing exploratory diagnostics are not reclassified as primary.

Stopped after this phase.

## 28 October 2026 — Final local feature-contract verification

Read this log and inspected Git status before verification. Preserved all pre-existing uncommitted work. No new regression was found, so no code or tests were changed. Only this progress log was updated.

Ran the three requested groups in separate pytest processes, using the existing local `venv` (Python 3.11.16, pytest 9.1.1):

| Exact command from repository root | Collected / passed / failed | Warnings | Duration |
|---|---|---|---|
| `venv/bin/python -m pytest tests/test_task_a_feature_contract.py -q -p no:cacheprovider` | 69 / **69** / 0 | 1 joblib physical-core detection fallback | 2.36 s |
| `venv/bin/python -m pytest tests/test_mechanistic_boundary.py -q -p no:cacheprovider` | 36 / **36** / 0 | 0 | 1.60 s |
| `venv/bin/python -m pytest tests/test_consensus_contract.py -q -p no:cacheprovider` | 74 / **74** / 0 | 0 | 2.12 s |

All **179** requested feature/mechanistic/consensus regression cases passed. These tests verify contracts and safeguards, not scientific measurement authenticity or primary-model performance. Actual consensus labels remain empty, mechanisms remain unverified, and primary scientific training remains blocked as documented above.

Historical leakage alert preserved:

- Compared the exact source text of `test_no_actual_benchmark_model_exceeds_threshold` with `git show HEAD:tests/test_ablation_leakage.py`: **identical**, including threshold **0.98**. The existing unrelated modification elsewhere in that test file predates this verification; the historical guard itself is unchanged.
- Historical result SHA-256 remains **`67e072a5fc06c4fa2b138a51522b300bb53685b6aba24acd3f9cc35f546c2f60`**, matching the forensic investigation. Stored `no_context_conflict` ROC-AUC remains **1.0**.
- Command: `venv/bin/python -m pytest tests/test_ablation_leakage.py::test_no_actual_benchmark_model_exceeds_threshold -q -p no:cacheprovider`.
- Result: **one collected, one failed, zero warnings**, 1.21 seconds, exit code 1. This is the retained **historical leakage alert** for the previously confirmed outcome-count/strength leakage, not a new feature-contract regression. It was not skipped, marked xfail, renamed, weakened or supplied a rewritten result.

Exact command for a clean full-suite invocation in a fresh process from the repository root, with pytest's cache provider disabled and no exclusions:

```sh
venv/bin/python -m pytest -q -p no:cacheprovider
```

The full suite was **not run** during this final verification, and no current full-suite count is claimed. The command deliberately retains the historical guard and therefore should still report its failure against the preserved historical artifact; “clean” does not mean suppressing that alert or rebuilding scientific data.

Final checks: `git diff --check` passed; `git diff --exit-code -- reports results data` passed. Git status was reinspected; no new code/test/data/result changes were introduced by verification. Stopped after verification.

## 28 October 2026 — Colab verification guide

Read this log, inspected Git status, CI/packaging/dependency configuration, test imports, environment lookups and file prerequisites. Added only `docs/COLAB_VERIFICATION.md` and this progress entry; preserved all pre-existing work. No application code, tests, datasets, model behavior or historical artifacts changed.

The guide covers a reviewed working-tree snapshot (including current uncommitted safeguards), exclusion of secrets/Git metadata/environments/caches/unnecessary outputs, retention of test-consumed historical fixtures, and a new local-only Git index for the test that invokes `git ls-files`. It explicitly identifies the Git-ignored `data/benchmark/private/pair_strata.csv` prerequisite for separate authorized transfer; absence must be reported, not patched with dummy data or hidden by skipping tests. Required processed JSON and audit/sensitivity images are listed.

Setup is grounded in `pyproject.toml` (Python >=3.10), CI (Python 3.11, CPU Torch/torchvision, core requirements, pytest/coverage, editable install), and the additional NetworkX test import dependency. The guide explains that the broad requirements file and minimal uv lock are not an exact environment lock, lists persistent notebook environment settings, and documents the public BiomedBERT download/fallback path without requiring secret credentials. It provides the exact full-suite command `venv/bin/python -m pytest -q -p no:cacheprovider`, Colab invocation, log/version capture with pytest's preserved pipeline exit code, and comparison of the exact known historical alert against all additional failures/errors/skips.

Documentation-only validation: reviewed commands and paths against repository sources; `git diff --check` passed and historical `reports/`, `results/`, and scientific data remain unchanged. No dependency installation, archive transfer, Colab execution or test-suite run was performed for this guide; no fresh-environment success is claimed. Stopped after documentation.

## 28 October 2026 — Latest verification summary

Latest verified local results (recorded from the separate runs above; no tests rerun for this update):

- Task A feature contract: **69 passed**, zero failed.
- Mechanistic boundary: **36 passed**, zero failed.
- Consensus contract: **74 passed**, zero failed.
- Historical AUC=1.0 guard: **still fails** for the preserved `no_context_conflict` result exceeding the unchanged 0.98 threshold. This remains the confirmed historical leakage alert, not a new contract regression.

Updated only `docs/PROGRESS.md`; no code, tests, data or historical results changed. Stopped after this documentation update.

## 28 October 2026 — Consensus-supervised Task A runner

Implemented one functional infrastructure phase after reading PROJECT_SPEC.md, this log and Git status. No AGENTS.md was found under the parent directory. Preserved all pre-existing working-tree edits.

- Added `src/training/task_a.py`, a runnable consensus-only LR/HGB baseline CV entry point. Reuses the shared consensus loader, curated feature builder and guarded preprocessing pipeline. Consensus rows define the pair universe; publication records/outcomes are never loaded. Disputed rows are counted and excluded from binary fitting/scoring.
- Added seeded repeated stratified grouped folds for herb-, drug- and pair-disjoint Task A evaluation (defaults: five folds, five repeats). Validates every fold before fitting; insufficient groups or single-class train/test folds block instead of being silently skipped. Scaling is fit separately within each fold. Pair grouping here has no literature inputs and does not implement Task B citation-cluster splitting.
- Added `--check-only` to validate data/features/folds without fitting. Successful CV emits uncalibrated per-fold AUC/Brier and pair-level OOF predictions as JSON; AUC above 0.98 raises an integrity alert. No result is written on blocked execution. Fixed model settings require no tuning on assessment labels.
- Added `tests/test_task_a_training.py`: synthetic, temporary fixtures only. Covers actual LR/HGB pipeline execution, grouped coverage/determinism, no-fit checks, disputed exclusion, no outcome fallback, empty labels, unverified/missing mechanisms, invalid folds, CLI no-output blocking and the AUC alert. Synthetic data never enters repository scientific tables or results.

Usage:

```sh
venv/bin/python -m src.training.task_a --check-only
# After independent consensus and mechanistic curation are ready:
venv/bin/python -m src.training.task_a --split herb --model lr --output results/task_a_consensus_lr_herb.json
```

Validation: `venv/bin/python -m pytest tests/test_task_a_training.py tests/test_task_a_feature_contract.py tests/test_consensus_contract.py -q -p no:cacheprovider` — **155 passed**, one existing joblib physical-core detection warning. `git diff --check` passed. Actual `--check-only` exited 2 with the expected empty-consensus blocker, without fitting or producing results. An initial missingness test incorrectly retained a known-null drug factor; corrected the synthetic fixture to make both factors unknown, preserving the existing extractor semantics.

Remaining blockers: real consensus labels are empty; real mechanisms still require verified per-value citations and complete approved features. Structural provenance checks cannot authenticate citations. No real-data training or scientific score is claimed. Task B agreement-supervised reliability training, calibration/conformal inference, full comparison/statistical reporting and fitted deployment artifacts remain later phases. Historical results and leakage guards are unchanged. Stopped after this implementation phase.

## 28 October 2026 — Supervised Task B reliability and residual aggregation

Read PROJECT_SPEC.md, this progress log, Git status, Task A runner, evidence schema, both aggregation modules and the existing lineage/source-registry boundaries. Preserved preceding working-tree edits. Implemented one Task B infrastructure phase; no Streamlit, scientific data, historical results or unrelated code changed.

Implemented:

- `src/models/aggregation.py`: added `ResidualEvidenceAggregator` implementing `w_i = R_i * (1 for primary else alpha)`, signed positive/negative masses, shrinkage `E = (S_pos-S_neg)/(S_pos+S_neg+kappa)`, `logit = mechanistic_prior_logit + lambda*E`, and contradiction `2*min(S_pos,S_neg)/(S_pos+S_neg+epsilon)`. Requires positive kappa, nonnegative lambda and alpha in [0,1]. Empty or wholly inconclusive evidence returns the original prior logit exactly. Diagnostics include masses, net evidence, contradiction, conflicting-evidence flag, study/inconclusive counts and absence of directional mass. The prior is also preserved at balanced conflict; contradiction distinguishes this from absent evidence.
- Replaced the official `train_reliability_model_entrypoint` placeholder with the guarded Task B fitter. Kept the prior empty-consensus guard and the old neural learned-delta classes for compatibility with exploratory callers; those classes are not the new primary path.
- `src/training/task_b.py`: added `SupervisedStudyReliability`, a regularized logistic model optimizing binary cross-entropy agreement against external consensus. Uses a fixed named design/lineage feature schema: controlled species/assay/route/preparation categories, dose-report presence, log sample size plus explicit missingness, annotated primary/echo status and structural citation-cluster size. Reuses the design extractor with absent-route/preparation defaults explicitly overridden to unknown, and the existing citation graph and agreement-target implementation. No text vectorization, pair/source identity columns, finding polarity, contradiction, legacy PK defaults or outcome-dependent quality is fed to R.
- `fit_task_b`: fits the guarded Task A mechanistic LR prior and reliability scaler/model only on explicitly supplied training pairs; learns alpha and residual lambda by bounded binary log-loss optimization on separate tuning pairs. Fixed kappa is not label-tuned. Pair overlap and transitive citation/source-cluster overlap are rejected, including common absent ancestors and intermediate supplied study records. Held-out outcomes are not used in fitting or retained by the topology guard. Prediction takes no consensus targets and rejects reuse of fitting/tuning pairs or their connected citation clusters. Features are computed only from studies belonging to the relevant partition.
- Disputed consensus pairs are counted/excluded from both optimization stages; inconclusive/unknown/conflicting study findings have no agreement targets and contribute no positive/negative mass. Unsupported findings/categories fail with an annotation/normalization error. Each binary partition and the agreement model must contain both target classes; insufficient data blocks rather than inventing targets. Known disputed pairs may be assessed later as unlabeled predictions; no disputed label is used to force a prediction or claimed abstention result.
- Consensus, verified and complete approved mechanisms, and the existing strict publication registry (`require_registered=True`) are required before fitting. Inference also requires verified mechanistic features and registered study sources. Primary/echo annotations must be supplied explicitly; no designation is inferred from a study's agreement or outcome.

API usage after real curation is ready:

```python
from src.models.aggregation import train_reliability_model_entrypoint

# train_pairs / tuning_pairs: separate DataFrames with herb_id and drug_id.
# studies: audited metadata with source_id, source_url_or_doi, study_id,
# cited_study_ids (parsed lists), study_finding and explicit is_primary (0/1).
# Optional design columns: species, evidence_type, route, preparation, dose,
# sample_size. Missing design is encoded explicitly, never defaulted to rigor.
model = train_reliability_model_entrypoint(
    studies=studies, train_pairs=train_pairs, tuning_pairs=tuning_pairs,
    consensus_path="data/benchmark/consensus_labels.csv",
    curation_path="data/curation/mechanistic_features.csv",
)
predictions = model.predict(heldout_pairs, heldout_studies)
```

Tests and verification:

- Added `tests/test_task_b_training.py` with **30 focused synthetic cases** covering aggregation equations, echo discount learning, exact zero/inconclusive-study fallback, balanced/unbalanced conflict, invalid inputs, planted agreement structure, excluded disputed/inconclusive targets, input allowlists, train-only prior/scaler/reliability fitting, pair/source/transitive lineage separation, assessment reuse rejection, and consensus/mechanistic/publication blocking before fit. Test publication references use a scoped mock verifier that asserts strict registration is requested; no synthetic source is added to the real registry. All fixture files live only in pytest temporary directories.
- Command: `venv/bin/python -m pytest tests/test_task_b_training.py tests/test_phase4_scaffolding.py tests/test_consensus_contract.py tests/test_task_a_training.py tests/test_task_a_feature_contract.py -q -p no:cacheprovider` — **195 passed**, one existing joblib physical-core detection warning. Initial collection exposed an incorrect imported lineage class name; corrected it to the repository's existing `EvidenceLineageAnalyzer` before the passing runs.
- Real default entry-point blocking is covered directly: empty consensus raises before any fit. No real-data training, performance score, result JSON or citation was fabricated.
- `git diff --check` and `git diff --exit-code -- data results reports` passed. No full-suite run; historical alerts remain unchanged.

Blockers and limits: real consensus labels remain empty and real per-value mechanisms require independent human verification/completion. Source-registry membership validates publication provenance, not independent consensus derivation, audited design accuracy or primary/echo annotation. Those remain curation responsibilities. The fitted model is uncalibrated infrastructure: no full repeated grouped benchmark, reliability ROC-AUC/ECE report, conformal/epistemic abstention, disputed-pair abstention evaluation, ablation suite or deployment integration is claimed. If tuning has no directional evidence/echoes, the relevant residual parameters cannot be inferred; the fit summary reports that support condition. Separable tuning data may drive large residual strengths, requiring later regularization/model-selection work before scientific use. Stopped after this phase.

## 28 October 2026 — Corrected Colab dependency setup and rerun commands

Documentation-only scope requested by the user. Inspected `docs/COLAB_VERIFICATION.md`, `pyproject.toml`, `requirements_core.txt`, `requirements.txt`, `uv.lock`, CI, relevant imports and `/Users/aditisharma/Downloads/hdi-verification.zip`. The supplied archive records Python **3.13.18**, exit **1**, empty `packages.txt`, and `/content/hdi/venv/bin/python: No module named pytest`. No tests ran. This establishes missing pytest in that interpreter; it does not establish the cause of the missed installation or a Python 3.13 compatibility failure.

Updated only `docs/COLAB_VERIFICATION.md` and this log; preserved all pre-existing application/model/test/data edits.

Corrected instructions:

- State the actual Python contract: project metadata says **>=3.10**, while CI and recorded local verification use **3.11**. The rerun deliberately chooses 3.11 for comparison, without claiming the project mandates exactly that version or that all Python 3.13 stacks are incompatible.
- Supply a complete Bash setup cell that stops on errors, bootstraps uv from the available Colab interpreter, explicitly installs managed Python 3.11, creates a fresh seeded environment at `/content/hdi-verify-venv`, and installs CPU Torch/Torchvision, all core requirements, pytest, pytest-cov, NetworkX and the editable project into that same environment. It does not use or assume the failed `/content/hdi/venv`. Existing setup/output paths are not overwritten.
- Explain that editable installation/`uv.lock` do not install or lock the runtime dependencies in this checkout. Distinguish required suite dependencies from optional database/app/chemistry/tracking/PyG integrations in the broad requirements file. Python-aware resolution is followed by actual checks; no exact cross-platform version reproduction is claimed.
- Add interpreter/prefix checks, pip availability and consistency checks, version/import provenance, Transformers and scikit-learn API smoke checks, Task A/B imports, nonempty package capture, preserved historical-result hash validation, full collection, setup logging and an explicit readiness marker. Preserve the new local Git-index/private-fixture guidance. All shell cells set their own working directory and required environment.
- Supply an exact full-suite cell that preserves pytest's real pipeline exit code separately from tee's logging status and saves the full log, final summary and JUnit report. A separate cell classifies only the exact historical test plus its unchanged `no_context_conflict` AUC alert as known; all additional failures/errors/skips remain visible. Interpreter/dependency/collection failures are never classified as the historical alert. Provide a download cell for logs/version/summary artifacts only.
- Checked managed interpreter/environment commands against official uv documentation and CPU installation guidance against official PyTorch documentation; links are included in the guide.

Validation: parsed all **three Bash cells** with `bash -n` and all **four Python snippets** with Python's AST parser. `git diff --check` and `git diff --exit-code -- data results reports` passed. The historical result SHA-256 remains `67e072a5fc06c4fa2b138a51522b300bb53685b6aba24acd3f9cc35f546c2f60`. No dependencies were installed, no application/model logic was modified, and no tests or Colab commands were executed for this documentation change. A fresh Colab installation and full-suite run remain unverified until the corrected commands are run and their actual outputs reviewed. Stopped after providing the corrected commands.

## 29 October 2026 — Returned independent Colab verification evidence

Inspected the user-returned `/Users/aditisharma/Downloads/hdi-verification-rerun.zip`, including setup log, interpreter/packages, orchestrator, pytest log, JUnit XML, structured summary, source manifest and snapshot-change report. Remote execution was performed by the user in Colab, not by the local assistant. The returned evidence confirms setup exit **0**, pytest exit **1**, summary exit **0**, and **428 tests: 425 passed, three failed, zero errors, zero skips**, with two warnings. Independently recounted the JUnit cases; counts agree with the log/summary. Setup/import/collection checks succeeded. No test-suite rerun was performed locally for this result review.

Snapshot: `/tmp/hdi-verification-20261029/hdi-source.zip`, SHA-256 `37aadd168e7161ac8e389800b9c4f4e16adb12dea3b4f03a2944f34437b9f88a`, source commit `daf20933ce18c2bbbe271f76c3a64a3b9a8e5295`. The reviewed archive contains **242 files** with per-file hashes and a working-tree manifest, including intended untracked Task A/B implementations/tests. Original Git metadata, credentials, environments, caches, private data, old notebook outputs and unnecessary binary artifacts were excluded. The ready-to-run notebook embeds that ZIP and was syntax/checksum validated locally. A connected Drive copy was uploaded, but the browser account could not access it; the user subsequently executed the supplied local notebook. No access controls were bypassed. Local packaging/manifest/notebook files are under `/tmp/hdi-verification-20261029/`.

Remote environment recorded in `packages.txt`: Python **3.11.18**, Torch **2.15.0+cpu**, Torchvision **0.29.2+cpu**, NumPy **2.4.6**, pandas **3.1.0**, scikit-learn **1.9.1**, SciPy **1.17.1**, Transformers **5.21.0**, NetworkX **3.6.1**, pytest **9.2.0**, pytest-cov **7.1.0**. These are actual Colab versions, distinct from the local Python **3.11.16** environment. Snapshot-change report is empty; no frozen input files changed during remote verification. The historical fixture hash remains `67e072a5fc06c4fa2b138a51522b300bb53685b6aba24acd3f9cc35f546c2f60`.

Failures, kept separate:

- Known historical alert: `tests/test_ablation_leakage.py::test_no_actual_benchmark_model_exceeds_threshold` still rejects the preserved `no_context_conflict` result above the unchanged threshold. Not suppressed or repaired by this run.
- Unresolved authorized-transfer prerequisite: `tests/test_phase4_scaffolding.py::test_private_stratification_metadata_is_gitignored` fails because the private directory/file was deliberately excluded. No dummy private data was created. Only a separately authorized transfer of the existing private fixture can satisfy this prerequisite.
- Additional documentation-integrity failure: `tests/test_report_numbers.py::test_all_reports_and_docs_numbers_match_json` identifies four forensic diagnostic values in this log's earlier leakage investigation (one evidence-strength endpoint and three intervention outputs) that have no matching result JSON. This is a provenance/documentation failure, not a model-training regression. The smallest safe follow-up is to preserve verifiable diagnostic outputs in an appropriately identified artifact and link the prose to them, or remove unsupported numerical claims; do not invent matching scientific results or weaken the test. No such unrelated repair was made during app integration.

The Colab run is **not fully passing**. It verifies the frozen pre-UI snapshot, not the app changes below. No standalone scientific model training or evaluation was initiated. Consensus and mechanistic curation remain scientific blockers.

## 29 October 2026 — Streamlit research-readiness integration

Inspected PROJECT_SPEC.md, this progress log, IMPLEMENTATION_AUDIT.md, Makefile/README app entry points, both Streamlit apps, the API backend and Task A/B interfaces. Confirmed `make app` launches `app/interactive_app.py`, which previously trained legacy graph/text fusion at startup on publication outcomes and exposed unsupported calibrated/clinical claims. The alternative `app/frontend.py` called an API with legacy/demo prediction fallbacks. Preserved all pre-existing uncommitted work.

Changed files:

- `app/research_status.py` (new): read-only readiness/evidence presenter. Reuses shared consensus validation, strict registered-source evidence validation, mechanistic provenance validation and the Task A feature-completeness boundary. Missing/invalid evidence is distinct from zero studies; invalid evidence rows are not silently discarded. Displays study findings separately from supplied external consensus categories (`interaction`, `no_interaction`, `disputed`), descriptive conflicting/inconclusive findings, provenance counts and precise readiness blockers. No inference, training, synthetic defaults or model deserialization occurs.
- `app/interactive_app.py`: activate the shared research view and stop before legacy imports/startup training/inference. Old prototype code remains inactive for reference; its predictions, metrics, graph claims and clinical/patent dossier are not rendered.
- `app/frontend.py`: use the same research view and stop before legacy API calls/demo output. `app/backend.py` is unchanged and remains a separate legacy service, not a supported redesigned inference endpoint.
- `tests/test_app_research_status.py` (new): **14 focused cases** cover empty real consensus/unverified mechanisms, external verdict separation, conflict, inconclusive/zero/unavailable evidence, strict evidence validation, incomplete mechanisms, and repeated execution of both entry points without fitting or API predictions. Core tests use a small Streamlit surface stub so they add no mandatory UI dependency to the core verification environment.
- `docs/PROGRESS.md`: this record and the returned Colab findings above.

Exact local validation performed:

1. Initial app test run, with real Streamlit AppTest startup/rerun coverage: `venv/bin/python -m pytest tests/test_app_research_status.py -q -p no:cacheprovider` — **14 passed**. The entry-point tests were then made independent of optional UI packages, preserving their fitting/API guards.
2. Final relevant suite: `venv/bin/python -m pytest tests/test_app_research_status.py tests/test_consensus_contract.py tests/test_task_a_feature_contract.py -q -p no:cacheprovider` — **157 passed**, one existing joblib physical-core detection warning.
3. Separate real `streamlit.testing.v1.AppTest` smoke check via `venv/bin/python`: both Streamlit entry points render and rerun after changing the herb selector, with fitting and API POST calls patched to fail if invoked. Both passed; prediction button disabled and no model metric widgets present. Two Streamlit bare-mode context warnings were emitted by the harness. No browser server or training run was launched.
4. `git diff --check` and `git diff --exit-code -- data results reports` passed. The Colab guide and historical fixture hashes still match the frozen snapshot. The running/completed Colab artifacts were read only and remain unchanged; no Colab rerun was requested.

Precise integration limit: Task A currently returns CV JSON, Task B returns an in-memory uncalibrated fitted object, and there is no persisted, provenance-checked inference bundle/app-loading contract. The canonical study table also lacks the Task B primary/echo and normalized finding annotations. Empty external consensus and unverified mechanisms prevent producing an authentic fitted bundle. Therefore the active UI safely withholds scientific predictions, learned contradiction scores, confidence, calibrated uncertainty and model abstention decisions. Withholding for readiness is explicitly distinguished from statistical abstention. No parallel predictor was built.

No immediate action is required from the user for this app phase. Future live predictions require independent consensus/mechanistic curation, audited Task B annotations and a reviewed fitted-artifact/calibration interface. The three Colab failures above remain separate follow-up work; no full-suite pass is claimed. Stopped after the app integration and result review.

## 29 October 2026 — Evidence-backed Colab failure review and Streamlit revalidation

Inspected the actual latest archive `/Users/aditisharma/Downloads/hdi-verification-rerun.zip` (SHA-256 `08cde017db33489f42eb70e3934bbe350eb487bc2fe9d93df0d9671b1c39b7e6`), not just the previous narrative. Extracted read-only copies of its JUnit XML, failure tracebacks, pytest/collection/setup logs, package list, interpreter version, manifest and summary into `/tmp/hdi-colab-failure-review/`. Original archive unchanged. JUnit and logs agree: **428 collected, 425 passed, three failed, no errors or skips**, setup exit **0**, pytest exit **1**. Recorded remote interpreter is **Python 3.11.18**, pytest **9.2.0**, NumPy **2.4.6**, pandas **3.1.0**, Torch **2.15.0+cpu**; complete installed versions remain in the extracted `packages.txt`. Local verification used the existing **Python 3.11.16** environment, not an asserted identical dependency stack. Reviewed current Git changes and preserved all unrelated uncommitted/untracked work.

Exact failing identities and findings:

1. `tests/test_ablation_leakage.py::test_no_actual_benchmark_model_exceeds_threshold` (`tests/test_ablation_leakage.py:201`). Reads the historical `no_context_conflict` entry, with no synthetic-fixture argument. Recomputed AUC **1.0** from its **26 saved predictions** without fitting a model. Artifact origin is commit `1daf3d39831882f6fbb6038ecaac3e05a1be7391`; inspected that commit's source directly. Historical benchmark lines 248/292/306 build test-pair summaries from held-out study outcomes, send full metadata, and derive targets from the same outcomes. Historical lineage partitions positive/negative records, emits outcome-derived strengths and raw counts (columns 0/1/6/8/9). Historical model defaults this mode to full features, overwrites column 6 with positive strength, and sends metadata through `full_lineage_proj` into R; R gates fused representations into the prediction head. Thus the ablation leaves a direct outcome-feature route intact. A read-only count reconstruction from saved study IDs confirms raw positive counts are zero for saved negatives and one/two for saved positives. Current code has a different guarded quality-only route; changing it cannot repair a preserved historical artifact. The threshold/guard is byte-identical to HEAD and was not weakened, relabeled, skipped or marked expected. No benchmark/model replay or historical artifact regeneration was performed.
2. `tests/test_phase4_scaffolding.py::test_private_stratification_metadata_is_gitignored` (`tests/test_phase4_scaffolding.py:348`). The test legitimately checks the repository's private-fixture storage convention: directory/file presence and an ignore rule, not clinical validity or model performance. The original local `data/benchmark/private/pair_strata.csv` exists, parses as **120 rows**, and is Git-ignored/untracked. Inspected only structure locally; did not print its rows, copy it into the archive, transmit it, or create replacements. It was deliberately excluded from Colab without transfer authorization. The unchanged test passes locally against this genuine file. It is recoverable from the existing local source if private transfer is explicitly authorized for a later remote verification; its absence remains a remote prerequisite failure rather than a model regression.
3. `tests/test_report_numbers.py::test_all_reports_and_docs_numbers_match_json` (`tests/test_report_numbers.py:118`). The four flagged values appeared in the earlier forensic narrative, but no corresponding diagnostic trace JSON or frozen weights were found in the repository or searched temporary diagnostic locations. `results/historical_audit_metrics.json` exists but contains different historical references, not these replay outputs. A retained temporary instrumentation script describes how they could have been generated; a script and repeated prose are not retained output evidence. Some other numerical claims passed the test only through membership in unrelated result values, which also does not authenticate their provenance. Corrected this log by withdrawing the unsupported exact replay, positive-strength ranges and quantitative intervention table/reliability/logit claims. Retained the independently inspectable historical source chain and saved-ranking recomputation, without claiming a unique quantitative causal decomposition. Did not add invented diagnostic JSON, round/hide unsupported numbers, weaken the report test, or change historical results.

Streamlit validation:

- Inspected `app/interactive_app.py`, `app/frontend.py`, `app/research_status.py` and `tests/test_app_research_status.py`. Both entry points render the read-only research view and call `st.stop()` before legacy code. Readiness distinguishes unavailable evidence from no studies and external consensus from publication findings. No fitted confidence/calibration/abstention output is produced. No app/model code change was needed in this phase.
- Real `streamlit.testing.v1.AppTest` smoke checks passed for both entry points at startup, after selecting the conflicting Ginseng/Digoxin pair, and after selecting the zero-study Saw Palmetto/Acetaminophen pair. Patched Torch training/optimizer step/loading, sklearn pipeline fitting/prediction, LR fitting, Task A/B fitting entry points, and HTTP GET/POST to raise if called. None was called. Scientific prediction remained disabled, no model metric widgets or probability/confidence/reliability table columns appeared, conflict/zero-study messages were present, and uncertainty/model abstention remained explicitly unavailable. Two bare-mode context warnings came from the harness.

Actual local test runs:

- Before documentation correction: `venv/bin/python -m pytest tests/test_ablation_leakage.py::test_no_actual_benchmark_model_exceeds_threshold tests/test_phase4_scaffolding.py::test_private_stratification_metadata_is_gitignored tests/test_report_numbers.py::test_all_reports_and_docs_numbers_match_json -q -p no:cacheprovider` — **one passed, two failed**, exit **1**. Reproduced the AUC and documentation failures; local private fixture passed.
- After correction: `venv/bin/python -m pytest tests/test_app_research_status.py tests/test_report_numbers.py tests/test_ablation_leakage.py tests/test_phase4_scaffolding.py::test_private_stratification_metadata_is_gitignored -q -p no:cacheprovider` — **40 passed, one failed**, exit **1**, two expected warning-fixture alerts. Sole failure is the unchanged historical AUC test above. All 14 app cases and all three report-number tests passed.

Changed only `docs/PROGRESS.md` during this phase; preceding Streamlit and research work remains intact. `git diff --check` and `git diff --exit-code -- data results reports` passed. The historical result and Colab-guide hashes match the archived manifest; Colab archive and historical artifacts are unchanged. No full local/remote suite rerun was performed. The full suite **cannot currently pass unchanged**, because the preserved historical AUC alert still fails locally; the private prerequisite also remains absent from the remote snapshot. The documentation correction has only been verified locally and does not retroactively change the archived Colab outcome.

Final documentation-only check after adding this review: `venv/bin/python -m pytest tests/test_report_numbers.py -q -p no:cacheprovider` — **three passed**, exit **0**. No additional model/app tests were repeated.

## 29 October 2026 — Redesigned leakage regressions and historical benchmark separation

Read this log and inspected the current Git diff before editing. Preserved all unrelated uncommitted and untracked files. Scope: trace the historical leak, compare the supported Task A/B paths, add focused end-to-end regression coverage, and propose a separate corrected benchmark path. Changed only the new `tests/test_redesigned_leakage_regression.py` and this log. No application/model implementation, historical artifact, test guard or Colab workflow changed.

### Verified source-to-evaluation trace

Inspected the artifact-origin source at commit `1daf3d39831882f6fbb6038ecaac3e05a1be7391`, not merely the current corrected metadata defaults:

1. **Source and target:** historical `benchmarks/run_verified_hdi_benchmark.py` uses each publication's `outcome` as its target. Its fallback target for conflicting/uncertain publications is not independent consensus truth.
2. **Feature construction:** `EvidenceLineageAnalyzer` partitions those same publications by outcome, calculates positive/negative strengths and raw counts, and exposes them in the full metadata vector. The benchmark computes separate summaries on the held-out partition itself and calls `to_full_metadata_vector()` while preparing its test tensors.
3. **Preprocessing:** the text encoder/projection fits only training texts, then transforms each partition's raw abstracts. That training-only fitting does not remove outcome language in assessment abstracts or the separately constructed outcome-derived metadata. Padding/default handling retains the polarity/count columns. The historical `no_context_conflict` branch chooses full metadata; it neutralizes context/conflict but copies positive strength into the margin slot and leaves positive/negative strengths and raw counts intact.
4. **Training:** historical model variants receive those metadata tensors during BCE training on publication targets (25 optimizer epochs). The full metadata projection supplies R; R controls fusion before the prediction head. This establishes a learnable route from study outcomes through predictor tensors, not just an unused logging field.
5. **Evaluation:** test forward passes receive metadata constructed from test findings. Temperature scaling fits on validation predictions/labels, then metrics compare calibrated test probabilities against targets from those same test publications. Validation-only scaling and study-disjoint partitioning cannot repair predictor features that already expose their own assessment outcomes. No historical training replay was run in this phase; the prior review's saved-probability recomputation remains the checked numerical evidence.

The preserved artifact still fails `tests/test_ablation_leakage.py::test_no_actual_benchmark_model_exceeds_threshold`. Its hash remains `67e072a5fc06c4fa2b138a51522b300bb53685b6aba24acd3f9cc35f546c2f60`. No threshold change, xfail, result relabeling or artifact regeneration is justified. The legacy benchmark runner remains an exploratory publication-outcome/raw-text path even though its current metadata guards improve on the historical implementation; those guards do not convert it into valid redesigned Task A/B evaluation.

### Redesigned boundary assessment

- **Task A:** `prepare_task_a` loads the independent consensus table and defines binary targets from `consensus_label`; disputed labels are excluded. `build_task_a_features` projects only pair lookup keys into the provenance-checked mechanistic extractor and reconstructs the fixed features, ignoring caller-supplied predictor aliases. Each CV fold fits its own guarded scaler/estimator pipeline; assessment rows only transform/predict. No publication table or publication-outcome fallback is used.
- **Task B reliability/prior:** `fit_task_b` selects targets from the external consensus lookup. Clear training-study findings define agreement targets for R; findings are not columns of R's named design/lineage feature matrix. The citation graph is built with outcome-free records. Prior and R/scaler fit only training pairs, while separate pair/citation-disjoint tuning pairs determine echo discount and residual scale. Inference takes no consensus file and uses only the fixed mechanistic builder plus design/lineage features for these components.
- **Task B aggregation:** study findings intentionally define positive/negative evidence masses. Changing a finding is allowed to change these masses and the residual prediction; requiring the final Task B prediction to ignore findings would violate the specification. The required isolation is that findings must not change R's predictors or the mechanistic prior, and assessment consensus must never enter inference or parameter fitting.
- No historical outcome-column leak was demonstrated in the supported redesigned routes, so no production patch was made. These are scoped implementation findings, not proof of clinical validity: false measurements/design annotations can be deliberately renamed to approved fields, and structural validators cannot establish the scientific independence of consensus/citations. Curation, appropriate benchmark splits and independent review remain necessary.

### New regression coverage

`tests/test_redesigned_leakage_regression.py` adds **14 synthetic cases**, reusing existing temporary fixtures and production entry points:

- Runs actual Task A LR/HGB grouped CV twice, injecting outcome counts, polarity masses, contradiction, raw labels/text and a caller-supplied approved mechanistic feature alias into the source rows. Captures actual estimator fitting after scaling and asserts canonical named columns, identical numeric inputs/targets and identical CV predictions.
- Fits Task B with/without the same contaminants in studies, consensus rows and partition inputs. Captures post-scaler LR inputs/targets for both R and the prior; they and tuning summaries/held-out outputs are identical.
- Captures R's actual target vector to verify agreement with external consensus, despite false publication-level consensus/agreement aliases.
- Changes only held-out consensus labels and verifies unchanged fitted prior/R parameters and residual tuning. Removes the temporary label file before inference and injects false consensus columns into pair/study inputs; predictions remain identical.
- Swaps held-out study findings: R features/scores and prior logits remain identical; positive/negative masses exchange, net evidence reverses and contradiction stays the same, demonstrating the permitted evidence-only channel.
- Rejects outcome-derived extra columns at both R fitting and prediction boundaries.

All fixtures are synthetic and temporary. No labels/citations or performance results were added to scientific tables or result files.

### Separate corrected benchmark path (proposal, not executed)

Use a new consensus benchmark orchestrator/output namespace, for example a future `benchmarks/run_consensus_hdi_benchmark.py` writing only to `results/consensus_hdi/`, leaving every historical artifact/guard untouched. Reuse the existing `src.training.task_a` runner for mechanistic-only grouped CV. For Task B, add an outer held-out pair/citation-component assessment loop around the existing fitter, with training and tuning partitions wholly inside each outer training partition; never repurpose Task A pair splits as citation isolation. Supply all studies for structural leakage checks, but fit features/targets only on their permitted partitions. Pool/report predictions with source/data/config manifests, independent consensus targets and explicit disputed/inconclusive handling. Learn calibration/abstention only within permitted training/calibration groups before reporting assessment performance; do not claim the current uncalibrated fitter implements that evaluation.

This route requires populated independently curated labels, verified complete mechanisms and normalized/audited Task B annotations. It is not authorized to overwrite the old JSON to make a guard green. A corrected new result would not make the existing immutable-artifact guard pass; the full current suite continues to surface that historical integrity alert. No new scientific benchmark was run.

### Exact test commands and results

1. Smallest new regression set first: `venv/bin/python -m pytest tests/test_redesigned_leakage_regression.py -q -p no:cacheprovider` — **14 passed**, exit **0**, one existing joblib physical-core fallback warning.
2. Relevant feature-contract/training/consensus coverage: `venv/bin/python -m pytest tests/test_redesigned_leakage_regression.py tests/test_task_a_feature_contract.py tests/test_task_a_training.py tests/test_task_b_training.py tests/test_consensus_contract.py -q -p no:cacheprovider` — **199 passed**, exit **0**, the same one warning.
3. Historical alert retained: `venv/bin/python -m pytest tests/test_ablation_leakage.py::test_no_actual_benchmark_model_exceeds_threshold -q -p no:cacheprovider` — **one failed**, exit **1**, on the unchanged stored historical result. This is not a new regression failure.

No full local/Colab suite was run. The scientific blockers remain missing consensus labels and unverified/incomplete mechanistic inputs; Task B still lacks a full calibrated outer assessment harness. No scientific validation is claimed.

Documentation integrity: `venv/bin/python -m pytest tests/test_report_numbers.py -q -p no:cacheprovider` — **three passed**, exit **0**. `git diff --check` and `git diff --exit-code -- data results reports` passed. Only the new regression file and this progress entry were changed in this phase; existing model/app/test work, Colab artifacts and the historical guard were preserved.


## 29 October 2026 — Pair-level mechanistic inspection and final readiness

Read PROJECT_SPEC.md, this progress log and Git status before changing files. Implementation checklist: review both entry points and shared read-only view; preserve startup stop guards and evidence/consensus/prediction distinctions; expose existing mechanistic curation records safely; test only the new inspection gap; document genuine prerequisites. Existing completed audits were not repeated.

Both entry points already call the shared view and stop before legacy training/API code. Kept that behavior. Added selected-entity mechanistic records to the read-only view, including original status, missing values, source and citation fields. Explicitly distinguish recorded inputs from verified mechanisms and predictions; not_tested does not establish no effect. Invalid tables are not displayed. No feature generation from unverified records, training, inference, scientific claims or new scientific data were introduced.

Changed in this phase: app/research_status.py, tests/test_app_research_status.py, docs/PROGRESS.md. New regression cases check exact selected-entity rows, preservation of unverified citations/missing measurements, copy isolation, absent entities and malformed-table withholding. Existing entry-point startup/rerun tests now also assert the mechanistic caveat and unverified records are rendered.

Validation: `venv/bin/python -m pytest tests/test_app_research_status.py -q -p no:cacheprovider` — **16 passed**, exit **0**. No full suite or historical benchmark rerun. This uses the existing Streamlit stub; no new real-browser smoke check is claimed.

Final-readiness checklist and remaining blockers:

- Read-only inspection of publication findings, independent consensus and mechanistic inputs is wired through both apps. Scientific predictions remain disabled.
- Populate independent external consensus labels with per-pair source/citation and retrieval date, blind annotation, circularity review and the specified double-annotated subset. Include supported binary labels and independently adjudicated disputed cases; study findings cannot substitute for targets.
- Curate complete, source-backed mechanistic measurements and explicit tested-null/unknown statuses for candidate entities and required pathways, including exposure provenance where applicable. Current unverified placeholders cannot enter real training.
- Audit normalized study_finding, primary/echo and citation-lineage annotations plus design metadata for Task B; column presence alone is not scientific verification.
- Complete Task B outer pair/citation-disjoint assessment and training-only calibration/conformal evaluation before making uncertainty, contradiction-based decision or abstention claims. Add a persisted provenance-checked inference bundle/loader before enabling UI predictions.
- Historical AUC integrity alert remains unresolved and intentionally preserved; historical results and unrelated work are untouched. Prior remote private-fixture prerequisite remains separate. No scientifically validated predictor or full-suite pass is claimed.

Next single recommended action: begin independent consensus annotation in data/benchmark/consensus_labels.csv under the specification's provenance and blinded-review rules. Mechanistic curation remains required before actual Task A/B training.

## 29 October 2026 — Human consensus annotation workflow

Read PROJECT_SPEC.md, this log, Git status, ConsensusPairRecord/consensus table loaders and tests/test_consensus_contract.py before editing. Inspected identifier-only coverage: existing study records contain **141 publications / 120 unique pairs**; pairs_to_label contains **120 unique candidates**, and random_unstudied contains **100 unique candidates**. These lists are not independent labels or verified negative cases. The validated canonical data/benchmark/consensus_labels.csv still contains exactly **0 real consensus labels**. No real pair was annotated automatically and no real reviewer workbooks were generated during validation.

Added src/data/consensus_annotation.py, docs/CONSENSUS_ANNOTATION.md and tests/test_consensus_annotation.py; updated only this log among pre-existing files. The CLI init command projects only validated herb/drug keys, discards any input findings/labels, deduplicates assignments and creates two separate reviewer files plus explicit adjudication, a frozen pair list and input-hash manifest. All cases start unresolved with blank reviewer/provenance fields. Unresolved is a staging status, not a new consensus target; the existing strict three-class training contract is unchanged.

The workflow requests actual external compendium citations, access date, rationale, stable reviewer identity, prediction-blinding and external-consensus attestations, and circularity/dependency notes. Every resolved export requires two documented independent responses and a separate adjudicator. All candidate pairs are assigned double review; completed-review counts are reported separately from assignments. Documented unresolved reviewer responses may lack inaccessible citations; unknown data must not be invented. Reviewer disagreement never automatically creates a disputed label. An explicit source-backed adjudicator decision is required even when reviewers agree.

Finalize creates only a new output directory containing contract-validated resolved rows, unresolved pair identifiers, completed double-review/disagreement counts, hashes and exact copies of the assignment/review inputs. It never merges into or overwrites the canonical table. Human source authentication, circularity assessment, reviewer independence and data-steward promotion remain required; software checks and attestations cannot prove them. No agreement statistic, clinical validation, mechanism, model prediction or scientific performance was fabricated.

Synthetic-only validation:

- Initial focused workflow/contract run: **86 passed, 5 failed**, exit **1**. Tests found loss of pair identifiers when indexed workbook rows were revalidated; corrected by restoring pair keys before decision validation.
- Following the fix: **91 passed**, exit **0**. Repeated after adding completed-review audit logic: **91 passed**, exit **0**.
- Final `venv/bin/python -m pytest tests/test_consensus_annotation.py tests/test_consensus_contract.py -q -p no:cacheprovider`: **94 passed**, exit **0**. Includes **20 new synthetic workflow cases** covering unresolved initialization/export, dropped outcome/label inputs, all supported adjudicated labels, no automatic disputed verdict, review requirements, provenance/date failures, duplicate/changed assignments, overwrite refusal, retained snapshots, incomplete-review counting, documented unresolved responses and CLI finalization. Existing consensus contract tests pass unchanged.

No full-suite run, real annotation validation, model fitting or historical benchmark rerun. Historical results, the AUC leakage guard, scientific data, and all unrelated uncommitted/untracked work are preserved. Genuine training blockers remain absent consensus targets, unverified/incomplete mechanisms, audited Task B annotations, and the calibrated assessment/inference interface described above.

Next concrete action: a human coordinator runs the documented init command for the existing candidate list, gives each reviewer only their own workbook and independent external-source access, then obtains their blinded annotations before adjudication. Do not directly fill the canonical table from study outcomes. The current number of real consensus labels remains **0**.

Documentation verification: `venv/bin/python -m pytest tests/test_report_numbers.py -q -p no:cacheprovider` — **3 passed**, exit **0**. `git diff --check` passed. A separate `git diff --exit-code -- data results reports` check passed. The leakage-test file still has its pre-existing quality-feature assertion change relative to HEAD; no edits were made to that file or its historical AUC guard in this phase.

## 29 October 2026 — Final integration and reproducibility audit

Inspected Git status before changes and reviewed the specification, progress and
annotation instructions. Existing modified and untracked work was retained.
No annotation implementation changes were needed. Clarified the CLI boundary in
CONSENSUS_ANNOTATION.md: only init/finalize exist; review/adjudication are manual,
and finalize validates while exporting. Added the actual Task A check-only command.

New synthetic integration coverage connects initialized reviewer workbooks through
explicit adjudication/export, the strict consensus loader and actual Task A CV.
It also confirms that an unresolved/header-only export blocks before mechanisms
are loaded and that unresolved reviewer workbooks cannot enter training.
No real workbooks, labels, citations or scientific results were generated.

Exact validation: `venv/bin/python -m pytest tests/test_annotation_training_integration.py -q -p no:cacheprovider`
initially exited 2 at collection due to a test-module import path; corrected to
package imports. The same command then exited 0: **2 passed**. Previously passed
annotation, contract, training and app tests were not rerun. No full suite or Colab.

Task B readiness verified by source inspection: fit_task_b accepts studies,
explicit train/tuning pair frames, consensus/curation paths and fixed kappa.
It fits agreement-supervised R and a mechanistic prior on training pairs, tunes
alpha/lambda on separate pairs, and returns FittedTaskB.predict for held-out pairs
without targets. Pair/citation-component checks exist. There is no Task B CLI,
outer repeated assessment orchestrator, persisted provenance-checked model loader,
or end-to-end calibration/conformal assessment interface. Existing calibration
utilities do not make this fitter calibrated. Remaining assessments include
training-only pooled OOF calibration, independent coverage/ECE/Brier/AURC,
contradiction/disputed abstention validation, ablations and uncertainty/statistical
comparisons. Real independent consensus, complete verified mechanisms, normalized
findings, audited source registration/lineage/primary-echo flags and design metadata
remain prerequisites; both agreement classes and binary train/tuning classes are
required. Column presence is not annotation verification.

Both Streamlit entry points call render_research_readiness and immediately st.stop
before legacy training/API imports. The shared view loads research tables and
renders a disabled prediction button; no fit or inference call is reachable in
that view. This is source verification, not a new browser smoke test.

Reproduce the human workflow from the repository root:

```sh
venv/bin/python -m src.data.consensus_annotation init --pairs data/benchmark/consensus_labels_pairs_to_label.csv --output annotation_round_1
# Independent humans complete reviewer_a.csv and reviewer_b.csv separately.
# A separate human adjudicator completes adjudication.csv after reviews lock.
venv/bin/python -m src.data.consensus_annotation finalize --directory annotation_round_1 --output annotation_export_1
venv/bin/python -m src.training.task_a --consensus annotation_export_1/consensus_labels.csv --curation data/curation/mechanistic_features.csv --check-only
venv/bin/streamlit run app/interactive_app.py
# Alternative entry point: venv/bin/streamlit run app/frontend.py
```

Use fresh output directory names. Finalize may legitimately export zero rows;
check-only is expected to block until human labels, mechanisms and fold coverage
are adequate. Software is ready for human annotation and read-only evidence
inspection; synthetic training integration is demonstrable. Real performance,
calibration, abstention and clinical predictions remain scientifically blocked.
Historical AUC guard/results and all scientific data remain untouched.

## 29 October 2026 — Minimal completion pass

Read both workflow documents and inspected Git status. No defect found; no code
changes. Verified CLI parser/help and new-directory-only writes: init/finalize
match the documented commands and cannot overwrite the existing canonical table.
Promotion remains an explicit manual human data-steward step after source,
circularity and reviewer checks; retain snapshots and unique pair identifiers.

Checks this pass:
- `venv/bin/python -m src.data.consensus_annotation init --help` — exit 0.
- `venv/bin/python -m src.data.consensus_annotation finalize --help` — exit 0.
- `venv/bin/python -m pytest tests/test_annotation_training_integration.py::test_unresolved_export_blocks_runner_before_mechanisms tests/test_task_a_training.py::test_check_only_never_fits -q -p no:cacheprovider` — 2 passed, exit 0. These exercise initialization/export and Task A validation without fitting; the earlier successful resolved-export integration was not rerun.
- `git diff --check` — passed.

Reproducible next human steps (fresh directory names, repository root):
```sh
venv/bin/python -m src.data.consensus_annotation init --pairs data/benchmark/consensus_labels_pairs_to_label.csv --output annotation_round_1
# Two independent blinded reviewers complete their separate workbooks;
# a third adjudicator records source-backed decisions after reviews lock.
venv/bin/python -m src.data.consensus_annotation finalize --directory annotation_round_1 --output annotation_export_1
venv/bin/python -m src.training.task_a --consensus annotation_export_1/consensus_labels.csv --curation data/curation/mechanistic_features.csv --check-only
```

Ready to demonstrate human annotation and read-only research inspection, not a
scientifically validated predictor. Blockers remain: independent consensus targets
(current canonical table is empty), complete verified mechanisms/exposure provenance,
audited Task B findings/design/source/lineage annotations, outer pair/citation-disjoint
assessment with training-only calibration/conformal and disputed-case abstention
validation, and a provenance-checked persisted inference interface before UI
predictions. The historical AUC integrity alert remains preserved and unresolved.
Next action: a human coordinator initializes assignments and distributes separate
workbooks to qualified blinded reviewers. No real assignments were generated here.
All existing changes/untracked files retained; no training, full suite, Colab,
scientific data generation or historical artifact edits in this pass.

## 29 October 2026 — Focused annotation command verification

Inspected Git status first and read PROJECT_SPEC.md and both workflow documents.
No concrete defect found; only this progress log changed. Parser/implementation
and CLI help confirm these documented commands (use fresh output directories):
```sh
venv/bin/python -m src.data.consensus_annotation init --pairs data/benchmark/consensus_labels_pairs_to_label.csv --output annotation_round_1
# Two blinded independent human reviews, then separate human adjudication.
venv/bin/python -m src.data.consensus_annotation finalize --directory annotation_round_1 --output annotation_export_1
```
Resolved exports validate both reviewer responses and adjudication, require three
distinct reviewer IDs, and preserve exact assignment/review snapshots plus hashes.
Unresolved cases stay pending; software cannot authenticate human independence or
sources. Outputs go to a new directory, with no automatic canonical promotion.
The documented human data-steward promotion and source/circularity review remain
required. No real assignments were created during this check.

Exact checks run:
- `venv/bin/python -m src.data.consensus_annotation init --help` — exit 0.
- `venv/bin/python -m src.data.consensus_annotation finalize --help` — exit 0.
- `venv/bin/python -m pytest tests/test_consensus_annotation.py tests/test_annotation_training_integration.py::test_unresolved_export_blocks_runner_before_mechanisms tests/test_task_a_training.py::test_check_only_never_fits -q -p no:cacheprovider` — **22 passed**, exit 0. Covers all annotation tests and non-fitting Task A integration. The resolved-export CV integration test was deliberately not run because it fits a model; its earlier result is recorded above.
- `venv/bin/python -c 'from src.data.evidence_schema import load_consensus_labels_table; labels, empty = load_consensus_labels_table(); print("Canonical consensus labels:", len(labels))'` — **0 real consensus labels**, exit 0.
- `git diff --check` — passed.

Ready for annotation/read-only demonstration. Remaining scientific blockers:
independent consensus targets, complete verified mechanisms and exposure provenance,
audited Task B study/design/source/lineage annotations, independent grouped outer
assessment with training-only calibration/conformal and disputed-case abstention
validation, and a provenance-checked persisted inference interface. Historical AUC
alert remains unresolved and unchanged. Next action: human coordinator runs init,
distributes separate review workbooks and arranges independent adjudication.
No model fitting, full suite, Colab, real annotation or historical artifact edits;
all unrelated changes and untracked files preserved.

## 29 October 2026 — Implementation closeout

Read the specification/progress and inspected Git status. Checked current Task A
consensus/feature loading, Task B fit/predict boundary, annotation CLI/export and
both app startup guards against the already recorded integration checks. Found
no new concrete integration defect. No source rewrites or new tests warranted;
annotation's prior 22-test check remains applicable and was not repeated.

Final targeted command:
`venv/bin/python -m pytest tests/test_app_research_status.py::test_streamlit_startup_never_fits_or_calls_legacy_api tests/test_task_b_training.py::test_real_entrypoint_blocked tests/test_task_a_training.py::test_cli_block_does_not_write_result -q -p no:cacheprovider`
— **4 passed**, exit 0. Confirms both app entry points stop before legacy calls,
and current real-data Task A/B paths block. No model training performed.
`git diff --check` — passed. Only docs/PROGRESS.md changed in this closeout;
all pre-existing modified/untracked work, historical results and AUC guard retained.

Demo from repository root:
```sh
venv/bin/streamlit run app/interactive_app.py
# Or: venv/bin/streamlit run app/frontend.py
venv/bin/python -m src.data.consensus_annotation init --pairs data/benchmark/consensus_labels_pairs_to_label.csv --output annotation_round_1
# Independent humans review separately; a third adjudicator records decisions.
venv/bin/python -m src.data.consensus_annotation finalize --directory annotation_round_1 --output annotation_export_1
venv/bin/python -m src.training.task_a --consensus annotation_export_1/consensus_labels.csv --curation data/curation/mechanistic_features.csv --check-only
```
Use fresh annotation directory names; no canonical promotion is automatic. The
last command intentionally blocks until labels, mechanisms and fold coverage are
adequate. These demo commands were documented, not executed in this closeout.

Implemented scope is ready for annotation and read-only evidence inspection.
The research specification is not scientifically complete: canonical labels remain
empty as last verified above; mechanisms/exposure provenance need human verification;
Task B findings/design/source/lineage annotations need auditing. Task B still needs
outer grouped assessment, training-only calibration/conformal and disputed-case
abstention validation, plus persisted provenance-checked inference before predictions.
These known unfinished interfaces are documented scope, not claimed as implemented.
Historical AUC alert remains unresolved. No additional small defect fix identified;
stop rather than repeat audits or expand the project. Single next action: human
coordinator initializes and distributes blinded consensus-review assignments.
No full suite, Colab, fabricated evidence or scientific validation claim.

## 29 October 2026 — Mechanistic identifier collision fix

Confirmed defect: validation checked raw entity/target pairs, while extraction
used string conversion and `.strip()` for dictionary keys. Whitespace-equivalent
conflicting rows could overwrite each other, selecting +0.5 or -0.5 solely by
row order. Validation and extraction now share a string-only canonical identifier
helper that strips surrounding whitespace. Duplicate canonical entity/target
records fail before matrix construction; neither row is chosen or aggregated.
Empty identifiers, role prefixes without suffixes, internal whitespace and
non-string identifiers fail explicitly. Case is preserved and extra target names
remain permitted. Duplicate curation/pair columns are rejected; CSV duplicate
headers are checked before pandas can silently rename them.

Inspected `src/features/task_a_contract.py`; its existing unique-column check
and delegation to the curated extractor require no source change. Added direct
extractor and Task A builder regressions for both row orders of conflicting
normalized records, including the reproduced +/-0.5 case. Tests also cover
malformed IDs, duplicate columns/CSV headers, accepted surrounding whitespace,
input non-mutation and preserved missing pathways. Numeric missing-value handling,
feature schemas and training missingness policy are unchanged.

Exact validation:
`venv/bin/python -m pytest tests/test_mechanistic_boundary.py tests/test_task_a_feature_contract.py tests/test_task_a_training.py tests/test_task_b_training.py tests/test_annotation_training_integration.py -q -p no:cacheprovider`
— **172 passed, 1 warning in 4.09s**, exit 0. The warning is joblib's physical-core
count detection falling back to logical cores. `git diff --check` — passed.
Only these relevant mechanistic, feature-contract and training tests ran; no full
suite or Colab. Historical benchmark artifacts, AUC leakage guard, consensus
labels and the pre-existing private annotation batch were not modified.

Compatibility: formerly accepted malformed IDs and whitespace-equivalent duplicate
records now raise ValueError and require explicit curation resolution. Valid
surrounding whitespace remains supported; no case folding, alias mapping or
scientific record repair is introduced.

## 29 October 2026 — Conformal engine fail-closed fix

Inspected the engine, both conformal test modules, HDIModel calibration/prediction
integration, and external-diagnostics dependent tests before changing behavior.
Removed the uncalibrated q=0.85 fallback: single, batch (including empty batches)
and evaluation calls now require calibration. Calibration validates finite [0,1]
probabilities, binary numeric labels without integer truncation, equal lengths,
nonempty calibration data and valid epsilon. Constructor thresholds and prediction
safety scores are also validated. Batch metadata lengths and evaluation labels
are checked explicitly. Failed recalibration invalidates stale fitted state.

Shared the identical calibration quantile implementation without changing its
capped finite-sample level or NumPy `method="higher"` behavior. Regression checks
cover both calibration methods across small and attainable sample sizes. Corrected
module/class documentation and logging: pooled OOF is empirical calibration, not
a formal coverage guarantee; split-conformal coverage requires the stated fixed
predictor/exchangeability assumptions, and the maximum-score small-sample fallback
does not establish nominal coverage. Existing decision routing remains unchanged.

Exact final validation:
`venv/bin/python -m pytest tests/test_conformal_abstention.py tests/test_conformal_overhaul.py tests/test_integrated_novelty.py tests/test_evaluation_integrity.py::test_external_diagnostics_use_fitted_sets_and_do_not_invent_uncertainty -q -p no:cacheprovider`
— **80 passed in 1.98s**, exit 0. Earlier targeted runs: **73 passed, 1 failed
in 2.06s** (new test incorrectly expected a singleton when q=1; corrected the
assertion), then **74 passed in 2.00s**, exit 0, before additional regressions.
`git diff --check` — passed. No full suite or Colab. The artifact-writing novelty
ablation experiment was not run; the HDIModel integration test covers its conformal
calibration/prediction path without writing benchmark artifacts.

Changed for this fix: `src/models/conformal_abstention.py`,
`tests/test_conformal_abstention.py`, and this appended progress entry. Preserved
all pre-existing workspace edits. Historical benchmark artifacts, AUC leakage guard,
consensus labels and private annotation batch were not modified.
Compatibility: callers relying on uncalibrated decisions now receive RuntimeError;
malformed inputs now receive ValueError; failed recalibration requires successful
calibration before reuse. Valid calibration thresholds and routing are preserved.


## 29 October 2026 — Trainer recalibration review follow-up

Inspected the full HDITrainer -> HDIModel -> ConformalTriStateEngine calibration
path and subsequent engine evaluation. The trainer now disables calibration at
entry and clears fitted state in a finally block on any unsuccessful attempt,
including graph/batch preparation, loader iteration, inference, engine calibration,
and post-calibration metadata validation. Successful calibration remains usable.
Added focused trainer regressions for all seven failure stages, each starting
with usable calibration and verifying a successful retry afterward.

Added an explicit nonmaximum finite-sample quantile regression for both held-out
and pooled OOF methods: n=10, epsilon=0.3, scores 0.0 through 0.9 yield q_hat=0.8,
strictly below the maximum. The quantile formula is unchanged. Trainer documentation
now qualifies nominal marginal coverage by the fixed predictor, exchangeability,
and attainable-rank assumptions, and identifies reported coverage as empirical
on calibration data. It disclaims nominal guarantees for pooled OOF and the
capped small-sample fallback.

Exact validation:
`venv/bin/python -m pytest tests/test_trainer_conformal.py tests/test_conformal_abstention.py tests/test_conformal_overhaul.py tests/test_evaluation_integrity.py -q -p no:cacheprovider`
— **94 passed in 4.22s**, exit 0. `git diff --check` — passed after removing
trailing whitespace introduced while wrapping the trainer method. Only relevant
trainer, conformal-abstention and evaluation-integrity tests ran; no full suite
or Colab.

Changed for this follow-up: `src/training/trainer.py`,
`tests/test_trainer_conformal.py`, `tests/test_conformal_abstention.py`, and this
appended progress entry. Preserved pre-existing workspace edits. Historical
benchmark artifacts, the AUC leakage guard, consensus labels, and the private
annotation batch were not modified.

Compatibility: a failed trainer recalibration now requires a successful retry
before conformal predictions can resume, including when the previous calibration
was usable or the new threshold had been fitted before metadata validation failed.
Successful thresholds, quantile rules, return values and decision routing remain
unchanged.

## 29 October 2026 — Leakage-safe evaluation Phase 1: partition contract

Added `src/training/evaluation_protocol.py` and
`tests/test_evaluation_protocol.py`. Inspected Task A's grouped split utility,
Task B's pair/source/citation overlap checks and tests, the evidence schema and
lineage graph, and the consensus identifier/label contract before implementation.
No legacy outcome/text-based split construction was reused. The new module is
stdlib-only and imports no training, feature, consensus-label, or model code.

`validate_partitions` accepts typed record iterables for an explicit pair universe,
explicit assignments, a pair/herb/drug regime, named partition roles, a protocol,
task A/B, topology, and an optional provenance seed. Protocols baseline, tuned,
probability_calibrated, conformal, and full require their documented distinct,
nonempty roles; full separates fitting, tuning, probability calibration,
conformal calibration, and assessment. Canonical scientific identifiers use the
consensus vocabulary without normalization. Disputed and zero-study pairs remain
in the supplied universe. Task B requires explicit topology (including an explicit
empty list for zero studies). Unassigned intermediate provenance rows use both
pair identifiers set to None; other topology pairs must be in the universe.

Shared study/source IDs, transitive citations and absent cited ancestors form
undirected evidence components. Pair nodes bridge multiple citation components;
herb/drug constraints join these further for the selected regime. Cross-partition
components fail explicitly; assignments are never synthesized or repaired.
Canonical manifests include version, protocol/configuration/seed, roles, stable
pair IDs, assignments, entities, evidence membership and constraint components.
Separate SHA-256 topology and assignment hashes cover canonical inputs, including
citation edges that do not change component connectivity. Input row order and
non-topology fields do not affect manifests. No file is written by validation.

Exact test commands and results:

- `venv/bin/python -m pytest tests/test_evaluation_protocol.py -q -p no:cacheprovider`
  — 39 passed, 6 failed in 2.13s, exit 1. Failures were in the new test helper's
  duplicate keyword handling; fixed the helper.
- `venv/bin/python -m pytest tests/test_evaluation_protocol.py tests/test_task_a_training.py tests/test_task_b_training.py tests/test_redesigned_leakage_regression.py -q -p no:cacheprovider`
  — 100 passed, 1 failed, 1 warning in 4.09s, exit 1. The new unknown-partition
  fixture used a reserved missing-identifier sentinel and expected the later
  unknown-partition error; changed it to an ordinary unlisted partition name.
- Same exact four-file command after correction and additional topology/hash and
  long-chain regressions — **106 passed, 1 warning in 4.13s**, exit 0 (50 new tests).
  Warning: joblib could not detect physical cores and used logical cores.

Coverage includes pair/herb/drug separation, every pair of the five roles,
shared study/source/citation ancestry, multiple-component pair bridges, explicit
infeasibility, malformed/duplicate/missing/extra inputs, all protocol role sets,
disputed and zero-study pairs, reordering, topology/assignment hash changes,
label/outcome/text/prediction perturbations, a 1,100-link chain, and fit/calibration
sentinels. Existing Task A AUC integrity guard tests passed unchanged. Only new
and directly related split/leakage tests ran; no full suite or benchmarks ran.

Limitations and compatibility risks: this is an additive contract only; no Task A
or B evaluation runners or integration are implemented. Existing callers must
explicitly adopt it. DataFrames require `.to_dict('records')`; parsed citation
lists and strict canonical identifiers are required, with no automatic cleanup.
Each selected role needs its own nonempty partition; no pooled/OOF role reuse is
supported. Complete audited topology is the caller's responsibility: omitted or
unrecorded provenance cannot be inferred. Structural validity does not attest
source authenticity, class balance, sufficient sample sizes, exchangeability,
scientific validation, predictive performance or conformal coverage. No source
registry checks, label filtering, assignment generation, fitting, calibration or
scoring occurs. Historical benchmarks, the AUC leakage guard, consensus labels,
and the pre-existing private annotation directory were not modified. No new
dependencies; no staging, commit or push.

Final `git diff --check` — passed, exit 0. `git status --short` showed only
these three implementation/documentation files plus the untouched pre-existing
`annotation_batch_24_seed20261029/` directory.

## 29 October 2026 — Phase 1 confirmed review fixes

Replaced repeated topology-record scans and component-node list membership checks
with a study-to-pair set index. Component membership now visits each component's
nodes and indexed associations, preserving sorted deterministic manifests and
existing connectivity/validation rules. Canonical sorting and hashing remain;
this does not claim the entire validator is strictly linear.

Topology rows are checked against `Mapping` before dictionary methods are used.
Unsupported containers raise a documented actionable `ValueError` directing
callers to dictionary records (or DataFrame `.to_dict('records')`). Valid mapping
wrappers and missing optional non-topology metadata remain accepted.

Added regressions for shared herbs/drugs within and across partitions under all
three regimes, a redundant citation edge changing the topology hash while all
component memberships stay identical, unsupported topology row containers, and
valid mapping wrappers. Replaced the old mostly unassigned long-chain fixture
with fully assigned 600/1,200-record cyclic chains. Each chain record belongs to
the fit pair, uses a distinct publication source, and participates in the evidence
component. An identifier-equality operation bound detects quadratic list searches
without wall-clock thresholds; reversed input verifies deterministic termination,
components and assignments. Existing coverage, role separation, absent ancestor,
source/citation, pair bridge, outcome/text/finding independence and no-fit
sentinel regressions remain in the targeted run.

Exact validation:

- `venv/bin/python -m pytest tests/test_evaluation_protocol.py tests/test_task_a_training.py tests/test_task_b_training.py tests/test_redesigned_leakage_regression.py -q -p no:cacheprovider`
  — **123 passed, 1 warning in 4.53s**, exit 0. Includes 67 partition-contract
  tests and 56 directly affected Task A/B split/leakage tests. Warning: joblib
  could not detect physical cores and used logical cores.
- `git diff --check` — passed, exit 0.

Changed only `src/training/evaluation_protocol.py`,
`tests/test_evaluation_protocol.py`, and this appended progress entry; preserved
pre-existing work. No Phase 2, full suite, Colab, staging, commit or push.
Historical benchmark artifacts, AUC leakage guard and consensus labels were
unchanged. Private annotation contents were neither inspected nor modified.

Remaining limitations: complete audited topology is still the caller's
responsibility; omitted relationships cannot be inferred. Structural validation
provides no scientific/performance or statistical-coverage guarantee. The new
work bound targets the confirmed component-membership regression, rather than
all possible graph shapes or total runtime. No runner integration or model
fitting was added.

## 29 October 2026 — Phase 2 release approval and evaluation preflight

Added `src/training/release_preflight.py`, `docs/dataset_release.md`, and
`tests/test_release_preflight.py`. Inspected the consensus schema/loader and
annotation finalize audit, mechanistic verifier/extractor, evidence schema,
Phase 1 partition validator and their focused tests before implementation.
Existing annotation exports record attestations/hashes but do not grant release
approval; the existing mechanistic guard is reused without weakening it.

API: `preflight_release(manifest_path=None, *, consensus_path, mechanistic_path,
task, protocol, regime, assignments=None, partition_roles=None, topology=None)`.
Returns deterministic JSON-safe prerequisite statuses/reasons; `report_json`
serializes canonically without writing files. The versioned release manifest
binds exact consensus/mechanistic bytes, contract versions, release approval,
source/provenance registries, two reviewer approvals, separate adjudicator
approval, and independence attestations. Known source overlap with Task B
topology contradicts the independence assertion and fails. No production manifest,
reviewer identity, citation, verification status or approval was invented.

Reports distinguish missing prerequisites (blocked) from invalid/contradictory
records (failed), retaining every check when overall status is blocked. Include
release/input/manifest hashes, software/schema versions, request, counts and
validated structural partition manifest. Preserve all three consensus states;
unresolved/inconclusive/empty labels cannot enter the approved consensus table.
Require all three classes per requested role. Reuse the existing mechanistic
verification and feature extraction contracts to check finite feature coverage;
unknown evidence is never made a negative consensus label. The existing extractor
returns zero pathway contribution when either factor is a known null, even if the
other factor is unknown; otherwise unknown factors remain NaN. Reuse Phase 1 only for structural
validation. Current repository data return blocked for absent approval manifest,
empty consensus labels and unverified mechanistic rows, with downstream
prerequisites explicitly blocked and no performance metrics.

Validation commands and exact results (no full suite or Colab):

- `venv/bin/python -m pytest tests/test_release_preflight.py -q -p no:cacheprovider`
  — initial **1 failed, 18 passed in 2.59s** (synthetic test fixture attempted a
  string assignment into a float column); corrected the fixture.
- `venv/bin/python -m pytest tests/test_release_preflight.py tests/test_evaluation_protocol.py tests/test_consensus_contract.py tests/test_consensus_annotation.py tests/test_evidence_schema.py tests/test_mechanistic_boundary.py tests/test_ablation_leakage.py tests/test_redesigned_leakage_regression.py -q -p no:cacheprovider`
  — first **2 failed, 281 passed, 3 warnings in 4.38s**. One synthetic missingness
  fixture still had a known-zero factor, correctly yielding zero; corrected to
  make both target factors unknown. The other failure was the historical AUC
  guard described below.
- Same eight-file command after fixture fixes/additional cases — **1 failed,
  290 passed, 3 warnings in 4.42s**.
- `venv/bin/python -m pytest tests/test_release_preflight.py -q -p no:cacheprovider`
  — **27 passed in 2.59s** after improved per-partition support diagnostics.
- Final same eight-file command after explicit source overlap regression —
  **1 failed, 291 passed, 3 warnings in 4.46s**, exit 1. All **28 new preflight
  tests passed**. Only failure:
  `tests/test_ablation_leakage.py::test_no_actual_benchmark_model_exceeds_threshold`:
  historical non-reference `no_context_conflict` ROC-AUC **1.0000 > 0.98**.
  The guard and historical artifacts remain untouched; this failure is not
  suppressed, skipped, or repaired. Warnings: two existing suspicious-performance
  warnings from synthetic metric tests and joblib physical-core detection fallback.
- `git diff --exit-code fc28fa3 -- data results reports benchmarks tests/test_ablation_leakage.py tests/test_redesigned_leakage_regression.py src/training/evaluation_protocol.py`
  — exit 0, confirming canonical data, historical artifacts and leakage/Phase 1
  files unchanged from the starting commit.
- `git diff --check` — exit 0.

No-fit/no-mutation regression installs fit/calibration/write sentinels, checks
caller input equality, and hashes only explicitly tracked canonical/historical
paths plus synthetic input files before/after preflight. Private annotation
contents are excluded: no directory discovery or packaging occurs, and explicit
paths into `annotation_batch_24_seed20261029/` (including resolved symlink targets)
are rejected before reading or hashing. That pre-existing directory remains
untracked and untouched. No staging, commit or push.

Limitations/compatibility risks: this is an additive gate; existing runners must
explicitly adopt it before training/reporting. Approval references and source
metadata are structural human attestations, not authenticated identities or
scientifically verified evidence. Omitted relationships and aliased source IDs
cannot be inferred. Registries are explicit inline reference mappings; preflight
does not fetch institutional approval records. Three-class support in every role
and complete ten-column mechanistic features for both tasks are conservative v1
policies, not statistical adequacy guarantees; binary evaluation or partial
feature policies need separately versioned contracts. Feasibility concerns the
supplied assignments only; no partition search is attempted. Existing schema
contracts are explicitly named v1 here without changing legacy schema semantics.
Software/environment versions are included, so byte-identical reports require
matching environments. No new dependencies or performance claims. The historical
AUC failure remains unresolved by design within this Phase 2 scope.

## 29 October 2026 — Phase 2 adversarial-review corrections (v2)

This entry supersedes the initial Phase 2 API/three-class eligibility policy above.
Changed only `src/training/release_preflight.py`,
`tests/test_release_preflight.py`, `docs/dataset_release.md`, and this progress
file. No Phase 3 work, production approval, dependency, canonical data change or
scientific feature-semantic change was introduced.

The manifest, consensus CSV and mechanistic CSV are captured once as immutable
bytes. Exact SHA-256 hashes, duplicate-header checks, parsing and the existing
schema/mechanistic verification functions all use those same snapshots. No parser
reopens a path. Three synthetic replacement regressions replace each input after
capture with invalid bytes, require prerequisite success from the original bytes,
verify original reported hashes and assert one snapshot read per input. Snapshot
capture does not lock files for later consumers or guarantee multi-file atomic
publication; approved manifest hashes must match the independently captured CSVs.

Canonical approval/reference/source/entity/partition identifiers now reject all
whitespace before uniqueness, lookup and overlap checks; records are never
normalized or repaired. Tests cover exact reviewer duplicates, whitespace reviewer
aliases, whitespace on either side of consensus/topology source overlap,
noncanonical adjudicator/provenance references, and consensus IDs. Contradictory
duplicate JSON keys fail instead of last-key-wins parsing. Attestations require
exact JSON true, not truthy numbers/strings. References remain human attestations,
not authenticated proof of reviewer identity, independence or source validity.

Request strings and record/role containers are validated before processing
identifiers, set membership or coverage. Malformed task/protocol/regime values,
including lists, dictionaries, booleans and ambiguous NumPy arrays, return
JSON-serializable deterministic failed diagnostics rather than TypeError.
Assignment/topology inputs must now be lists/tuples of mappings (not generators
or DataFrames); nested identifiers and citation lists are checked before use.

Release/preflight schema versions are now 2, with target policy schema 1.
Both request and approved manifest must explicitly bind `binary_consensus_v1`
and `binary_discrimination_v1` for Task A/B. Only interaction/no_interaction
support is required in each requested role. Disputed records remain distinct in
the unchanged pair universe, partition validation, feature checks and reported
counts, outside binary targets. No target array is created. Unresolved and
inconclusive cases remain staging/evidence states and are rejected in approved
consensus input without coercion or silent dropping. Unsupported, missing and
manifest-mismatched policies fail closed. This matches existing Task A/B binary
targets; it does not certify probability calibration/conformal coverage merely
because a calibration partition exists.

`counts.per_role` now includes pair/herb/drug counts, declared evidence component
and constraint-group counts, plus per-class pair/group support including disputed.
Groups come directly from Phase 1's regime/topology components; overlapping
class-group counts cannot be summed as independent samples. No repository
contract defines defensible minimum independent sample sizes. Consequently the
`statistical_support` prerequisite is always blocked and `eligible` remains
false. `contract_status` separately reports implemented prerequisite
passed/failed/blocked status; it is explicitly not permission to fit/report.
Overall `status` retains all blocked gates and individual failures. No thresholds
or claim of statistically sufficient evaluation support was invented.

Investigated Task B `validate_studies` and `validate_study_sources`: the latter
requires source URLs/DOIs and registered references through
`DEFAULT_SOURCE_REGISTRY.verify_reference(..., require_registered=True)`.
Topology-only records cannot satisfy the actual study-release contract. Chose
the explicitly permitted narrowed scope: `task_b_evidence_release` is blocked,
with named missing gates, and report scope is
`dataset_and_partition_prerequisites_only`. Existing Task B source verification
was neither bypassed nor claimed to have run. A future actual evidence-release
contract would need bound evidence bytes and those gates; it is not implemented.

Corrected the earlier missing-evidence wording: the unchanged extractor returns
zero pathway contribution when either factor is known null, even if its partner
is unknown; otherwise unknown factors remain NaN. Zero pathway contribution is
not a no-interaction consensus label. Ten-feature completeness remains the
conservative preflight feature policy for both tasks.

Exact focused validation (no full suite or Colab):

- `venv/bin/python -m pytest tests/test_release_preflight.py -q -p no:cacheprovider`
  — **28 passed in 2.68s**, exit 0 after adapting the existing tests to narrowed
  scope/policies; then **78 passed in 3.97s**, exit 0 after bypass regressions.
- `venv/bin/python -m pytest tests/test_release_preflight.py tests/test_evaluation_protocol.py tests/test_consensus_contract.py tests/test_consensus_annotation.py tests/test_evidence_schema.py tests/test_mechanistic_boundary.py tests/test_task_a_training.py tests/test_task_b_training.py -q -p no:cacheprovider`
  — **350 passed, 1 warning in 6.13s**, exit 0; final rerun after additional
  canonical-reference, exact-true and malformed-input coverage:
  **360 passed, 1 warning in 6.24s**, exit 0. Includes **91 preflight tests**.
  Warning: existing joblib physical-core detection fallback. These existing
  training contract tests use synthetic fitting; preflight itself never fits.
- `venv/bin/python -m pytest tests/test_ablation_leakage.py::test_no_actual_benchmark_model_exceeds_threshold -q -p no:cacheprovider`
  — **1 failed in 1.38s**, exit 1: historical `no_context_conflict` ROC-AUC
  **1.0000 > 0.98**. This is the same unchanged baseline failure established in
  the prior read-only review, not suppressed or repaired here.
- `git diff --exit-code fc28fa3 -- data results reports benchmarks tests/test_ablation_leakage.py tests/test_redesigned_leakage_regression.py src/training/evaluation_protocol.py`
  — exit 0: canonical datasets, historical artifacts, AUC guard/tests and Phase 1
  remain unchanged from the starting commit.
- `git diff --check` — exit 0.

Compatibility/remaining limitations: v1 manifests require explicit reviewed v2
migration; no automatic approval upgrade. Callers must supply explicit target and
metric policies and adapt to the intentionally narrowed result: prerequisite
success never grants full eligibility. Statistical adequacy remains unestablished,
Task B evidence-release validation remains outstanding, source aliases/omitted
relationships cannot be independently discovered, and no external approval is
authenticated. The API is additive; existing runners still require adoption.
Reports are deterministic for the same snapshots, request and software environment.
No performance metrics or scientific validity claims are generated. Private
`annotation_batch_24_seed20261029/` contents were not inspected, discovered,
hashed, modified or packaged, and the directory remains untracked. No staging,
commit or push.

## 29 October 2026 — Phase 2 final four audit fixes

Changed only `src/training/release_preflight.py`,
`tests/test_release_preflight.py`, `docs/dataset_release.md`, and this progress
file, addressing the four remaining findings without changing target policies,
statistical-support blocking, the Task B topology-only limitation or mechanistic
semantics. `eligible` remains false; successful implemented prerequisites do not
establish scientific validity or full evaluation readiness.

Every source-registry entry is now validated before decision approval checks,
including unused entries. Reject malformed objects, missing fields, noncanonical
source/approval identifiers, missing metadata, invalid retrieval dates and
unapproved verification assertions. Reject repeated source IDs or exact citation
strings across registry entries, including the demonstrated same-citation alias
that selected a different ID to evade topology overlap. Multiple decisions must
reuse a single canonical registry entry instead of duplicating aliases. This is
recorded internal consistency only; different citations/URLs for the same real
source and fabricated approvals cannot be authenticated by this software.

All input snapshots have a 32 MiB operational limit, read with a limit-plus-one
bound. Oversize inputs fail the input check and have no hash; accepted snapshots
retain exact-byte hashing/parsing. CSV header and pandas parser failures are
converted to stable failed diagnostics. UTF-8 JSON nesting is checked lexically
before decoding (maximum 64 levels, ignoring escaped/quoted delimiters), with a
fallback RecursionError diagnostic. Reject NaN, Infinity, -Infinity and floating
point overflow such as 1e999 anywhere in a manifest, even unused fields. Invalid
non-string release versions cannot leak into a returned report. Reports from
these error paths remain canonically serializable and deterministic.

Manifest schema success is emitted after validation, canonical release-version
checks precede its success result, and the check accumulator guarantees one
result per prerequisite without replacing a failure with success. The test helper
now requires exactly one matching check instead of silently taking the first.
Added regressions for malformed/whitespace manifests and all demonstrated alias,
parser and serialization failures, plus valid unused sources, quoted JSON
delimiters and bounded input reads. Synthetic fixture metadata remain explicitly
synthetic. No scientific records or approvals were created or repaired.

Exact focused validation:

- `venv/bin/python -m pytest tests/test_release_preflight.py -q -p no:cacheprovider`
  — initial **2 failed, 89 passed in 4.08s**, exit 1: the existing consensus-source
  whitespace assertions expected failure in `decision_provenance`, whereas the
  new earlier all-entry `source_registry` gate correctly failed first. Updated
  those expectations to the earlier gate; no validation was weakened.
- `venv/bin/python -m pytest tests/test_release_preflight.py tests/test_evaluation_protocol.py tests/test_consensus_contract.py tests/test_mechanistic_boundary.py -q -p no:cacheprovider`
  — **323 passed in 5.32s**, exit 0, including **125 preflight tests**; no warnings.
- `git diff --exit-code fc28fa3 -- data results reports benchmarks tests/test_ablation_leakage.py tests/test_redesigned_leakage_regression.py src/training/evaluation_protocol.py`
  — exit 0, confirming canonical datasets, historical artifacts, leakage guard
  and its tests, and Phase 1 remain unchanged.
- `git diff --check` — exit 0.

Remaining limitations/compatibility: the new operational size/depth limits and
strict all-entry/alias checks can reject formerly accepted manifests and require
explicit human review/restructuring. Limits are parser/resource constraints, not
scientific thresholds. Source uniqueness uses exact recorded citation strings,
not online identity resolution. Statistical adequacy remains unestablished,
Task B actual evidence-release checks remain outstanding, existing runners still
require explicit adoption, and report reproducibility requires the same snapshots,
request and software environment. The known historical AUC guard failure remains
unresolved; neither that guard nor artifacts were modified or rerun in this fix.
No full suite, Colab, Phase 3 work, staging, commit or push. Private annotation
contents were not inspected, read, hashed, staged or packaged; the private
directory remains untracked.

## 29 October 2026 — Phase 3.1 minimal evaluation orchestrator

Implemented `src/training/evaluation_orchestrator.py` with versioned request and
report TypedDict schemas, runtime request validation, canonical serialization,
explicit caller-declared synthetic/empirical provenance, software/schema versions,
input hashes, plan hash and actionable prerequisite results. Documented the API
and limits in `docs/evaluation_orchestrator.md`.

Added a snapshot-based Phase 2 entry point sharing the existing contract. The
orchestrator captures each permitted input once; hashes and parsing consume the
same immutable bytes even when paths are replaced after capture. The gate
requires the literal Phase 2 `eligible=True`, a passed contract and passing
prerequisites. Contract success alone never grants eligibility. Task A retains
mechanistic requirements; Task B retains its separate actual-evidence-release
blocker and cannot claim readiness from topology alone. Current statistical
support remains blocked. Phase 3.1 always blocks because execution is not
implemented, even if a future preflight grants eligibility. Blocked reports omit
predictions and metrics, with no estimator/backend hooks or output path accepted.

Exact focused validation:

- `venv/bin/python -m pytest tests/test_evaluation_orchestrator.py tests/test_release_preflight.py -q -p no:cacheprovider`
  — first attempt: **1 collection error in 2.09s**, exit 2 (fixture import needed
  the `tests.` package prefix); second attempt: **3 failed, 137 passed in 5.03s**,
  exit 1 (bound default reader hid monkeypatched readers in existing Phase 2
  snapshot tests). Corrected the import and resolved the reader dynamically;
  existing Phase 2 tests were not changed.
- `venv/bin/python -m pytest tests/test_evaluation_orchestrator.py tests/test_release_preflight.py tests/test_evaluation_protocol.py -q -p no:cacheprovider`
  — intermediate **207 passed in 5.05s**, exit 0; after five additional edge-case
  tests, final **212 passed in 5.45s**, exit 0, no warnings. Includes 20 new
  orchestrator tests, 125 existing release tests and 67 partition tests.
- Runtime tripwires reject estimator module calls, fitting, calibration,
  prediction and scoring, plus write modes through builtins/Path open, across
  blocked Task A/B, invalid/missing/malformed inputs and contract-only success.
  Further tests cover replacement after capture, canonical deterministic reports,
  plan binding, inaccessible inputs, invalid request read prevention, missing or
  nonliteral eligibility flags and a true flag with failed prerequisites. The
  synthetic future-eligibility stub still cannot execute anything.
- `git diff --exit-code 999ca9cf385fb5c2024aa748af29b23a1204c097 -- data results reports benchmarks tests/test_ablation_leakage.py tests/test_redesigned_leakage_regression.py`
  — exit 0. Canonical data, historical artifacts and these guard tests unchanged.
- `git diff --check` — exit 0.

Limitations: no model adapters, fitting, scoring, calibration, conformal prediction
or bootstrap inference; no production execution path or bypass. No labels,
approvals, sample thresholds, metrics or coverage guarantees were invented.
Provenance is declared, not authenticated. Snapshots are per-file captures rather
than an atomic multi-file transaction; approved manifest hashes still bind the
approved versions. Plan hashes preserve supplied record ordering. Determinism
requires identical request, input bytes and software environment. Existing runners
are not migrated. Full Task B evidence-release validation and statistical adequacy
remain outside this phase. No full suite, AUC guard, Colab, staging, commit or push.
The private annotation directory remains untouched and untracked; its contents
were not inspected, hashed or packaged. Stopping here for review.

## 2026-10-29 — snapshot research components and strict role boundaries

Implemented `src/training/pipeline_components.py` and 27 synthetic regression
cases in `tests/test_pipeline_components.py`. Updated the production
orchestrator's blocker explanation and `docs/evaluation_orchestrator.md`.
No production execution switch or eligibility override was introduced.

Implemented stages:

- Immutable snapshot adapters with exact manifest hash binding and no path reads;
  explicit role validation preserves pair/herb/drug and Task B provenance-component
  constraints through the existing partition validator. Mutation detection binds
  copied role data and fitted estimator state.
- Working Task A LR/HGB fit and explicit candidate tuning interfaces use the
  existing approved feature builder and pipeline. Fit preprocessing sees only
  fit rows; tuning selects without refitting. Role composition supports all five
  roles and requests assessment targets only after predictions and sets exist.
- Strict ordered identity, vocabulary and probability validation; unavailable
  metrics carry explicit reasons. Average precision and trapezoidal PR-AUC are
  separate. No interval or significance inference is implemented. The new scorer
  retains the .98 AUC integrity alert; existing guard code/tests are untouched.
- Temperature calibration and binary split-conformal score-set interfaces with
  role provenance, fixed-predictor binding, mutation checks and explicit failed
  states. Calibration failure never silently becomes identity calibration;
  failed refits invalidate previous state. Unsupported coverage certification
  stays blocked, including after successful synthetic score-set construction.
- Actual Task B study-snapshot validation invokes existing study, registered-source
  and reliability feature checks and compares exact topology. The Task B snapshot
  adapter separately exposes prior/reliability fitting and residual tuning, both
  fail-closed pending an approved actual-evidence-release contract. It does not
  invoke the old path-based runner. This is an intentional implemented blocker,
  not a claim that Task B fitting or scientific evidence approval is complete.

Exact focused commands and results (no full suite or Colab):

1. `venv/bin/python -m pytest tests/test_evaluation_orchestrator.py tests/test_release_preflight.py tests/test_evaluation_protocol.py -q -p no:cacheprovider`
   — **212 passed in 5.48s**, exit 0.
2. `venv/bin/python -m pytest tests/test_pipeline_components.py -q -p no:cacheprovider`
   — **21 passed in 3.92s**, exit 0, one joblib physical-core detection warning.
3. `venv/bin/python -m pytest tests/test_pipeline_components.py tests/test_evaluation_orchestrator.py tests/test_release_preflight.py tests/test_evaluation_protocol.py tests/test_task_a_training.py tests/test_task_b_training.py tests/test_task_a_feature_contract.py -q -p no:cacheprovider`
   — **350 passed in 9.80s**, exit 0, same warning.
4. Final after added failure-state, composition, mapping and malformed-column
   cases: `venv/bin/python -m pytest tests/test_pipeline_components.py tests/test_evaluation_orchestrator.py tests/test_release_preflight.py tests/test_evaluation_protocol.py tests/test_task_a_training.py tests/test_task_b_training.py tests/test_task_a_feature_contract.py tests/test_redesigned_leakage_regression.py -q -p no:cacheprovider`
   — **366 passed in 11.01s**, exit 0, same warning. Includes existing leakage
   regression tests unchanged. No test failures occurred in these runs.
5. `git diff --exit-code 3e25e4e70093e46005d5950d8feb4cef9785c6cf -- data results reports benchmarks src/training/task_a.py src/training/task_b.py tests/test_ablation_leakage.py tests/test_redesigned_leakage_regression.py`
   — exit 0; historical data/results and existing Task A/B runners and named
   leakage guards unchanged.
6. `git diff --check` — exit 0.

Remaining blockers and compatibility limits:

- Phase 2 still returns actual `eligible=False`. Statistical sample adequacy has
  no approved repository contract. Current empty consensus, unverified mechanisms
  and incomplete Task B evidence validation remain blockers. No training,
  performance metrics or empirical evaluation was run on that release.
- Task B source/topology consistency cannot authenticate evidence, human approval,
  consensus independence or completeness. The actual study release still needs a
  reviewed schema/approval contract. Extra intermediate provenance nodes in the
  study-snapshot adapter are conservatively unsupported, though the existing
  partition validator still supports intermediate topology nodes.
- No exchangeability or finite-sample coverage guarantee is established. Binary
  conformal score calibration excludes disputed decisions; no guarantee is made
  about the broader disputed/unresolved population. No scientific cutoffs,
  statistical support thresholds, approvals, labels or citations were invented.
- Components are separate low-level APIs for direct testing, not production
  authorization. Snapshot structural validation is not a replacement for release
  preflight. Production remains blocked even under a synthetic future-eligibility
  stub. Historical runners are not migrated and retain their existing APIs.
- Temperature optimizer bounds are numerical limits; boundary optima fail.
  Fitted identities hash serialized state without deserializing input and depend
  on software versions. Component reports contain DataFrames rather than the
  production JSON schema. Snapshot capture remains per-file, not atomic.
- No known failing component tests; one environment-only joblib core-count warning
  remains. The full suite was intentionally not run.

Historical artifacts and existing leakage guards remain unchanged. The private
annotation directory remains untracked and untouched: no contents were inspected,
read, hashed, discovered, staged or packaged. Only tracked status was queried;
no staging, commit or push occurred. Stopping for review.

## 2026-10-29 — Phase 3.2 adversarial defect fixes

This entry supersedes the earlier claim that public component execution could be
kept separate from release authorization by documentation alone. The adversarial
review correctly reproduced a public eligibility bypass, arbitrary fitted-wrapper
acceptance, and snapshot rebinding with mismatched cached tables.

Changed files:

- `src/training/pipeline_components.py`: public structural validators plus a
  fail-closed execution facade. All new public fit/tune/predict/score/calibration/
  conformal/assessment paths reject execution through the orchestrator, including
  manually allocated objects and private synthetic predictors passed as inputs.
  No public synthetic flag, caller approval mapping or contract-only success can
  authorize execution. A returning gate function still cannot dispatch algorithms.
- `src/training/evaluation_orchestrator.py`: centralized authorization rejection
  and explicit blocker diagnostics. The orchestrator issues no execution grants.
  Actual Phase 2 eligibility remains false. A future grant issuer must originate
  here after preflighting exact captured bytes, require literal `eligible is True`
  and every prerequisite, and bind the complete plan/snapshot/fit provenance. No
  such approved issuance contract exists; the implementation deliberately fails
  closed instead of inventing one or enabling a positive execution path.
- New private `src/training/_pipeline_algorithms.py`: existing research algorithms
  moved behind a private test surface. It cannot issue production authorization.
  Snapshot binding is read-only and checks object identity, bytes, cached tables
  and role manifest before use; duck-typed caller table providers are rejected.
  Arbitrary fitted-estimator construction is rejected. Only successful fit-role
  fitting registers a private predictor with a frozen fit record: fit-role hash,
  task, explicit configuration, approved feature contract, canonical plan hash,
  exact snapshot hashes, derived-table hash and fitted-estimator hash. Registration
  binds the original data/estimator objects; absent, copied or forged records and
  changed state fail before prediction or calibration. Returned records are copies.
  Task B now checks role-data integrity before structural validation. Nonfinite
  temperature optimizer parameters and malformed binary-loss targets fail closed.
- `tests/test_pipeline_components.py`: existing algorithm tests now explicitly
  import the private module. Added public gate denial for caller-created grants,
  actual false-eligibility reproduction, manual predictor construction and forged
  records, snapshot/derived-table mutations and rebinding, copied registration,
  different-predictor calibration/conformal reuse, and nonfinite optimizer state.
  Informative overlapping synthetic features test unchanged fitted estimator state,
  selected configuration, calibration temperature, nonconstant probabilities and
  sets under changed assessment labels. There are now 42 component cases.
- `docs/evaluation_orchestrator.md`: corrected public/private execution contracts,
  provenance behavior, compatibility break and remaining blockers.

Exact focused validation (no full suite, Colab or skipped requested files):

1. `PYTHONDONTWRITEBYTECODE=1 venv/bin/python -m pytest tests/test_pipeline_components.py tests/test_evaluation_orchestrator.py -q -p no:cacheprovider`
   — initial refactor: **47 passed in 8.00s**, exit 0. After new regression cases:
   **1 failed, 59 passed in 11.18s**, exit 1. A test-body placement mistake left
   endpoint-calibration assertions in the wrong function, causing NameError for
   `monkeypatch`; restored those assertions to their original test. Rerun:
   **60 passed in 10.85s**, exit 0. Each run had the existing joblib core-count warning.
2. Final after the defensive provenance-copy and final failure-state cases:
   `PYTHONDONTWRITEBYTECODE=1 venv/bin/python -m pytest tests/test_pipeline_components.py tests/test_evaluation_orchestrator.py tests/test_release_preflight.py tests/test_evaluation_protocol.py tests/test_task_a_training.py tests/test_task_b_training.py tests/test_task_a_feature_contract.py tests/test_redesigned_leakage_regression.py tests/test_calibration_audit.py tests/test_conformal_abstention.py tests/test_conformal_overhaul.py tests/test_interaction_model_and_calibration.py tests/test_trainer_conformal.py -q -p no:cacheprovider`
   — **480 passed in 18.10s**, exit 0, one existing joblib physical-core detection
   warning (logical-core fallback). All algorithm fitting/scoring in these tests
   uses synthetic fixtures; no current-release empirical execution occurred.
3. `git diff --exit-code 3e25e4e70093e46005d5950d8feb4cef9785c6cf -- data results reports benchmarks src/training/task_a.py src/training/task_b.py src/training/evaluation.py src/training/evaluator.py src/training/ablation.py tests/test_ablation_leakage.py tests/test_redesigned_leakage_regression.py`
   — exit 0. Historical/canonical artifacts, existing runners and named leakage
   guards remain unchanged. The AUC > .98 guard was retained in private scoring.
4. `git diff --check` — exit 0. `git diff --cached --quiet` — exit 0.

Remaining blockers/precise limitations:

- All new public production execution is intentionally disabled, even for a caller
  claiming eligible=true. No reviewed issuance contract or scientific eligibility
  exists, so a production positive path was not added. Snapshot parsing and static
  feature construction remain structural/read-only operations, not authorization.
- Current consensus/mechanistic/statistical support and Task B evidence approval
  blockers remain. Task B prior/reliability fitting and residual tuning are still
  separate blocked interfaces. No real-data performance, scientific approval,
  intervals, significance, exchangeability or coverage guarantees were created.
- Private fit registrations establish algorithm provenance within this process,
  not evidence authenticity, source independence or release authorization. Python
  private modules are not a security sandbox: arbitrary code that imports private
  internals or rewrites module globals is outside the public API contract. We do
  not claim interpreter-level isolation. The public backend never dispatches to
  these test algorithms; private records cannot unlock it.
- Intentional compatibility break: previous public component execution calls now
  raise PermissionError. Internal algorithms and version-dependent pickle hashes
  are not portable production APIs. Existing historical runners are unchanged and
  not migrated. Snapshot capture remains per-file, not an atomic transaction.
- No remaining failures in the requested focused tests; the full suite was not run.

The private annotation directory remains untracked and completely untouched. Its
contents were not read, inspected, discovered, hashed, staged or packaged. Only
tracked status was queried; no staging, commit or push occurred. Stop for review.

## 2026-10-29 — research presentation, explanation, official-form and paper drafts

Reviewed the current public facade, private fit registration/snapshot checks,
orchestrator and regression tests. No further Phase 3.2 execution-code change was
needed: public execution remains unconditionally blocked, private predictors need
registered immutable fit-role provenance, and snapshot/table mutation is checked.
Current Phase 2 eligibility remains false. No empirical fitting or performance
reporting was enabled. Prior uncommitted pipeline changes are retained for review.

Implemented this session:

- Improved the existing Streamlit site rather than adding a duplicate framework.
  `app/interactive_app.py` and `app/frontend.py` now offer a Project presentation
  view and retain the existing read-only Evidence explorer. Both still stop
  before inactive legacy prototype execution. `app/project_presentation.py` and
  `app/assets/research_presentation.html` add an academic presentation with hero,
  problem, proposed approach, accessible ordered architecture flow, separate
  tasks, safeguards, status, limitations/next steps and actual local-document
  downloads. No external fonts/assets, fake metrics, new dependencies or API calls.
  Responsive grids, visible focus, native disclosures and reduced-motion styling
  are included. Real rendering tests cover both views and entry points.
- `explanation.md`: comprehensive beginner guide to architecture, terminology,
  task inputs/outputs, feature/data/source contracts, role/disjointness controls,
  setup, status, failure modes, glossary, future eligible workflow and professor
  pitches/questions. Established methods are separated from candidate novelty.
- `docs/IDF-B_draft.md`: review draft following the actual official repository PDF,
  `docs/Patent_Form_Details.pdf`, Document 02-IPR-R003, issue 2 / 1 February 2024. Preserves
  all ten numbered field names and the nine-level TRL structure; no TRL is ticked.
  Student/advisor information, approvals, timelines/budgets, prior-art verification
  and institutional declarations are explicitly marked for input. It must be
  transferred into the official form and reviewed; no signature/date/approval was
  invented. Older patent Markdown and the source PDF remain unchanged.
- `docs/research_paper_draft.md`: methodology/engineering draft with prominent empty
  consensus, unverified mechanisms and blocked evaluation disclosures in abstract,
  results/status and limitations. Related work explicitly awaits verified sources;
  no external citation or superiority/clinical/coverage claim was invented.
- `README.md`: current, accurate entry point replaces superseded clinical/
  performance headlines, links all deliverables and distinguishes historical work.
  Historical benchmark artifacts themselves were not modified.
- `docs/website.md`: launch/navigation/setup/design and verification instructions.
- `tests/test_project_presentation.py`: three cases check fragment destinations,
  download targets and actual Streamlit rendering/navigation between both views
  with fitting/prediction tripwires. `tests/test_app_research_status.py` changes
  only the fake View selector so existing explorer startup safeguards still run.

Exact checks/results:

1. One combined focused run after code changes:
   `PYTHONDONTWRITEBYTECODE=1 venv/bin/python -m pytest tests/test_pipeline_components.py tests/test_evaluation_orchestrator.py tests/test_release_preflight.py tests/test_evaluation_protocol.py tests/test_task_a_training.py tests/test_task_b_training.py tests/test_task_a_feature_contract.py tests/test_redesigned_leakage_regression.py tests/test_calibration_audit.py tests/test_conformal_abstention.py tests/test_conformal_overhaul.py tests/test_interaction_model_and_calibration.py tests/test_trainer_conformal.py tests/test_app_research_status.py tests/test_project_presentation.py -q -p no:cacheprovider`
   — **499 passed in 19.62s**, exit 0, one existing joblib physical-core detection
   warning (logical-core fallback). Includes the prior 480 cases, 16 app cases
   and three presentation cases. No failed/skipped test cases; no full suite,
   Colab or redundant test rerun. No eligible current-release model was fitted.
2. Website compilation (no JavaScript/Node build exists):

   ```sh
   PYTHONDONTWRITEBYTECODE=1 venv/bin/python - <<'PY'
   import py_compile
   paths=['app/frontend.py','app/interactive_app.py','app/project_presentation.py','app/research_status.py']
   for i,p in enumerate(paths):
       py_compile.compile(p,cfile=f'/private/tmp/hdi-website-{i}.pyc',doraise=True)
       print('COMPILED',p)
   PY
   ```

   — four files compiled successfully, exit 0; compiled outputs are outside the
   repository. Streamlit 1.62.0 AppTest renders both entry points and view switches
   without exceptions; download controls match all six fixed local documents.
3. Local preview:
   `PYTHONDONTWRITEBYTECODE=1 venv/bin/python -m streamlit run app/interactive_app.py --server.headless true --server.address 127.0.0.1 --server.port 8517 --server.fileWatcherType none --browser.gatherUsageStats false`
   — first attempt failed with sandbox PermissionError binding the socket.
   Approved escalation of the same command started Uvicorn/Streamlit successfully
   on localhost; file watching and usage telemetry were disabled. Native Chrome
   showed the presentation's accessibility structure and rendered document controls.
   Section-link URL changes were observed, but scrolling/mobile layout were not
   conclusively verified before native browser automation reported no windows.
   Thus browser visual QA is partial, not a full responsive/accessibility audit.
   No browser extension/IAB surface was available. Server stopped cleanly, exit 0.
4. Official form inspection: pypdf/pdfplumber and Poppler were unavailable in the
   active environment. Native Swift PDFKit extracted both source PDFs successfully;
   AppKit rendered both official-form pages to temporary PNGs, visually reviewed.
   Source PDF has ten numbered fields and a nine-column TRL matrix. No PDF was
   edited/generated and no packages installed. CoreGraphics emitted a PDF warning,
   but both official form pages rendered with readable headings and matrix.
5. A local Markdown-target check over README, explanation and the three new docs
   passed; the IDF-B numbered-section check found exactly 1–10 in order, exit 0.
6. `git diff --exit-code 3e25e4e70093e46005d5950d8feb4cef9785c6cf -- data results reports benchmarks patent src/training/task_a.py src/training/task_b.py src/training/evaluation.py src/training/evaluator.py src/training/ablation.py tests/test_ablation_leakage.py tests/test_redesigned_leakage_regression.py`
   — exit 0. Historical/canonical artifacts, original patent material, existing
   runners and named guards unchanged. No weakening of the AUC alert.
7. `git diff --check` — exit 0. Final tracked differences plus explicitly named new
   deliverables reviewed without untracked-tree discovery. No staging/commit/push.

Remaining limits: the scientific project is not empirically complete. Independent
consensus, verified complete mechanisms, actual Task B evidence approval,
statistical support/inference criteria and a reviewed authorization issuer remain
blocked. Public APIs stay disabled, not merely conditional on a caller report.
Private Python algorithms are not an interpreter security sandbox. External
literature/patent verification, institutional form review/TRL selection and a full
manual mobile/accessibility review remain pending. Drafts are not an approved
paper, patent submission or clinical system. No real-data performance claims.

The private annotation directory remains untracked and completely untouched:
contents were not read, inspected, discovered, hashed, staged or packaged. No full
repository discovery, packaging or Colab execution occurred. Stop for review.

## CI follow-up for f88f5fa

GitHub Actions run 40782632385, job 123147872166, failed at **Run unit test suite**
(installation succeeded): seven failures, 835 passes, two skips. Four app failures
were reproduced with pandas 3.1.0 / NumPy 2.4.6 and Python string storage: Enum
comparisons differ from Arrow storage. Normalize findings to their string values
before dtype inference and compare string values; add a Python-backend regression.
The report-number guard also mistook the official form's dotted issue date for a
metric; spell out the same date in this log and the IDF-B draft. Tests/guards are
not relaxed and historical metrics are not changed.

Targeted reproduction: four app cases failed under the CI string backend before
normalization. The seven CI failure cases locally produced two failures/five
passes (the genuine ignored metadata exists locally, and Arrow masked the enum
issue). A temporary clean-directory reproduction of the private-metadata check
failed as CI does, without reading or transferring that metadata.

Verification, with CI's pandas/NumPy installed only in /private/tmp:

```sh
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=/private/tmp/hdi-ci-pandas31:. venv/bin/python - <<'PY'
import pandas as pd
import pytest
pd.options.mode.string_storage='python'
raise SystemExit(pytest.main(['tests/test_app_research_status.py','tests/test_report_numbers.py','-q','-p','no:cacheprovider']))
PY
```

Result: **20 passed in 2.70s**, 13 existing Pandas4Warning messages from mechanistic
notes handling. `git diff --check` passed. No full suite or Colab run.

**CI is not yet green:** the unchanged AUC guard correctly rejects historical
`no_context_conflict` ROC-AUC 1.0; a clean checkout also lacks intentionally ignored
`data/benchmark/private/pair_strata.csv`. Neither can be honestly solved by changing
metrics, weakening tests, manufacturing dummy data or publishing private metadata.
They remain explicit blockers requiring separate scientific/data decisions. The
workflow, authorization gate and canonical/historical artifacts remain unchanged.
No staging, commit or push; the private annotation directory remains untouched.

## Independent evidence-review preparation — 30 October 2026

Implemented `src/data/review_packet.py` to prepare, not scientifically approve,
explicitly supplied candidate inputs. Each original CSV is read once and its exact
bytes saved in coordinator snapshots; protected-input hashes detect accidental
replacement. Identifier-only consensus assignments reuse the existing independent
review/adjudication workflow. Mechanism review rows start pending with blank source,
interpretation and reviewer fields. No label, verification, eligibility or model
prediction is generated. Output directories cannot overwrite existing work.

Added `docs/evidence_review_protocol.md` and a README entry explaining the CLI,
reviewer blinding, evidence/circularity checks, mechanism interpretation checklist,
and handoff to existing scientific release requirements. Coordinator snapshots
can contain labels/predictions and must not be distributed to blinded reviewers.
Hashes are not proof against manifest replacement or scientific authentication.
Mechanism worksheets cannot be promoted automatically to verified evidence.

Exact focused validation:

```sh
PYTHONDONTWRITEBYTECODE=1 venv/bin/python -m pytest tests/test_review_packet.py tests/test_consensus_annotation.py tests/test_annotation_batch.py tests/test_release_preflight.py tests/test_pipeline_components.py -q -p no:cacheprovider
```

Result: **201 passed in 14.77s**, one existing joblib physical-core detection
warning. Eight new synthetic tests cover unresolved/blinded preparation, exact
snapshots, original-file replacement, frozen-input mutation, malformed manifests,
duplicate headers/keys and missing schema columns. No real review packet or real
labels were produced. No full suite, network literature retrieval or empirical
training/evaluation was run.

Remaining prerequisites: advisor-approved outcome/inclusion rules; explicitly
selected permitted evidence inputs; independent qualified reviewers and adjudicator;
source-level mechanism verification; feasible disjoint roles; remaining Task B
scientific contracts and reviewed production authorization. Current eligibility
and disabled public execution are unchanged. Existing guards, canonical data,
historical artifacts and training runners were not edited. The excluded private
annotation directory was not inspected, read, discovered, hashed or modified.

## Public primary-source pilot and readiness diagnostics — 30 October 2026

Collected four actual primary-publication candidates through PubMed/PMC searches
for St John's wort pharmacokinetics and PXR activation. Added the public pilot at
`docs/review_batches/public_pilot_20261030/`, with three existing canonical pair
identities, three existing mechanism candidate rows, frozen preparation snapshots,
unresolved blinded workbooks, pending mechanism review and source summaries.
PubMed abstracts were inspected; indexed PMC mechanism Results/Figure 3 excerpts
were available, but direct PMC access challenged the browser. Full-text review,
independent human adjudication and scientific mechanism verification are not
claimed. Citation links, access limitations, retrieval date, pending status and
exact-string overlap counts are recorded in `source_candidates.json`.

Added `src/data/review_readiness.py` and five synthetic regression tests. The CLI
reads explicitly supplied snapshots, validates consensus structure, preserves
disputed cases, and reports necessary group-count conditions for the existing
protocol roles. It cannot issue eligibility, establish statistical support,
authenticate reference independence, or construct Task B components without
verified topology. The current pilot report has zero binary reference pairs;
canonical consensus remains empty. Even fully labeling this pilot would not
supply five disjoint nonempty roles. No partitions or scientific thresholds were
invented, and no fitting or performance reporting occurred.

Exact focused command:

```sh
PYTHONDONTWRITEBYTECODE=1 venv/bin/python -m pytest tests/test_review_readiness.py tests/test_review_packet.py tests/test_consensus_annotation.py tests/test_evaluation_protocol.py tests/test_evaluation_orchestrator.py tests/test_pipeline_components.py -q -p no:cacheprovider
```

Result: **162 passed in 11.08s**, one existing joblib core-detection warning.
An additional explicit-path integrity check verified packet hashes, regeneration
of the stored readiness report, pending source statuses and membership of all
pilot pair identities in the canonical evidence table. No full suite or Colab.

Remaining work cannot be replaced by software assertions: independent reference
review, source-level biological interpretation and quantitative feature mapping,
expanded evidence coverage, Task B scientific contracts, statistical adequacy
criteria, and a reviewed production authorization issuer. The public APIs and
Phase 2 gate remain blocked. Canonical data, historical results, existing runners
and leakage guards are unchanged. Private annotation contents were never accessed
or discovered. No staging, commit or push.

## Expanded mechanistic curation panel — 30 October 2026

Corrected the presentation: the primary baseline has five CYP enzymes plus one
transporter, not six enzymes plus a transporter. Mechanistic evidence still comes
from scientific sources; publication outcomes/text are excluded as predictors.
Added `src/data/mechanistic_panel.py`, `tests/test_mechanistic_panel.py` and
`docs/mechanistic_panel_expansion.md`; updated README and explanation. The versioned
curation scope covers seven enzymes and nine transporters, grounded in FDA Table
1-1 and Table 3-1 (links and scope limits in the new documentation). FDA relevance
is not herb-specific evidence or approval of model features.

Generated `docs/review_batches/expanded_panel_20261030/` using:

```sh
PYTHONDONTWRITEBYTECODE=1 venv/bin/python -m src.data.mechanistic_panel --curation data/curation/mechanistic_features.csv --output docs/review_batches/expanded_panel_20261030
```

The generator captured exact input bytes and prepared 522 additional unverified,
not-tested entity-target candidates for 58 existing entities. Numeric fields are
blank, not zero. These are a curation queue, not measured data or model inputs.
The primary allowlist and general extractor remain unchanged. Receptor-mediated
regulation and transporter compartments require separate feature-design review.

Validation:

```sh
PYTHONDONTWRITEBYTECODE=1 venv/bin/python -m pytest tests/test_mechanistic_panel.py tests/test_task_a_feature_contract.py tests/test_pipeline_components.py -q -p no:cacheprovider
```

Initial result: 114 passed, one failed, one existing joblib warning, in 10.94s.
The new test incorrectly treated the existing verification tuple as a Boolean;
fixed it to assert the exact `(False, 0, 15)` result. All 113 unchanged feature/
component tests passed in that run. Focused regression after correcting the test
and changing presentation copy:

```sh
PYTHONDONTWRITEBYTECODE=1 venv/bin/python -m pytest tests/test_mechanistic_panel.py tests/test_project_presentation.py -q -p no:cacheprovider
```

Result: **5 passed in 2.24s**. `git diff --check` passed. Explicit diff checks confirm
canonical data, historical results, training modules, primary feature contract
and named leakage guards unchanged. No full suite or real-data fitting. Private
annotation contents remain untouched. No staging, commit or push.

Broader model execution remains unimplemented pending verified measurements,
reviewed feature semantics/missingness policy and scientific eligibility. More
candidate targets do not demonstrate increased accuracy; an eligible independent
comparison of fixed panels remains future empirical work.
