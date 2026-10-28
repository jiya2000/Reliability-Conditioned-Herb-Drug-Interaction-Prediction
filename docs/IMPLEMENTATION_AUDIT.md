# Implementation audit

Audited 28 October 2026 against `PROJECT_SPEC.md`, including the current uncommitted extraction repair. Inspected repository inventory, data/schema contracts, model and training entrypoints, evaluation, tests, configs/CI, Streamlit/API, and reporting artifacts. No applicable repository `AGENTS.md` was found. Only this audit document was edited; no implementation was performed. Existing result files are historical artifacts, not evidence that the redesign works.

## Verified dataset inventory

Counts below were recomputed using Python `csv.DictReader`, excluding headers. Publication authenticity and clinical conclusions were not independently verified online.

| File under `data/` | Verified contents |
|---|---|
| `benchmark/verified_botanical_hdi.csv` | 141 rows and distinct study IDs; 120 distinct pairs; 18 herbs; 40 drugs |
| Same table, records per pair | 105 pairs have one record, 11 have two, two have three, two have four; record multiplicity does not establish independent studies |
| `benchmark/consensus_labels.csv` | Header only: zero consensus targets |
| `benchmark/consensus_labels_pairs_to_label.csv`, `consensus_labels_pairs_shuffled.csv` | 120 candidate rows each; not additional labels |
| `benchmark/consensus_labels_random_unstudied.csv` | 100 candidate rows; not verified clinical negatives |
| `curation/mechanistic_features.csv` | 722 rows; all sources are `UNVERIFIED` |
| `external/napdi_template.csv` | Five template records; executed external-validation test reports four pairs overlapping its training partition and zero study overlaps |

`data/drugbank.tab` and `data/twosides.csv` contain HTML 403 responses. `data/raw/` is absent. `data/manifest.yaml` declares expected assets and licenses; those declarations do not establish availability or verified licensing.

## Implemented and incomplete

| Area | Status and evidence |
|---|---|
| Studies and provenance | **Implemented infrastructure:** `src/data/evidence_schema.py:86`, `source_registry.py:58`, and `public_data_loader.py:62` provide records, source registration, and real-data validation. The CSV retains study findings as `outcome`, plus design fields and citation IDs. It lacks dedicated sample-size, assay-type, and standardized-extract fields; `evidence_schema.py:481` extracts only part of the specified design contract and supplies defaults for missing route/preparation. Registry checks are not independent contemporary source authentication. |
| Consensus truth | **Blocking/missing:** the table is empty. `src/models/aggregation.py:35` blocks empty labels, but its loader accepts any nonempty label without validating provenance or class. `src/data/evidence_schema.py:384–432` validates IDs/classes, not unique pairs, mandatory citations or annotation agreement; `:435` permits legacy-label fallback. No completed blinded double-annotation/κ evidence was verified. |
| Mechanistic features | **Boundary repaired, scientific input blocked:** `src/data/mechanistic_curation.py:69–247` validates measurement roles/ranges/statuses and maps inhibition/induction polarity times drug fraction/substrate weight. It preserves unknowns as NaN and rejects unverified tables even with `allow_unverified=True`. The old nonexistent `value` access is gone. Output is nine target-activity features plus absolute sum, not the completed clinical prior: max aggregation, NTI, sourced tissue reachability, and clinical exposure direction remain absent. `docs/mechanistic_features.md` explicitly describes these limits. |
| Task A | **Incomplete:** `src/models/feature_switched_models.py:87–155` offers curated/legacy feature switching and LR/HGB/GAT comparisons, but uses study `outcome` targets; LR has no missingness handling for the new NaN output. `scripts/run_phase5_task_a.py:93–203,694` uses legacy profiles and study labels and appropriately marks results exploratory. No complete consensus-supervised pair-level Task A pipeline exists. |
| Task B | **Scaffolding only:** `src/models/aggregation.py:71–163` and `aggregation_model.py:27–124` duplicate aggregation models with different defaults. They add unrestricted learned evidence deltas to a fixed scalar prior, rather than implementing pair-specific prior + nonnegative coefficient × shrunk normalized evidence. Agreement targets and vote baselines exist, but the training entrypoint explicitly raises `NotImplementedError` (`aggregation.py:301`). Learned echo utilities exist in `citation_dag.py:450`; they do not establish integration into the specified residual model or complete ablation family. |
| Reusable components | Citation DAG/lineage, PK hepatic/gut branches, context parsers, masking, calibration/conformal classes, grouped splitters, bootstrap/statistical utilities and shortcut probes exist (`src/models/{citation_dag,pk_reachability,conformal_abstention}.py`, `src/features/text_masking.py`, `src/training/{evaluation,calibration,statistical_tests}.py`). They require clean end-to-end integration and clinical validation. |

## Correctness, leakage and delivery blockers

- **P1 — Wrong prediction task/raw text:** `benchmarks/run_verified_hdi_benchmark.py:240–306` encodes raw abstracts and predicts publication outcomes. `src/training/evaluation.py:1094–1180` repeats this legacy architecture. Training-only vocabulary fitting does not prevent held-out outcome-word shortcuts or replace independent consensus truth.
- **P1 — Outcome-derived reliability input:** `src/data/evidence_lineage.py:139–152` includes contradiction in `quality_only`; `:416–480` derives it from positive/negative study findings. It is not a strictly design/lineage reliability input. Streamlit uses full metadata (`app/interactive_app.py:383–416`), including polarity. Keep findings confined to Task B aggregation and audit actual model tensors.
- **P1 — Assessment-label calibration:** `src/training/evaluation.py:1254–1278` fits pooled temperature on outer-test labels and evaluates those same predictions. It also combines rescaled probabilities with previously computed decisions. Calibration must be learned within outer training groups; uncertainty intervals must respect repeated predictions and pair/entity dependence.
- **P1 — Pair and citation separation are separate modes:** `src/training/evaluation.py:663–690` groups by citation component OR pair. Pair-disjoint mode does not also guarantee citation-component separation. Task B needs joint grouping and assertions before using cross-pair citation dependencies.
- **P1 — Streamlit is not the redesigned system:** `app/interactive_app.py:360–459` trains legacy fusion at startup on study outcomes with a single validation split. Queries reuse full-dataset abstracts (`:591–624`); zero literature still enters fusion rather than a guaranteed pair-specific Task A prior. The conformal engine uses an unfitted default (`:573–580`; `src/models/conformal_abstention.py:286`), so coverage is unsupported. Hardcoded legacy headline claims remain (`:939–960,1027`). Browser interaction was not tested.
- **P2 — Alternative entrypoints diverge:** `main.py:128`, `src/data/data_pipeline.py:45,172`, and `configs/*.yaml` retain synthetic/DrugBank graph/text training. `src/data/real_data.py:88` encodes identity through ID-seeded vectors; `src/data/dataset.py:143` samples unobserved pairs as negatives. These are auxiliary development paths, not valid consensus-labeled HDI tasks. `app/backend.py:319–348` maps unknown entities to existing graph nodes by hash; `:442` generates demo probabilities. Unify artifact and inference contracts before deployment.
- **P2 — Claim provenance is partial:** `tests/test_report_numbers.py` checks decimal membership anywhere in result JSON, not claim/run/model/split correspondence. README, paper/patent, Streamlit and multiple report/result directories retain legacy claims. No redesigned performance claim is established by this audit.

**Spec ambiguity:** R1 prohibits polarity in “any model,” while Task B explicitly requires findings. Define the prohibition for Task A and reliability inputs, permitting findings only in the Task B evidence aggregator. Also distinguish target-activity sign from clinical drug-exposure sign before finalizing Task A features.

## Verified tests and limits

- `venv/bin/python -m pytest -q -p no:cacheprovider`: **236 collected; 234 passed, two failed, 22 warnings** on this audit run.
- `tests/test_ablation_leakage.py:201`: stored non-reference `no_context_conflict` result exceeds the high-AUC guard. Investigate data/features and artifact provenance; a failed guard alone does not identify the root cause.
- `tests/test_external_validation_overlap.py:88`: expects five two-class conformal sets; execution produces four. Reconcile behavior and expectation using calibration/trigger diagnostics rather than changing the assertion blindly.
- Targeted `tests/test_mechanistic_boundary.py` + `tests/test_phase3_and_ci_checks.py`: **46 passed, one warning**. Synthetic fixtures establish extraction behavior, not scientific citation validity.
- All **156 repository Python files** returned by `rg --files -g '*.py'` passed `ast.parse`. `.venv/bin/python` lacks pytest; the separate `venv` environment was used.
- Existing tests cover many module/split/lineage/PK/calibration invariants, but do not establish consensus-supervised training, specified residual aggregation, disputed-pair abstention or deployed UI correctness. `.github/workflows/ci.yml` exists; fresh-environment CI was not executed. Tests can generate report/output artifacts; tracked working-tree scope was checked afterward.

## Prioritized implementation plan

1. **P1 — Freeze contracts and curate truth:** resolve R1/sign semantics; require unique consensus pairs, external citations, circularity notes, retrieval dates and blinded double annotation. Keep disputed cases out of binary fitting; unstudied candidates are not negatives.
2. **P1 — Complete the mechanistic foundation:** curate per-value sources; specify units, missingness, NTI, max/sum summaries and tissue reachability. Preserve provenance blocking and add downstream missingness handling. Do not redo the already repaired extraction boundary.
3. **P1 — Build clean Task A:** fit consensus-supervised pair-level LR/HGB and rule baselines; use repeated herb/drug/pair-and-cluster disjoint assessment. Keep identities/raw text explicitly labeled as leaky diagnostics.
4. **P1 — Build Task B:** consolidate scaffolds; cross-fit agreement-supervised reliability on clean design/lineage inputs, implement the exact residual equation/constraints and echo tuning, and test collapse to each pair’s prior. Match vote baselines and all specified ablations.
5. **P1 — Isolate evaluation:** inner OOF calibration, untouched outer assessment, joint grouping, disputed/hard-negative/sparse-rich slices, reliability AUC/ECE, grouped intervals and corrected paired comparisons. Resolve both current failing checks.
6. **P2 — Deliver reproducibly:** isolate test outputs, reproduce in a fresh environment, persist model/calibrator/data manifests, connect one Streamlit/API inference path, and regenerate all claims from linked result fields.

**Assessment:** substantial reusable infrastructure and legacy experiments exist. The redesigned scientific system remains blocked by absent consensus truth and verified mechanisms, and lacks the specified clean training, aggregation and evaluation pipeline.
