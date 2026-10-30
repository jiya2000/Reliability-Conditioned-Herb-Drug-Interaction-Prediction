# Work completed in this conversation

**Follow-up update:** a separate real-data assay workflow has now been implemented
and trained. See section 11 below. Sections 1–10 preserve the earlier session's
status; statements that no assay model has been trained are historical.

Status recorded: 30 October 2026. Repository:
`Reliability-Conditioned-Herb-Drug-Interaction-Prediction`.
Latest pushed implementation commit: `55e6ae2` on `origin/main`.

## Current position

We have working research-pipeline safeguards, a presentation website, research
explanations/drafts, evidence-review preparation tools and a real official-source
ChEMBL assay dataset. **We have not established predictive accuracy for clinical
herb–drug interactions. No real-data model was trained in these implementation
and source-population steps.**

The original datasets remain unchanged at the user's request. They are not a
verified empirical release. Downloaded assay data is a separate compound–target
dataset, not a replacement for whole-herb measurements or clinical reference
labels. Public Task A/B execution remains disabled and Phase 2 eligibility false.

## 1. Pipeline safeguards and research presentation

Completed Phase 3.2 work includes snapshot-bound inputs, role isolation, predictor
fit provenance, strict scoring and calibration/conformal interfaces. Public
production APIs remain blocked: caller-created contexts, synthetic fixtures and
reports claiming success cannot authorize execution. Private synthetic algorithms
exercise components without enabling a production bypass.

The existing Streamlit application was extended with a project presentation and
read-only evidence explorer. Added beginner-friendly `explanation.md`, an IDF-B
technical draft following the repository's official form, a research-paper draft
and website instructions. Drafts explicitly describe absent scientific validation;
no institutional approval, clinical readiness or empirical success was invented.

Relevant starting points:

- `src/training/pipeline_components.py`
- `src/training/_pipeline_algorithms.py`
- `src/training/evaluation_orchestrator.py`
- `app/interactive_app.py`, `app/project_presentation.py`
- `docs/website.md`, `explanation.md`
- `docs/IDF-B_draft.md`, `docs/research_paper_draft.md`

Website command:

```sh
venv/bin/python -m streamlit run app/interactive_app.py
```

## 2. CI diagnosis and targeted fixes

CI for `f88f5fa` exposed a pandas string-storage difference: comparing string enums
inside pandas produced incorrect evidence counts on the CI backend. Converted
findings to ordinary string values before comparison and added a regression.
An official form's dotted issue date also triggered a historical metric-number
scanner; spelled out the same date without changing the form's meaning.

The targeted CI-backend verification passed 20 tests. Two unrelated pre-existing
CI blockers remained and were later reconfirmed in CI #29 for `e3cb602`:

1. `tests/test_ablation_leakage.py:201` rejects historical
   `no_context_conflict` ROC-AUC 1.0 because it exceeds the unchanged 0.98 guard.
2. `tests/test_phase4_scaffolding.py:348` requires intentionally ignored
   `data/benchmark/private/pair_strata.csv`, absent on a clean GitHub runner.

CI #29 reported **873 passed, two failed, two skipped**. Both failures were
reproduced using targeted checks; the discussion-intake tests passed. No historical
score was rewritten, private metadata fabricated or leakage test disabled.
**CI is not claimed green.**

## 3. Evidence-review preparation

Implemented reproducible preparation tools:

- `src/data/review_packet.py`: captures exact explicitly supplied input bytes,
  creates unresolved blinded consensus assignments and pending mechanism review.
- `src/data/review_readiness.py`: reports reference-label availability and necessary
  group-count conditions for existing role protocols; never grants eligibility.
- `docs/evidence_review_protocol.md`: instructions for source review, independence,
  circularity checks, mechanism interpretation and adjudication.

Created `docs/review_batches/public_pilot_20261030/` with three candidate pairs
and four actual publication candidates:

| Publication | Official PubMed link |
| --- | --- |
| St John's wort–digoxin study | https://pubmed.ncbi.nlm.nih.gov/10546917/ |
| St John's wort–cyclosporine study | https://pubmed.ncbi.nlm.nih.gov/12580993/ |
| Comparative St John's wort/Echinacea–digoxin study | https://pubmed.ncbi.nlm.nih.gov/18214850/ |
| PXR/CYP3A4 mechanism study | https://pubmed.ncbi.nlm.nih.gov/10852961/ |

All four pilot PMID/DOI pairs matched downloaded NCBI metadata. This verifies
bibliographic consistency, not full scientific interpretation or independent
clinical consensus. The pilot remains unresolved and is too small for the full
five-role evaluation protocol.

## 4. Expanded biological scope

Corrected an inaccurate description: current primary Task A uses **five CYP
enzymes plus one transporter**, not six enzymes plus a transporter:
CYP3A4, CYP2C9, CYP2C19, CYP2D6, CYP1A2 and ABCB1. The six effects plus their
absolute sum form seven primary model inputs. Signs express target activity,
not a universal clinical-risk direction.

Added `src/data/mechanistic_panel.py` and an expanded curation scope containing
seven CYP enzymes and nine transporters. It generated 522 additional **unverified
candidate rows**, not measurements, under
`docs/review_batches/expanded_panel_20261030/`. The fixed primary allowlist was
not silently expanded. FDA tables support general target relevance, not specific
herb effects. See `docs/mechanistic_panel_expansion.md`.

“Biology only” means exclusion of publication outcomes, raw abstracts and entity
encodings as features. It does not mean the biological measurements can be
obtained without reading scientific sources.

## 5. Discussion-platform signals

Added `src/data/discussion_signals.py`, tests and
`docs/task_b_sources_and_discussions.md`.

The new intake accepts explicitly supplied minimal metadata, requires permission/
privacy references, rejects extra raw/user fields and caller-supplied labels or
weights, and groups known thread/URL/repost relationships. Every output remains
quarantined with `task_b_eligible=false` and no consensus label. Declared permission
references are not authenticated approvals. Unlinked duplicates remain a limitation.

No real social posts were collected. Discussion frequency does not establish
causality, independence, incidence or reliability. No discussion records were
merged into scientific evidence or used to adjust predictions.

## 6. Critical corrections from the source audit

Earlier descriptions of the original data as verified were too strong. Direct
source retrieval established the following:

| Original table | Actual findings |
| --- | --- |
| `data/benchmark/verified_botanical_hdi.csv` | 141 evidence rows, 140 source IDs, 120 pairs. Source IDs do not establish 140 independent verified papers. |
| `data/curation/mechanistic_features.csv` | 722 rows, 58 entities, 14 target identifiers. All statuses `not_tested`; all assay-value fields blank; every source/citation UNVERIFIED. This is a generated curation grid, not laboratory measurements. |
| `data/benchmark/consensus_labels.csv` | Header only, zero decisions. No clinical panel has supplied reference labels. |
| `data/external/napdi_template.csv` | Five example records, not an authenticated NaPDI export; all five DOI/PMID pairs mismatch current PubMed metadata. |

Downloaded original PubMed metadata/available abstracts for the canonical IDs:
139 unique PMIDs requested, 131 returned. Eight unique missing PMIDs correspond
to nine evidence rows. Across all 141 rows: one matching DOI, 130 mismatched or
uncomparable cases, nine missing-PMID rows and one non-PubMed row. Some returned
papers concern unrelated subjects. This is a substantial source-integrity problem,
not merely missing formatting.

Added `src/data/pubmed_download.py` and source-audit reports. Official clinical
reference pages from NIH NCCIH and NCI PDQ were downloaded separately. They are
not ready-made independent consensus labels. No verified official benchmark
matching our exact pairs/outcome/independence requirements was identified.

`docs/DATASET_SOURCE_AUDIT.md` covers dataset roles and source links. It also notes:

- HERB and Dryad are declared resources, but checked expected local inputs are
  absent. Dryad source filenames differ from loader expectations.
- InterPAD is a phytochemical–anticancer-drug research resource, not automatically
  a whole-herb clinical consensus benchmark.
- The DrugBank derivative lacks sufficient authenticated local release/license/
  extraction provenance; it is auxiliary, not herb–drug reference truth.
- Manifest declarations such as `license_verified: true` are not fresh proof.

The user subsequently instructed that original datasets remain unchanged. They
were preserved historically rather than automatically corrected with guessed
mappings or replacement labels.

## 7. Real official ChEMBL data downloaded

Initially downloaded 361 human-target IC50/Ki records for quercetin, CHEMBL50,
from EMBL-EBI. Added `src/data/chembl_assay_review.py` to retain provenance,
preserve censoring and flag unsupported/ambiguous records. This compound-specific
batch is not a replacement for all herbs and drugs.

To avoid the multi-gigabyte full ChEMBL download, implemented the API importer
`src/data/chembl_dataset.py` and populated a larger separate dataset directly
from official ChEMBL_37.

Local and committed dataset directory:
`data/external/chembl_assays_20261030/`.

| Target | Downloaded records |
| --- | ---: |
| CYP3A4 | 6,642 |
| CYP2D6 | 3,349 |
| CYP2C9 | 3,107 |
| CYP2C19 | 1,737 |
| CYP1A2 | 1,764 |
| ABCB1/P-gp | 2,234 |
| Total | 18,833 |

The dataset contains **10,554 distinct compounds**, **2,036 referenced source
documents** and **16,018 structurally screened candidate records**. These are
record counts, not independent clinical samples or accuracy results.

Retrieval filters: human SINGLE PROTEIN targets resolved through exact UniProt
accessions; IC50, exact relation, nM units, non-null pChEMBL and declared API
assay-confidence score 9. All pages were retrieved. Raw responses, structures,
assay descriptions, document identifiers/DOIs/PMIDs and integrity hashes remain
available. Target-assignment confidence is not clinical reliability.

Important files:

- `assay_records.csv`: all downloaded records and review flags.
- `assay_research_candidates.csv`: structurally screened subset.
- `manifest.json`: filters, version, completion status, counts and hashes.
- `raw/response_*.json`: 46 exact API responses.
- `docs/chembl_api_dataset.md`: provenance, commands and usage limits.
- `docs/source_audits/chembl_assay_dataset_20261030.json`: audit summary.

Actual validation checked hashes, complete pagination/counts, activity uniqueness,
source-to-CSV identities, per-target counts, positive finite quantities, structure
presence and continued exclusion from production eligibility. The first ad hoc
validation script mistook an API status count for a list; it was corrected and
all checks passed on rerun. No downloaded records were changed to obtain a pass.

This provides real data for a future retrospective **compound-assay prediction**
workflow. It does not supply induction effects, drug clearance fractions,
whole-herb composition/exposure, independent clinical labels or a trained model.
The original production Task A/B gate was not bypassed.

## 8. Validation record

Focused runs only; no full local suite, Colab or empirical model fitting.
Representative results, with exact commands and additional details in
`docs/PROGRESS.md`:

| Work | Result |
| --- | --- |
| Earlier pipeline/frontend focused validation | 499 passed; engineering checks, not accuracy |
| CI pandas/backend correction | 20 passed, existing pandas warnings |
| Review preparation and release/component checks | 201 passed, one existing joblib warning |
| Public pilot/readiness and role checks | 162 passed, one existing joblib warning |
| Expanded panel initial run | 114 passed, one new-test failure: Boolean misuse of a tuple return; corrected |
| Corrected panel/presentation checks | 5 passed |
| Discussion plus broad focused integration | 552 passed, one existing joblib warning |
| PubMed parser | 4 passed |
| ChEMBL review/panel/feature checks | 83 passed, one existing joblib warning |
| Source/consensus/preflight checks | 164 passed |
| API importer and review adapter initial run | 17 passed |
| Final API importer, adapter, component and orchestrator checks | 80 passed in 11.06s, one existing joblib warning |

Final API-related test command:

```sh
PYTHONDONTWRITEBYTECODE=1 venv/bin/python -m pytest tests/test_chembl_dataset.py tests/test_chembl_assay_review.py tests/test_pipeline_components.py tests/test_evaluation_orchestrator.py -q -p no:cacheprovider
```

Passing engineering tests do not establish biological correctness, predictive
performance, statistical significance, uncertainty validity or clinical readiness.

## 9. Commits pushed during this conversation

| Commit | Work |
| --- | --- |
| `f88f5fa` | Fail-closed pipeline stabilization and research presentation drafts |
| `ea986ad` | CI evidence counting and form-date formatting |
| `ff44a44` | Evidence-review workflow and public literature pilot |
| `5d0c41e` | Expanded mechanistic curation scope and corrected baseline description |
| `e3cb602` | Quarantined discussion intake and integration checks |
| `c6a43df` | Verified CI #29 blocker diagnosis |
| `bbf2fd1` | Official-source provenance audit and retrieval tools |
| `55e6ae2` | Populated ChEMBL API dataset, captures, importer and tests |

The last commit explicitly includes the separate populated assay dataset:
49 data files, approximately 45.6 MB, plus implementation/documentation changes.
Earlier temporary PubMed, quercetin-only and guidance downloads were not included
in that publication. Their audit summaries document local temporary locations;
those temporary files are not guaranteed to persist indefinitely.

## 10. Remaining work

For an assay-based professor demonstration, the next substantive task is a
separate retrospective compound-assay modeling/evaluation workflow: prespecified
endpoint and filters, duplicate/conflict handling, appropriate molecular features,
compound/source separation, honest metrics and website integration. It has not
been implemented or run yet. The downloaded data makes that work possible; it
is not itself the predictive experiment.

For the original clinical herb–drug aim, unresolved requirements include trusted
study mappings, appropriate verified mechanism/context data, independent reference
labels, remaining Task B scientific contracts, feasible role assignments,
statistical adequacy/inference design and reviewed production authorization.
A smaller retrospective assay project can proceed without new laboratory work,
but must not be presented as validated clinical herb–drug prediction.

Throughout these changes, original canonical datasets, historical benchmark
artifacts, existing training runners and AUC/leakage guards were preserved.
`annotation_batch_24_seed20261029/` was not inspected, read, discovered, hashed,
modified, staged or included in any publication.

This `tillnow.md` is a conversation summary, not a scientific approval or an
assertion that the project or CI is complete.

## 11. Follow-up implementation: real-data assay experiment

Implemented `src/assay/` with source-integrity verification, conservative
inhibition-assay screening, canonical structure grouping, document-balanced
aggregation, explicit conflict statistics, compound/scaffold/source splitting,
four regression comparisons, validation-only model selection, separate conformal
calibration, test predictions, bootstrap MAE intervals and saved inference bundles.
Source splitting purges structures crossing roles and logs every removed activity.
An independent verifier recomputes saved metrics and checks split separation.

Executed two actual experiments from the official local ChEMBL dataset:

- Scaffold holdout: selected-model pooled MAE about 0.50 pIC50 over 2,116 test pairs.
- Publication holdout with compound purging: pooled MAE about 0.66 pIC50 over
  2,076 test pairs. Several target-specific R² values are negative. Some observed
  interval coverages are below nominal; the reports retain those limitations.

The assay-consistency weighting ablation uses training-measurement dispersion.
It is not the proposed agreement-supervised clinical reliability model. No
whole-herb inference, herb-to-compound mapping, clinical labels or exposure values
were fabricated. Original data and clinical Task A/B gates remain unchanged.

Full fitted artifacts are local under ignored `artifacts/assay/scaffold/` and
`artifacts/assay/source/`. Summary JSON, split audits, Markdown reports and saved
test-prediction CSVs are retained separately under `results/` and `reports/`.
No commit or remote push was performed in this follow-up.

The Streamlit site now has a **Compound assay experiment** view showing actual
results and an explicit SMILES inference form. It does not train on startup.
Inference returns assay log-potency intervals, descriptive molecular similarity
and training-compound flags, never a clinical interaction verdict.

Validation: 219 focused assay/source/frontend/component/orchestrator/preflight
tests passed, with one existing joblib core-count warning; three report-number
integrity tests passed separately. Both actual runs passed the independent
verifier. Streamlit AppTest rendered the new view and submitted SMILES successfully,
producing six target estimates. No full-suite or CI-green claim is made.

Commands, endpoint definition, model settings, split semantics, outputs and
remaining clinical requirements are documented in `docs/assay_workflow.md`.

## 12. Presentation-ready integrated research demonstration

Added `app/demo.py`, `app/screening_demo.py`, `src/assay/screening.py` and the
separate `data/demo/public_evidence.json` catalog. The catalog contains four
pair-specific records from three PubMed human publications, with abstract-level
source checks, explicit comparator orientation and pending independent review.
It does not repair or replace the historical dataset or create consensus labels.

The interface combines herb/drug selection, live hyperforin or curcumin assay
inference, cited pathway mappings, human study findings, duplicate-citation
handling, preparation-context checks, an induction/inhibition guard, validation
metrics and a downloadable JSON trace. Echinacea has no invented constituent
structure. Clinical abstention remains explicit; no clinical predictive accuracy
or scientific novelty is asserted.

Launched locally at `http://127.0.0.1:8501` using the dedicated demo entry point
and opened it in Chrome. Verified the actual model chart in the browser.
Twenty-three focused screening, assay, presentation and report-integrity tests
passed, including live AppTest flows. See `docs/presentation_demo.md` for restart
instructions and a short professor-facing walkthrough. No remote publication
or commit was performed.
