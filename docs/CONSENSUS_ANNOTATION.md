# Independent consensus annotation workflow

The canonical `data/benchmark/consensus_labels.csv` currently contains **0 labels**. Candidate tables and individual publication findings are not ground truth. This workflow stages human decisions separately and never edits the canonical table or trains a model.

## Prepare assignments

### Prepared 24-pair batch: human handoff

`annotation_batch_24_seed20261029/` has been verified against its identifier-only
candidate hash, seeded selection, selection manifest and assignment hashes.
It contains 24 pairs covering 18 herbs and 23 drugs. Both reviewer workbooks
contain exactly those pairs, with every decision `unresolved` and all review
metadata blank; no publication findings, predictions or outcome-derived inputs
are included. The blank `consensus_label` decision slot is initialized to the
staging status `unresolved`, not a scientific label.

Distribute separately (paths relative to the repository root):

| Recipient | Send only this workbook |
|---|---|
| Reviewer A | `annotation_batch_24_seed20261029/review_assignments/reviewer_a.csv` |
| Reviewer B | `annotation_batch_24_seed20261029/review_assignments/reviewer_b.csv` |

Provide each reviewer this guide and independent external-reference access.
Do not send the whole batch directory or share a folder granting access to both
workbooks. These are separate regular files; filesystem separation alone does
not enforce reviewer access controls. The coordinator must arrange separate
delivery and keep each returned review private until both reviews are locked.

Keep these files private to the coordinator during independent review:

- `annotation_batch_24_seed20261029/selection_manifest.json`
- `annotation_batch_24_seed20261029/selected_pairs.csv`
- `annotation_batch_24_seed20261029/review_assignments/manifest.json`
- `annotation_batch_24_seed20261029/review_assignments/pairs.csv`
- `annotation_batch_24_seed20261029/review_assignments/adjudication.csv`

Preserve frozen assignments/manifests and original review returns. Keep
`adjudication.csv` with the coordinator until both independent reviews lock;
then provide it and the locked reviews to a separate third adjudicator. Every
resolved decision requires explicit source-backed adjudication, including
agreements. Keep reviewers blinded to model predictions, project publication
findings and one another's decisions during independent review. Missing evidence
remains unresolved; never infer a negative or invent a citation. Source details,
access dates, rationale, attestations and circularity checks follow the fields
below. This handoff does not certify scientific validity or actual delivery.

From the repository root, create a new directory (the command refuses an existing directory):

```sh
venv/bin/python -m src.data.consensus_annotation init \
  --pairs data/benchmark/consensus_labels_pairs_to_label.csv \
  --output annotation_round_1
```

The existing study table has 141 publications covering 120 distinct pairs; the candidate file above covers those 120 pairs. To include the separate 100 unstudied candidates, add `data/benchmark/consensus_labels_random_unstudied.csv` after `--pairs`. These candidates are not verified negatives. You may also supply another pair CSV or the study table itself: only validated `herb_id` and `drug_id` are retained, duplicates are collapsed, and any input findings/labels are discarded. All assignments start `unresolved` with blank provenance and reviewer fields. No real template-generation command was executed during implementation validation.

Give `reviewer_a.csv` and `reviewer_b.csv` separately to two qualified reviewers. Each should see only their own workbook, identifiers, and independent external references until their review is locked. The coordinator retains `pairs.csv` and `manifest.json`; do not edit these frozen assignments. This workflow assigns every pair to double review, exceeding the specification's minimum subset requirement if completed. It does not claim that assignment alone constitutes completed review.

## Complete independent reviews

Keep the column names and pair identifiers unchanged. Use a CSV-aware editor and preserve identifier strings. Do not copy labels from study outcomes, abstracts, model predictions, candidate lists, or legacy profiles. Consult authoritative external clinical references specified in PROJECT_SPEC.md; source access and scientific judgment must be supplied by human reviewers.

| Field | Reviewer action |
|---|---|
| `consensus_label` | Choose `interaction`, `no_interaction`, `disputed`, or leave `unresolved`. Unresolved is a staging status only; the training contract does not accept it. |
| `source_name` | Name the actual external compendium/reference, including edition/version where available. |
| `source_url_or_citation` | Provide an exact monograph/section/page, stable URL or bibliographic citation supporting this pair-level verdict. For disputed cases identify the conflicting authoritative references or explicit conflict statement. Multiple references can share this quoted CSV cell. |
| `retrieval_date` | Actual source-access date in YYYY-MM-DD format. |
| `reviewer_id` | Stable reviewer identifier, supplied by the reviewer; not a generated decision. |
| `rationale` | Explain how the external source supports the label, its clinical context and scope (preparation, route, dose), and any limitations. For unresolved cases explain what is missing or inaccessible. |
| `blinded_to_predictions` | Enter `yes` only if the review was performed blind to model predictions. |
| `external_consensus_only` | Enter `yes` only if the verdict was obtained from external curated consensus rather than derived from the project's publication findings. |
| `circularity_notes` | Document upstream primary citations, possible overlap with project evidence, cross-reference checks and remaining dependencies. No external source should be assumed independent merely because it is a compendium. |

A completed unresolved response requires reviewer ID, rationale, both attestations and circularity notes; unavailable citations need not be invented. Untouched unresolved rows remain pending. Never interpret absence of a monograph, an unstudied pair, or a mechanistic hypothesis as `no_interaction`. Reviewer disagreement alone is not `disputed`: that label needs support from authoritative clinical sources. If independence or source support cannot be established, retain unresolved status. Structural validators cannot authenticate citations or detect dishonest attestations; source and independence review remain human responsibilities.

## Adjudicate and export

Lock the original independent review files before discussing disagreements. A separate adjudicator fills `adjudication.csv` using the same fields and their own reviewer ID. Document both original positions and the source-backed reason for the final choice in `rationale`. All resolved decisions, including agreements, require explicit adjudication in this conservative workflow. An adjudicator may retain `unresolved` even when reviewers agree. Do not overwrite the original review decisions after discussion; they are needed for pre-adjudication agreement analysis.

```sh
venv/bin/python -m src.data.consensus_annotation finalize \
  --directory annotation_round_1 \
  --output annotation_export_1
```

The output must be a new directory. Resolved exports require documented responses from two distinct reviewers plus a third adjudicator, provenance validation through the existing consensus contract, and unchanged assignment membership. A reviewer may document an unresolved response without inventing source metadata; only the adjudicator's source-backed resolved verdict can become an export row. No voting or automatic disagreement resolution occurs.

Outputs:

- `consensus_labels.csv`: only explicitly adjudicated resolved decisions, including rationale and adjudicator metadata; compatible with the existing loader. May be header-only.
- `unresolved_pairs.csv`: all pairs still lacking a final verdict.
- `audit.json`: completed double-review count, pre-adjudication disagreement pair identifiers, input/output hashes and limitations. Incomplete reviewer slots are not counted as completed disagreements.
- `annotation_snapshot/`: exact assignment, manifest and review files supporting that export. Keep these with the export; hashes alone cannot reconstruct a decision.

Before promoting any rows to the canonical table, a human data steward must inspect sources, circularity, identities, blinding and resolution notes. Preserve the snapshot and report pre-adjudication agreement on a prespecified eligible subset (including how unresolved responses were handled); compute Cohen's kappa only when actual adequate reviews exist. This tool supplies raw review records and completion counts, not a fabricated agreement statistic or scientific certification. Promotion/merging into the canonical table is deliberately manual and must retain one unique row per pair and the audit trail. No clinical validation is implied by a structurally valid export.

## Command and training boundary

The CLI has exactly two subcommands: `init` and `finalize` (inspect with
`venv/bin/python -m src.data.consensus_annotation --help`). Review and
adjudication are manual CSV operations, not CLI subcommands. `finalize` performs
structural validation and export together; there is no standalone `validate`
command. A successful header-only export means cases remain unresolved, not that
training is ready.

After human source review, check an export directly without promoting it:

```sh
venv/bin/python -m src.training.task_a \
  --consensus annotation_export_1/consensus_labels.csv \
  --curation data/curation/mechanistic_features.csv --check-only
```

This checks labels, verified mechanistic coverage and grouped fold feasibility
without fitting. Empty exports and unresolved staging workbooks fail closed.
Disputed labels are valid consensus records but excluded from binary training;
both binary classes and adequate groups are required. Current real data cannot
pass this readiness check. Passing it would not establish clinical validity.
