# Evidence preparation and human review

This protocol prepares work for scientific review. It does not establish predictive
accuracy, authenticate reviewers, or grant release eligibility. No new real-world
labels or mechanism verifications have been produced by this workflow.

## Decisions to make with the advisor before annotation

Write down the intended outcome, population, exposure/formulation, dose context,
and evidence inclusion criteria. Decide whether the reference label describes an
externally established interaction or another endpoint. A mechanistic effect is
not automatically clinical harm. Specify eligible reference sources and document
how overlap with model evidence will be excluded or audited. Record the protocol
version and reviewer instructions before reviewers see predictions or results.
These decisions remain pending; this document does not approve them.

## Prepare a packet

Use explicitly selected, permitted CSV inputs. The pair table must contain
`herb_id` and `drug_id`; the mechanism table must use the existing
`CURATION_COLUMNS` schema in `src/data/mechanistic_curation.py`. Do not point this
command at a private annotation workspace. Do not publish raw review packets
without checking privacy and source redistribution permissions.

```sh
venv/bin/python -m src.data.review_packet prepare \
  --pairs /path/to/permitted_candidate_pairs.csv \
  --mechanisms /path/to/permitted_mechanistic_candidates.csv \
  --output /path/to/new_review_packet
venv/bin/python -m src.data.review_packet verify \
  --directory /path/to/new_review_packet
```

The output directory must be new. Each source file is read once; its exact bytes
are saved under `coordinator_snapshots/`. The manifest hashes those bytes and the
normalized assignment table. `verify` detects accidental replacement; a person
who can rewrite both the manifest and snapshots can defeat these hashes. This is
an audit aid, not a security boundary or a production snapshot authorization.

Keep coordinator snapshots away from blinded reviewers: they can contain old
labels, predictions, or other excluded columns. Distribute the separate reviewer
workbooks from `consensus_review/`, which contain pair identities, unresolved
labels, and blank review fields. The generator ignores supplied labels and scores
when constructing assignments. Mechanism review belongs to a separate evidence
curation process; do not merge its notes into Task A's label review.

## Independent consensus review

Assign two qualified reviewers and a separate adjudicator using the existing
`src.data.consensus_annotation` workflow. Each reviewer should independently
consult approved external reference evidence and record:

- Source name, exact citation, and retrieval date.
- Rationale and relevant population/exposure context.
- Reviewer identity and truthful blinding/independence attestations.
- Circularity checks: whether the reference reuses the same studies or claims as
  the model inputs, and the actual origin of the reference conclusion.

Use only the existing label vocabulary. `interaction`, `no_interaction`, and
`disputed` are explicit decisions; `unresolved` remains a staging state. Absence
of a paper or an unsupported mechanism is not a negative reference label.
Disagreement is not automatically resolved by voting or converted to `disputed`.
A reviewer may leave a case unresolved with a documented explanation.

```sh
venv/bin/python -m src.data.consensus_annotation finalize \
  --directory /path/to/new_review_packet/consensus_review \
  --output /path/to/new_adjudicated_export
```

The existing finalizer checks structural completeness and distinct reviewer IDs,
exports adjudicated decisions into a new directory, and preserves unresolved
pairs. Names and attestations cannot prove expertise, independent work, or source
truth. Human oversight must establish those facts. This export does not update
canonical labels or make a release eligible.

## Mechanism review checklist

`mechanism_review.csv` starts every entity-target candidate as `pending`. It is a
review worksheet, not an ingestible verified curation table. For each candidate:

1. Confirm entity identity, formulation/constituent and biological target.
2. Retrieve the original source; record DOI/PMID or another traceable identifier
   and an exact supporting passage, table, figure or page location.
3. Record assay/species/population, direction, units and experimental conditions.
4. Check measurement interpretation, relevant exposure and applicability. Keep
   measured effects, tested no effect and not tested distinct.
5. Record uncertainty, conflicting reports, rationale, reviewer and review date.
6. Have a qualified reviewer check the interpretation. Only then propose a
   separate change to the actual curation table and source registry for review.

The generator deliberately does not infer verification from a DOI, a registry
flag, a worksheet status, or a syntactically complete citation. Existing curation,
source and evidence-to-topology checks remain mandatory. No automated promotion
from this worksheet to production evidence is provided.

## After review, before evaluation

Review sample feasibility and independent provenance components. Select the
intended pair-, herb-, or drug-disjoint generalization question before examining
assessment results. Task B also needs source-component isolation. Freeze separate
fit, tuning, probability calibration, conformal calibration and assessment roles;
do not silently relax an infeasible assignment. Do not pick sample counts,
statistical thresholds, or coverage guarantees merely to make execution possible.

Run the existing release preflight against the proposed release through the
reviewed release workflow. All prerequisites must pass. Production execution APIs
currently remain disabled pending an approved authorization issuer and remaining
scientific contracts; this packet never enables them. Calibration, abstention and
performance claims require an eligible release, completed execution interfaces,
and an appropriate prespecified evaluation and inference design.

## What can be done now

Engineering can prepare reproducible packets, validate structure, detect duplicate
assignments and preserve audit records. Reviewers must supply independent labels
and verified biological interpretations. Literature retrieval and a real review
batch remain pending explicitly approved inputs and an agreed scientific scope.
No network retrieval, real-data fitting or empirical evaluation occurs here.

Focused synthetic checks:

```sh
PYTHONDONTWRITEBYTECODE=1 venv/bin/python -m pytest tests/test_review_packet.py tests/test_consensus_annotation.py tests/test_annotation_batch.py tests/test_release_preflight.py tests/test_pipeline_components.py -q -p no:cacheprovider
```
