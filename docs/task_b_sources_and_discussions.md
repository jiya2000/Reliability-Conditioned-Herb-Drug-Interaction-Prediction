# Task B sources and discussion-signal intake

## What is actually available

On 30 October 2026, the existing canonical evidence CSV contains 141 evidence
rows, 140 distinct source identifiers, 141 distinct study identifiers and 120
herb–drug pairs. Its design fields say 121 clinical, 16 case reports and four
RCT records. These are table inventories, not independently authenticated counts
of papers, patients or independent experiments. Source aliases and citation reuse
can change the effective independent evidence count. The filename “verified” and
membership in the registry do not establish that all scientific interpretations
have been checked.

The recent public pilot adds four publication candidates for review, separately
from the canonical table. It contains three pair assignments and one mechanistic
publication; these are not four new eligible Task B studies or independent
reference labels. PubMed abstracts were consulted and indexed PMC mechanism
excerpts inspected. Full-text and human interpretation review remain incomplete.
The expanded 16-target panel adds curation work, not additional publications or
measured target values.

The pilot links are in [its README](review_batches/public_pilot_20261030/README.md).
PubMed indexes literature; PMC hosts accessible full texts. A PubMed record is
traceability, not a guarantee that conclusions generalize. Task B requires actual
source verification, correct outcome extraction, context and shared-source/citation
provenance, together with independent consensus. There is no scientifically
justified fixed paper quota. We need enough independent evidence and reference
pairs for the requested disjoint roles, not simply more publication rows.

## Correct interpretation of Task B

The implemented components distinguish study findings from design/lineage
features. Citation graphs can expose *recorded* repetition, not prove that every
review or citation is an echo. Primary/echo annotations need verification;
independent replication must not be discarded as repetition.

Agreement supervision against external consensus is the intended reliability
learning target. A human trial is not assigned a universally valid trust score
merely by its type. An absence of detected interaction is not a negative risk
score for all contexts. Inconclusive findings contribute no directional signal;
a low reliability weight need not imply mathematically zero adjustment. The
production snapshot adapter still blocks prior/reliability fitting and residual
tuning pending the scientific release contract. This is not a deployed system
that has already established how much to trust every paper.

## Should discussions be considered?

Yes, as a distinct **hypothesis and review-triage source**. A legitimate platform
can host inaccurate reports. Confounding, self-selection, uncertain products,
missing doses/timing, reposts and publicity bias prevent treating discussion volume
as independent causal evidence or a denominator for event rates.

The [ENCePP methodological guide, chapter 8](https://encepp.europa.eu/encepp-toolkit/methodological-guide/chapter-8-approaches-data-collection_en)
discusses digital-media data and notes that cited studies did not demonstrate
earlier signal detection in their settings. The [FDA FAERS reporting/review material](https://www.fda.gov/media/165667/download)
explains that spontaneous reports do not establish causation. FAERS is a reporting
system, not a social discussion platform. Faster publication is not proof of a
more reliable signal.

## Implemented interface

`src/data/discussion_signals.py` accepts explicitly supplied, minimal metadata in
a new quarantine format. It does not scrape platforms or export publication
records. This keeps existing scientific data contracts unchanged.

Input is a JSON list with exactly these string fields:

| Field | Meaning |
| --- | --- |
| record_id | Local opaque record identifier; never an account name |
| platform | Source platform name; not a reliability classification |
| source_url | HTTPS locator without credentials, query parameters or fragments |
| herb_id, drug_id | Explicit reviewed entity mappings, not guessed from text |
| event_term | Minimal nonidentifying reported-event term, not a confirmed outcome |
| retrieval_date | ISO calendar date |
| thread_id | Opaque local thread key for grouping, not author identity |
| repost_of | Local record ID of known repost, or empty string |
| collection_permission_ref | Reference to externally reviewed collection/access permission |
| privacy_review_ref | Reference to externally reviewed privacy/ethics handling |

Those references are declarations, not approvals authenticated by this code.
Metadata fields can still contain identifying information if misused; manual
privacy review is required. The schema rejects extra fields such as raw narratives,
usernames, clinical labels and caller-provided reliability weights. No input-byte
snapshot is copied into output. Only a digest and validated metadata are written.
Do not publish intake outputs containing personal information or sensitive URLs.

```sh
venv/bin/python -m src.data.discussion_signals \
  --input /path/to/permitted_reviewed_metadata.json \
  --output /path/to/new_quarantine_directory
```

`docs/templates/discussion_signals.json` is an empty input template. No real social
posts have been collected: platform-specific access permissions, an actual
permitted source and privacy handling have not been provided. The implementation
is tested with synthetic metadata, never portrayed as real experiences.

Every accepted record remains quarantined with `task_b_eligible=false`, no
consensus label and `evidence_type=social_signal`, which is not a permitted
scientific EvidenceType. Same-thread, exact-URL and explicit repost connections
form review components. Unlinked copies and shared patient identity are not
reliably detectable; component counts cannot become patient counts. There is
no automatic promotion, reliability scoring or adjustment of predictions.

## Project integration and limits

Tests check previous public snapshot/readiness artifacts, new expanded curation
integrity, rejection of social rows by Task B validation, and public execution
blocking. This verifies compatibility at the supported preparation boundaries;
it does not mean these candidate files are eligible for model training. No paper
or social signal has been silently merged into canonical evidence or reference
labels. Future use as model evidence requires a separately reviewed schema,
source-specific validation and independence protocol, reference evaluation,
permission/privacy review and actual release eligibility.
