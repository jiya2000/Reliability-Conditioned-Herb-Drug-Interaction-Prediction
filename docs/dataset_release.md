# Dataset release approval and preflight v2

`src.training.release_preflight.preflight_release` checks dataset and partition
prerequisites before supervised Task A/B work. It does not grant evaluation or
scientific approval. No repository contract supplies defensible independent
sample-size thresholds; therefore `statistical_support` is always blocked and
`eligible` is always false. Task B additionally blocks `task_b_evidence_release`:
topology is checked, but the actual study release is not supplied to this API.

`contract_status` (`passed`/`failed`/`blocked`) summarizes the implemented dataset,
policy and partition checks, excluding those two explicitly unresolved gates.
It must never be used as permission to fit or publish performance. Overall
`status` retains all unresolved gates and reasons; `scope` is
`dataset_and_partition_prerequisites_only`. Existing runners do not automatically
invoke this additive API. No fitting, calibration, scoring or writes occur here.

Call with an explicit JSON manifest path and keyword arguments `consensus_path`,
`mechanistic_path`, `task` (`A`/`B`), `target_policy`, `metric_policy`, `protocol`,
`regime`, `assignments`, `partition_roles`, and (for B) `topology`. Protocols,
regimes, assignment and source/citation topology records follow
`evaluation_protocol.py` unchanged. Record containers must be lists/tuples of
mappings, including parsed citation lists; generators and DataFrames are rejected.
Malformed arguments produce failed prerequisite checks, not unchecked set/hash
operations. Invalid non-string request values are represented as null in reports.
`report_json(report)` returns canonical JSON without writing files. A missing
manifest yields blocked diagnostics.

## Manifest format

The release manifest is a JSON object with these required v2 fields. Unknown
fields confer no approval. There is deliberately no approved production example:
only synthetic test fixtures currently demonstrate prerequisite success.

| Field | Contract |
| --- | --- |
| `schema_versions` | Exact mapping `{"release_manifest":"2","preflight":"2","target_policy":"1","consensus":"1","mechanistic":"1","evidence":"1","evaluation_protocol":"1"}` |
| `target_policy`, `metric_policy` | Must explicitly match the requested supported policies below |
| `release_version` | Nonempty release identifier |
| `approval_status` | `approved`, `pending`, or `unapproved`; only `approved` passes |
| `input_sha256` | Object with `consensus` and `mechanistic`: exact lowercase SHA-256 of each complete CSV's bytes |
| `approval_refs` | Nonempty list of release approval record references |
| `source_registry` | Reference-keyed mapping to source records described below |
| `provenance_registry` | Reference-keyed mapping to objects with a nonempty `reference` to the provenance audit |
| `mechanistic_provenance` | Mapping from every exact mechanistic `source` value to an object with `reference` and `approval_ref` |
| `decisions` | Exactly one approval decision for every consensus pair, no extras or duplicates |

Duplicate JSON keys are rejected, including contradictory repeated approval fields.
JSON must be UTF-8 (an optional UTF-8 BOM is accepted), with at most 64 nested
object/array levels. Nonstandard constants (`NaN`, `Infinity`, `-Infinity`) and
numbers that overflow to infinity are rejected, including in unused fields.
Every input snapshot is limited to 32 MiB; at most limit-plus-one bytes are read
to detect oversize inputs, which receive a failed input check and no hash.
CSV header/parser failures and excessive JSON nesting produce deterministic
failed diagnostics, not uncaught parser exceptions. Reports never retain an
invalid non-string release version.
Version 1 release manifests fail closed; migration requires explicit policy binding
and review, never an automatic approval upgrade.

Versions identify the contracts adopted here, including the existing consensus,
mechanistic and evidence schemas, which previously had no machine-readable
version constants. Any future semantic change requires a version review.

Approval, reviewer, adjudicator, registry and independence reference identifiers,
release versions, source/study IDs, partition names and entity IDs must be
nonempty canonical strings without whitespace. Boundary whitespace is rejected
before lookup, uniqueness and overlap checks; no normalization/repair occurs.
Free-text citations and provenance descriptions are not identifier fields.

Each source registry record has `source_name`, `source_url_or_citation`, and
`retrieval_date` matching the corresponding consensus CSV row, plus a canonical `source_id`,
`verification_status: "approved"` and an `approval_ref`. A source may be reused
across decisions when its metadata match. These are recorded human assertions;
preflight does not visit URLs, verify citations, or authenticate approval records.
Every source-registry entry is checked, even when unused by a decision. Each must
be an object with the required fields, canonical source/approval IDs, nonempty
source metadata, an ISO retrieval date and recorded `approved` verification.
Source IDs and exact citation strings must each be unique across registry entries;
reuse one canonical entry for multiple decisions. Duplicate entries and a citation
assigned multiple source IDs are rejected rather than treated as independent
sources. Different citation spellings/URLs for the same real publication cannot
be independently resolved here; this is internal consistency, not authentication.

Each decision has canonical `herb_id`, `drug_id`, an unchanged `consensus_label`,
`approval_status: "approved"`, `source_ref`, `provenance_ref`,
`reviewer_approval_refs` (exactly two distinct approval record references),
`adjudicator_approval_ref` (distinct from both), and
`independence_approval_ref` pointing to the independence audit. References must
resolve institutionally to the actual reviewers/adjudicator; distinct strings
alone cannot prove distinct people. The `attestations` object must explicitly set
all four fields to JSON boolean `true`:

- `independent_reviewers`
- `blinded_to_predictions`
- `external_consensus_only`
- `independent_of_evaluation_evidence`

The provenance audit must substantiate these assertions, including independence
of consensus sources from the evaluation evidence. The software checks presence
and internal consistency; it cannot independently establish their truth or detect
omitted evidence relationships. An explicitly recorded consensus `source_id`
shared with Task B topology contradicts the independence attestation and fails. Annotation `finalize` audit hashes/review records
can be referenced by a release approval, but an annotation export never grants
release approval automatically. Do not reference the private annotation directory
from this contract. Public release provenance must be approved separately.

## Target and metric policies

Partition protocols select separated roles; they do not define the target.
Only these explicitly versioned combinations are supported:

| Task | Target policy | Metric policy | Minimum class contract |
| --- | --- | --- | --- |
| A | `binary_consensus_v1` | `binary_discrimination_v1` | At least one interaction and one no_interaction pair in every requested role |
| B | `binary_consensus_v1` | `binary_discrimination_v1` | Same binary support; actual evidence release remains unchecked |

This matches the existing Task A/B binary target vocabulary. All approved
`disputed` records remain in the pair universe, partition checks, feature checks
and class/group counts; they are outside binary targets, never negative and never
required to exist. This API creates no target array or filtered export.
`unresolved` is staging only and inconclusive findings remain study evidence;
keep them in separate staging/evidence files. They fail validation in an approved
consensus CSV rather than being dropped or coerced. Missing, unsupported or
manifest-mismatched policies fail closed. This metric policy checks binary class
support for discrimination only: it does not certify probability calibration,
conformal coverage, or metric estimability from sufficient independent samples.
Calibration/conformal partition protocols still check their declared role
separation and binary support, without a corresponding statistical guarantee.

## Diagnostics and independent support

Checks have `passed`, `failed`, or `blocked` status plus a reason. Missing files,
approval, attestations, labels, topology coverage, or features block. Invalid
schemas/arguments, hash mismatches, contradictory approvals, insufficient binary
class support, and infeasible supplied partitions fail. Overall status is
`blocked` if any prerequisite is missing/unestablished, otherwise `failed` if any
check failed. Each prerequisite has exactly one result; failures cannot be
overwritten by a later passed result. Individual failures remain visible. `contract_status` gives failure
precedence over missing implemented prerequisites; `eligible` stays false.

`counts.per_role` reports pair, herb, drug, evidence component and constraint-group
counts plus class-specific pair and constraint-group counts, including disputed.
Constraint groups are exactly the Phase 1 components: pair groups under pair
regime, shared herbs under herb regime, shared drugs under drug regime, joined
with declared evidence components for Task B. Different class-specific counts
can include the same group, so they must not be summed as independent groups.
No evidence components are inferred for Task A without topology. Counts encode
recorded structural constraints, not proven statistical independence. No numeric
adequacy threshold is invented: the always-blocked statistical support check
prevents any claim of sufficient evaluation support.

The existing mechanistic validator/verifier is reused unchanged, then the existing
feature extractor computes only the nine target features and summary to check
finite completeness. It does not fit a model. Any unverified row blocks the
entire input. Missing features block; `not_tested` stays unknown and an explicitly
sourced `tested_no_effect` retains its distinct meaning. This policy currently
requires all ten features for all pairs, including for Task B. No imputation,
scientific record repair, or legacy feature fallback is authorized.

Task B requires at least one topology record for every approved pair, plus the
existing source/citation component separation. Task A requires mechanistic
coverage. Distinct pair/herb/drug counts, label counts, and missing feature cell
counts are reported where computable. The structural validator decides feasibility
of the supplied assignments only; it does not search for alternative assignments.

Each input file is read once into an immutable bytes snapshot. Header checks,
CSV parsing and existing schema/verifier calls use those snapshots only, and
SHA-256 hashes cover those exact bytes. Replacement after capture cannot switch
the validated content. This does not guarantee the filesystem remains unchanged
after return or that three independently captured files were published atomically;
manifest input hashes detect inconsistent release snapshots.

Reports include exact manifest/CSV hashes, release version, request parameters,
contract and software versions, and (when valid) the structural partition manifest
with its assignment/topology hashes. There is no timestamp or performance metric.
Repeat calls in the same software environment produce the same report. Reports
record implemented prerequisites only; automated checks never establish scientific validity,
source authenticity, exchangeability, conformal coverage or predictive quality.

All files are explicit inputs; no discovery, packaging, or directory hashing
occurs. Paths into `annotation_batch_24_seed20261029/`, including resolved symlink
targets, are refused before content access. Historical artifacts are never output
targets. No production approval manifest is created by this implementation.

Task B's existing `validate_studies` checks the study schema/findings, and
`validate_study_sources` requires nonempty source URLs/DOIs plus
`DEFAULT_SOURCE_REGISTRY.verify_reference(..., require_registered=True)` for
every publication. This API receives topology rather than those records and does
not invoke those gates or imply they passed. A future evidence-release contract
must bind the actual evidence bytes and apply those checks before full eligibility
can be considered. It is outside this Phase 2 correction.

The existing extractor is unchanged: a known zero factor establishes zero pair
contribution even if the other factor is unknown; otherwise unknown factors
remain NaN and block feature completeness. This is a mechanistic pathway rule,
not a no-interaction consensus label or scientific record repair.
