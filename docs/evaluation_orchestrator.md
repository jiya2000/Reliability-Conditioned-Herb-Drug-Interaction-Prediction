# Phase 3.1 evaluation request and report contract

`src.training.evaluation_orchestrator.evaluate_request(request)` returns a JSON-safe
`hdi.evaluation-report/v1` mapping without writing files. `report_json(report)`
serializes canonical JSON (sorted keys, no timestamps or nonfinite numbers).
TypedDict schemas document the API; request validation is performed at runtime.

Request v1 requires exactly these fields:

```python
{
    'schema': 'hdi.evaluation-request/v1',
    'task': 'A',                       # A or B
    'protocol': 'baseline',            # evaluation_protocol documented protocols
    'regime': 'pair',                  # pair, herb, drug
    'inputs': {
        'consensus': 'path/to/consensus.csv',
        'mechanistic': 'path/to/mechanistic.csv',
        'manifest': 'path/to/release.json',
    },
    'plan': {
        'assignments': [],             # explicit Phase 2 records
        'partition_roles': {'fit': 'fit', 'assessment': 'assessment'},
        'target_policy': 'binary_consensus_v1',
        'metric_policy': 'binary_discrimination_v1',
        # Task B additionally requires explicit audited topology records.
    },
    'provenance': 'empirical',          # or synthetic; caller declaration only
}
```

This shape example supplies no decisions or approvals and is not eligible.
Null input paths denote missing prerequisites. Unsupported fields (including
output paths or execution callbacks) fail validation before reading input files.
Plan record validation uses Phase 2; omitted prerequisites block or fail there.

Each permitted path is captured once with Phase 2's bounded reader. The
`preflight_release_snapshots` entry point hashes and parses those immutable bytes
using the same Phase 2 contract as `preflight_release`. Paths are never reopened
for preflight. Snapshots are per input, not an atomic multi-file filesystem
transaction; release manifest hashes detect inconsistent approved input versions.

Reports include task, protocol, regime, exact-byte input SHA-256 hashes (null
when unavailable), a canonical plan SHA-256 binding task/protocol/regime and the
supplied plan, software/schema versions, prerequisite diagnostics, declared
provenance and the full Phase 2 result. Provenance is explicitly unverified;
synthetic declarations cannot confer empirical validity. Reproducibility requires
the same request, bytes and software environment. The plan hash reflects the
supplied record ordering; it does not assert semantic equivalence across reordered
plans. Input paths are not included in the plan hash.

The gate requires literal `eligible is True`, `contract_status == 'passed'`, and
all collected prerequisites passing. Phase 2 currently always returns
`eligible=False`: statistical support is unestablished. Task A requires verified
mechanistic evidence; Task B additionally retains the actual evidence-release
blocker, which topology-only validation cannot satisfy. No minimum sample sizes,
labels, approvals or coverage guarantees are inferred here.

The production orchestrator still always returns `status='blocked'` and
`eligible=False`, even if a future preflight grants eligibility. Predictions and
metrics are absent. Existing historical runners are not redirected by this API.
There is no production gate bypass. Component APIs described below are available
for direct synthetic testing; they are not a release authorization mechanism.

## Phase 3.2 public boundary after adversarial review

`pipeline_components` exposes structural validation and disabled execution APIs.
Every public fit, tune, predict, score, probability calibration, conformal
calibration/set generation, and assessment entry point fails with PermissionError
through the orchestrator authorization check. Manual contexts, arbitrary fitted
estimators, synthetic fixtures, caller-created approval records, and even a
caller-supplied `eligible=True` report cannot authorize execution. Constructors
bypassed with `object.__new__` still cannot execute public methods. There is no
synthetic/test flag on the public boundary and no dispatch to internal algorithms.

The orchestrator currently issues **no execution grants**, even if a future
preflight reports eligibility. This is an intentional fail-closed implementation:
the repository lacks approved statistical/evidence-release support and an
execution-authorization issuance contract. A future implementation must issue
non-caller-created authorization from the orchestrator only after preflighting its
captured immutable bytes, requiring literal `eligible is True`, a passed contract
and every prerequisite passing, and binding snapshots, plan and fitted-role
provenance. A report mapping is not an authorization token. Merely changing a gate
function to return does not enable the current public backend.

The previous direct-execution component APIs are intentionally incompatible:
callers now receive PermissionError. Existing historical Task A/B runners retain
their APIs and code, and are not routed into the new production boundary.

## Private synthetic algorithms and bindings

Algorithm implementations live in `src.training._pipeline_algorithms`, a private
module used directly by synthetic tests. Their fit registrations are **not**
production release authorizations and cannot be passed into the public APIs to
execute. Python module privacy is an API convention, not a sandbox against code
that imports private internals, rewrites module globals or monkeypatches Python.
Production execution stays disabled rather than claiming such interpreter-level
security. Do not use this private test surface as an empirical entry point.

`SnapshotTables.from_snapshots` accepts exactly consensus/mechanistic/manifest
immutable bytes, never paths. Parsing checks strict consensus states, curation,
and manifest hash binding. It does not authenticate approval or replace preflight.
`RoleData` accepts the exact snapshot class rather than a caller table provider,
copies derived tables and the explicit JSON plan, validates all required roles
and requested pair/herb/drug and Task B provenance constraints, and checks state
before use. Its public snapshot property is read-only; checks additionally bind
snapshot object identity and bytes, cached tables and the validated role manifest.
Returned tables are copies. Private rebinding, in-place table mutation and forced
mutation of frozen snapshot fields are detected before algorithm use.

The private Task A fit factory alone registers a fitted predictor after successful
fit-role fitting. Arbitrary estimator wrapping and manually allocated/cloned
predictors are rejected. Each registered predictor has a frozen fit record with
verified fit-role identities/targets/features hash, task, explicit configuration,
feature allowlist, canonical validated-plan hash, exact snapshot hashes, derived
table hash and fitted-estimator hash. Records returned to callers are defensive
copies. Every prediction and calibration operation verifies registration, original
data/estimator bindings and record consistency. Copying or fabricating a fit record
does not create a registration. Hashes are software-version-dependent mutation
checks; they do not authenticate scientific evidence or independent annotation.

Private Task A LR/HGB retain the approved feature builder/pipeline. Preprocessing
and candidate fitting see fit-role binary decisions only. Explicit candidate
selection uses tuning BCE without refitting on tuning. LR supports C; HGB supports
learning_rate, max_leaf_nodes and l2_regularization. Disputed decisions remain in
the universe and predictions but outside binary optimization/scoring; unresolved
or inconclusive consensus strings fail rather than become negatives. Assessment
targets are requested only after predictions and sets are constructed. Informative
synthetic tests compare estimator state, selected configuration, calibration
temperature, nonconstant predictions and sets after assessment-label changes.

## Scoring, calibration and Task B limitations

Private strict scoring validates exact ordered unique pair identity, canonical
labels and finite numeric probabilities in [0,1]. Undefined metrics return None
with reasons, never fabricated defaults. Average precision and trapezoidal PR-AUC
are distinct. Binary log loss clips validated probabilities to machine epsilon
as a numeric convention and validates its binary targets. The existing AUC > .98
alert is retained; existing guard code and tests are unchanged. No justified
inference design exists for intervals or significance.

Private temperature calibration requires a registered fixed predictor, the same
role context, both classes and interior fitting probabilities. Optimizer failures,
nonfinite objectives/parameters and log-temperature boundary solutions fail;
there is no identity fallback. Numerical bounds [-10,10] are not scientific
criteria. Failed refits invalidate earlier state. Conformal calibration uses its
own role and a fixed predictor/calibrator; genuine different-predictor reuse is
rejected. The rank is ceil((n+1)*(1-alpha)); rank n+1 produces all-label sets.
Coverage certification always stays blocked: disjointness and rank arithmetic do
not establish exchangeability or finite-sample coverage for this release.

Read-only Task B validation accepts bounded JSON study bytes, invokes existing
study/registered-source/reliability checks, and compares exact source/study/citation
topology after citation-list set normalization. Extra intermediate study records
remain unsupported pending a release schema. The adapter checks its role context
before validation. Prior/reliability fitting and residual tuning both remain
blocked. Source/topology consistency cannot supply the missing actual-evidence
approval and independence contract. Synthetic source stubs remain test-local.
Current empty consensus, unverified mechanisms, missing statistical support and
incomplete Task B release validation continue to block empirical execution.
