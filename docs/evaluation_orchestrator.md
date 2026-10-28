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

Phase 3.1 has no execution backend and always returns `status='blocked'` and
`eligible=False`, even if a future preflight grants eligibility. Predictions and
metrics are absent. No estimator, model adapter, fitting, calibration, prediction,
scoring, conformal or bootstrap implementation is provided. Existing historical
runners are not redirected by this API. There is no production gate bypass.
