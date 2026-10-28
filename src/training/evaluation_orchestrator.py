"""Phase 3.1 read-only orchestration. No evaluation execution backend exists."""
from __future__ import annotations

import hashlib
from collections.abc import Mapping
from typing import TypedDict

from src.training import release_preflight as release
from src.training.evaluation_protocol import canonical_manifest_json, _REQUIRED

REQUEST_SCHEMA = 'hdi.evaluation-request/v1'
REPORT_SCHEMA = 'hdi.evaluation-report/v1'


class EvaluationRequest(TypedDict):
    schema: str
    task: str
    protocol: str
    regime: str
    inputs: dict[str, str | None]
    plan: dict
    provenance: str


class EvaluationReport(TypedDict):
    schema: str
    status: str
    eligible: bool
    task: str | None
    protocol: str | None
    regime: str | None
    input_sha256: dict
    plan_sha256: str | None
    schema_versions: dict
    software_versions: dict
    provenance: dict
    prerequisites: list[dict]
    preflight: dict | None


def report_json(report: EvaluationReport) -> str:
    return canonical_manifest_json(report)


def evaluate_request(request: EvaluationRequest) -> EvaluationReport:
    """Validate, snapshot, and preflight; return a deterministic blocked report.

    Provenance is a caller declaration, never an authentication of evidence.
    Plan fields mirror Phase 2 arguments; absent prerequisites are not invented.
    Paths are read only after request validation and never reopened. No output
    path, estimator, callbacks, or production eligibility override is accepted.
    """
    checks = []
    plan_hash = None
    safe = {}
    plan = {}
    provenance = None
    try:
        if not isinstance(request, Mapping):
            raise ValueError('Supply an evaluation request mapping')
        # Freeze all JSON request data before validating or reading inputs.
        frozen = release._manifest_snapshot(canonical_manifest_json(dict(request)).encode('utf-8'))
        safe = {k: frozen.get(k) if isinstance(frozen.get(k), str) else None
                for k in ('task', 'protocol', 'regime')}
        if set(frozen) != {'schema', 'task', 'protocol', 'regime', 'inputs', 'plan', 'provenance'}:
            raise ValueError('Require exactly schema, task, protocol, regime, inputs, plan, provenance')
        if frozen['schema'] != REQUEST_SCHEMA:
            raise ValueError('Use ' + REQUEST_SCHEMA)
        if safe['task'] not in {'A', 'B'} or safe['protocol'] not in _REQUIRED or safe['regime'] not in {'pair', 'herb', 'drug'}:
            raise ValueError('Require task A/B, documented protocol, pair/herb/drug regime')
        provenance = frozen['provenance']
        if provenance not in ('synthetic', 'empirical'):
            raise ValueError('Declare provenance synthetic or empirical; declaration is not verification')
        inputs = frozen['inputs']
        if not isinstance(inputs, dict) or set(inputs) != {'consensus', 'mechanistic', 'manifest'}:
            raise ValueError('inputs must explicitly name consensus, mechanistic, manifest')
        for path in inputs.values():
            if path is not None:
                if not isinstance(path, str) or not path:
                    raise ValueError('Input paths must be nonempty strings or null for missing input')
                release._path(path)
        plan = frozen['plan']
        allowed = {'assignments', 'partition_roles', 'topology', 'target_policy', 'metric_policy'}
        if not isinstance(plan, dict) or set(plan) - allowed:
            raise ValueError('plan accepts only ' + ', '.join(sorted(allowed)))
        plan_hash = hashlib.sha256(canonical_manifest_json(dict(**safe, plan=plan)).encode('utf-8')).hexdigest()
        checks.append(dict(prerequisite='request_schema', status='passed', reason='Supported request v1'))
    except (ValueError, TypeError, OSError, RecursionError) as exc:
        checks.append(dict(prerequisite='request_schema', status='failed', reason=str(exc)))
        inputs = None

    snapshots = dict.fromkeys(('consensus', 'mechanistic', 'manifest'))
    preflight = None
    if inputs is not None:
        for name, path in sorted(inputs.items()):
            try:
                snapshots[name] = None if path is None else release._snapshot(path)
            except (ValueError, TypeError, OSError) as exc:
                snapshots[name] = None
                checks.append(dict(prerequisite=name + '_snapshot', status='failed',
                                   reason='Supply a readable permitted input: ' + str(exc)))
        try:
            preflight = release.preflight_release_snapshots(snapshots, **safe, **plan)
            checks.extend(preflight['checks'])
            gate = (preflight.get('eligible') is True and
                    preflight.get('contract_status') == 'passed' and
                    all(c.get('status') == 'passed' for c in checks))
            checks.append(dict(prerequisite='evaluation_eligibility',
                               status='passed' if gate else 'blocked',
                               reason='Require actual Phase 2 eligible=true and all prerequisites passed'))
        except (ValueError, TypeError, OSError, KeyError, AttributeError) as exc:
            checks.append(dict(prerequisite='release_preflight', status='failed',
                               reason='Repair release inputs: ' + str(exc)))
    checks.append(dict(prerequisite='execution_backend', status='blocked',
                       reason='Phase 3.1 validates only; execution is not implemented'))
    return dict(schema=REPORT_SCHEMA, status='blocked', eligible=False,
                **{k: safe.get(k) for k in ('task', 'protocol', 'regime')},
                input_sha256={k: hashlib.sha256(v).hexdigest() if v is not None else None
                              for k, v in sorted(snapshots.items())},
                plan_sha256=plan_hash,
                schema_versions=dict(request=REQUEST_SCHEMA, report=REPORT_SCHEMA, **release.VERSIONS),
                software_versions=dict(package='hdi-prediction 0.1.0',
                                       python=release.platform.python_version(),
                                       numpy=release.np.__version__, pandas=release.pd.__version__),
                provenance=dict(declared=provenance if provenance in ('synthetic', 'empirical') else None,
                                verified=False, scope='Caller declaration only; no empirical validity certified'),
                prerequisites=checks, preflight=preflight)
