"""Synthetic orchestration tests; no empirical readiness or approvals asserted."""
import builtins
import copy
import hashlib
import sys
from pathlib import Path

import pytest

from tests.test_release_preflight import release  # synthetic Phase 2 fixture
from src.training import release_preflight as preflight
from src.training.evaluation_orchestrator import evaluate_request, report_json, REQUEST_SCHEMA


def request_for(release, task='A'):
    return dict(schema=REQUEST_SCHEMA, task=task, protocol='baseline', regime='pair',
                provenance='synthetic',
                inputs={n: str(release[n + '_path']) for n in ('consensus', 'mechanistic', 'manifest')},
                plan={k: copy.deepcopy(release[k]) for k in
                      ('assignments', 'partition_roles', 'target_policy', 'metric_policy')})


@pytest.mark.parametrize('case', ['A', 'B', 'invalid', 'missing', 'malformed', 'contract_only'])
def test_blocked_has_no_execution_or_writes(release, monkeypatch, case):
    request = request_for(release, 'B' if case == 'B' else 'A')
    if case == 'B':
        request['plan']['topology'] = []
    elif case == 'invalid':
        request['schema'] = 'unsupported'
    elif case == 'missing':
        request['inputs']['manifest'] = None
    elif case == 'malformed':
        Path(request['inputs']['consensus']).write_bytes(b'broken')
    elif case == 'contract_only':
        monkeypatch.setattr(preflight, 'preflight_release_snapshots', lambda *a, **k:
                            dict(eligible=False, contract_status='passed', checks=[]))

    original_open = builtins.open
    original_path_open = Path.open
    def guarded_open(file, mode='r', *args, **kwargs):
        assert not any(c in mode for c in 'wax+'), 'output write attempted'
        return original_open(file, mode, *args, **kwargs)
    def guarded_path_open(path, mode='r', *args, **kwargs):
        assert not any(c in mode for c in 'wax+'), 'output write attempted'
        return original_path_open(path, mode, *args, **kwargs)
    monkeypatch.setattr(builtins, 'open', guarded_open)
    monkeypatch.setattr(Path, 'open', guarded_path_open)
    # Runtime tripwire covers existing/future estimators and execution operations,
    # without offering a backend hook that could bypass the production gate.
    def forbid_execution(frame, event, arg):
        if event != 'call':
            return
        name = frame.f_code.co_name
        module = frame.f_globals.get('__name__', '')
        assert name not in {'fit', 'fit_transform', 'predict', 'predict_proba', 'score',
                            'calibrate', 'compute_comprehensive_metrics', 'run_repeated_nested_cv'}
        assert not module.startswith(('sklearn.', 'src.models.', 'src.training.calibration'))
    old_profile = sys.getprofile()
    try:
        sys.setprofile(forbid_execution)
        report = evaluate_request(request)
    finally:
        sys.setprofile(old_profile)
    assert report['status'] == 'blocked' and report['eligible'] is False
    assert 'predictions' not in report and 'metrics' not in report
    assert report['provenance']['verified'] is False
    if case == 'A':
        assert report['preflight']['contract_status'] == 'passed'
        assert not report['preflight']['eligible']
    if case == 'B':
        assert any(c['prerequisite'] == 'task_b_evidence_release' and c['status'] == 'blocked'
                   for c in report['prerequisites'])
    if case == 'contract_only':
        assert next(c for c in report['prerequisites'] if c['prerequisite'] == 'evaluation_eligibility')['status'] == 'blocked'


def test_snapshot_replacement_never_reopens(release, monkeypatch):
    request = request_for(release)
    expected = {n: hashlib.sha256(Path(p).read_bytes()).hexdigest()
                for n, p in request['inputs'].items()}
    original = preflight._snapshot
    seen = []
    def replace_after_capture(path):
        snapshot = original(path)
        seen.append(path)
        Path(path).write_bytes(b'REPLACED AFTER SNAPSHOT')
        return snapshot
    monkeypatch.setattr(preflight, '_snapshot', replace_after_capture)
    report = evaluate_request(request)
    assert len(seen) == 3 and len(set(seen)) == 3
    assert report['input_sha256'] == expected == report['preflight']['input_sha256']
    assert report['preflight']['contract_status'] == 'passed'


def test_serialization_and_plan_binding(release):
    request = request_for(release)
    first = evaluate_request(request)
    reordered = dict(reversed(list(request.items())))
    assert report_json(first) == report_json(evaluate_request(reordered))
    assert first['plan_sha256']
    request['plan']['partition_roles']['fit'] = 'assessment'
    assert evaluate_request(request)['plan_sha256'] != first['plan_sha256']


@pytest.mark.parametrize('change', [
    lambda r: r.update(provenance=[]),
    lambda r: r.update(plan=[]),
    lambda r: r.update(inputs={}),
    lambda r: r.update(task=[]),
    lambda r: r.update(output_path='forbidden.json'),
    lambda r: r['inputs'].update(consensus='annotation_batch_24_seed20261029/never-open.csv'),
])
def test_invalid_request_does_not_read_inputs(release, monkeypatch, change):
    request = request_for(release)
    change(request)
    def forbidden(*a, **k):
        pytest.fail('Invalid request reached input reader')
    monkeypatch.setattr(preflight, '_snapshot', forbidden)
    report = evaluate_request(request)
    assert report['prerequisites'][0]['status'] == 'failed'
    assert report_json(report)


def test_true_flag_still_requires_passing_prerequisites(release, monkeypatch):
    monkeypatch.setattr(preflight, 'preflight_release_snapshots', lambda *a, **k:
                        dict(eligible=True, contract_status='passed', checks=[
                            dict(prerequisite='synthetic_failure', status='failed', reason='fixture')]))
    report = evaluate_request(request_for(release))
    assert next(c for c in report['prerequisites'] if c['prerequisite'] == 'evaluation_eligibility')['status'] == 'blocked'
    assert report['status'] == 'blocked'


@pytest.mark.parametrize('flag', [None, 1, 'true', True])
def test_no_execution_backend_even_with_future_eligibility(release, monkeypatch, flag):
    result = dict(contract_status='passed', checks=[])
    if flag is not None:
        result['eligible'] = flag
    monkeypatch.setattr(preflight, 'preflight_release_snapshots', lambda *a, **k: result)
    report = evaluate_request(request_for(release))
    gate = next(c for c in report['prerequisites'] if c['prerequisite'] == 'evaluation_eligibility')
    assert gate['status'] == ('passed' if flag is True else 'blocked')
    assert report['status'] == 'blocked' and report['eligible'] is False
    assert 'metrics' not in report and 'predictions' not in report


def test_unreadable_snapshot_blocks_and_is_serializable(release, monkeypatch):
    def denied(path):
        raise PermissionError('fixture: supply readable input')
    monkeypatch.setattr(preflight, '_snapshot', denied)
    report = evaluate_request(request_for(release))
    assert all(v is None for v in report['input_sha256'].values())
    assert sum(c['status'] == 'failed' for c in report['prerequisites']) >= 3
    assert report_json(report)
