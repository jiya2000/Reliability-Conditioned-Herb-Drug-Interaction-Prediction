"""Entire prerequisite-passing release is synthetic; no scientific approval is asserted."""
import copy
import hashlib
import json
from pathlib import Path

import pandas as pd
import pytest

from src.data.mechanistic_curation import CANONICAL_TARGETS, CURATION_COLUMNS
from src.training.release_preflight import VERSIONS, preflight_release, report_json


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


@pytest.fixture
def release(tmp_path):
    rows, mechanics, assignments, decisions = [], [], [], []
    source = dict(source_name='SYNTHETIC FIXTURE ONLY',
                  source_url_or_citation='SYNTHETIC bibliography fixture; not clinical evidence',
                  retrieval_date='2026-10-28')
    for i, label in enumerate(['interaction', 'no_interaction', 'disputed'] * 2):
        pair = dict(herb_id=f'HERB_SYNTH{i}', drug_id=f'DRUG_SYNTH{i}')
        rows.append(dict(pair, consensus_label=label, **source))
        assignments.append(dict(pair, partition='fit' if i < 3 else 'assessment'))
        decisions.append(dict(pair, consensus_label=label, approval_status='approved',
            reviewer_approval_refs=['SYNTHETIC_A', 'SYNTHETIC_B'],
            adjudicator_approval_ref='SYNTHETIC_C', source_ref='SYNTHETIC_SOURCE',
            provenance_ref='SYNTHETIC_PROVENANCE', independence_approval_ref='SYNTHETIC_INDEPENDENCE',
            attestations={k: True for k in ('independent_reviewers', 'blinded_to_predictions',
                'external_consensus_only', 'independent_of_evaluation_evidence')}))
        for entity in pair.values():
            for target in CANONICAL_TARGETS:
                r = dict.fromkeys(CURATION_COLUMNS, '')
                r.update(entity=entity, target=target, status='tested_no_effect',
                         source='SYNTHETIC', citation='SYNTHETIC ONLY')
                mechanics.append(r)
    c, m, manifest = [tmp_path / n for n in ('consensus.csv', 'mechanistic.csv', 'release.json')]
    pd.DataFrame(rows).to_csv(c, index=False)
    pd.DataFrame(mechanics).to_csv(m, index=False)
    data = dict(target_policy='binary_consensus_v1', metric_policy='binary_discrimination_v1', schema_versions=VERSIONS, release_version='SYNTHETIC-1', approval_status='approved',
        input_sha256=dict(consensus=digest(c), mechanistic=digest(m)), approval_refs=['SYNTHETIC_RELEASE'],
        source_registry={'SYNTHETIC_SOURCE': dict(source, source_id='SYNTHETIC_CONSENSUS', verification_status='approved', approval_ref='SYNTHETIC')},
        provenance_registry={'SYNTHETIC_PROVENANCE': {'reference': 'SYNTHETIC ONLY'}},
        mechanistic_provenance={'SYNTHETIC': dict(reference='SYNTHETIC ONLY', approval_ref='SYNTHETIC')},
        decisions=decisions)
    manifest.write_text(json.dumps(data))
    return dict(manifest_path=manifest, consensus_path=c, mechanistic_path=m,
                task='A', protocol='baseline', regime='pair', assignments=assignments,
                target_policy='binary_consensus_v1', metric_policy='binary_discrimination_v1',
                partition_roles={'fit': 'fit', 'assessment': 'assessment'})


def change_manifest(release, change):
    p = release['manifest_path']
    data = json.loads(p.read_text())
    change(data)
    p.write_text(json.dumps(data))


def check(report, name):
    matches = [c for c in report['checks'] if c['prerequisite'] == name]
    assert len(matches) == 1, matches
    return matches[0]['status']


def test_eligible_deterministic(release):
    r = preflight_release(**release)
    assert r['contract_status'] == 'passed', r
    assert r['status'] == 'blocked' and not r['eligible']
    assert check(r, 'statistical_support') == 'blocked'
    assert r['counts']['labels'] == dict(interaction=2, no_interaction=2, disputed=2)
    assert report_json(r) == report_json(preflight_release(**release))


@pytest.mark.parametrize('mutation,field,status', [
    (lambda m: m.update(approval_status='unapproved'), 'release_approval', 'blocked'),
    (lambda m: m['decisions'][0].pop('attestations'), 'decision_provenance', 'blocked'),
    (lambda m: m['decisions'][0]['attestations'].update(independent_reviewers=False), 'decision_provenance', 'failed'),
    (lambda m: m['decisions'][0].update(adjudicator_approval_ref='SYNTHETIC_A'), 'decision_provenance', 'failed'),
    (lambda m: m['input_sha256'].update(consensus='0' * 64), 'consensus_hash', 'failed'),
    (lambda m: m.pop('mechanistic_provenance'), 'mechanistic_provenance', 'blocked'),
    (lambda m: m['source_registry']['SYNTHETIC_SOURCE'].update(source_name='contradiction'), 'decision_provenance', 'failed'),
])
def test_manifest_contract(release, mutation, field, status):
    change_manifest(release, mutation)
    r = preflight_release(**release)
    assert not r['eligible']
    assert check(r, field) == status


@pytest.mark.parametrize('label', ['', 'inconclusive', 'unresolved'])
def test_unresolved_not_labels(release, label):
    p = release['consensus_path']
    frame = pd.read_csv(p)
    frame.loc[0, 'consensus_label'] = label
    frame.to_csv(p, index=False)
    assert check(preflight_release(**release), 'consensus_schema') == 'failed'


def test_empty_consensus(release):
    p = release['consensus_path']
    pd.read_csv(p).iloc[:0].to_csv(p, index=False)
    assert check(preflight_release(**release), 'consensus_labels') == 'blocked'


@pytest.mark.parametrize('mode,expected', [('unverified', 'mechanistic_verification'),
                                          ('missing', 'feature_completeness')])
def test_mechanistic(release, mode, expected):
    p = release['mechanistic_path']
    frame = pd.read_csv(p)
    if mode == 'unverified':
        frame.loc[0, 'source'] = 'UNVERIFIED'
    else:
        frame.loc[[0, len(CANONICAL_TARGETS)], 'status'] = 'not_tested'
        frame['notes'] = 'SYNTHETIC lookup only'
    frame.to_csv(p, index=False)
    assert check(preflight_release(**release), expected) == 'blocked'


def test_class_and_partition_support(release):
    release['assignments'][0]['partition'] = 'assessment'
    assert check(preflight_release(**release), 'class_support') == 'failed'
    release['assignments'][0]['partition'] = 'fit'
    release['regime'] = 'herb'
    p = release['consensus_path']
    frame = pd.read_csv(p)
    frame.loc[3, 'herb_id'] = frame.loc[0, 'herb_id']
    frame.to_csv(p, index=False)
    release['assignments'][3]['herb_id'] = frame.loc[0, 'herb_id']
    assert check(preflight_release(**release), 'partition_structure') == 'failed'


def test_task_b_coverage(release):
    release.update(task='B', topology=[])
    assert check(preflight_release(**release), 'evidence_coverage') == 'blocked'


def test_repository_blocked():
    r = preflight_release(task='A', protocol='baseline', regime='pair')
    assert r['status'] == 'blocked'
    assert check(r, 'release_approval') == 'blocked'
    assert check(r, 'consensus_labels') == 'blocked'
    assert check(r, 'mechanistic_verification') == 'blocked'
    assert 'metrics' not in r


def test_no_fit_no_mutation(release, monkeypatch):
    from sklearn.linear_model import LogisticRegression
    from src.models.conformal_abstention import ConformalTriStateEngine
    def forbidden(*a, **kw):
        pytest.fail('Preflight must not fit, calibrate or write')
    monkeypatch.setattr(LogisticRegression, 'fit', forbidden)
    monkeypatch.setattr(ConformalTriStateEngine, 'calibrate', forbidden)
    originals = copy.deepcopy(release)
    # Explicit tracked paths only: excludes private annotation and avoids discovery.
    import subprocess
    tracked = subprocess.check_output(['git', 'ls-files', 'data', 'results', 'reports', 'benchmarks',
                                       'tests/test_redesigned_leakage_regression.py'], text=True).splitlines()
    paths = [Path(p) for p in tracked] + [release[k] for k in
        ('manifest_path', 'consensus_path', 'mechanistic_path')]
    before = {str(p): digest(p) for p in paths if p.is_file()}
    monkeypatch.setattr(Path, 'write_text', forbidden)
    monkeypatch.setattr(pd.DataFrame, 'to_csv', forbidden)
    assert preflight_release(**release)['contract_status'] == 'passed'
    assert release == originals
    assert before == {str(p): digest(p) for p in paths if p.is_file()}


def test_private_directory_excluded(tmp_path):
    p = tmp_path / 'annotation_batch_24_seed20261029' / 'consensus.csv'
    r = preflight_release(consensus_path=p, task='A', protocol='baseline', regime='pair')
    assert check(r, 'consensus_input') == 'failed'


@pytest.mark.parametrize('field', ['release_version', 'approval_refs', 'source_registry'])
def test_missing_manifest_prerequisite(release, field):
    change_manifest(release, lambda m: m.pop(field))
    assert preflight_release(**release)['status'] == 'blocked'


@pytest.mark.parametrize('value', [None, [], 'unsupported'])
def test_malformed_manifest(release, value):
    release['manifest_path'].write_text(json.dumps(value))
    r = preflight_release(**release)
    assert not r['eligible']
    assert check(r, 'manifest_schema') == 'failed'


def test_missing_mechanistic_values_and_columns(release):
    p = release['mechanistic_path']
    frame = pd.read_csv(p)
    frame.loc[0, 'status'] = 'measured'
    frame.to_csv(p, index=False)
    assert check(preflight_release(**release), 'mechanistic_verification') == 'blocked'
    frame.drop(columns='value_fm').to_csv(p, index=False)
    assert check(preflight_release(**release), 'mechanistic_verification') == 'failed'


def test_task_b_eligible_and_infeasible(release):
    release.update(task='B', topology=[dict(a, study_id=f'SYNTHETIC_{i}',
        source_id=f'SYNTHETIC_{i}', cited_study_ids=[])
        for i, a in enumerate(release['assignments'])])
    assert preflight_release(**release)['contract_status'] == 'passed'
    release['topology'][3]['source_id'] = release['topology'][0]['source_id']
    assert check(preflight_release(**release), 'partition_structure') == 'failed'


def test_source_independence_contradiction(release):
    release.update(task='B', topology=[dict(a, study_id=f'SYNTHETIC_{i}',
        source_id='SYNTHETIC_CONSENSUS' if i == 0 else f'SYNTHETIC_{i}', cited_study_ids=[])
        for i, a in enumerate(release['assignments'])])
    assert check(preflight_release(**release), 'decision_provenance') == 'failed'


@pytest.mark.parametrize('input_name', ['manifest', 'consensus', 'mechanistic'])
def test_snapshot_replacement_cannot_change_validated_bytes(release, monkeypatch, input_name):
    import src.training.release_preflight as module
    paths = {n: release[n + '_path'] for n in ('manifest', 'consensus', 'mechanistic')}
    original_bytes = {n: p.read_bytes() for n, p in paths.items()}
    snapshot = module._snapshot
    reads = []
    def replace_after_capture(path):
        reads.append(str(path))
        captured = snapshot(path)
        if Path(path) == paths[input_name]:
            # Synthetic fixture only: a path reopened for parsing would now fail.
            Path(path).write_bytes(b'REPLACED AFTER SNAPSHOT: NOT VALID DATA')
        return captured
    monkeypatch.setattr(module, '_snapshot', replace_after_capture)
    report = preflight_release(**release)
    assert report['contract_status'] == 'passed'
    assert sorted(reads) == sorted(str(p) for p in paths.values())
    for n in paths:
        assert report['input_sha256'][n] == hashlib.sha256(original_bytes[n]).hexdigest()
    assert paths[input_name].read_bytes() != original_bytes[input_name]


@pytest.mark.parametrize('refs', [
    ['SYNTHETIC_A', 'SYNTHETIC_A'],
    ['SYNTHETIC_A', ' SYNTHETIC_A '],
    [' SYNTHETIC_A', 'SYNTHETIC_B'],
    ['SYNTHETIC_A', 'SYNTHETIC_B '],
])
def test_noncanonical_or_duplicate_reviewer_refs(release, refs):
    change_manifest(release, lambda m: m['decisions'][0].update(reviewer_approval_refs=refs))
    r = preflight_release(**release)
    assert check(r, 'decision_provenance') == 'failed'
    assert r['contract_status'] == 'failed'


@pytest.mark.parametrize('side', ['consensus', 'topology'])
@pytest.mark.parametrize('padding', [' ', '\t'])
def test_whitespace_source_overlap_cannot_pass(release, side, padding):
    release.update(task='B', topology=[dict(a, study_id=f'SYNTHETIC_{i}',
        source_id='SYNTHETIC_CONSENSUS' if i == 0 else f'SYNTHETIC_{i}', cited_study_ids=[])
        for i, a in enumerate(release['assignments'])])
    if side == 'consensus':
        change_manifest(release, lambda m: m['source_registry']['SYNTHETIC_SOURCE'].update(
            source_id=padding + 'SYNTHETIC_CONSENSUS' + padding))
        prerequisite = 'source_registry'
    else:
        release['topology'][0]['source_id'] = padding + 'SYNTHETIC_CONSENSUS' + padding
        prerequisite = 'topology_arguments'
    r = preflight_release(**release)
    assert check(r, prerequisite) == 'failed'
    assert r['contract_status'] == 'failed'


@pytest.mark.parametrize('field', ['task', 'protocol', 'regime'])
@pytest.mark.parametrize('value', [[], {}, None, 42, True])
def test_malformed_request_is_deterministic_report(release, field, value):
    release[field] = value
    r = preflight_release(**release)
    assert check(r, 'request') == 'failed'
    assert not r['eligible']
    assert report_json(r) == report_json(preflight_release(**release))


@pytest.mark.parametrize('field,value', [
    ('herb_id', []), ('drug_id', {}), ('study_id', []), ('source_id', {}),
    ('cited_study_ids', [None]), ('herb_id', ' HERB_SYNTH0'),
])
def test_malformed_topology_identifier(release, field, value):
    release.update(task='B', topology=[dict(release['assignments'][0], study_id='SYNTHETIC',
        source_id='SYNTHETIC', cited_study_ids=[])])
    release['topology'][0][field] = value
    r = preflight_release(**release)
    assert check(r, 'topology_arguments') == 'failed'
    assert report_json(r) == report_json(preflight_release(**release))


@pytest.mark.parametrize('field,value', [
    ('assignments', 42), ('assignments', {}), ('assignments', [None]),
    ('assignments', [dict(herb_id=[], drug_id='DRUG_SYNTH0', partition='fit')]),
    ('partition_roles', []), ('partition_roles', {'fit': []}), ('topology', 42),
])
def test_malformed_record_arguments(release, field, value):
    release[field] = value
    r = preflight_release(**release)
    assert r['contract_status'] == 'failed'
    assert report_json(r) == report_json(preflight_release(**release))


@pytest.mark.parametrize('field,value', [('target_policy', None),
    ('target_policy', 'tri_state_v1'), ('metric_policy', None),
    ('metric_policy', 'conformal_coverage_v1'), ('target_policy', [])])
def test_unsupported_or_missing_policy_fails(release, field, value):
    release[field] = value
    assert check(preflight_release(**release), 'target_metric_policy') == 'failed'


def test_release_policy_must_match_request(release):
    change_manifest(release, lambda m: m.pop('target_policy'))
    assert check(preflight_release(**release), 'release_policy') == 'failed'


def test_binary_support_preserves_disputed_and_allows_no_disputed(release):
    r = preflight_release(**release)
    assert check(r, 'class_support') == 'passed'
    assert r['counts']['per_role']['fit']['class_pairs']['disputed'] == 1
    c = release['consensus_path']
    frame = pd.read_csv(c)
    frame.loc[frame.consensus_label != 'disputed'].to_csv(c, index=False)
    release['assignments'] = [a for i, a in enumerate(release['assignments']) if i not in (2, 5)]
    def remove_disputed(m):
        m['decisions'] = [d for d in m['decisions'] if d['consensus_label'] != 'disputed']
        m['input_sha256']['consensus'] = digest(c)
    change_manifest(release, remove_disputed)
    r = preflight_release(**release)
    assert r['contract_status'] == 'passed'
    assert r['counts']['per_role']['fit']['class_pairs'] == dict(
        interaction=1, no_interaction=1, disputed=0)
    assert not r['eligible']


@pytest.mark.parametrize('regime', ['pair', 'herb', 'drug'])
def test_support_by_declared_constraint_group(release, regime):
    release['regime'] = regime
    if regime != 'pair':
        field = regime + '_id'
        c = release['consensus_path']
        frame = pd.read_csv(c)
        for i in range(6):
            value = frame.loc[0 if i < 3 else 3, field]
            frame.loc[i, field] = value
            release['assignments'][i][field] = value
        frame.to_csv(c, index=False)
        def update(m):
            m['input_sha256']['consensus'] = digest(c)
            for i, d in enumerate(m['decisions']):
                d[field] = frame.loc[i, field]
        change_manifest(release, update)
    r = preflight_release(**release)
    assert r['contract_status'] == 'passed'
    for role in ('fit', 'assessment'):
        counts = r['counts']['per_role'][role]
        assert counts['pairs'] == 3
        assert counts['constraint_groups'] == (3 if regime == 'pair' else 1)
        assert counts['class_constraint_groups'] == dict(interaction=1, no_interaction=1, disputed=1)
        assert counts['evidence_components'] == 0
    assert check(r, 'statistical_support') == 'blocked'
    assert not r['eligible']


def test_task_b_is_explicitly_topology_only(release):
    release.update(task='B', topology=[dict(a, study_id=f'SYNTHETIC_{i}',
        source_id='SYNTHETIC_SHARED' if i < 3 else 'SYNTHETIC_OTHER', cited_study_ids=[])
        for i, a in enumerate(release['assignments'])])
    r = preflight_release(**release)
    assert r['contract_status'] == 'passed'
    assert check(r, 'task_b_evidence_release') == 'blocked'
    for counts in r['counts']['per_role'].values():
        assert counts['evidence_components'] == 1
        assert counts['constraint_groups'] == 1
    assert not r['eligible']


def test_ambiguous_array_request_is_reported(release):
    import numpy as np
    release['task'] = np.array(['A', 'B'])
    release['target_policy'] = np.array(['binary_consensus_v1', 'other'])
    r = preflight_release(**release)
    assert check(r, 'request') == 'failed'
    assert check(r, 'target_metric_policy') == 'failed'
    assert report_json(r) == report_json(preflight_release(**release))


def test_duplicate_json_approval_status_fails(release):
    p = release['manifest_path']
    text = p.read_text()
    p.write_text(text[:-1] + ', "approval_status": "unapproved", "approval_status": "approved"}')
    assert check(preflight_release(**release), 'manifest_schema') == 'failed'


def test_v1_manifest_fails_closed(release):
    change_manifest(release, lambda m: m['schema_versions'].update(release_manifest='1', preflight='1'))
    assert check(preflight_release(**release), 'manifest_schema') == 'failed'


@pytest.mark.parametrize('field', ['adjudicator_approval_ref', 'source_ref',
                                  'provenance_ref', 'independence_approval_ref'])
def test_noncanonical_other_approval_identifiers(release, field):
    def corrupt(m):
        m['decisions'][0][field] = ' ' + m['decisions'][0][field]
    change_manifest(release, corrupt)
    assert check(preflight_release(**release), 'decision_provenance') == 'failed'


@pytest.mark.parametrize('value', ['true', 1, None, [], {}])
def test_attestation_requires_exact_true(release, value):
    change_manifest(release, lambda m: m['decisions'][0]['attestations'].update(
        independent_of_evaluation_evidence=value))
    assert check(preflight_release(**release), 'decision_provenance') == 'failed'


def test_noncanonical_consensus_not_silently_normalized(release):
    p = release['consensus_path']
    frame = pd.read_csv(p)
    frame.loc[0, 'herb_id'] = ' ' + frame.loc[0, 'herb_id']
    frame.to_csv(p, index=False)
    assert check(preflight_release(**release), 'consensus_schema') == 'failed'


def assert_deterministic_unique_report(release):
    r = preflight_release(**release)
    names = [c['prerequisite'] for c in r['checks']]
    assert len(names) == len(set(names))
    assert report_json(r) == report_json(preflight_release(**release))
    assert not r['eligible']
    return r


@pytest.mark.parametrize('entry', [None, [], {},
    dict(source_id=' UNUSED ', source_name='SYNTHETIC', source_url_or_citation='SYNTHETIC',
         retrieval_date='2026-10-28', verification_status='approved', approval_ref='SYNTHETIC'),
    dict(source_id='SYNTHETIC_UNUSED', source_name='SYNTHETIC', source_url_or_citation='SYNTHETIC',
         retrieval_date='2026-02-30', verification_status='approved', approval_ref='SYNTHETIC'),
    dict(source_id='SYNTHETIC_UNUSED', source_name='SYNTHETIC', source_url_or_citation='SYNTHETIC',
         retrieval_date='2026-10-28', verification_status='unapproved', approval_ref='SYNTHETIC')])
def test_every_source_registry_entry_is_validated(release, entry):
    change_manifest(release, lambda m: m['source_registry'].update(SYNTHETIC_UNUSED=entry))
    r = assert_deterministic_unique_report(release)
    assert check(r, 'source_registry') == 'failed'
    assert r['contract_status'] == 'failed'


@pytest.mark.parametrize('alias_kind', ['same_id', 'same_citation', 'exact_duplicate'])
def test_source_alias_overlap_cannot_pass(release, alias_kind):
    def alias(m):
        entry = dict(m['source_registry']['SYNTHETIC_SOURCE'])
        if alias_kind == 'same_id':
            entry['source_url_or_citation'] = 'SYNTHETIC second citation'
        elif alias_kind == 'same_citation':
            entry['source_id'] = 'SYNTHETIC_ALIAS_ID'
        m['source_registry']['SYNTHETIC_ALIAS'] = entry
        for d in m['decisions']:
            d['source_ref'] = 'SYNTHETIC_ALIAS'
    change_manifest(release, alias)
    release.update(task='B', topology=[dict(a, study_id=f'SYNTHETIC_{i}',
        source_id='SYNTHETIC_CONSENSUS' if i == 0 else f'SYNTHETIC_{i}', cited_study_ids=[])
        for i, a in enumerate(release['assignments'])])
    r = assert_deterministic_unique_report(release)
    assert check(r, 'source_registry') == 'failed'
    assert r['contract_status'] == 'failed'


def test_valid_unused_source_registry_entry(release):
    def extra(m):
        entry = dict(m['source_registry']['SYNTHETIC_SOURCE'], source_id='SYNTHETIC_OTHER',
                     source_url_or_citation='SYNTHETIC unrelated citation')
        m['source_registry']['SYNTHETIC_OTHER'] = entry
    change_manifest(release, extra)
    assert assert_deterministic_unique_report(release)['contract_status'] == 'passed'


@pytest.mark.parametrize('input_name,prerequisite', [('consensus', 'consensus_schema'),
    ('mechanistic', 'mechanistic_verification')])
@pytest.mark.parametrize('payload', [b'X' * 140000 + b'\n', b'"unclosed header\n',
    b'first,second\n"unterminated data\n'])
def test_csv_parser_errors_are_diagnostic(release, input_name, prerequisite, payload):
    p = release[input_name + '_path']
    p.write_bytes(payload)
    r = assert_deterministic_unique_report(release)
    assert check(r, prerequisite) == 'failed'
    assert r['input_sha256'][input_name] == hashlib.sha256(payload).hexdigest()


def test_excessive_json_nesting_is_diagnostic(release):
    payload = b'{"nested":' + b'[' * 1100 + b'0' + b']' * 1100 + b'}'
    release['manifest_path'].write_bytes(payload)
    r = assert_deterministic_unique_report(release)
    assert check(r, 'manifest_schema') == 'failed'
    assert r['input_sha256']['manifest'] == hashlib.sha256(payload).hexdigest()


@pytest.mark.parametrize('constant', ['NaN', 'Infinity', '-Infinity', '1e999'])
@pytest.mark.parametrize('field', ['release_version', 'unused_field'])
def test_nonfinite_json_cannot_enter_report(release, constant, field):
    p = release['manifest_path']
    data = json.loads(p.read_text())
    data.pop(field, None)
    p.write_text(json.dumps(data)[:-1] + ', "' + field + '": ' + constant + '}')
    r = assert_deterministic_unique_report(release)
    assert check(r, 'manifest_schema') == 'failed'
    assert r['release_version'] is None


@pytest.mark.parametrize('field,value', [('release_version', ' SYNTHETIC-1 '),
    ('approval_refs', [' SYNTHETIC_APPROVAL ']), ('release_version', 12),
    ('input_sha256', []), ('schema_versions', None)])
def test_malformed_manifest_has_one_result_per_prerequisite(release, field, value):
    change_manifest(release, lambda m: m.update({field: value}))
    r = assert_deterministic_unique_report(release)
    assert check(r, 'manifest_schema') == 'failed'
    assert r['contract_status'] == 'failed'


def test_json_depth_check_ignores_quoted_delimiters(release):
    change_manifest(release, lambda m: m.update(unused_field='[' * 200 + '\\"' + ']' * 200))
    assert assert_deterministic_unique_report(release)['contract_status'] == 'passed'


@pytest.mark.parametrize('input_name', ['manifest', 'consensus', 'mechanistic'])
def test_input_size_limit_is_diagnostic(release, monkeypatch, input_name):
    import src.training.release_preflight as module
    limit = max(p.stat().st_size for p in (release['manifest_path'], release['consensus_path'],
                                          release['mechanistic_path']))
    release[input_name + '_path'].write_bytes(b'X' * (limit + 1))
    monkeypatch.setattr(module, 'MAX_INPUT_BYTES', limit)
    r = assert_deterministic_unique_report(release)
    assert check(r, input_name + '_input') == 'failed'
    assert r['input_sha256'][input_name] is None
