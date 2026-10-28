"""Synthetic component checks; never release eligibility or empirical results."""
import copy
import hashlib
import json

import numpy as np
import pandas as pd
import pytest

from src.data.mechanistic_curation import CANONICAL_TARGETS, CURATION_COLUMNS
from src.training import _pipeline_algorithms as pc
from src.training import pipeline_components as production


def context(*, assessment_flip=False, protocol='full', task='A', topology=None, informative=False):
    roles = ['fit', 'tuning', 'probability_calibration', 'conformal_calibration', 'assessment']
    if protocol == 'baseline':
        roles = ['fit', 'assessment']
    labels, mechanics, assignments = [], [], []
    for j, role in enumerate(roles):
        for i in range(6):
            index = j * 6 + i
            pair = dict(herb_id=f'HERB_{index}', drug_id=f'DRUG_{index}')
            label = 'interaction' if i % 2 else 'no_interaction'
            if assessment_flip and role == 'assessment':
                label = 'no_interaction' if i % 2 else 'interaction'
            if i == 5:
                label = 'disputed'
            labels.append(dict(pair, consensus_label=label, source_name='SYNTHETIC',
                source_url_or_citation='SYNTHETIC', retrieval_date='2026-10-29'))
            assignments.append(dict(pair, partition=role))
            for entity in pair.values():
                for target in CANONICAL_TARGETS:
                    row = dict.fromkeys(CURATION_COLUMNS, '')
                    row.update(entity=entity, target=target, status='tested_no_effect',
                               source='SYNTHETIC', citation='SYNTHETIC')
                    if informative:
                        # Deliberately overlapping classes, not perfect separation.
                        if entity.startswith('HERB_'):
                            row.update(status='measured', value_ic50_uM=[1., 4., 2., 8., 3., 6.][i],
                                       exposure_conc_uM=1.)
                        else:
                            row.update(status='measured', value_fm=[.15, .5, .3, .8, .6, .4][i],
                                       value_substrate_flag=1.)
                    mechanics.append(row)
    c = pd.DataFrame(labels).to_csv(index=False).encode()
    m = pd.DataFrame(mechanics).to_csv(index=False).encode()
    manifest = json.dumps(dict(input_sha256=dict(consensus=hashlib.sha256(c).hexdigest(),
                                                mechanistic=hashlib.sha256(m).hexdigest()))).encode()
    snapshots = pc.SnapshotTables(c, m, manifest)
    return pc.RoleData(snapshots, assignments=assignments, partition_roles={r: r for r in roles},
                       regime='pair', protocol=protocol, task=task, topology=topology)


@pytest.mark.parametrize('family', ['lr', 'hgb'])
def test_assessment_labels_do_not_affect_fit_tune_preprocess_or_predictions(family):
    first, second = context(), context(assessment_flip=True)
    params = [{'C': .1}, {'C': 1.}] if family == 'lr' else [{'max_leaf_nodes': 2}]
    a = pc.tune_task_a(first, family=family, candidates=params)
    b = pc.tune_task_a(second, family=family, candidates=params)
    np.testing.assert_array_equal(a._estimator['scale'].mean_, b._estimator['scale'].mean_)
    np.testing.assert_array_equal(a.predict(first.rows('assessment')).probability,
                                  b.predict(second.rows('assessment')).probability)
    assert pc.assess(first, a)['integrity']['disputed_pairs'] == 1


def test_fit_reads_only_fit_targets_and_tuning_only_its_targets(monkeypatch):
    data = context()
    original = data.rows
    seen = []
    def guarded(role, *, targets=False):
        if targets:
            assert role in {'fit', 'tuning'}
            seen.append(role)
        return original(role, targets=targets)
    monkeypatch.setattr(data, 'rows', guarded)
    pc.tune_task_a(data, family='lr', candidates=[{'C': .1}, {'C': 1.}])
    assert seen[0] == 'tuning' and seen.count('fit') >= 2
    assert set(seen) == {'fit', 'tuning'}


def test_snapshot_replacement_and_no_path_access(tmp_path, monkeypatch):
    data = context(protocol='baseline')
    path = tmp_path / 'input.csv'
    path.write_bytes(data.snapshots.consensus)
    captured = path.read_bytes()
    path.write_bytes(b'REPLACED')
    monkeypatch.setattr('pathlib.Path.open', lambda *a, **k: pytest.fail('snapshot reopened a path'))
    assert captured == data.snapshots.consensus
    model = pc.fit_task_a(data)
    assert len(model.predict(data.rows('assessment'))) == 6


def test_malformed_snapshots_and_states():
    data = context()
    with pytest.raises(ValueError, match='immutable'):
        pc.SnapshotTables(bytearray(data.snapshots.consensus), b'', b'')
    with pytest.raises(ValueError, match='exact'):
        pc.SnapshotTables(data.snapshots.consensus, data.snapshots.mechanistic, b'{}').tables()
    for label in ['unresolved', 'inconclusive', '', None, 0]:
        with pytest.raises(ValueError):
            pc.binary_targets([label])
    assert pc.binary_targets(['disputed'])[0].size == 0


def test_role_and_model_mutation_fail_closed():
    data = context()
    model = pc.fit_task_a(data)
    model._estimator['model'].coef_[0, 0] += 1
    with pytest.raises(ValueError, match='changed'):
        model.predict(data.rows('assessment'))
    data = context()
    data.manifest['regime'] = 'herb'
    with pytest.raises(ValueError, match='changed'):
        data.rows('fit')


def test_role_assignment_failures():
    data = context(protocol='baseline')
    assignments = copy.deepcopy(data.manifest['assignments'])
    assignments[0]['partition'] = 'missing'
    with pytest.raises(ValueError, match='Unknown'):
        pc.RoleData(data.snapshots, assignments=assignments, partition_roles={'fit': 'fit', 'assessment': 'assessment'},
                    regime='pair', protocol='baseline')
    with pytest.raises(ValueError, match='roles'):
        pc.RoleData(data.snapshots, assignments=data.manifest['assignments'],
                    partition_roles={'fit': 'fit', 'assessment': 'assessment'}, regime='pair', protocol='full')


def scoring_rows(labels, p):
    targets = pd.DataFrame(dict(herb_id=[f'HERB_{i}' for i in range(len(labels))],
                                drug_id=[f'DRUG_{i}' for i in range(len(labels))], consensus_label=labels))
    return targets, targets[pc.KEYS].assign(probability=p)


@pytest.mark.parametrize('p', [[np.nan], [np.inf], [-.1], [1.1], ['.5'], [True], [[.5]]])
def test_bad_probabilities(p):
    with pytest.raises(ValueError):
        pc.probabilities(p, 1)


def test_undefined_metrics_and_strict_identity():
    t, p = scoring_rows(['interaction', 'disputed'], [.4, .6])
    result = pc.strict_score(t, p)
    assert result['metrics']['roc_auc'] == dict(value=None, reason='Both binary classes required by scoring policy')
    assert result['intervals']['status'] == 'unavailable'
    t, p = scoring_rows(['disputed'], [.5])
    assert all(m['value'] is None and m['reason'] for m in pc.strict_score(t, p)['metrics'].values())
    t, p = scoring_rows(['interaction', 'no_interaction'], [.4, .6])
    with pytest.raises(ValueError, match='identities'):
        pc.strict_score(t, p.iloc[::-1])
    with pytest.raises(ValueError, match='Duplicate'):
        pc.strict_score(pd.concat([t, t]), pd.concat([p, p]))


def test_average_precision_is_not_trapezoidal_area_and_guard_preserved():
    t, p = scoring_rows(['interaction', 'no_interaction', 'interaction', 'no_interaction'], [.9, .8, .7, .1])
    m = pc.strict_score(t, p)['metrics']
    assert m['average_precision']['value'] != m['trapezoidal_pr_auc']['value']
    p.probability = [.9, .1, .8, .2]
    with pytest.raises(ValueError, match='AUC integrity alert'):
        pc.strict_score(t, p)


def test_calibration_failure_has_no_identity_fallback(monkeypatch):
    data = context()
    model = pc.fit_task_a(data)
    calibrator = pc.ProbabilityCalibration()
    monkeypatch.setattr(pc, 'minimize_scalar', lambda *a, **k: type('Failure', (), {'success': False})())
    with pytest.raises(ValueError, match='optimizer'):
        calibrator.fit(data, model)
    assert calibrator.status == 'failed' and calibrator.reason
    with pytest.raises(ValueError, match='unchanged'):
        calibrator.transform([.5], model)


def test_calibration_fixed_identity_and_conformal(monkeypatch):
    data = context()
    model = pc.fit_task_a(data)
    # Constant synthetic probabilities yield an interior flat objective.
    c = pc.ProbabilityCalibration().fit(data, model)
    conformal = pc.ConformalCalibration().fit(data, model, alpha=.1, calibrator=c)
    assert conformal.coverage_certification['status'] == 'blocked'
    assert conformal.threshold == 1.0  # rank n+1: conservative all-label sets
    output = conformal.predict_sets(data, model, calibrator=c)
    assert output.include_interaction.all() and output.include_no_interaction.all()
    c.temperature *= 2
    with pytest.raises(ValueError, match='unchanged'):
        conformal.predict_sets(data, model, calibrator=c)


def test_calibration_role_context_and_unsupported_protocol():
    data = context()
    model = pc.fit_task_a(data)
    with pytest.raises(ValueError, match='context'):
        pc.ProbabilityCalibration().fit(context(), model)
    baseline = context(protocol='baseline')
    with pytest.raises(ValueError, match='absent'):
        pc.ProbabilityCalibration().fit(baseline, pc.fit_task_a(baseline))
    for alpha in [0, 1, np.nan, True, '0.1']:
        with pytest.raises(ValueError, match='alpha'):
            pc.ConformalCalibration().fit(data, model, alpha=alpha)


def study_release(data):
    studies = [dict(**p, study_id=f'SYNTHETIC_{i}', source_id=f'SYNTHETIC_SOURCE_{i}',
                    source_url_or_doi='SYNTHETIC', cited_study_ids=[],
                    study_finding='inconclusive', is_primary=1)
               for i, p in enumerate(data._labels[pc.KEYS].to_dict('records'))]
    topology = [{k: row[k] for k in pc.KEYS + ['study_id', 'source_id', 'cited_study_ids']} for row in studies]
    return json.dumps({'studies': studies}).encode(), topology


def test_task_b_actual_sources_required_and_topology_consistent(monkeypatch):
    a = context()
    snapshot, topology = study_release(a)
    with pytest.raises(ValueError, match='Unverified'):
        pc.validate_task_b_release(snapshot, topology=topology, pairs=a._labels[pc.KEYS])
    monkeypatch.setattr('src.data.source_registry.DEFAULT_SOURCE_REGISTRY.verify_reference',
                        lambda *a, **k: (True, 'SYNTHETIC test verifier only'))
    result = pc.validate_task_b_release(snapshot, topology=topology, pairs=a._labels[pc.KEYS])
    assert result['status'] == 'blocked' and result['source_validation'] == 'passed'
    wrong = copy.deepcopy(topology)
    wrong[0]['cited_study_ids'] = ['SYNTHETIC_UNDECLARED']
    with pytest.raises(ValueError, match='topology differ'):
        pc.validate_task_b_release(snapshot, topology=wrong, pairs=a._labels[pc.KEYS])
    b = context(task='B', topology=topology)
    adapter = pc.TaskBSnapshotAdapter(b, study_snapshot=snapshot, topology=topology)
    with pytest.raises(RuntimeError, match='prior/reliability fitting blocked'):
        adapter.fit_prior_reliability()
    with pytest.raises(RuntimeError, match='residual tuning blocked'):
        adapter.tune_residual(None)


def test_task_b_cross_role_provenance_isolation():
    data = context()
    _, topology = study_release(data)
    topology[6]['cited_study_ids'] = [topology[0]['study_id']]
    with pytest.raises(ValueError, match='overlap'):
        context(task='B', topology=topology)


def test_complete_role_composition_assessment_independence():
    a = pc.run_task_a_roles(context(), candidates=[{'C': .1}, {'C': 1}], alpha=.2)
    b = pc.run_task_a_roles(context(assessment_flip=True), candidates=[{'C': .1}, {'C': 1}], alpha=.2)
    pd.testing.assert_frame_equal(a['predictions'], b['predictions'])
    pd.testing.assert_frame_equal(a['prediction_sets'], b['prediction_sets'])
    assert a['coverage_certification']['status'] == 'blocked'


def test_failed_calibration_refit_invalidates_previous_fit(monkeypatch):
    data = context()
    model = pc.fit_task_a(data)
    c = pc.ProbabilityCalibration().fit(data, model)
    monkeypatch.setattr(pc, 'minimize_scalar', lambda *a, **k:
                        type('Boundary', (), {'success': True, 'fun': .5, 'x': 10.})())
    with pytest.raises(ValueError, match='boundary'):
        c.fit(data, model)
    assert c.status == 'failed'
    with pytest.raises(ValueError):
        c.transform([.4], model)


def test_conformal_failure_invalidates_previous_fit_and_frozen_state():
    data = context()
    model = pc.fit_task_a(data)
    c = pc.ConformalCalibration().fit(data, model, alpha=.2)
    c.threshold = .123
    with pytest.raises(ValueError, match='unchanged'):
        c.predict_sets(data, model)
    with pytest.raises(ValueError, match='alpha'):
        c.fit(data, model, alpha=0)
    assert c.status == 'failed' and c.reason


def test_reject_unsupported_parameters_before_fit():
    data = context()
    with pytest.raises(ValueError, match='Unsupported'):
        pc.fit_task_a(data, parameters={'class_weight': 'balanced'})
    with pytest.raises(ValueError, match='candidate'):
        pc.tune_task_a(data, family='lr', candidates=[])
    with pytest.raises(ValueError, match='Baseline'):
        pc.run_task_a_roles(context(protocol='baseline'), candidates=[{}])


def test_snapshot_mapping_hash_alignment_and_scoring_columns():
    data = context()
    mapping = {k: getattr(data.snapshots, k) for k in ('consensus', 'mechanistic', 'manifest')}
    assert pc.SnapshotTables.from_snapshots(mapping).hashes == data.snapshots.hashes
    with pytest.raises(ValueError, match='exactly'):
        pc.SnapshotTables.from_snapshots(dict(mapping, studies=b'{}'))
    targets, predictions = scoring_rows(['interaction'], [.5])
    with pytest.raises(ValueError, match='columns'):
        pc.strict_score(targets, predictions.drop(columns='probability'))


def test_calibration_rejects_single_class_and_endpoint_predictions(monkeypatch):
    data = context()
    model = pc.fit_task_a(data)
    original = data.rows
    def single_class(role, *, targets=False):
        rows = original(role, targets=targets)
        if targets and role == 'probability_calibration':
            rows['consensus_label'] = 'interaction'
        return rows
    monkeypatch.setattr(data, 'rows', single_class)
    c = pc.ProbabilityCalibration()
    with pytest.raises(ValueError, match='Both binary'):
        c.fit(data, model)
    assert c.status == 'failed'
    monkeypatch.setattr(data, 'rows', original)
    original_predict = model.predict
    monkeypatch.setattr(model, 'predict', lambda pairs: original_predict(pairs).assign(probability=0.))
    with pytest.raises(ValueError, match='interior'):
        c.fit(data, model)
    assert c.status == 'failed'


@pytest.mark.parametrize('grant', [None, True, object(), {'contract_status': 'passed'},
    {'eligible': False, 'contract_status': 'passed'},
    {'eligible': True, 'contract_status': 'passed', 'checks': []}])
def test_public_execution_has_no_caller_issuable_authorization(grant):
    data = context()
    # Private synthetic predictors cannot authorize public calibration/prediction.
    predictor = pc.fit_task_a(data)
    targets, predictions = scoring_rows(['interaction', 'no_interaction'], [.4, .6])
    calls = [
        lambda: production.fit_task_a(data, authorization=grant),
        lambda: production.tune_task_a(data, family='lr', candidates=[{}], authorization=grant),
        lambda: production.run_task_a_roles(data, candidates=[{}], alpha=.2, authorization=grant),
        lambda: production.strict_score(targets, predictions, authorization=grant),
        lambda: production.binary_loss(np.array([0, 1]), [.4, .6], authorization=grant),
        lambda: production.assess(data, predictor, authorization=grant),
        lambda: production.role_predictions(data, predictor, 'assessment', authorization=grant),
        lambda: production.FixedPredictor(predictor._estimator, data, authorization=grant),
        lambda: production.ProbabilityCalibration().fit(data, predictor, authorization=grant),
        lambda: production.ProbabilityCalibration().transform([.5], predictor, authorization=grant),
        lambda: production.ConformalCalibration().fit(data, predictor, alpha=.2, authorization=grant),
        lambda: production.ConformalCalibration().predict_sets(data, predictor, authorization=grant),
        lambda: object.__new__(production.FixedPredictor).predict(data.rows('assessment'), authorization=grant),
        lambda: object.__new__(production.TaskBSnapshotAdapter).fit_prior_reliability(authorization=grant),
        lambda: object.__new__(production.TaskBSnapshotAdapter).tune_residual(None, authorization=grant),
    ]
    for call in calls:
        with pytest.raises(PermissionError, match='orchestrator-issued'):
            call()


def test_original_false_eligibility_bypass_is_blocked():
    from src.training.release_preflight import preflight_release_snapshots
    data = context()
    snapshots = {k: getattr(data.snapshots, k) for k in ('consensus', 'mechanistic', 'manifest')}
    report = preflight_release_snapshots(snapshots, task='A', regime='pair', protocol='full',
        assignments=data.manifest['assignments'], partition_roles=data.manifest['partition_roles'],
        target_policy='binary_consensus_v1', metric_policy='binary_discrimination_v1')
    assert report['eligible'] is False
    with pytest.raises(PermissionError):
        production.run_task_a_roles(data, candidates=[{}], alpha=.2, authorization=report)


def test_no_public_dispatch_even_if_gate_returns(monkeypatch):
    monkeypatch.setattr('src.training.evaluation_orchestrator._require_execution_authorization', lambda *a: None)
    with pytest.raises(PermissionError, match='no authorization issuance'):
        production.fit_task_a(context())


def test_snapshot_rebinding_and_cached_table_mutation_rejected():
    data = context()
    other = context(assessment_flip=True)
    with pytest.raises(AttributeError):
        data.snapshots = other.snapshots
    # Even private rebinding and in-place byte changes are detected before use.
    data._snapshots = other.snapshots
    with pytest.raises(ValueError, match='changed'):
        pc.fit_task_a(data)
    data = context()
    data._labels.loc[0, 'consensus_label'] = 'interaction'
    with pytest.raises(ValueError, match='changed'):
        data.rows('fit')
    data = context()
    object.__setattr__(data.snapshots, 'consensus', other.snapshots.consensus)
    with pytest.raises(ValueError, match='changed'):
        data.rows('fit')


def test_caller_table_provider_and_forged_fit_records_rejected():
    data = context()
    class TableProvider:
        hashes = data.snapshots.hashes
        def tables(self):
            return data.snapshots.tables()
    with pytest.raises(ValueError, match='exact immutable'):
        pc.RoleData(TableProvider(), assignments=data.manifest['assignments'],
                    partition_roles=data.manifest['partition_roles'], regime='pair', protocol='full')
    # Reproduce assessment-trained-estimator wrapping without issuing provenance.
    from src.features.task_a_contract import task_a_training_pipeline
    from sklearn.linear_model import LogisticRegression
    rows = data.rows('assessment', targets=True)
    y, mask = pc.binary_targets(rows.consensus_label, both=True)
    estimator = task_a_training_pipeline(LogisticRegression()).fit(data.features(rows.loc[mask]), y)
    with pytest.raises(ValueError, match='trusted fit provenance'):
        pc.FixedPredictor(estimator, data, model='forged')
    forged = object.__new__(pc.FixedPredictor)
    forged._data, forged._estimator = data, estimator
    forged._state = pc.digest(estimator)
    forged.fit_record = {'task': 'A', 'fit_role_hash': 'forged'}
    for operation in [lambda: forged.predict(rows),
                      lambda: pc.ProbabilityCalibration().fit(data, forged),
                      lambda: pc.ConformalCalibration().fit(data, forged, alpha=.2)]:
        with pytest.raises(ValueError, match='trusted fit provenance'):
            operation()


def test_registered_fit_provenance_is_immutable_and_bound():
    from dataclasses import FrozenInstanceError
    data = context()
    model = pc.fit_task_a(data, parameters={'C': .1})
    record = model.provenance
    assert record.task == 'A' and record.fit_role_hash and record.plan_hash
    assert record.snapshot_hashes == tuple(sorted(data.snapshots.hashes.items()))
    assert record.feature_contract == pc.TASK_A_FEATURES
    assert json.loads(record.configuration) == dict(family='lr', parameters={'C': .1})
    assert record.derived_tables_hash == pc.digest((data._labels, data._curation))
    with pytest.raises(FrozenInstanceError):
        record.task = 'B'
    with pytest.raises(AttributeError):
        model.provenance = record
    identity = model.identity
    object.__setattr__(record, 'configuration', 'forged external copy')
    assert model.identity == identity
    model.check()
    # Copying a fit record or predictor does not create a registered fitted object.
    clone = copy.copy(model)
    with pytest.raises(ValueError, match='trusted fit provenance'):
        pc.ProbabilityCalibration().fit(data, clone)


def test_informative_assessment_independence_including_selection_and_calibrator():
    outputs = []
    for flip in [False, True]:
        data = context(assessment_flip=flip, informative=True)
        assert data.features(data.rows('fit')).iloc[:, 0].nunique() > 1
        model = pc.tune_task_a(data, family='lr', candidates=[{'C': .1}, {'C': 1.}, {'C': 10.}])
        c = pc.ProbabilityCalibration().fit(data, model)
        cc = pc.ConformalCalibration().fit(data, model, alpha=.2, calibrator=c)
        p = pc.role_predictions(data, model, 'assessment', calibrator=c)
        assert np.ptp(p.probability) > .01
        outputs.append((pc.digest(model._estimator), model.provenance.configuration,
                        c.temperature, p, cc.predict_sets(data, model, calibrator=c)))
    assert outputs[0][:3] == outputs[1][:3]
    pd.testing.assert_frame_equal(outputs[0][3], outputs[1][3])
    pd.testing.assert_frame_equal(outputs[0][4], outputs[1][4])


def test_calibrator_and_conformal_reject_different_registered_predictor():
    data = context(informative=True)
    a = pc.fit_task_a(data, parameters={'C': .1})
    b = pc.fit_task_a(data, parameters={'C': 1.})
    c = pc.ProbabilityCalibration().fit(data, a)
    cc = pc.ConformalCalibration().fit(data, a, alpha=.2, calibrator=c)
    with pytest.raises(ValueError, match='matching fixed predictor'):
        c.transform([.5], b)
    with pytest.raises(ValueError, match='unchanged'):
        cc.predict_sets(data, b, calibrator=c)


def test_returned_tables_cannot_mutate_cached_snapshot_binding():
    data = context()
    labels, mechanics, manifest = data.snapshots.tables()
    labels.loc[0, 'consensus_label'] = 'interaction'
    mechanics.loc[0, 'source'] = 'UNVERIFIED'
    manifest.clear()
    data.check()
    assert data.rows('fit', targets=True).iloc[0].consensus_label == 'no_interaction'
    model = pc.fit_task_a(data)
    model._data = context()
    with pytest.raises(ValueError, match='binding changed'):
        pc.ProbabilityCalibration().fit(data, model)


def test_nonfinite_temperature_optimizer_parameter_fails_closed(monkeypatch):
    data = context()
    model = pc.fit_task_a(data)
    monkeypatch.setattr(pc, 'minimize_scalar', lambda *a, **k:
                        type('Nonfinite', (), {'success': True, 'fun': .5, 'x': np.nan})())
    c = pc.ProbabilityCalibration()
    with pytest.raises(ValueError, match='optimizer'):
        c.fit(data, model)
    assert c.status == 'failed'
