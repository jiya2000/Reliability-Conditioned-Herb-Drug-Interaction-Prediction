"""Scientific invariants of the separate retrospective assay experiment."""
import json

import numpy as np
import pandas as pd
import pytest

from src.assay.data import aggregate, assign_roles, canonical_structure, fingerprints, scaffold_key
from src.assay.experiment import bootstrap_mae, conformal_radius, run
from src.assay.predict import load_experiment, predict_smiles
from src.assay.verify import verify_experiment


def test_structure_equivalence_and_salt_grouping():
    assert canonical_structure('CCO') == canonical_structure('OCC')
    assert canonical_structure('CCO.[Na+]') == canonical_structure('CCO')
    assert canonical_structure('C[C@H](O)CC') == canonical_structure('C[C@@H](O)CC')
    assert canonical_structure('[Na+].[Cl-]') is None
    assert canonical_structure('invalid') is None
    assert scaffold_key('CCC') == scaffold_key('CCCC')
    assert np.array_equal(fingerprints(['CCO'])[0], fingerprints(['OCC'])[0])


def records(n=120):
    return pd.DataFrame([dict(activity_id=i, structure='C' * (i + 1), scaffold=f'scaffold_{i}',
                              target_symbol='CYP3A4', document_chembl_id=f'doc_{i // 2}',
                              assay_chembl_id=f'assay_{i // 2}', molecule_chembl_id=f'mol_{i}',
                              pic50=4 + (i % 8) / 4) for i in range(n)])


@pytest.mark.parametrize('mode', ['compound', 'scaffold', 'source'])
def test_roles_reproducible_disjoint_and_outcome_independent(mode):
    frame = records()
    assigned, purged, audit = assign_roles(frame, mode)
    changed = frame.copy()
    changed['pic50'] *= -100
    other, _, _ = assign_roles(changed, mode)
    assert assigned.role.tolist() == other.role.tolist()
    assert len(assigned) + len(purged) == len(frame)
    assert set(assigned.role) == {'train', 'validation', 'calibration', 'test'}
    for overlap in audit['overlaps'].values():
        assert overlap['structure'] == 0
        if mode == 'source':
            assert overlap['document_chembl_id'] == 0
        if mode == 'scaffold':
            assert overlap['scaffold'] == 0


def test_source_split_purges_repeated_molecules_across_documents():
    frame = records()
    # Each compound has a second publication, ensuring relationships cross roles.
    repeated = frame.copy()
    repeated.activity_id += 1000
    repeated.document_chembl_id = [f'repeat_{i}' for i in range(len(frame))]
    assigned, purged, audit = assign_roles(pd.concat([frame, repeated]), 'source')
    assert len(purged) > 0
    assert assigned.groupby('structure').role.nunique().max() == 1
    assert not any(v['document_chembl_id'] for v in audit['overlaps'].values())


def test_document_balanced_aggregation_preserves_conflict():
    frame = pd.DataFrame([dict(activity_id=i, structure='CCO', target_symbol='CYP3A4',
                               role='train', document_chembl_id=doc, molecule_chembl_id='mol', pic50=value)
                          for i, (doc, value) in enumerate([('a', 3), ('a', 3), ('a', 3), ('b', 7)])])
    sample = aggregate(frame).iloc[0]
    assert sample.pic50 == 5  # documents balanced, not measurement-count weighted
    assert sample.spread == 4
    assert sample.documents == 2 and sample.measurements == 4
    assert json.loads(sample.activity_ids) == [0, 1, 2, 3]


def test_finite_sample_conformal_order_statistic():
    assert conformal_radius(np.arange(1, 11), np.zeros(10), .1) == 10
    with pytest.raises(ValueError, match='Insufficient'):
        conformal_radius([1, 2], [0, 0], .1)
    with pytest.raises(ValueError):
        conformal_radius([1], [0], 0)


def test_bootstrap_clusters_all_targets_of_a_structure():
    frame = pd.DataFrame({'structure': ['a', 'a', 'b'], 'observed_pic50': [1., 3., 2.],
                          'predicted_pic50': [0., 0., 0.]})
    ci = bootstrap_mae(frame)
    assert ci['lower'] == ci['upper'] == 2


def test_end_to_end_artifacts_inference_and_tamper_detection(tmp_path, monkeypatch):
    import src.assay.experiment as experiment
    source = tmp_path / 'source'
    source.mkdir()
    (source / 'manifest.json').write_text('{}')
    frame = records(140)
    manifest = {'outputs': {}, 'source_status': {'chembl_db_version': 'test-fixture'}}
    monkeypatch.setattr(experiment, 'load_records', lambda _: (frame, pd.DataFrame(columns=['activity_id', 'exclusion_reason']), manifest))
    destination = tmp_path / 'run'
    report = run(source, destination, mode='compound')
    assert report['complete'] and not report['clinical_task_a_eligible']
    assert len(report['targets']) == 1
    bundle, loaded = load_experiment(destination)
    assert verify_experiment(destination)['verified']
    result = predict_smiles(bundle, 'CCO')
    assert result['predictions'][0]['clinical_interaction_verdict'] == 'unavailable'
    assert result['predictions'][0]['lower_pic50'] <= result['predictions'][0]['upper_pic50']
    predictions = pd.read_csv(destination / 'test_predictions.csv')
    selected = predictions[predictions.selected]
    assert loaded['selected_micro_test']['mae'] == pytest.approx(
        (selected.observed_pic50 - selected.predicted_pic50).abs().mean())
    with pytest.raises(FileExistsError):
        run(source, destination, mode='compound')
    assigned, _, _ = assign_roles(frame, 'compound')
    poisoned = frame.copy()
    poisoned.loc[assigned.role == 'test', 'pic50'] += 100
    monkeypatch.setattr(experiment, 'load_records', lambda _: (poisoned, pd.DataFrame(columns=['activity_id', 'exclusion_reason']), manifest))
    second = tmp_path / 'poisoned_test'
    run(source, second, mode='compound')
    other_bundle, other_report = load_experiment(second)
    assert loaded['targets']['CYP3A4']['validation'] == other_report['targets']['CYP3A4']['validation']
    assert bundle['targets']['CYP3A4']['radius'] == other_bundle['targets']['CYP3A4']['radius']
    assert np.allclose(bundle['targets']['CYP3A4']['model'].predict(fingerprints(['CCO'])),
                       other_bundle['targets']['CYP3A4']['model'].predict(fingerprints(['CCO'])))
    with (destination / 'samples.csv').open('a') as stream:
        stream.write('tampered')
    with pytest.raises(ValueError, match='Changed assay artifact'):
        load_experiment(destination)


def test_official_row_screen_excludes_species_conflicts_and_wrong_endpoint(tmp_path, monkeypatch):
    import src.assay.data as data
    good = dict(activity_id=1, canonical_smiles='CCO', standard_type='IC50', standard_relation='=',
                standard_units='nM', organism='Homo sapiens', assay_type='A',
                assay_description='Inhibition of human CYP3A4', standard_value=1000, pchembl_value=6.,
                document_chembl_id='doc', target_symbol='CYP3A4')
    rows = [good, dict(good, activity_id=2, assay_description='Inhibition in guinea pig heart'),
            dict(good, activity_id=3, standard_relation='<'), dict(good, activity_id=4, pchembl_value=9.),
            dict(good, activity_id=5, assay_description='Transport substrate accumulation')]
    pd.DataFrame(rows).to_csv(tmp_path / 'assay_research_candidates.csv', index=False)
    monkeypatch.setattr(data, 'verify_dataset', lambda _: {})
    retained, excluded, _ = data.load_records(tmp_path)
    assert retained.activity_id.tolist() == [1]
    assert 'nonhuman_description' in excluded.loc[excluded.activity_id == 2, 'exclusion_reason'].iloc[0]
    assert retained.pic50.tolist() == [6.]


def test_assay_view_renders_without_startup_fit(monkeypatch):
    pytest.importorskip('streamlit')
    from pathlib import Path
    from streamlit.testing.v1 import AppTest
    from sklearn.ensemble import ExtraTreesRegressor

    def forbid(*args, **kwargs):
        pytest.fail('Assay view attempted startup fitting')
    monkeypatch.setattr(ExtraTreesRegressor, 'fit', forbid)
    root = Path(__file__).resolve().parents[1]
    app = AppTest.from_file(str(root / 'app/interactive_app.py')).run(timeout=30)
    app.selectbox(key='research_view').set_value('Compound assay experiment').run(timeout=30)
    assert not app.exception
    assert any('Official-data compound assay experiment' in item.value for item in app.header)
