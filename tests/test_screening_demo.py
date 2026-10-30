"""Behavioral checks for evidence-linked research screening."""
from copy import deepcopy
from pathlib import Path

import pytest

from src.assay.screening import load_catalog, screen_pair, summarize_evidence

ROOT = Path(__file__).resolve().parents[1]


def test_duplicate_citations_never_increase_unique_evidence():
    catalog = load_catalog(ROOT)
    rows = catalog['studies']
    before = summarize_evidence(rows, "St John's wort", 'Digoxin')
    repeated = summarize_evidence(rows + [deepcopy(rows[0])] * 8, "St John's wort", 'Digoxin')
    assert before['finding'] == repeated['finding']
    assert before['unique_publications'] == repeated['unique_publications'] == 2
    assert repeated['duplicate_records_removed'] == 8


def test_conflicting_duplicate_extractions_are_rejected():
    rows = load_catalog(ROOT)['studies']
    conflict = dict(rows[0], ratio=.1)
    with pytest.raises(ValueError, match='Conflicting'):
        summarize_evidence(rows + [conflict], "St John's wort", 'Digoxin')


def test_unknown_pairs_and_different_contexts_never_become_safe():
    catalog = load_catalog(ROOT)
    result = screen_pair(catalog, None, 'Turmeric', 'Digoxin', 'different_or_unknown')
    assert result['clinical_prediction'].startswith('ABSTAIN')
    assert not result['evidence']['studies']
    assert any('differs' in reason for reason in result['missing_requirements'])
    result = screen_pair(catalog, None, 'Echinacea', 'Digoxin')
    assert result['clinical_prediction'].startswith('ABSTAIN')
    assert result['evidence']['studies'][0]['direction'] == 'no_change_reported'


def test_induction_guard_disables_inhibition_exposure_ratio(monkeypatch):
    import src.assay.screening as screening
    monkeypatch.setattr(screening, 'predict_smiles', lambda *_: {'predictions': [
        dict(target='ABCB1', implied_ic50_uM=1., lower_pic50=5., upper_pic50=7.)]})
    result = screen_pair(load_catalog(ROOT), object(), "St John's wort", 'Digoxin', exposure_uM=1.)
    assert 'illustrative_exposure_over_ic50' not in result['assay_estimates'][0]
    assert any('Induction parameters' in reason for reason in result['missing_requirements'])


def test_reported_ratio_orientation_and_interval():
    result = screen_pair(load_catalog(ROOT), None, "St John's wort", 'Cyclosporine')
    study = result['evidence']['studies'][0]
    assert study['ratio'] == pytest.approx(1 / 1.83)
    assert study['ratio_ci'][0] < study['ratio'] < study['ratio_ci'][1]
    with pytest.raises(ValueError):
        screen_pair(load_catalog(ROOT), None, 'Turmeric', 'Digoxin', exposure_uM=float('nan'))


def test_presentation_demo_paths_without_fitting(monkeypatch):
    pytest.importorskip('streamlit')
    from sklearn.ensemble import ExtraTreesRegressor
    from streamlit.testing.v1 import AppTest

    if not (ROOT / 'artifacts/assay/scaffold/report.json').exists():
        pytest.skip('Locally trained experiment not installed; core screening tests still run')

    def forbid(*_, **__):
        pytest.fail('Demo attempted training')
    monkeypatch.setattr(ExtraTreesRegressor, 'fit', forbid)
    app = AppTest.from_file(str(ROOT / 'app/demo.py')).run(timeout=60)
    assert not app.exception
    for herb in ['Turmeric', 'Echinacea', "St John's wort"]:
        app.selectbox(key='screen_herb').set_value(herb).run(timeout=30)
        assert not app.exception
    app.checkbox[0].check().run(timeout=30)
    assert not app.exception
    assert any('1 duplicate records removed' in item.value for item in app.caption)
    app.radio[0].set_value('Different or unknown preparation / dose').run(timeout=30)
    assert not app.exception
    assert any('selected context differs' in item.value for item in app.warning)
