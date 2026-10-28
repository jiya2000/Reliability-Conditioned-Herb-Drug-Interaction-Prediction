"""Read-only app tests; all modified tables are temporary synthetic fixtures."""
from pathlib import Path

import pandas as pd
import pytest
import runpy
import sys
from types import ModuleType

from app.research_status import ResearchState, describe_pair, load_research_state
from src.data.mechanistic_curation import CURATION_COLUMNS
from src.features.task_a_contract import TASK_A_TARGETS

ROOT = Path(__file__).resolve().parent.parent


@pytest.fixture
def research_root(tmp_path):
    (tmp_path / 'data/benchmark').mkdir(parents=True)
    (tmp_path / 'data/curation').mkdir(parents=True)
    # Reuse existing registered references; modified outcomes are test-only.
    studies = pd.read_csv(ROOT / 'data/benchmark/verified_botanical_hdi.csv').iloc[:2].copy()
    studies.loc[1, 'outcome'] = 'no_interaction_observed'
    studies.to_csv(tmp_path / 'data/benchmark/verified_botanical_hdi.csv', index=False)
    columns = ['herb_id', 'drug_id', 'consensus_label', 'source_name', 'source_url_or_citation', 'retrieval_date']
    pd.DataFrame(columns=columns).to_csv(tmp_path / 'data/benchmark/consensus_labels.csv', index=False)
    rows = []
    for entity in [studies.iloc[0].herb_id, studies.iloc[0].drug_id]:
        for target in TASK_A_TARGETS:
            row = dict.fromkeys(CURATION_COLUMNS, '')
            row.update(entity=entity, target=target, status='tested_no_effect',
                       source='SYNTHETIC_FIXTURE', citation='SYNTHETIC_FIXTURE')
            rows.append(row)
    pd.DataFrame(rows).to_csv(tmp_path / 'data/curation/mechanistic_features.csv', index=False)
    return tmp_path


def test_real_readiness_blocks_without_training():
    state = load_research_state(ROOT)
    assert state.evidence_available and len(state.studies) == 141
    assert any('labels are empty' in s for s in state.blockers)
    assert any('remain unverified' in s for s in state.blockers)
    assert any('inference bundle' in s for s in state.blockers)
    assert any('primary/echo' in s for s in state.blockers)


def test_conflicting_findings_do_not_invent_consensus(research_root):
    state = load_research_state(research_root)
    row = state.studies.iloc[0]
    pair = describe_pair(state, row.herb_id, row.drug_id)
    assert pair['conflicting_findings']
    assert pair['positive_findings'] == pair['negative_findings'] == 1
    assert pair['consensus'] is None
    assert 'probability' not in pair and 'confidence' not in pair


@pytest.mark.parametrize('verdict', ['interaction', 'no_interaction', 'disputed'])
def test_only_external_consensus_defines_verdict(research_root, verdict):
    studies = pd.read_csv(research_root / 'data/benchmark/verified_botanical_hdi.csv')
    pd.DataFrame([dict(herb_id=studies.iloc[0].herb_id, drug_id=studies.iloc[0].drug_id,
        consensus_label=verdict, source_name='SYNTHETIC_FIXTURE',
        source_url_or_citation='SYNTHETIC_FIXTURE', retrieval_date='2026-10-29')]).to_csv(
            research_root / 'data/benchmark/consensus_labels.csv', index=False)
    state = load_research_state(research_root)
    pair = describe_pair(state, studies.iloc[0].herb_id, studies.iloc[0].drug_id)
    assert pair['consensus']['consensus_label'] == verdict
    assert any('inference bundle' in s for s in state.blockers)
    assert not any('labels are empty' in s for s in state.blockers)


def test_zero_studies_does_not_mean_negative(research_root):
    state = load_research_state(research_root)
    pair = describe_pair(state, 'HERB_UNSTUDIED', 'DRUG_UNSTUDIED')
    assert pair['evidence_status'] == 'no_studies'
    assert pair['consensus'] is None
    assert not pair['conflicting_findings']


def test_pair_mechanisms_preserve_unverified_records_without_inference():
    state = load_research_state(ROOT)
    row = state.studies.iloc[0]
    pair = describe_pair(state, row.herb_id, row.drug_id)
    expected = state.mechanisms.loc[state.mechanisms.entity.isin([row.herb_id, row.drug_id])]
    pd.testing.assert_frame_equal(pair['mechanisms'], expected)
    assert not expected.empty
    assert expected.citation.eq('UNVERIFIED').all()
    assert expected.value_ic50_uM.isna().all()
    assert describe_pair(state, 'HERB_UNSTUDIED', 'DRUG_UNSTUDIED')['mechanisms'].empty
    pair['mechanisms'].loc[:, 'citation'] = 'TEST_ONLY'
    pd.testing.assert_frame_equal(state.mechanisms.loc[expected.index], expected)


def test_invalid_mechanisms_are_not_displayed(research_root):
    path = research_root / 'data/curation/mechanistic_features.csv'
    frame = pd.read_csv(path)
    frame.loc[0, 'status'] = 'invalid'
    frame.to_csv(path, index=False)
    state = load_research_state(research_root)
    assert state.mechanisms.empty
    assert any('Mechanistic inputs unavailable' in s for s in state.blockers)


def test_missing_tables_fail_closed(tmp_path):
    state = load_research_state(tmp_path)
    pair = describe_pair(state, 'HERB_TEST', 'DRUG_TEST')
    assert pair['evidence_status'] == 'unavailable'
    assert len(state.blockers) == 4


def test_invalid_evidence_is_not_silently_dropped(research_root):
    path = research_root / 'data/benchmark/verified_botanical_hdi.csv'
    frame = pd.read_csv(path)
    frame.loc[0, 'source_id'] = 'UNVERIFIED'
    frame.loc[0, 'source_url_or_doi'] = 'UNVERIFIED'
    frame.to_csv(path, index=False)
    state = load_research_state(research_root)
    assert not state.evidence_available
    assert state.studies.empty
    assert any('evidence validation errors' in s for s in state.blockers)


def test_missing_mechanistic_values_block_even_with_provenance(research_root):
    path = research_root / 'data/curation/mechanistic_features.csv'
    frame = pd.read_csv(path)
    frame = frame.loc[frame.target.ne('CYP3A4')]
    frame.to_csv(path, index=False)
    state = load_research_state(research_root)
    assert any('missing mechanistic values' in s for s in state.blockers)


@pytest.mark.parametrize('outcome', ['unknown', 'conflicting', 'inconclusive'])
def test_inconclusive_evidence_is_not_negative(outcome):
    state = ResearchState(studies=pd.DataFrame([dict(herb_id='HERB_TEST', drug_id='DRUG_TEST', outcome=outcome)]),
                          evidence_available=True)
    pair = describe_pair(state, 'HERB_TEST', 'DRUG_TEST')
    assert pair['negative_findings'] == 0
    assert pair['inconclusive_findings'] == 1
    assert pair['conflicting_findings'] == (outcome == 'conflicting')


@pytest.mark.parametrize('entrypoint', ['app/interactive_app.py', 'app/frontend.py'])
def test_streamlit_startup_never_fits_or_calls_legacy_api(entrypoint, monkeypatch):
    from sklearn.pipeline import Pipeline
    import src.training.task_a
    import src.training.task_b

    def forbidden(*args, **kwargs):
        raise AssertionError('Startup attempted fitting, prediction or API demo')
    monkeypatch.setattr(Pipeline, 'fit', forbidden)
    monkeypatch.setattr(src.training.task_a, 'run_task_a', forbidden)
    monkeypatch.setattr(src.training.task_b, 'fit_task_b', forbidden)
    requests = ModuleType('requests')
    requests.post = forbidden
    monkeypatch.setitem(sys.modules, 'requests', requests)

    # Exercise both entry points without requiring optional UI dependencies in
    # the core verification environment. A separate local AppTest smoke check
    # verifies real Streamlit rendering where the application is installed.
    class StopUI(Exception):
        pass
    st = ModuleType('streamlit')
    calls = []
    def stop():
        raise StopUI()
    st.stop = stop
    for name in ['set_page_config', 'title', 'warning', 'caption', 'subheader',
                 'write', 'info', 'error', 'dataframe', 'button']:
        setattr(st, name, lambda *args, _name=name, **kwargs: calls.append((_name, args, kwargs)))
    st.selectbox = lambda label, options, **kwargs: 'Evidence explorer' if label == 'View' else options[0]
    monkeypatch.setitem(sys.modules, 'streamlit', st)
    for _ in range(2):  # Streamlit reruns must also stop before legacy code.
        with pytest.raises(StopUI):
            runpy.run_path(str(ROOT / entrypoint), run_name='__main__')
    assert any(name == 'warning' and 'Research readiness' in args[0]
               for name, args, kwargs in calls)
    assert any(name == 'button' and kwargs['disabled'] for name, args, kwargs in calls)
    assert any(name == 'info' and 'model abstention are unavailable' in args[0]
               for name, args, kwargs in calls)
    assert any(name == 'caption' and 'not verified mechanisms or model predictions' in args[0]
               for name, args, kwargs in calls)
    assert any(name == 'dataframe' and 'citation' in args[0]
               and args[0].citation.eq('UNVERIFIED').all()
               for name, args, kwargs in calls)
