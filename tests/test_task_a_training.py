"""Synthetic infrastructure tests only; no scientific labels or citations."""
import numpy as np
import pandas as pd
import pytest

from src.data.evidence_schema import ConsensusLabelsNotPopulatedError
from src.data.mechanistic_curation import CURATION_COLUMNS, UnverifiedCurationError
from src.features.task_a_contract import TASK_A_TARGETS, TaskAFeatureContractError
from src.training.task_a import main, prepare_task_a, run_task_a, task_a_splits


@pytest.fixture
def tables(tmp_path):
    labels = pd.DataFrame([dict(herb_id=f'HERB_{i}', drug_id=f'DRUG_{i}',
        consensus_label='interaction' if i % 2 else 'no_interaction',
        source_name='SYNTHETIC_FIXTURE', source_url_or_citation='SYNTHETIC_FIXTURE',
        retrieval_date='2026-10-28') for i in range(24)])
    rows = []
    for entity in list(labels.herb_id) + list(labels.drug_id):
        for target in TASK_A_TARGETS:
            row = dict.fromkeys(CURATION_COLUMNS, '')
            row.update(entity=entity, target=target, status='tested_no_effect',
                       source='SYNTHETIC_FIXTURE', citation='SYNTHETIC_FIXTURE')
            rows.append(row)
    curation = pd.DataFrame(rows)
    lp, cp = tmp_path / 'labels.csv', tmp_path / 'curation.csv'
    labels.to_csv(lp, index=False)
    curation.to_csv(cp, index=False)
    return lp, cp, labels, curation


@pytest.mark.parametrize('mode', ['herb', 'drug', 'pair'])
def test_grouped_coverage_and_determinism(tables, mode):
    lp, cp, _, _ = tables
    pairs, X, y, _ = prepare_task_a(lp, cp)
    folds = task_a_splits(pairs, y, mode=mode, n_splits=3, repeats=2)
    again = task_a_splits(pairs, y, mode=mode, n_splits=3, repeats=2)
    for repeat in range(2):
        assert sorted(np.concatenate([f[3] for f in folds if f[0] == repeat])) == list(range(24))
    for a, b in zip(folds, again):
        assert np.array_equal(a[3], b[3])
        assert not set(pairs.iloc[a[2]].herb_id) & set(pairs.iloc[a[3]].herb_id)


@pytest.mark.parametrize('model', ['lr', 'hgb'])
def test_real_pipeline_cv_on_constant_synthetic_features(tables, model):
    lp, cp, _, _ = tables
    result = run_task_a(lp, cp, n_splits=3, repeats=1, model=model)
    assert len(result['predictions']) == 24
    assert len(result['folds']) == 3
    assert all(f['roc_auc'] == .5 for f in result['folds'])


def test_check_only_never_fits(tables, monkeypatch):
    lp, cp, _, _ = tables
    monkeypatch.setattr('src.training.task_a.task_a_training_pipeline',
                        lambda *a: pytest.fail('check-only attempted fitting'))
    assert run_task_a(lp, cp, n_splits=3, check_only=True)['predictions'] == []


def test_disputed_exclusion_and_no_outcome_fallback(tables):
    lp, cp, labels, _ = tables
    labels.loc[0, 'consensus_label'] = 'disputed'
    labels['outcome'] = 'interaction_present'
    labels.to_csv(lp, index=False)
    pairs, X, y, disputed = prepare_task_a(lp, cp)
    assert disputed == 1 and len(y) == 23
    assert y.tolist() == [i % 2 for i in range(1, 24)]
    assert 'outcome' not in X


def test_empty_labels_block_before_curation(tables):
    lp, cp, labels, _ = tables
    labels.iloc[:0].to_csv(lp, index=False)
    with pytest.raises(ConsensusLabelsNotPopulatedError):
        prepare_task_a(lp, 'nonexistent.csv')


def test_unverified_and_missing_features_block(tables):
    lp, cp, _, curation = tables
    curation.loc[0, 'source'] = 'UNVERIFIED'
    curation.to_csv(cp, index=False)
    with pytest.raises(UnverifiedCurationError):
        prepare_task_a(lp, cp)
    curation.loc[~(curation.entity.isin(['HERB_0', 'DRUG_0']) &
                    curation.target.eq(TASK_A_TARGETS[0]))].to_csv(cp, index=False)
    with pytest.raises(TaskAFeatureContractError, match='missing mechanistic'):
        prepare_task_a(lp, cp)


def test_degenerate_folds_rejected(tables):
    lp, cp, _, _ = tables
    pairs, X, y, _ = prepare_task_a(lp, cp)
    with pytest.raises(ValueError, match='lacks both classes'):
        task_a_splits(pairs, np.zeros(len(y)), n_splits=3)
    with pytest.raises(ValueError, match='Insufficient'):
        task_a_splits(pairs, y, n_splits=30)


def test_cli_block_does_not_write_result(tables, tmp_path):
    lp, cp, labels, _ = tables
    labels.iloc[:0].to_csv(lp, index=False)
    output = tmp_path / 'result.json'
    with pytest.raises(SystemExit) as error:
        main(['--consensus', str(lp), '--curation', str(cp), '--output', str(output)])
    assert error.value.code == 2
    assert not output.exists()


def test_auc_integrity_alert_is_enforced(tables, monkeypatch):
    lp, cp, _, _ = tables
    monkeypatch.setattr('src.training.task_a.roc_auc_score', lambda *a: 1.0)
    with pytest.raises(ValueError, match='AUC integrity alert'):
        run_task_a(lp, cp, n_splits=3, repeats=1)
