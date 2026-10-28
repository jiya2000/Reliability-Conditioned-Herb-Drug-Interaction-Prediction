"""Synthetic end-to-end predictor/target isolation; no scientific data/results.

Reuse the existing temporary table fixtures and production entry points. Findings
are allowed Task B evidence, never Task A or study-reliability predictors.
"""
import numpy as np
import pandas as pd
import pytest
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import HistGradientBoostingClassifier

from tests.test_task_a_training import tables as task_a_tables
from tests.test_task_b_training import fixture as task_b_tables
from src.features.task_a_contract import TASK_A_FEATURES
from src.training.task_a import run_task_a
from src.training.task_b import (
    RELIABILITY_FEATURES, SupervisedStudyReliability, fit_task_b,
    reliability_features, directions, select_studies,
)

OUTCOME_FIELDS = ('outcome', 'raw_label', 'original_text', 'S_pos', 'S_neg',
                  'raw_positive_count', 'raw_negative_count', 'contradiction_strength')


def contaminate(frame):
    """Deliberately extreme, already numeric outcome-like fields and text."""
    frame = frame.copy()
    for i, field in enumerate(OUTCOME_FIELDS):
        frame[field] = (np.arange(len(frame)) % 2 * 2 - 1) * (i + 1) * 100000
    frame['original_text'] = 'positive outcome negative outcome HERB_identity'
    # Even caller-supplied approved feature names must be reconstructed for A.
    frame['signed_effect_CYP3A4'] = frame['S_pos']
    return frame


@pytest.mark.parametrize('model,estimator', [('lr', LogisticRegression), ('hgb', HistGradientBoostingClassifier)])
def test_task_a_cv_ignores_outcome_fields_at_actual_estimator_boundary(task_a_tables, monkeypatch, model, estimator):
    lp, cp, labels, _ = task_a_tables
    fit_calls = []
    original_fit = estimator.fit

    def fit(self, X, y, *args, **kwargs):
        assert isinstance(X, pd.DataFrame)
        assert tuple(X.columns) == TASK_A_FEATURES
        fit_calls.append((X.copy(), np.asarray(y).copy()))
        return original_fit(self, X, y, *args, **kwargs)

    monkeypatch.setattr(estimator, 'fit', fit)
    baseline = run_task_a(lp, cp, n_splits=3, repeats=1, model=model)
    first_calls = list(fit_calls)
    fit_calls.clear()
    contaminate(labels).to_csv(lp, index=False)
    changed = run_task_a(lp, cp, n_splits=3, repeats=1, model=model)
    assert changed == baseline
    assert len(first_calls) == len(fit_calls) == 3
    for (before_X, before_y), (after_X, after_y) in zip(first_calls, fit_calls):
        pd.testing.assert_frame_equal(before_X, after_X)
        np.testing.assert_array_equal(before_y, after_y)


def test_task_b_fitting_and_scaling_ignore_outcome_predictor_injection(task_b_tables, monkeypatch):
    args, labels, _ = task_b_tables
    fit_calls = []
    original_fit = LogisticRegression.fit

    def fit(self, X, y, *a, **kw):
        fit_calls.append((np.asarray(X).copy(), np.asarray(y).copy()))
        return original_fit(self, X, y, *a, **kw)

    monkeypatch.setattr(LogisticRegression, 'fit', fit)
    baseline = fit_task_b(**args)
    first_calls = list(fit_calls)
    fit_calls.clear()
    contaminate(labels).to_csv(args['consensus_path'], index=False)
    changed_args = dict(args, studies=contaminate(args['studies']),
                        train_pairs=contaminate(args['train_pairs']),
                        tuning_pairs=contaminate(args['tuning_pairs']))
    changed = fit_task_b(**changed_args)
    # R and prior see identical post-scaler values and targets, not just column names.
    assert len(first_calls) == len(fit_calls) == 2
    assert first_calls[0][0].shape[1] == len(RELIABILITY_FEATURES)
    assert first_calls[1][0].shape[1] == len(TASK_A_FEATURES)
    for (before_X, before_y), (after_X, after_y) in zip(first_calls, fit_calls):
        np.testing.assert_array_equal(before_X, after_X)
        np.testing.assert_array_equal(before_y, after_y)
    assert baseline.fit_summary_ == changed.fit_summary_
    heldout = labels.iloc[9:][['herb_id', 'drug_id']]
    pd.testing.assert_frame_equal(baseline.predict(heldout, args['studies']),
                                  changed.predict(contaminate(heldout), changed_args['studies']))


def test_task_b_agreement_targets_are_external_consensus_not_publication_aliases(task_b_tables, monkeypatch):
    args, labels, _ = task_b_tables
    studies = select_studies(args['studies'], args['train_pairs'])
    truth = labels.set_index(['herb_id', 'drug_id']).consensus_label.to_dict()
    expected = np.array([int((direction == 1) == (truth[(row.herb_id, row.drug_id)] == 'interaction'))
                         for row, direction in zip(studies.itertuples(), directions(studies))])
    captured = []
    original_fit = SupervisedStudyReliability.fit

    def fit(self, X, targets):
        assert tuple(X.columns) == RELIABILITY_FEATURES
        captured.append(np.asarray(targets).copy())
        return original_fit(self, X, targets)

    monkeypatch.setattr(SupervisedStudyReliability, 'fit', fit)
    # A publication alias that falsely declares all studies correct cannot define y_R.
    args['studies'] = contaminate(args['studies'])
    args['studies']['consensus_label'] = 'interaction'
    args['studies']['agreement'] = 1
    fit_task_b(**args)
    assert len(captured) == 1
    np.testing.assert_array_equal(captured[0], expected)
    assert set(expected) == {0, 1}


def test_task_b_heldout_targets_do_not_fit_or_predict(task_b_tables):
    args, labels, _ = task_b_tables
    heldout = labels.iloc[9:][['herb_id', 'drug_id']]
    baseline = fit_task_b(**args)
    before = baseline.predict(heldout, args['studies'])
    # Modify only assessment truth, leaving fitting/tuning labels untouched.
    labels.loc[9:, 'consensus_label'] = 'no_interaction'
    labels.to_csv(args['consensus_path'], index=False)
    after = fit_task_b(**args)
    assert baseline.fit_summary_ == after.fit_summary_
    np.testing.assert_array_equal(baseline.reliability_.pipeline_[-1].coef_,
                                  after.reliability_.pipeline_[-1].coef_)
    np.testing.assert_array_equal(baseline.prior_[-1].coef_, after.prior_[-1].coef_)
    # Prediction must not even read a target file; the only label file is removed.
    args['consensus_path'].unlink()
    pair_input = contaminate(heldout).assign(consensus_label='disputed')
    study_input = contaminate(args['studies']).assign(consensus_label='interaction')
    pd.testing.assert_frame_equal(before, after.predict(pair_input, study_input))


def test_task_b_finding_changes_only_permitted_evidence_at_inference(task_b_tables):
    args, labels, _ = task_b_tables
    model = fit_task_b(**args)
    heldout = labels.iloc[9:][['herb_id', 'drug_id']]
    studies = select_studies(args['studies'], heldout)
    swapped = studies.copy()
    swapped['study_finding'] = swapped.study_finding.map({'positive': 'negative', 'negative': 'positive'})
    before_X, after_X = reliability_features(studies), reliability_features(swapped)
    pd.testing.assert_frame_equal(before_X, after_X)
    np.testing.assert_array_equal(model.reliability_.predict(before_X), model.reliability_.predict(after_X))
    before, after = model.predict(heldout, studies), model.predict(heldout, swapped)
    np.testing.assert_array_equal(before.prior_logit, after.prior_logit)
    np.testing.assert_allclose(before.S_pos, after.S_neg)
    np.testing.assert_allclose(before.S_neg, after.S_pos)
    np.testing.assert_allclose(before.net_evidence, -after.net_evidence)
    np.testing.assert_allclose(before.contradiction, after.contradiction)
    assert (before.net_evidence != 0).any(), 'Fixture must exercise directional evidence'


@pytest.mark.parametrize('field', OUTCOME_FIELDS)
def test_reliability_rejects_outcome_columns_on_fit_and_prediction(task_b_tables, field):
    args, _, _ = task_b_tables
    X = reliability_features(args['studies'])
    y = np.tile([1, 0, 0], 12)
    model = SupervisedStudyReliability().fit(X, y)
    dirty = X.assign(**{field: y})
    with pytest.raises(ValueError, match='allowlist'):
        SupervisedStudyReliability().fit(dirty, y)
    with pytest.raises(ValueError, match='allowlist'):
        model.predict(dirty)
