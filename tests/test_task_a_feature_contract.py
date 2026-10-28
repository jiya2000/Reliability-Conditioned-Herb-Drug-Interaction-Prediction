"""Feature-boundary tests using synthetic measurements/targets, never scientific data."""
import numpy as np
import pandas as pd
import pytest
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import HistGradientBoostingClassifier

from src.data.mechanistic_curation import CURATION_COLUMNS, CANONICAL_TARGETS, UnverifiedCurationError
from src.features.task_a_contract import (
    TASK_A_FEATURES, TASK_A_TARGETS, TaskAFeatureContractError,
    build_task_a_features, task_a_training_pipeline, validate_task_a_features,
)
from src.models.feature_switched_models import FeatureSwitchedBenchmark

# Includes raw, already numeric, one-hot, vectorized and prefixed variants. An
# allowlist must reject unknown names, not rely on these blacklist spellings.
PROHIBITED = [
    "herb_id", "drug_id", "herb_name", "drug_name", "HERB_FIXTURE", "DRUG_FIXTURE",
    "herb_id_HERB_FIXTURE", "drug_id_DRUG_FIXTURE", "encoded_herb_id", "identity_hash",
    "original_text", "abstract", "tfidf_0", "embedding_0", "masked_text_0",
    "outcome", "study_finding", "consensus_label", "raw_label", "effect_direction",
    "clinical_significance", "S_pos", "S_neg", "net_evidence_margin",
    "raw_positive_count", "raw_negative_count", "N_raw_pos", "N_raw_neg",
    "contradiction_strength", "undiscounted_contradiction_strength",
    "species", "route", "preparation", "dose", "sample_size", "mean_source_quality",
    "study_id", "source_id", "citation_echo_ratio", "citation_cluster_size",
    "remainder__raw_positive_count", "numeric__outcome", "unrecognized_numeric_feature",
    "signed_effect_CYP2E1", "signed_effect_SLCO1B1", "signed_effect_NR1I2",
    "NTI", "rho_reach",  # Not yet available from an approved sourced builder.
]


@pytest.fixture
def curation():
    rows = []
    for target in CANONICAL_TARGETS:
        for entity in ["HERB_FIXTURE_A", "HERB_FIXTURE_B", "DRUG_FIXTURE"]:
            row = dict.fromkeys(CURATION_COLUMNS, "")
            row.update(entity=entity, target=target, status="measured",
                       source="SYNTHETIC_FIXTURE", citation="SYNTHETIC_FIXTURE")
            if entity.startswith("HERB_"):
                row["value_ki_uM"] = 2
            else:
                row["value_fm"] = 0.25
            rows.append(row)
    return pd.DataFrame(rows)


@pytest.fixture
def pairs():
    return pd.DataFrame({"herb_id": ["HERB_FIXTURE_A", "HERB_FIXTURE_B"] * 3,
                         "drug_id": ["DRUG_FIXTURE"] * 6}, index=[10, 20, 30, 40, 50, 60])


@pytest.fixture
def X(pairs, curation):
    return build_task_a_features(pairs, curation)


def fail_if_fit(*args, **kwargs):
    pytest.fail("Prohibited feature reached estimator.fit")


@pytest.mark.parametrize("column", PROHIBITED)
def test_prohibited_numeric_columns_rejected_before_scaling_or_training(X, column, monkeypatch):
    pipe = task_a_training_pipeline(LogisticRegression())
    monkeypatch.setattr(pipe.named_steps["scale"], "fit", fail_if_fit)
    monkeypatch.setattr(pipe.named_steps["model"], "fit", fail_if_fit)
    # Already numeric, so dtype-based selection/scaling could otherwise admit it.
    poisoned = X.assign(**{column: [0, 1, 0, 1, 0, 1]})
    with pytest.raises(TaskAFeatureContractError, match="unapproved"):
        pipe.fit(poisoned, [0, 1, 0, 1, 0, 1])


@pytest.mark.parametrize("estimator", [LogisticRegression, HistGradientBoostingClassifier])
def test_actual_estimator_receives_only_named_mechanistic_columns(X, estimator, monkeypatch):
    model = estimator()
    original_fit = model.fit
    seen = []

    def checked_fit(features, y, **kwargs):
        assert isinstance(features, pd.DataFrame)
        assert tuple(features.columns) == TASK_A_FEATURES
        assert features.shape == (6, 7)
        seen.append(features.copy())
        return original_fit(features, y, **kwargs)

    monkeypatch.setattr(model, "fit", checked_fit)
    pipe = task_a_training_pipeline(model)
    pipe.fit(X, [0, 1, 0, 1, 0, 1])
    assert len(seen) == 1
    assert pipe.predict_proba(X).shape == (6, 2)
    # Transform/prediction is checked too, not just initial fitting.
    with pytest.raises(TaskAFeatureContractError, match="unapproved"):
        pipe.predict_proba(X.assign(outcome=1))


@pytest.mark.parametrize("column", ["herb_id_HERB_FIXTURE", "tfidf_0", "raw_positive_count"])
def test_preprocessor_cannot_append_features_before_estimator_fit(X, column, monkeypatch):
    pipe = task_a_training_pipeline(LogisticRegression())
    original_transform = pipe.named_steps["scale"].fit_transform

    def poisoned_transform(features, y=None, **kwargs):
        return original_transform(features, y, **kwargs).assign(**{column: 1.0})

    monkeypatch.setattr(pipe.named_steps["scale"], "fit_transform", poisoned_transform)
    monkeypatch.setattr(pipe.named_steps["model"], "fit", fail_if_fit)
    with pytest.raises(TaskAFeatureContractError, match="unapproved"):
        pipe.fit(X, [0, 1, 0, 1, 0, 1])


def test_preprocessor_cannot_hide_schema_by_returning_an_array(X, monkeypatch):
    pipe = task_a_training_pipeline(LogisticRegression())
    monkeypatch.setattr(pipe.named_steps["scale"], "fit_transform", lambda features, y=None, **kw: features.to_numpy())
    monkeypatch.setattr(pipe.named_steps["model"], "fit", fail_if_fit)
    with pytest.raises(TaskAFeatureContractError, match="named"):
        pipe.fit(X, [0, 1, 0, 1, 0, 1])


def test_upstream_column_transformer_passthrough_does_not_bypass_contract(X):
    upstream = ColumnTransformer(
        [("mechanistic", "passthrough", list(TASK_A_FEATURES))],
        remainder="passthrough", verbose_feature_names_out=False,
    ).set_output(transform="pandas")
    encoded = upstream.fit_transform(X.assign(encoded_herb_id=1, positive_count=1))
    with pytest.raises(TaskAFeatureContractError, match="unapproved"):
        task_a_training_pipeline(LogisticRegression()).fit(encoded, [0, 1, 0, 1, 0, 1])


def test_builder_projects_only_pair_keys_and_recomputes_summary(pairs, curation, monkeypatch):
    import src.features.task_a_contract as contract
    original = contract.extract_curated_feature_matrix
    seen = []

    def checked_extract(raw_pairs, table):
        assert list(raw_pairs.columns) == ["herb_id", "drug_id"]
        seen.append(True)
        return original(raw_pairs, table)

    monkeypatch.setattr(contract, "extract_curated_feature_matrix", checked_extract)
    base = build_task_a_features(pairs, curation)
    poisoned = pairs.copy()
    for column in PROHIBITED:
        if column not in {"herb_id", "drug_id"}:
            poisoned[column] = 999
    # Even approved names on the raw study table cannot override sourced values.
    for column in TASK_A_FEATURES:
        poisoned[column] = 999
    actual = build_task_a_features(poisoned, curation)
    pd.testing.assert_frame_equal(actual, base)
    assert seen == [True, True]
    assert tuple(actual.columns) == TASK_A_FEATURES
    np.testing.assert_allclose(actual.iloc[:, :6], -0.25)
    # Nine-pathway extractor sum would be 2.25. Primary sum is six pathways only.
    np.testing.assert_allclose(actual["sum_abs_signed_effect"], 1.5)
    assert tuple(TASK_A_TARGETS) == ("CYP3A4", "CYP2C9", "CYP2C19", "CYP2D6", "CYP1A2", "ABCB1")


def test_key_names_are_lookup_only(pairs, curation):
    base = build_task_a_features(pairs, curation)
    renamed_pairs = pairs.replace({"HERB_FIXTURE_A": "HERB_OTHER_A", "HERB_FIXTURE_B": "HERB_OTHER_B",
                                   "DRUG_FIXTURE": "DRUG_OTHER"})
    renamed_table = curation.replace({"HERB_FIXTURE_A": "HERB_OTHER_A", "HERB_FIXTURE_B": "HERB_OTHER_B",
                                     "DRUG_FIXTURE": "DRUG_OTHER"})
    pd.testing.assert_frame_equal(base, build_task_a_features(renamed_pairs, renamed_table))


@pytest.mark.parametrize("kind", ["array", "duplicate", "missing", "string", "bool", "complex", "infinity", "nan"])
def test_schema_and_missingness_fail_closed(X, kind):
    bad = X.copy()
    if kind == "array": bad = bad.to_numpy()
    elif kind == "duplicate": bad = pd.concat([bad, bad.iloc[:, [0]]], axis=1)
    elif kind == "missing": bad = bad.drop(columns=TASK_A_FEATURES[0])
    elif kind == "string": bad[TASK_A_FEATURES[0]] = "0.2"
    elif kind == "bool": bad[TASK_A_FEATURES[0]] = True
    elif kind == "complex": bad[TASK_A_FEATURES[0]] = 1j
    elif kind == "infinity": bad[TASK_A_FEATURES[0]] = np.inf
    elif kind == "nan": bad[TASK_A_FEATURES[0]] = np.nan
    with pytest.raises(TaskAFeatureContractError):
        validate_task_a_features(bad)


def test_scaling_is_fit_on_training_only_and_column_order_is_canonical(X):
    train = X.iloc[:4].copy()
    train.iloc[:, 0] = [-1, -0.5, 0.5, 1]
    test = X.iloc[4:].copy()
    test.iloc[:, 0] = 999  # Fixture perturbation, never scientific data.
    pipe = task_a_training_pipeline(LogisticRegression()).fit(train, [0, 1, 0, 1])
    before = pipe.named_steps["scale"].mean_.copy()
    normal = pipe.predict_proba(test)
    np.testing.assert_allclose(normal, pipe.predict_proba(test.loc[:, list(reversed(TASK_A_FEATURES))]))
    np.testing.assert_array_equal(pipe.named_steps["scale"].mean_, before)
    np.testing.assert_allclose(before, train.mean().to_numpy())


def test_real_unverified_mechanisms_still_blocked(pairs):
    with pytest.raises(UnverifiedCurationError):
        build_task_a_features(pairs)


def test_missingness_preserved_during_build_but_cannot_enter_training(pairs, curation):
    table = curation[curation.target != "CYP3A4"]
    features = build_task_a_features(pairs, table)
    assert features["signed_effect_CYP3A4"].isna().all()
    assert features["sum_abs_signed_effect"].isna().all()
    with pytest.raises(TaskAFeatureContractError, match="missing mechanistic"):
        task_a_training_pipeline(LogisticRegression()).fit(features, [0, 1, 0, 1, 0, 1])


def test_curated_runner_uses_contract_and_cannot_fall_back_to_study_targets(pairs, curation, monkeypatch):
    import src.models.feature_switched_models as switched
    monkeypatch.setattr(switched, "load_mechanistic_curation", lambda path: curation)
    benchmark = FeatureSwitchedBenchmark(feature_set="new")
    features, status = benchmark.get_feature_matrix(pairs.assign(outcome="interaction_present"))
    assert tuple(features.columns) == TASK_A_FEATURES
    assert status == "curated_mechanistic"
    monkeypatch.setattr(switched.LogisticRegression, "fit", fail_if_fit)
    monkeypatch.setattr(switched.HistGradientBoostingClassifier, "fit", fail_if_fit)
    monkeypatch.setattr(switched.SmallGATClassifier, "__init__", fail_if_fit)
    # Deliberately omit outcome: the guard must trigger before any legacy target access.
    with pytest.raises(NotImplementedError, match="consensus-supervised"):
        benchmark.run_cv(pairs)
