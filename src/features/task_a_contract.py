"""Fail-closed feature boundary for primary Task A; not a new model/trainer.

Only the six pathways explicitly specified in PROJECT_SPEC section 4.1 are
currently approved. IDs may be used upstream to join curation rows, never as X.
See docs/task_a_feature_contract.md for provenance and unresolved prerequisites.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from pandas.api.types import is_numeric_dtype, is_bool_dtype, is_complex_dtype
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import HistGradientBoostingClassifier

from src.data.mechanistic_curation import CANONICAL_TARGETS, extract_curated_feature_matrix

# Keep this explicit: adding a target to the general extractor must not silently
# expand the primary model's inputs. Signs describe target activity, not exposure.
TASK_A_TARGETS = ("CYP3A4", "CYP2C9", "CYP2C19", "CYP2D6", "CYP1A2", "ABCB1")
TASK_A_FEATURES = tuple(f"signed_effect_{t}" for t in TASK_A_TARGETS) + ("sum_abs_signed_effect",)


class TaskAFeatureContractError(ValueError):
    """Primary Task A received unapproved or untraceable feature columns."""


def validate_task_a_features(X: pd.DataFrame, *, allow_missing: bool = False) -> pd.DataFrame:
    """Reject extras (including already encoded/scaled extras), not silently drop.

    Positional arrays/sparse matrices are intentionally rejected: width alone
    cannot establish which raw columns entered a preprocessor. A caller can lie
    by renaming contaminated values; names cannot authenticate scientific origin.
    The trusted builder below is the supported raw-data route.
    """
    if not isinstance(X, pd.DataFrame):
        raise TaskAFeatureContractError("Task A requires a named pandas DataFrame, not positional features")
    if not X.columns.is_unique:
        raise TaskAFeatureContractError("Duplicate Task A feature columns")
    missing = [c for c in TASK_A_FEATURES if c not in X.columns]
    extra = [c for c in X.columns if c not in TASK_A_FEATURES]
    if missing or extra:
        raise TaskAFeatureContractError(f"Task A feature allowlist violation: missing={missing}, unapproved={extra}")
    for c in TASK_A_FEATURES:
        dtype = X[c].dtype
        if not is_numeric_dtype(dtype) or is_bool_dtype(dtype) or is_complex_dtype(dtype):
            raise TaskAFeatureContractError(f"Task A feature {c} must contain real numeric measurements")
    frame = X.loc[:, list(TASK_A_FEATURES)].copy()
    values = frame.to_numpy(dtype=float, na_value=np.nan)
    if np.isinf(values).any():
        raise TaskAFeatureContractError("Task A features contain infinite values")
    if not allow_missing and np.isnan(values).any():
        raise TaskAFeatureContractError("Task A training blocked: missing mechanistic values; no imputation policy approved")
    return frame


def build_task_a_features(pairs: pd.DataFrame, curation: pd.DataFrame | None = None) -> pd.DataFrame:
    """Project only join keys into the provenance-checked extractor.

    Discard all study/text/design/lineage columns before extraction, including
    any caller-supplied column with an approved output name. Recompute the sum
    from the six selected pathways, never reuse the general nine-pathway sum.
    Unknown mechanisms stay NaN at extraction; training rejects them explicitly.
    """
    if not isinstance(pairs, pd.DataFrame) or not pairs.columns.is_unique:
        raise TaskAFeatureContractError("Pair input requires unique named columns")
    if not {"herb_id", "drug_id"}.issubset(pairs.columns):
        raise TaskAFeatureContractError("Pair input requires herb_id and drug_id join keys")
    matrix = extract_curated_feature_matrix(pairs.loc[:, ["herb_id", "drug_id"]].copy(), curation)
    if matrix.shape != (len(pairs), len(CANONICAL_TARGETS) + 1):
        raise TaskAFeatureContractError("Unexpected curated extractor output schema")
    effects = matrix[:, [CANONICAL_TARGETS.index(t) for t in TASK_A_TARGETS]]
    values = np.column_stack([effects, np.abs(effects).sum(axis=1)])
    frame = pd.DataFrame(values, index=pairs.index, columns=TASK_A_FEATURES)
    return validate_task_a_features(frame, allow_missing=True)


class TaskAFeatureContract(TransformerMixin, BaseEstimator):
    """Validate both fit and transform, and retain names in canonical order."""

    def fit(self, X, y=None):
        validate_task_a_features(X)
        self.feature_names_in_ = np.array(TASK_A_FEATURES, dtype=object)
        self.n_features_in_ = len(TASK_A_FEATURES)
        return self

    def transform(self, X):
        return validate_task_a_features(X)

    def get_feature_names_out(self, input_features=None):
        if input_features is not None and tuple(input_features) != TASK_A_FEATURES:
            raise TaskAFeatureContractError("Unexpected preprocessor feature names")
        return np.array(TASK_A_FEATURES, dtype=object)


def task_a_training_pipeline(estimator) -> Pipeline:
    """Guard the existing LR/HGB estimators before AND after fixed preprocessing.

    No column discovery, categorical encoding, text vectorization, passthrough,
    feature selection by dtype, or missingness indicators. Scaling fits only on
    the partition passed to Pipeline.fit. This factory does not load labels or
    implement consensus training/CV, which remains blocked in the old runner.
    """
    if type(estimator) not in (LogisticRegression, HistGradientBoostingClassifier):
        raise TaskAFeatureContractError("Primary Task A supports only the existing LR/HGB estimators")
    return Pipeline([
        ("input_contract", TaskAFeatureContract()),
        ("scale", StandardScaler().set_output(transform="pandas")),
        ("training_contract", TaskAFeatureContract()),
        ("model", estimator),
    ])
