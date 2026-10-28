"""Supervised Task B fitting infrastructure, without calibration/performance claims.

Call fit_task_b with explicitly separated train_pairs and tuning_pairs (DataFrames
of herb_id/drug_id), audited study rows, and external consensus/curation paths.
Prediction takes no consensus labels. Scientific source independence, primary/echo
annotations and design extraction still require human verification.
"""
from __future__ import annotations

from types import SimpleNamespace
import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.special import expit
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from src.data.evidence_schema import (
    extract_study_design_fields, extract_study_finding, StudyFinding,
    load_consensus_labels_table,
)
from src.data.evidence_lineage import EvidenceLineageAnalyzer
from src.data.mechanistic_curation import load_mechanistic_curation
from src.features.task_a_contract import build_task_a_features, task_a_training_pipeline, validate_task_a_features
from src.models.aggregation import ReliabilityTarget, ResidualEvidenceAggregator

KEYS = ['herb_id', 'drug_id']
DESIGN_CATEGORIES = {
    'species': ('human', 'rat', 'mouse', 'animal', 'in_vitro', 'unknown'),
    'assay': ('rct', 'clinical', 'observational', 'case_report', 'animal', 'in_vitro',
              'mechanistic', 'in_silico', 'meta_analysis', 'inconclusive', 'unknown'),
    'route': ('oral', 'iv', 'unknown'),
    'preparation': ('standardized_extract', 'whole_crude', 'ethanolic_extract',
                    'isolated_constituent', 'unknown'),
}
RELIABILITY_FEATURES = tuple(f'{field}_{value}' for field, values in DESIGN_CATEGORIES.items()
                             for value in values) + (
    'dose_reported', 'log_sample_size', 'sample_size_missing', 'is_primary', 'log_cluster_size')
FINDINGS = {'interaction_present', 'positive', 'no_interaction_observed', 'negative',
            'inconclusive', 'unknown', 'conflicting', 'null'}


def pair_keys(pairs):
    if not isinstance(pairs, pd.DataFrame) or not pairs.columns.is_unique or not set(KEYS) <= set(pairs):
        raise ValueError('Pairs require unique herb_id/drug_id columns')
    for field, prefix in zip(KEYS, ['HERB_', 'DRUG_']):
        if not pairs[field].map(lambda v: isinstance(v, str) and v == v.strip()
                               and v.startswith(prefix)).all():
            raise ValueError(f'Invalid {field}')
    keys = list(pairs[KEYS].itertuples(index=False, name=None))
    if len(set(keys)) != len(keys):
        raise ValueError('Duplicate pairs')
    return keys


def validate_studies(studies):
    required = set(KEYS + ['source_id', 'study_id', 'cited_study_ids', 'study_finding', 'is_primary'])
    if not isinstance(studies, pd.DataFrame) or not studies.columns.is_unique or not required <= set(studies):
        raise ValueError(f'Studies require unique columns including {sorted(required)}')
    pair_keys(studies[KEYS].drop_duplicates())
    for field in ['source_id', 'study_id']:
        if not studies[field].map(lambda v: isinstance(v, str) and bool(v.strip()) and v == v.strip()).all():
            raise ValueError(f'Missing or noncanonical {field}')
    if studies.duplicated(KEYS + ['source_id']).any():
        raise ValueError('Duplicate publication rows for pair')
    if not studies.cited_study_ids.map(lambda v: isinstance(v, list) and all(
            isinstance(s, str) and bool(s.strip()) and s == s.strip() for s in v)).all():
        raise ValueError('cited_study_ids must be parsed lists of nonempty study IDs')
    if not studies.study_finding.isin(FINDINGS).all():
        raise ValueError('Unknown study_finding; explicitly annotate inconclusive evidence')
    if not studies.is_primary.isin([0, 1]).all() or studies.is_primary.isna().any():
        raise ValueError('is_primary must be explicitly annotated as 0/1')


def validate_study_sources(studies):
    """Keep the existing publication registry gate; no source/outcome data becomes X."""
    from src.data.source_registry import DEFAULT_SOURCE_REGISTRY
    if 'source_url_or_doi' not in studies:
        raise ValueError('Task B studies require source_url_or_doi for strict traceability')
    for row in studies.itertuples():
        if not isinstance(row.source_url_or_doi, str) or not row.source_url_or_doi.strip():
            raise ValueError('Task B publication requires a nonempty source_url_or_doi')
        valid, reason = DEFAULT_SOURCE_REGISTRY.verify_reference(
            row.source_id, row.source_url_or_doi, require_registered=True)
        if not valid:
            raise ValueError(f'Unverified Task B publication: {reason}')


def lineage_components(studies):
    """Reuse the citation graph, also connect repeated source IDs across pairs."""
    records = [SimpleNamespace(study_id=r.study_id, source_id=r.source_id,
               cited_study_ids=r.cited_study_ids, outcome=None)
               for r in studies.itertuples()]
    graph = EvidenceLineageAnalyzer().build_lineage_graph(records).to_undirected()
    source_studies = {}
    for record in records:
        previous = source_studies.setdefault(record.source_id, record.study_id)
        graph.add_edge(previous, record.study_id)
    import networkx as nx
    return list(nx.connected_components(graph))


def assert_disjoint(left_pairs, right_pairs, studies):
    left, right = set(pair_keys(left_pairs)), set(pair_keys(right_pairs))
    if left & right:
        raise ValueError('Task B pair overlap between fitting/tuning/assessment partitions')
    membership = list(studies[KEYS].itertuples(index=False, name=None))
    left_ids = set(studies.loc[[k in left for k in membership], 'study_id'])
    right_ids = set(studies.loc[[k in right for k in membership], 'study_id'])
    for component in lineage_components(studies):
        if component & left_ids and component & right_ids:
            raise ValueError('Task B citation/source cluster overlap between partitions')


def select_studies(studies, pairs):
    keys = set(pair_keys(pairs))
    return studies.loc[[k in keys for k in studies[KEYS].itertuples(index=False, name=None)]].reset_index(drop=True)


def reliability_features(studies):
    """Fixed design vocabulary and structural lineage only; no free text encoding.

    Missing categories have explicit unknown encodings; missing sample size has
    a separate indicator. No outcome-based quality, PK defaults or raw text.
    Unsupported category spellings fail rather than discovering categories.
    """
    validate_studies(studies)
    components = lineage_components(studies)
    sizes = {study: len(set(studies.loc[studies.study_id.isin(c), 'source_id']))
             for c in components for study in c}
    rows = []
    for _, row in studies.iterrows():
        raw = row.to_dict()
        # Override the legacy helper's oral/standardized defaults for absent fields.
        for field in ['species', 'evidence_type', 'route', 'preparation']:
            raw.setdefault(field, None)
        design = extract_study_design_fields(raw)
        values = []
        for field, categories in DESIGN_CATEGORIES.items():
            value = design[field] or 'unknown'
            if value not in categories:
                raise ValueError(f'Unsupported {field}: {value}; normalize audited metadata first')
            values.extend(float(value == c) for c in categories)
        size = row.get('sample_size', np.nan)
        missing = pd.isna(size)
        if not missing and (isinstance(size, (str, bool)) or not np.isfinite(size) or size <= 0):
            raise ValueError('sample_size must be a positive numeric value or missing')
        values.extend([float(design['dose'] is not None), 0.0 if missing else np.log1p(size),
                       float(missing), float(row.is_primary), np.log1p(sizes[row.study_id])])
        rows.append(values)
    return pd.DataFrame(rows, columns=RELIABILITY_FEATURES, dtype=float)


def directions(studies):
    mapping = {StudyFinding.POSITIVE: 1, StudyFinding.NEGATIVE: -1, StudyFinding.NULL: 0}
    return np.array([mapping[extract_study_finding(v)] for v in studies.study_finding], dtype=int)


class SupervisedStudyReliability:
    """Regularized logistic BCE agreement model; X must be named design/lineage."""
    @staticmethod
    def validate(X):
        if not isinstance(X, pd.DataFrame) or tuple(X.columns) != RELIABILITY_FEATURES:
            raise ValueError('Reliability feature allowlist violation')
        if not all(pd.api.types.is_numeric_dtype(t) for t in X.dtypes) or not np.isfinite(X.to_numpy(dtype=float)).all():
            raise ValueError('Reliability features must be finite numeric values')

    def fit(self, X, targets):
        self.validate(X)
        y = np.asarray(targets)
        if y.shape != (len(X),) or not np.isin(y, [0, 1]).all() or len(np.unique(y)) != 2:
            raise ValueError('Reliability fitting requires both agreement classes')
        self.pipeline_ = make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000, random_state=42))
        self.pipeline_.fit(X, y)
        return self

    def predict(self, X):
        self.validate(X)
        return self.pipeline_.predict_proba(X)[:, 1] if len(X) else np.empty(0)


def aggregate_pairs(pairs, prior, studies, reliability, aggregator):
    keys = list(studies[KEYS].itertuples(index=False, name=None))
    o = directions(studies)
    result = []
    for key, logit in zip(pair_keys(pairs), prior):
        mask = np.array([k == key for k in keys], dtype=bool)
        info = aggregator.aggregate(float(logit), o[mask], reliability[mask],
                                    studies.loc[mask, 'is_primary'].to_numpy())
        result.append(dict(zip(KEYS, key), **info, probability=float(expit(info['logit']))))
    return pd.DataFrame(result)


class FittedTaskB:
    """Predict held-out pairs without reading targets; reject known pair/cluster reuse."""
    def predict(self, pairs, studies):
        pair_keys(pairs)
        validate_studies(studies)
        validate_study_sources(studies)
        # Retain intermediate lineage nodes when checking overlap, but never use
        # unrelated studies to compute the assessment feature matrix.
        combined = pd.concat([self.fit_lineage_, studies], ignore_index=True)
        assert_disjoint(self.fit_pairs_, pairs, combined)
        studies = select_studies(studies, pairs)
        X = validate_task_a_features(build_task_a_features(pairs, self.curation_))
        prior = self.prior_.decision_function(X)
        r = self.reliability_.predict(reliability_features(studies))
        return aggregate_pairs(pairs, prior, studies, r, self.aggregator_)


def fit_task_b(*, studies, train_pairs, tuning_pairs,
               consensus_path='data/benchmark/consensus_labels.csv',
               curation_path='data/curation/mechanistic_features.csv', kappa=1.0):
    """Fit prior/R on train only; learn alpha/lambda on separate tuning pairs.

    Disputed pairs are excluded from both optimization stages, never coerced to
    negative. Inconclusive studies receive no agreement target or polarity mass.
    A held-out evaluation partition is deliberately absent from this fit API.
    """
    labels, _ = load_consensus_labels_table(consensus_path, require_labels=True)
    train_keys, tune_keys = pair_keys(train_pairs), pair_keys(tuning_pairs)
    lookup = labels.set_index(KEYS).consensus_label.to_dict()
    if not set(train_keys + tune_keys) <= set(lookup):
        raise ValueError('Every fitting/tuning pair requires independent consensus')
    validate_studies(studies)
    assert_disjoint(train_pairs, tuning_pairs, studies)
    # All requested pairs, including disputed ones, require verified mechanisms.
    curation = load_mechanistic_curation(curation_path)
    all_pairs = pd.concat([train_pairs[KEYS], tuning_pairs[KEYS]], ignore_index=True)
    validate_task_a_features(build_task_a_features(all_pairs, curation))
    validate_study_sources(studies)
    train = train_pairs.loc[[lookup[k] != 'disputed' for k in train_keys], KEYS].reset_index(drop=True)
    tune = tuning_pairs.loc[[lookup[k] != 'disputed' for k in tune_keys], KEYS].reset_index(drop=True)
    y_train = np.array([int(lookup[k] == 'interaction') for k in pair_keys(train)])
    y_tune = np.array([int(lookup[k] == 'interaction') for k in pair_keys(tune)])
    if len(np.unique(y_train)) != 2 or len(np.unique(y_tune)) != 2:
        raise ValueError('Binary train and tuning partitions each require both consensus classes')
    train_studies, tune_studies = select_studies(studies, train), select_studies(studies, tune)
    # Compute reliability targets with the existing agreement-target implementation.
    o = directions(train_studies)
    study_y = np.array([int(lookup[k] == 'interaction') for k in
                        train_studies[KEYS].itertuples(index=False, name=None)])
    targets, valid = ReliabilityTarget(enforce_validated_labels=False).compute_target(o, study_y)
    train_features = reliability_features(train_studies)
    tune_features = reliability_features(tune_studies)
    # Validate kappa and all inputs before fitting.
    ResidualEvidenceAggregator(kappa=kappa)
    model = FittedTaskB()
    model.reliability_ = SupervisedStudyReliability().fit(train_features.loc[valid], targets)
    model.prior_ = task_a_training_pipeline(LogisticRegression(max_iter=1000, random_state=42))
    model.prior_.fit(build_task_a_features(train, curation), y_train)
    prior = model.prior_.decision_function(build_task_a_features(tune, curation))
    r = model.reliability_.predict(tune_features)

    def loss(parameters):
        aggregator = ResidualEvidenceAggregator(alpha=parameters[0], evidence_scale=parameters[1], kappa=kappa)
        logits = aggregate_pairs(tune, prior, tune_studies, r, aggregator).logit.to_numpy()
        return float(np.mean(np.logaddexp(0, logits) - y_tune * logits))

    # Bounds implement alpha in [0,1], lambda >= 0; kappa is fixed, not label-tuned.
    optimum = minimize(loss, [0.15, 1.0], method='L-BFGS-B', bounds=[(0, 1), (0, None)])
    if not optimum.success or not np.isfinite(optimum.fun):
        raise RuntimeError(f'Task B residual optimization failed: {optimum.message}')
    model.aggregator_ = ResidualEvidenceAggregator(optimum.x[0], optimum.x[1], kappa)
    model.curation_ = curation.copy()
    model.fit_pairs_ = all_pairs.copy()
    # Topology only: preserve transitive links without retaining held-out findings.
    model.fit_lineage_ = studies[KEYS + ['source_id', 'study_id', 'cited_study_ids']].copy(deep=True)
    model.fit_summary_ = dict(train_pairs=len(train), tuning_pairs=len(tune),
        disputed_pairs_excluded=len(all_pairs) - len(train) - len(tune),
        agreement_studies=int(valid.sum()), inconclusive_train_studies=int((~valid).sum()),
        alpha=float(optimum.x[0]), evidence_scale=float(optimum.x[1]), kappa=kappa,
        tuning_echo_studies=int(tune_studies.is_primary.eq(0).sum()),
        tuning_has_directional_evidence_and_echoes=bool(len(tune_studies) and
            (directions(tune_studies) != 0).any() and tune_studies.is_primary.eq(0).any()))
    return model
