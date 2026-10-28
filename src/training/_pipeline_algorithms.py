"""Private algorithms for synthetic component tests, never production execution.

All inputs are captured bytes or explicit role partitions. No legacy evaluation
runner, filesystem discovery, metric fallback, or coverage certification is used.
"""
from __future__ import annotations

from dataclasses import dataclass, replace
import hashlib
import math
import pickle
from weakref import WeakKeyDictionary

import numpy as np
import pandas as pd
from scipy.optimize import minimize_scalar
from scipy.special import expit, logit
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (average_precision_score, precision_recall_curve,
                             roc_auc_score, auc)

from src.data.evidence_schema import validate_consensus_labels_table
from src.data.mechanistic_curation import validate_curation_dataframe, is_curation_verified
from src.features.task_a_contract import TASK_A_FEATURES, build_task_a_features, task_a_training_pipeline, validate_task_a_features
from src.training import release_preflight as release
from src.training.evaluation_protocol import validate_partitions, canonical_manifest_json
from src.training import task_b

KEYS = ['herb_id', 'drug_id']
STATES = {'interaction', 'no_interaction', 'disputed'}


def digest(value):
    return hashlib.sha256(pickle.dumps(value, protocol=5)).hexdigest()


def probabilities(values, n):
    raw = np.asarray(values)
    if raw.dtype.kind not in 'fiu' or raw.shape != (n,):
        raise ValueError('Probabilities require an aligned real numeric vector')
    p = raw.astype(float)
    if not np.isfinite(p).all() or ((p < 0) | (p > 1)).any():
        raise ValueError('Probabilities must be finite and in [0,1]')
    return p


def binary_targets(labels, *, both=False):
    labels = np.asarray(labels)
    if labels.ndim != 1 or not set(labels.tolist()) <= STATES:
        raise ValueError('Noncanonical consensus label vocabulary; unresolved states cannot become negatives')
    mask = labels != 'disputed'
    y = (labels[mask] == 'interaction').astype(int)
    if both and len(np.unique(y)) != 2:
        raise ValueError('Both binary consensus classes required')
    return y, mask


@dataclass(frozen=True)
class SnapshotTables:
    """Exact captured bytes; parsing never reopens source paths."""
    consensus: bytes
    mechanistic: bytes
    manifest: bytes

    @classmethod
    def from_snapshots(cls, snapshots):
        """Consume the orchestrator/preflight mapping without accepting paths."""
        if set(snapshots) != {'consensus', 'mechanistic', 'manifest'}:
            raise ValueError('Require exactly consensus, mechanistic, manifest snapshots')
        return cls(**snapshots)

    def __post_init__(self):
        for value in (self.consensus, self.mechanistic, self.manifest):
            if not isinstance(value, bytes) or len(value) > release.MAX_INPUT_BYTES:
                raise ValueError('Require bounded immutable input snapshots')

    @property
    def hashes(self):
        return {k: hashlib.sha256(getattr(self, k)).hexdigest()
                for k in ('consensus', 'mechanistic', 'manifest')}

    def tables(self):
        labels = release._csv_snapshot(self.consensus)
        valid, errors = validate_consensus_labels_table(labels)
        if not valid or labels.empty:
            raise ValueError('Populated valid consensus required: ' + '; '.join(errors))
        binary_targets(labels.consensus_label)
        # Strict identity validation, without the legacy loader's normalization.
        task_b.pair_keys(labels)
        curation = release._csv_snapshot(self.mechanistic)
        validate_curation_dataframe(curation)
        if not is_curation_verified(curation)[0]:
            raise ValueError('Verified mechanistic records required')
        manifest = release._manifest_snapshot(self.manifest)
        if not isinstance(manifest, dict) or manifest.get('input_sha256') != {
                k: self.hashes[k] for k in ('consensus', 'mechanistic')}:
            raise ValueError('Manifest must bind exact consensus/mechanistic snapshots')
        return labels, curation, manifest


class RoleData:
    """Freeze assignments and input bytes; expose only a requested role's targets.

    This is a component boundary, not an eligibility token. Assessment targets
    are accessed only by the final scorer. Structural validation ignores labels.
    """
    def __init__(self, snapshots, *, assignments, partition_roles, regime,
                 protocol, task='A', topology=None):
        if type(snapshots) is not SnapshotTables:
            raise ValueError('Require exact immutable SnapshotTables, not caller table providers')
        self._snapshots = snapshots
        self._snapshot_identity = snapshots
        labels, curation, _ = snapshots.tables()
        plan = release._manifest_snapshot(canonical_manifest_json(dict(
            assignments=assignments, partition_roles=partition_roles,
            topology=topology)).encode())
        self.manifest = validate_partitions(labels[KEYS].to_dict('records'),
            plan['assignments'], partition_roles=plan['partition_roles'],
            regime=regime, protocol=protocol, task=task, topology=plan['topology'])
        self._labels = labels.copy(deep=True)
        self._curation = curation.copy(deep=True)
        self._state = digest((self.snapshots, self.manifest, self._labels, self._curation))

    @property
    def snapshots(self):
        return self._snapshots

    def check(self):
        if (self._snapshots is not self._snapshot_identity or
                digest((self.snapshots, self.manifest, self._labels, self._curation)) != self._state):
            raise ValueError('Role data changed after validation')

    def rows(self, role, *, targets=False):
        self.check()
        partitions = [p for p, r in self.manifest['partition_roles'].items() if r == role]
        if len(partitions) != 1:
            raise ValueError('Requested role is absent or ambiguous: ' + role)
        keys = [(r['herb_id'], r['drug_id']) for r in self.manifest['assignments']
                if r['partition'] == partitions[0]]
        frame = self._labels.set_index(KEYS).loc[keys].reset_index()
        return frame[KEYS + (['consensus_label'] if targets else [])].copy()

    def features(self, pairs):
        self.check()
        return validate_task_a_features(build_task_a_features(pairs[KEYS], self._curation))


@dataclass(frozen=True)
class _FitProvenance:
    task: str
    fit_role_hash: str
    configuration: str
    feature_contract: tuple
    plan_hash: str
    snapshot_hashes: tuple
    derived_tables_hash: str
    estimator_hash: str


# Process-local registration, not a caller-supplied fit-record field. Public APIs
# cannot issue/register predictors. These records certify private test fitting
# only and MUST NOT be interpreted as production release authorization.
_FITTED = WeakKeyDictionary()


def _fit_record(data, estimator, configuration):
    data.check()
    rows = data.rows('fit', targets=True)
    _, mask = binary_targets(rows.consensus_label, both=True)
    return _FitProvenance(task=data.manifest['task'],
        fit_role_hash=digest((rows, data.features(rows.loc[mask]))),
        configuration=configuration, feature_contract=TASK_A_FEATURES,
        plan_hash=hashlib.sha256(canonical_manifest_json(data.manifest).encode()).hexdigest(),
        snapshot_hashes=tuple(sorted(data.snapshots.hashes.items())),
        derived_tables_hash=digest((data._labels, data._curation)),
        estimator_hash=digest(estimator))


class FixedPredictor:
    """Private factory-issued predictor; arbitrary fitted wrappers are rejected."""
    def __init__(self, *args, **kwargs):
        raise ValueError('Missing trusted fit provenance: use the internal fit algorithm')

    @property
    def provenance(self):
        record = _FITTED.get(self)
        if record is None:
            raise ValueError('Missing trusted fit provenance')
        # Do not expose the registered record by reference, even though frozen.
        return replace(record[2])

    @property
    def identity(self):
        return digest(self.provenance)

    def check(self):
        record = _FITTED.get(self)
        if record is None:
            raise ValueError('Missing trusted fit provenance')
        data, estimator, provenance = record
        if self._data is not data or self._estimator is not estimator:
            raise ValueError('Fitted predictor binding changed')
        self._data.check()
        if _fit_record(data, estimator, provenance.configuration) != provenance:
            raise ValueError('Fitted predictor or fit provenance changed after freezing')

    def predict(self, pairs):
        self.check()
        task_b.pair_keys(pairs)
        p = self._estimator.predict_proba(self._data.features(pairs))[:, 1]
        return pairs[KEYS].assign(probability=probabilities(p, len(pairs)))


def fit_task_a(data, *, family='lr', parameters=None):
    """Fit preprocessing and estimator on fit only; explicit fixed parameters."""
    if data.manifest['task'] != 'A' or family not in {'lr', 'hgb'}:
        raise ValueError('Task A requires LR or HGB')
    parameters = {} if parameters is None else dict(parameters)
    allowed = {'C'} if family == 'lr' else {'learning_rate', 'max_leaf_nodes', 'l2_regularization'}
    if set(parameters) - allowed:
        raise ValueError('Unsupported model parameter')
    rows = data.rows('fit', targets=True)
    y, mask = binary_targets(rows.consensus_label, both=True)
    estimator = (LogisticRegression(max_iter=1000, random_state=42, **parameters)
                 if family == 'lr' else HistGradientBoostingClassifier(random_state=42, **parameters))
    pipeline = task_a_training_pipeline(estimator)
    pipeline.fit(data.features(rows.loc[mask]), y)
    # Registration occurs only after successful fit on the verified fit-role rows.
    predictor = object.__new__(FixedPredictor)
    predictor._estimator, predictor._data = pipeline, data
    configuration = canonical_manifest_json(dict(family=family, parameters=parameters))
    _FITTED[predictor] = (data, pipeline, _fit_record(data, pipeline, configuration))
    return predictor


def tune_task_a(data, *, family, candidates):
    """Select by tuning BCE; never refit on tuning or inspect assessment labels."""
    if not isinstance(candidates, (list, tuple)) or not candidates:
        raise ValueError('Supply an explicit nonempty candidate list')
    rows = data.rows('tuning', targets=True)
    y, mask = binary_targets(rows.consensus_label, both=True)
    best = None
    for parameters in candidates:
        model = fit_task_a(data, family=family, parameters=parameters)
        p = model.predict(rows[KEYS]).probability.to_numpy()[mask]
        loss = binary_loss(y, p)
        if best is None or loss < best[0]:
            best = (loss, model)
    return best[1]


def binary_loss(y, p):
    y = np.asarray(y)
    if y.ndim != 1 or not len(y) or not np.isin(y, [0, 1]).all():
        raise ValueError('Binary loss requires nonempty binary targets')
    # Clipping is a declared numeric log-loss convention, never a probability repair.
    p = np.clip(probabilities(p, len(y)), np.finfo(float).eps, 1 - np.finfo(float).eps)
    return float(-np.mean(y * np.log(p) + (1 - y) * np.log1p(-p)))


def strict_score(targets, predictions):
    """Exact ordered identity join; disputed retained outside binary metrics."""
    left, right = task_b.pair_keys(targets), task_b.pair_keys(predictions)
    if 'consensus_label' not in targets or 'probability' not in predictions:
        raise ValueError('Scoring requires consensus_label and probability columns')
    if left != right:
        raise ValueError('Prediction/target identities or ordering differ')
    y, mask = binary_targets(targets.consensus_label)
    p = probabilities(predictions.probability, len(targets))[mask]
    metrics = {}
    def record(name, value=None, reason=None):
        metrics[name] = dict(value=value, reason=reason)
    if not len(y):
        for name in ('roc_auc', 'average_precision', 'trapezoidal_pr_auc', 'brier', 'log_loss'):
            record(name, reason='No binary consensus targets')
    else:
        record('brier', float(np.mean((p - y) ** 2)))
        record('log_loss', binary_loss(y, p))
        if len(np.unique(y)) != 2:
            for name in ('roc_auc', 'average_precision', 'trapezoidal_pr_auc'):
                record(name, reason='Both binary classes required by scoring policy')
        else:
            score = float(roc_auc_score(y, p))
            if score > .98:
                raise ValueError(f'Task A AUC integrity alert: AUC={score}')
            record('roc_auc', score)
            record('average_precision', float(average_precision_score(y, p)))
            precision, recall, _ = precision_recall_curve(y, p)
            record('trapezoidal_pr_auc', float(auc(recall, precision)))
    return dict(metrics=metrics, binary_pairs=len(y), disputed_pairs=int((~mask).sum()),
                intervals=dict(status='unavailable', reason='No justified inference design'),
                significance=dict(status='unavailable', reason='No justified inference design'))


def role_predictions(data, predictor, role, *, calibrator=None):
    predictor.check()
    if predictor._data is not data:
        raise ValueError('Predictor and roles must share the same validated input context')
    predictor.check()
    frame = predictor.predict(data.rows(role))
    if calibrator is not None:
        frame['probability'] = calibrator.transform(frame.probability, predictor)
    return frame


class ProbabilityCalibration:
    """Temperature scaling with explicit failure; no identity fallback."""
    def __init__(self):
        self.status = 'unfitted'

    def fit(self, data, predictor):
        self.status = 'failed'
        self.reason = 'Calibration did not complete'
        try:
            predictor.check()
            rows = data.rows('probability_calibration', targets=True)
            y, mask = binary_targets(rows.consensus_label, both=True)
            p = role_predictions(data, predictor, 'probability_calibration').probability.to_numpy()[mask]
            # Endpoint logits cannot be temperature-scaled without a new policy.
            if ((p <= 0) | (p >= 1)).any():
                raise ValueError('Temperature calibration requires strictly interior probabilities')
            z = logit(p)
            def objective(t):
                logits = z / np.exp(t)
                return float(np.mean(np.logaddexp(0, logits) - y * logits))
            # Numerical bounds are optimizer limits, not scientific thresholds.
            optimum = minimize_scalar(objective, bounds=(-10., 10.), method='bounded')
            if (not optimum.success or not np.isfinite(optimum.fun) or not np.isfinite(optimum.x)
                    or abs(optimum.x) >= 9.99):
                raise ValueError('Temperature optimizer failed or reached numerical boundary')
            self.temperature = float(np.exp(optimum.x))
            self.predictor_id = predictor.identity
            self.role_provenance = digest((data.manifest, rows[KEYS], 'probability_calibration'))
            self.identity = digest((self.predictor_id, self.role_provenance, self.temperature))
            self._state = (self.temperature, self.predictor_id, self.role_provenance)
            self.status, self.reason = 'fitted', None
            return self
        except (ValueError, TypeError, RuntimeError) as exc:
            self.reason = str(exc)
            raise

    def check(self, predictor):
        predictor.check()
        if (self.status != 'fitted' or predictor.identity != self.predictor_id
                or self._state != (self.temperature, self.predictor_id, self.role_provenance)):
            raise ValueError('Require unchanged fitted calibrator and matching fixed predictor')

    def transform(self, values, predictor):
        self.check(predictor)
        p = probabilities(values, len(values))
        return expit(logit(p) / self.temperature)


class ConformalCalibration:
    """Binary split-conformal score sets; coverage certification stays blocked."""
    def __init__(self):
        self.status = 'unfitted'
        self.coverage_certification = dict(status='blocked', reason=
            'Finite-sample design and exchangeability are not established')

    def fit(self, data, predictor, *, alpha, calibrator=None):
        self.status = 'failed'
        self.reason = 'Conformal calibration did not complete'
        try:
            predictor.check()
            if isinstance(alpha, bool) or not isinstance(alpha, (float, int)) or not 0 < alpha < 1:
                raise ValueError('alpha must be finite and in (0,1)')
            rows = data.rows('conformal_calibration', targets=True)
            y, mask = binary_targets(rows.consensus_label, both=True)
            p = role_predictions(data, predictor, 'conformal_calibration', calibrator=calibrator).probability.to_numpy()[mask]
            scores = np.where(y == 1, 1 - p, p)
            rank = math.ceil((len(scores) + 1) * (1 - alpha))
            # rank n+1 is the conservative all-label set, not a clipped quantile.
            self.threshold = 1.0 if rank > len(scores) else float(np.sort(scores)[rank - 1])
            self.predictor_id = predictor.identity
            self.calibrator_id = None if calibrator is None else calibrator.identity
            self.role_provenance = digest((data.manifest, rows[KEYS], 'conformal_calibration'))
            self._state = (self.threshold, self.predictor_id, self.calibrator_id, self.role_provenance)
            self.status, self.reason = 'fitted', None
            return self
        except (ValueError, TypeError, RuntimeError) as exc:
            self.reason = str(exc)
            raise

    def predict_sets(self, data, predictor, *, role='assessment', calibrator=None):
        if (self.status != 'fitted' or self.predictor_id != predictor.identity
                or self.calibrator_id != (None if calibrator is None else calibrator.identity)
                or self._state != (self.threshold, self.predictor_id, self.calibrator_id, self.role_provenance)):
            raise ValueError('Conformal calibration requires unchanged predictor/calibrator')
        frame = role_predictions(data, predictor, role, calibrator=calibrator)
        p = frame.probability.to_numpy()
        frame['include_no_interaction'] = p <= self.threshold
        frame['include_interaction'] = 1 - p <= self.threshold
        return frame


def validate_task_b_release(snapshot, *, topology, pairs):
    """Validate actual immutable study rows and exact declared topology.

    No evidence-release approval schema exists yet: successful structural/source
    validation is deliberately insufficient to authorize fitting.
    """
    if not isinstance(snapshot, bytes) or len(snapshot) > release.MAX_INPUT_BYTES:
        raise ValueError('Require bounded immutable Task B study snapshot')
    topology = release._records(topology, 'topology')
    if topology is None:
        raise ValueError('Explicit Task B topology required')
    payload = release._manifest_snapshot(snapshot)
    if not isinstance(payload, dict) or set(payload) != {'studies'} or not isinstance(payload['studies'], list):
        raise ValueError('Task B snapshot requires an explicit studies list')
    studies = pd.DataFrame(payload['studies'])
    task_b.validate_studies(studies)
    task_b.validate_study_sources(studies)
    task_b.reliability_features(studies)
    if not set(task_b.pair_keys(studies[KEYS].drop_duplicates())) <= set(task_b.pair_keys(pairs)):
        raise ValueError('Study release contains undeclared pairs')
    fields = KEYS + ['study_id', 'source_id', 'cited_study_ids']
    def canonical(rows):
        return sorted(canonical_manifest_json({k: sorted(set(r[k])) if k == 'cited_study_ids'
                                              else r[k] for k in fields}) for r in rows)
    # Extra intermediate nodes require a future explicit evidence-release schema.
    if canonical(studies[fields].to_dict('records')) != canonical(topology):
        raise ValueError('Actual study release and declared topology differ')
    return dict(status='blocked', studies_sha256=hashlib.sha256(snapshot).hexdigest(),
                source_validation='passed', topology_consistency='passed',
                reason='No approved evidence-release attestation/independence contract exists')


class TaskBSnapshotAdapter:
    """Fail-closed fitting boundary pending an approved actual evidence release.

    Prior/reliability fitting and residual tuning are distinct operations. Neither
    may reuse the legacy path reader or treat topology as evidence authorization.
    """
    def __init__(self, data, *, study_snapshot, topology):
        data.check()
        if data.manifest['task'] != 'B':
            raise ValueError('Task B adapter requires Task B role assignments')
        pairs = data._labels[KEYS]
        self.validation = validate_task_b_release(study_snapshot, topology=topology, pairs=pairs)
        # Also bind supplied topology to the already validated role manifest.
        assignments = [{k: r[k] for k in KEYS + ['partition']} for r in data.manifest['assignments']]
        checked = validate_partitions(pairs.to_dict('records'), assignments,
            partition_roles=data.manifest['partition_roles'], regime=data.manifest['regime'],
            protocol=data.manifest['protocol'], task='B', topology=topology)
        if checked['topology_hash'] != data.manifest['topology_hash']:
            raise ValueError('Task B topology differs from role assignment validation')
        self.status = 'blocked'
        self.reason = self.validation['reason']

    def fit_prior_reliability(self):
        raise RuntimeError('Task B prior/reliability fitting blocked: ' + self.reason)

    def tune_residual(self, fixed_prior_reliability):
        raise RuntimeError('Task B residual tuning blocked: ' + self.reason)


def assess(data, predictor, *, calibrator=None):
    """Compute predictions before requesting assessment targets."""
    predictions = role_predictions(data, predictor, 'assessment', calibrator=calibrator)
    return dict(predictions=predictions,
                integrity=strict_score(data.rows('assessment', targets=True), predictions))


def run_task_a_roles(data, *, family='lr', candidates=None, alpha=None):
    """Compose components for a supplied context, without granting eligibility.

    This low-level API is for direct component tests. Production callers must
    use evaluate_request, which remains blocked by actual Phase 2 eligibility.
    No data paths or gate overrides are accepted here.
    """
    protocol = data.manifest['protocol']
    if protocol == 'baseline':
        if candidates is not None or alpha is not None:
            raise ValueError('Baseline has no tuning or conformal role')
        predictor = fit_task_a(data, family=family)
    else:
        predictor = tune_task_a(data, family=family, candidates=candidates)
    calibrator = None
    if protocol in {'full', 'probability_calibrated'}:
        calibrator = ProbabilityCalibration().fit(data, predictor)
    conformal = None
    if protocol in {'full', 'conformal'}:
        conformal = ConformalCalibration().fit(data, predictor, alpha=alpha, calibrator=calibrator)
    elif alpha is not None:
        raise ValueError('alpha requires a conformal calibration role')
    # Generate all outputs before requesting any assessment targets.
    sets = None if conformal is None else conformal.predict_sets(data, predictor, calibrator=calibrator)
    report = assess(data, predictor, calibrator=calibrator)
    report.update(predictor_id=predictor.identity,
                  calibrator_id=None if calibrator is None else calibrator.identity,
                  prediction_sets=sets,
                  coverage_certification=dict(status='blocked', reason=
                      'No justified finite-sample exchangeability design'),
                  scope='component_execution_only; no release eligibility conferred')
    return report
