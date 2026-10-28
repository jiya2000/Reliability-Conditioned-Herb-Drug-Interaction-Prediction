"""Fail-closed public Phase 3.2 boundary.

Only structural validators are usable without release authorization. The production
orchestrator currently issues no execution grants. Private synthetic algorithms
live in _pipeline_algorithms and cannot authorize this API. No test/synthetic flag,
caller report, fitted estimator or manually constructed context enables execution.
"""
from src.training import evaluation_orchestrator as _orchestrator
from src.training._pipeline_algorithms import (
    SnapshotTables, RoleData, probabilities, binary_targets, validate_task_b_release,
)

__all__ = ['SnapshotTables', 'RoleData', 'probabilities', 'binary_targets',
           'validate_task_b_release', 'fit_task_a', 'tune_task_a', 'binary_loss',
           'strict_score', 'role_predictions', 'assess', 'run_task_a_roles',
           'FixedPredictor', 'ProbabilityCalibration', 'ConformalCalibration',
           'TaskBSnapshotAdapter']


def _blocked(*args, authorization=None, **kwargs):
    _orchestrator._require_execution_authorization(authorization)
    # Never dispatch to algorithms merely because a gate function returns.
    raise PermissionError('Production backend has no authorization issuance contract')


fit_task_a = _blocked
tune_task_a = _blocked
binary_loss = _blocked
strict_score = _blocked
role_predictions = _blocked
assess = _blocked
run_task_a_roles = _blocked


class FixedPredictor:
    def __init__(self, *args, **kwargs):
        _blocked(*args, **kwargs)

    check = _blocked
    predict = _blocked


class ProbabilityCalibration:
    def __init__(self):
        self.status = 'unfitted'
        self.reason = 'Production execution requires orchestrator authorization'

    fit = _blocked
    check = _blocked
    transform = _blocked


class ConformalCalibration:
    def __init__(self):
        self.status = 'unfitted'
        self.coverage_certification = dict(status='blocked', reason=
            'Finite-sample design and exchangeability are not established')

    fit = _blocked
    predict_sets = _blocked


class TaskBSnapshotAdapter:
    """Read-only evidence validation; all execution remains blocked."""
    def __init__(self, data, *, study_snapshot, topology):
        from src.training._pipeline_algorithms import TaskBSnapshotAdapter as _Validator
        validated = _Validator(data, study_snapshot=study_snapshot, topology=topology)
        self.validation = validated.validation.copy()
        self.status, self.reason = 'blocked', validated.reason

    fit_prior_reliability = _blocked
    tune_residual = _blocked
