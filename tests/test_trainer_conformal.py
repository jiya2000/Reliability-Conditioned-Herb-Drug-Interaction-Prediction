"""Trainer recalibration must fail closed across the entire call path."""
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from src.models.conformal_abstention import ConformalTriStateEngine
from src.models.hdi_model import HDIModel
from src.training.trainer import HDITrainer


class CalibrationModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conformal_engine = ConformalTriStateEngine()
        self.failure = None
        self.engine_calls = 0

    def forward(self, **kwargs):
        if self.failure == 'inference':
            raise RuntimeError('inference failed')
        return {
            'probabilities': torch.tensor([0.1, 0.9]),
            'reliability_scores': torch.tensor([
                float('nan') if self.failure == 'reliability' else 0.8, 0.8
            ]),
        }

    def calibrate_conformal(self, *args, **kwargs):
        self.engine_calls += 1
        return HDIModel.calibrate_conformal(self, *args, **kwargs)


@pytest.fixture
def calibration_case():
    model = CalibrationModel()
    model.conformal_engine.calibrate([0.2, 0.8], [0, 1])
    trainer = SimpleNamespace(model=model, device=torch.device('cpu'))
    graph = {name: torch.zeros(1) for name in ('node_features', 'edge_index', 'edge_type')}
    batch = {
        'source_indices': torch.tensor([0, 0]),
        'target_indices': torch.tensor([0, 0]),
        'labels': torch.tensor([0, 1]),
        'metadata': torch.zeros(2, 8),
    }
    return trainer, graph, batch


@pytest.mark.parametrize('failure', [
    'graph', 'batch', 'loader', 'inference', 'calibration', 'conflict', 'reliability'
])
def test_trainer_failed_recalibration_invalidates_state(calibration_case, failure):
    trainer, graph, batch = calibration_case
    model = trainer.model
    engine = model.conformal_engine
    loader = [batch]
    epsilon = 0.1
    if failure == 'graph':
        del graph['node_features']
    elif failure == 'batch':
        del batch['labels']
    elif failure == 'loader':
        def broken_loader():
            yield batch
            raise RuntimeError('loader failed')
        loader = broken_loader()
    elif failure in ('inference', 'reliability'):
        model.failure = failure
    elif failure == 'calibration':
        epsilon = 0
    elif failure == 'conflict':
        batch['metadata'][0, 5] = float('nan')

    with pytest.raises((KeyError, RuntimeError, ValueError)):
        HDITrainer.calibrate_conformal(trainer, loader, graph, epsilon=epsilon)
    assert model.engine_calls == int(failure in ('calibration', 'conflict', 'reliability'))
    assert not engine.is_calibrated
    assert engine.q_hat is None
    assert engine.calibration_scores is None
    assert not engine.sample_size_valid
    assert engine.sample_size_warning == ''
    with pytest.raises(RuntimeError, match='calibration'):
        engine.predict_single(0.9)

    # A valid retry restores usable calibration after every failure stage.
    model.failure = None
    graph['node_features'] = torch.zeros(1)
    batch['labels'] = torch.tensor([0, 1])
    batch['metadata'] = torch.zeros(2, 8)
    q_hat, metrics = HDITrainer.calibrate_conformal(trainer, [batch], graph)
    assert engine.is_calibrated
    assert q_hat == pytest.approx(0.1)
    np.testing.assert_allclose(engine.calibration_scores, [0.1, 0.1], atol=1e-7)
    assert metrics.empirical_coverage == 1.0
    assert engine.predict_single(0.9).predicted_label == 1
