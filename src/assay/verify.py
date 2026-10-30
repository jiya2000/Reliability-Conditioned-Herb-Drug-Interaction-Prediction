"""Independently recompute saved test metrics and required role separations."""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from src.assay.predict import load_experiment


def verify_experiment(directory):
    directory = Path(directory)
    _, report = load_experiment(directory)
    predictions = pd.read_csv(directory / 'test_predictions.csv')
    samples = pd.read_csv(directory / 'samples.csv')
    records = pd.read_csv(directory / 'record_roles.csv')
    if records.activity_id.duplicated().any() or samples.sample_id.duplicated().any():
        raise ValueError('Duplicate activity or aggregated sample')
    if samples.groupby('structure').role.nunique().max() != 1:
        raise ValueError('Structure crosses roles')
    mode = report['config']['split']
    group = {'scaffold': 'scaffold', 'compound': 'structure', 'source': 'document_chembl_id'}[mode]
    if records.groupby(group).role.nunique().max() != 1:
        raise ValueError('Split group crosses roles')
    for target, result in report['targets'].items():
        if result['selected_model'] != min(report['config']['models'], key=lambda name: result['validation'][name]['mae']):
            raise ValueError('Model selection differs from prespecified validation rule')
        target_samples = samples[(samples.target == target) & (samples.role == 'test')].set_index('sample_id')
        target_predictions = predictions[predictions.target == target]
        for name, score in result['test'].items():
            frame = target_predictions[target_predictions.model == name]
            if frame.sample_id.duplicated().any() or set(frame.sample_id) != set(target_samples.index):
                raise ValueError('Missing or duplicated held-out predictions')
            expected = target_samples.loc[frame.sample_id]
            if not np.allclose(frame.observed_pic50, expected.pic50, atol=1e-12, rtol=0):
                raise ValueError('Saved test outcomes differ from held-out samples')
            if frame.structure.tolist() != expected.structure.tolist():
                raise ValueError('Prediction structures differ from held-out samples')
            y, p = frame.observed_pic50, frame.predicted_pic50
            recomputed = {'n': len(frame), 'mae': mean_absolute_error(y, p),
                          'rmse': np.sqrt(mean_squared_error(y, p)),
                          'r2': r2_score(y, p) if len(y) > 1 and np.ptp(y) > 0 else None}
            for metric, value in recomputed.items():
                if value is None:
                    if score[metric] is not None:
                        raise ValueError('Undefined metric recorded as finite')
                elif not np.isclose(value, score[metric], atol=1e-12, rtol=0):
                    raise ValueError(f'Metric mismatch: {target}/{name}/{metric}')
        selected = target_predictions[target_predictions.model == result['selected_model']]
        if not selected.selected.all() or int(target_predictions.selected.sum()) != len(selected):
            raise ValueError('Wrong selected-model flags')
        radius = result['interval']['radius_pic50']
        if not (np.allclose(selected.lower_pic50, selected.predicted_pic50 - radius)
                and np.allclose(selected.upper_pic50, selected.predicted_pic50 + radius)):
            raise ValueError('Interval endpoints differ from calibrated radius')
        coverage = np.mean(np.abs(selected.observed_pic50 - selected.predicted_pic50) <= radius)
        if not np.isclose(coverage, result['interval']['test_coverage'], atol=1e-12, rtol=0):
            raise ValueError('Interval coverage mismatch')
    selected = predictions[predictions.selected]
    if not np.isclose(mean_absolute_error(selected.observed_pic50, selected.predicted_pic50),
                      report['selected_micro_test']['mae'], atol=1e-12, rtol=0):
        raise ValueError('Micro MAE mismatch')
    return {'verified': True, 'split': mode, 'targets': len(report['targets']),
            'selected_test_predictions': len(selected), 'fit_performed': False}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--experiment', type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(verify_experiment(args.experiment), indent=2))


if __name__ == '__main__':
    main()
