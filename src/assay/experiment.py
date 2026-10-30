"""Train and evaluate retrospective assay regressors with untouched test roles.

Run: python -m src.assay.experiment --dataset PATH --output NEW_DIRECTORY
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
from hashlib import sha256
import json
import math
from pathlib import Path
import platform

import joblib
import numpy as np
import pandas as pd
import rdkit
import sklearn
from sklearn.dummy import DummyRegressor
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from src.assay.data import PROTOCOL, aggregate, assign_roles, fingerprints, load_records

MODEL_NAMES = ('median', 'ridge', 'extra_trees', 'consistency_weighted_extra_trees')
LIMITATIONS = [
    'Compound-target assay potency only; no whole-herb or clinical interaction prediction.',
    'Text screening is conservative and incomplete; assay protocols and substrates remain heterogeneous.',
    'Repeat consistency weights measure dispersion, not study truthfulness or clinical reliability.',
    'Scaffold/compound holdouts may share publications; consult the explicit overlap audit.',
    'Source holdouts purge cross-role structures; purging changes the evaluated population.',
    'Conformal intervals assume exchangeable calibration/test errors; grouped distribution shift can violate coverage.',
    'Bootstrap intervals describe held-out compounds conditional on this fitted split, not split-to-split uncertainty.',
    'No tautomer normalization, external validation, induction, exposure or drug-clearance inference.',
]


def metrics(y, prediction):
    y, prediction = np.asarray(y), np.asarray(prediction)
    return {'n': len(y), 'mae': float(mean_absolute_error(y, prediction)),
            'rmse': float(np.sqrt(mean_squared_error(y, prediction))),
            'r2': float(r2_score(y, prediction)) if len(y) > 1 and np.ptp(y) > 0 else None}


def conformal_radius(y, prediction, alpha=.10):
    if not 0 < alpha < 1:
        raise ValueError('alpha must be strictly between zero and one')
    scores = np.sort(np.abs(np.asarray(y) - np.asarray(prediction)))
    if not len(scores) or not np.isfinite(scores).all():
        raise ValueError('Finite calibration scores required')
    rank = math.ceil((len(scores) + 1) * (1 - alpha))
    if rank > len(scores):
        raise ValueError('Insufficient calibration samples for finite conformal interval')
    return float(scores[rank - 1])


def bootstrap_mae(frame, repeats=200, seed=20261030):
    """Resample structures, retaining all target predictions of a drawn compound."""
    errors = (frame.observed_pic50 - frame.predicted_pic50).abs()
    groups = pd.DataFrame({'structure': frame.structure, 'error': errors}).groupby('structure').error.agg(['sum', 'count'])
    rng = np.random.default_rng(seed)
    sums, counts = groups['sum'].to_numpy(), groups['count'].to_numpy()
    values = []
    for _ in range(repeats):
        index = rng.integers(0, len(groups), len(groups))
        values.append(float(sums[index].sum() / counts[index].sum()))
    return {'lower': float(np.quantile(values, .025)), 'upper': float(np.quantile(values, .975)),
            'resamples': repeats, 'unit': 'canonical structure', 'conditional_on': 'single fitted split'}


def new_model(name, seed):
    if name == 'median':
        return DummyRegressor(strategy='median')
    if name == 'ridge':
        return Ridge(alpha=100., solver='lsqr')
    if name in MODEL_NAMES[2:]:
        return ExtraTreesRegressor(n_estimators=128, min_samples_leaf=2,
                                   max_features=.5, n_jobs=2, random_state=seed)
    raise ValueError('Unknown model')


def run(dataset, output, mode='scaffold', seed=20261030, alpha=.10):
    if not 0 < alpha < 1:
        raise ValueError('alpha must be strictly between zero and one')
    output = Path(output)
    if output.exists():
        raise FileExistsError('Use a new output directory to preserve previous experiments')
    records, excluded, source = load_records(dataset)
    assigned, purged, split_audit = assign_roles(records, mode, seed)
    samples = aggregate(assigned)
    structures = sorted(samples.structure.unique())
    matrix = fingerprints(structures)
    lookup = {value: i for i, value in enumerate(structures)}
    output.mkdir(parents=True, exist_ok=False)
    # A completed report is written last. Partial outputs never authorize inference.
    config = {'schema': 'hdi.assay-experiment/v1', 'protocol': PROTOCOL,
              'split': mode, 'seed': seed, 'alpha': alpha,
              'models': list(MODEL_NAMES), 'selection': 'minimum validation MAE per target; fixed model order breaks ties',
              'hyperparameters': {'ridge_alpha': 100., 'trees': 128, 'min_samples_leaf': 2, 'max_features': .5},
              'minimum_role_samples': {'train': 40, 'validation': 10, 'calibration': max(10, math.ceil(1 / alpha)), 'test': 10},
              'source_manifest_sha256': sha256((Path(dataset) / 'manifest.json').read_bytes()).hexdigest(),
              'source_outputs': source['outputs']}
    (output / 'config.json').write_text(json.dumps(config, indent=2) + '\n')
    excluded[['activity_id', 'exclusion_reason']].to_csv(output / 'excluded_records.csv', index=False)
    purged[['activity_id', 'role', 'structure', 'document_chembl_id']].to_csv(output / 'purged_records.csv', index=False)
    samples.to_csv(output / 'samples.csv', index=False)
    assigned[['activity_id', 'structure', 'scaffold', 'document_chembl_id', 'assay_chembl_id', 'role']].to_csv(
        output / 'record_roles.csv', index=False)
    (output / 'split_audit.json').write_text(json.dumps(split_audit, indent=2) + '\n')
    prediction_rows, summaries, bundles, skipped = [], {}, {}, {}
    for target, frame in samples.groupby('target', sort=True):
        counts = frame.role.value_counts().to_dict()
        if any(counts.get(role, 0) < minimum for role, minimum in config['minimum_role_samples'].items()):
            skipped[target] = {'reason': 'insufficient role samples', 'counts': counts}
            continue
        print(f'{target}: {counts}', flush=True)
        roles = {}
        for role in PROTOCOL['roles']:
            subset = frame[frame.role == role]
            roles[role] = (subset, matrix[[lookup[s] for s in subset.structure]], subset.pic50.to_numpy())
        train, x_train, y_train = roles['train']
        validation, x_val, y_val = roles['validation']
        calibration, x_cal, y_cal = roles['calibration']
        test, x_test, y_test = roles['test']
        models, validation_scores = {}, {}
        for name in MODEL_NAMES:
            model = new_model(name, seed)
            kwargs = {'sample_weight': train.weight.to_numpy()} if name == MODEL_NAMES[-1] else {}
            model.fit(x_train, y_train, **kwargs)
            models[name] = model
            validation_scores[name] = metrics(y_val, model.predict(x_val))
        # Select before reading calibration residuals or producing test metrics.
        selected = min(MODEL_NAMES, key=lambda name: validation_scores[name]['mae'])
        model = models[selected]
        radius = conformal_radius(y_cal, model.predict(x_cal), alpha)
        test_scores = {}
        for name, candidate in models.items():
            prediction = candidate.predict(x_test)
            test_scores[name] = metrics(y_test, prediction)
            for row, value in zip(test.itertuples(), prediction):
                is_selected = name == selected
                prediction_rows.append(dict(sample_id=row.sample_id, structure=row.structure, target=target,
                                            model=name, selected=is_selected, observed_pic50=row.pic50,
                                            predicted_pic50=float(value),
                                            lower_pic50=float(value - radius) if is_selected else None,
                                            upper_pic50=float(value + radius) if is_selected else None,
                                            measurements=row.measurements, spread=row.spread))
        selected_prediction = model.predict(x_test)
        summaries[target] = {'role_counts': counts, 'selected_model': selected, 'validation': validation_scores,
                             'test': test_scores, 'interval': {'alpha': alpha, 'radius_pic50': radius,
                                 'test_coverage': float(np.mean(np.abs(y_test - selected_prediction) <= radius)),
                                 'mean_width_pic50': 2 * radius, 'calibration_n': len(calibration)},
                             'train_conflict_pairs_spread_gt_1_log_unit': int((train.spread > 1).sum())}
        bundles[target] = {'model': model, 'selected_model': selected, 'radius': radius,
                           'train_structures': train.structure.tolist(), 'train_fingerprints': x_train,
                           'train_min': float(y_train.min()), 'train_max': float(y_train.max())}
    if not summaries:
        raise ValueError('No target meets prespecified role sizes; inspect split audit, do not claim a completed run')
    predictions = pd.DataFrame(prediction_rows)
    selected_rows = predictions[predictions.selected]
    for target in summaries:
        summaries[target]['selected_test_mae_ci95'] = bootstrap_mae(selected_rows[selected_rows.target == target], seed=seed)
    predictions.to_csv(output / 'test_predictions.csv', index=False)
    bundle = {'schema': 'hdi.assay-model/v1', 'config': config, 'targets': bundles, 'limitations': LIMITATIONS}
    joblib.dump(bundle, output / 'model.joblib', compress=3)
    report = {'schema': 'hdi.assay-results/v1', 'complete': True,
              'created_at': datetime.now(timezone.utc).isoformat(),
              'clinical_task_a_eligible': False, 'clinical_task_b_eligible': False,
              'config': config, 'source_version': source['source_status']['chembl_db_version'],
              'counts': {'source_candidates': len(records) + len(excluded), 'retained_records': len(records),
                         'excluded_records': len(excluded), 'purged_records': len(purged),
                         'aggregated_samples': len(samples), 'distinct_structures': len(structures)},
              'targets': summaries, 'skipped_targets': skipped,
              'selected_micro_test': metrics(selected_rows.observed_pic50, selected_rows.predicted_pic50),
              'selected_micro_mae_ci95': bootstrap_mae(selected_rows, seed=seed),
              'limitations': LIMITATIONS,
              'versions': {'python': platform.python_version(), 'rdkit': rdkit.__version__,
                           'sklearn': sklearn.__version__, 'numpy': np.__version__, 'pandas': pd.__version__}}
    report['artifacts'] = {path.name: sha256(path.read_bytes()).hexdigest()
                           for path in sorted(output.iterdir()) if path.is_file()}
    (output / 'report.json').write_text(json.dumps(report, indent=2, allow_nan=False) + '\n')
    write_markdown(report, output / 'report.md')
    return report


def write_markdown(report, destination):
    lines = ['# Retrospective ChEMBL compound-assay experiment', '',
             'Endpoint: pIC50. These are held-out assay predictions, not clinical herb–drug predictions.', '',
             f"Split: {report['config']['split']}; seed: {report['config']['seed']}. Model selection uses validation MAE.", '',
             '| Target | Selected model | Test n | MAE | RMSE | R² | Interval coverage |',
             '| --- | --- | ---: | ---: | ---: | ---: | ---: |']
    for target, result in report['targets'].items():
        score = result['test'][result['selected_model']]
        r2 = 'undefined' if score['r2'] is None else f"{score['r2']:.3f}"
        lines.append(f"| {target} | {result['selected_model']} | {score['n']} | {score['mae']:.3f} | "
                     f"{score['rmse']:.3f} | {r2} | {result['interval']['test_coverage']:.3f} |")
    lines += ['', 'Complete baseline results, confidence intervals and role counts are in `report.json`.', '',
              'Cross-role publication/scaffold overlaps and all source-split purges are in `split_audit.json`.', '',
              *['- ' + item for item in report['limitations']], '']
    Path(destination).write_text('\n'.join(lines))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dataset', type=Path, default=Path('data/external/chembl_assays_20261030'))
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--split', choices=['scaffold', 'compound', 'source'], default='scaffold')
    parser.add_argument('--seed', type=int, default=20261030)
    parser.add_argument('--alpha', type=float, default=.10)
    args = parser.parse_args()
    report = run(args.dataset, args.output, args.split, args.seed, args.alpha)
    print(json.dumps(report['selected_micro_test'], indent=2))


if __name__ == '__main__':
    main()
