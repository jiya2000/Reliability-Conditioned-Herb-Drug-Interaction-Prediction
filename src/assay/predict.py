"""Inference from an explicitly supplied, trusted local assay experiment bundle."""
from __future__ import annotations

import argparse
from hashlib import sha256
import json
from pathlib import Path

import joblib
import numpy as np

from src.assay.data import PROTOCOL, canonical_structure, fingerprints


def load_experiment(directory):
    """Integrity checks detect changed artifacts; they do not authenticate publishers.

    joblib uses pickle. Only load an experiment generated locally or obtained
    from a trusted publisher, never an uploaded/untrusted bundle.
    """
    directory = Path(directory)
    report = json.loads((directory / 'report.json').read_text())
    if (report.get('schema') != 'hdi.assay-results/v1' or report.get('complete') is not True
            or report.get('clinical_task_a_eligible') is not False
            or report.get('clinical_task_b_eligible') is not False):
        raise ValueError('Incomplete or invalid assay experiment')
    required = {'model.joblib', 'config.json', 'samples.csv', 'record_roles.csv',
                'split_audit.json', 'test_predictions.csv', 'excluded_records.csv', 'purged_records.csv'}
    if set(report.get('artifacts', {})) != required:
        raise ValueError('Missing assay artifacts')
    for name, expected in report['artifacts'].items():
        if sha256((directory / name).read_bytes()).hexdigest() != expected:
            raise ValueError(f'Changed assay artifact: {name}')
    bundle = joblib.load(directory / 'model.joblib')
    if (bundle.get('schema') != 'hdi.assay-model/v1' or bundle.get('config') != report['config']
            or bundle['config']['protocol'] != PROTOCOL):
        raise ValueError('Incompatible assay model protocol')
    return bundle, report


def predict_smiles(bundle, smiles):
    structure = canonical_structure(smiles)
    if structure is None:
        raise ValueError('Enter valid organic-molecule SMILES')
    x = fingerprints([structure])
    rows = []
    for target, item in sorted(bundle['targets'].items()):
        value = float(item['model'].predict(x)[0])
        radius = item['radius']
        train = item['train_fingerprints']
        intersection = np.logical_and(train, x).sum(axis=1)
        union = np.logical_or(train, x).sum(axis=1)
        similarity = np.divide(intersection, union, out=np.zeros(len(train)), where=union != 0)
        nearest = int(similarity.argmax())
        seen = structure in item['train_structures']
        rows.append({'target': target, 'predicted_pic50': value,
                     'lower_pic50': value - radius, 'upper_pic50': value + radius,
                     'implied_ic50_uM': float(10 ** np.clip(6 - value, -300, 300)),
                     'nearest_training_tanimoto': float(similarity[nearest]),
                     'nearest_training_structure': item['train_structures'][nearest],
                     'seen_in_training': seen, 'model': item['selected_model'],
                     'status': 'training-compound illustration' if seen else 'unvalidated prospective assay estimate',
                     'clinical_interaction_verdict': 'unavailable'})
    return {'structure': structure, 'endpoint': PROTOCOL['endpoint'],
            'interval_nominal_coverage': 1 - bundle['config']['alpha'],
            'predictions': rows, 'limitations': bundle['limitations']}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--experiment', type=Path, required=True,
                        help='Trusted locally generated experiment directory (contains pickle-based model)')
    parser.add_argument('--smiles', required=True)
    args = parser.parse_args()
    bundle, _ = load_experiment(args.experiment)
    print(json.dumps(predict_smiles(bundle, args.smiles), indent=2, allow_nan=False))


if __name__ == '__main__':
    main()
