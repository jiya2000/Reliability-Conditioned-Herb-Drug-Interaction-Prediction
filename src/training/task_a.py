"""Consensus-only Task A entry point: python -m src.training.task_a --check-only.

Fixed LR/HGB baselines; no tuning, calibration, Task B, or publication claims.
Provenance validation is structural and still requires independent human review.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss, roc_auc_score
from sklearn.model_selection import StratifiedGroupKFold

from src.data.evidence_schema import load_consensus_labels_table
from src.data.mechanistic_curation import load_mechanistic_curation
from src.features.task_a_contract import (
    TASK_A_FEATURES, build_task_a_features, task_a_training_pipeline,
    validate_task_a_features,
)


def prepare_task_a(consensus_path, curation_path):
    """Use consensus as the pair universe; never read publication outcomes.

    Disputed pairs are counted but excluded from binary fitting and scoring.
    Unknown mechanisms block training rather than becoming imputed negatives.
    """
    labels, _ = load_consensus_labels_table(consensus_path, require_labels=True)
    disputed = int(labels.consensus_label.eq('disputed').sum())
    pairs = labels.loc[labels.consensus_label.ne('disputed')].reset_index(drop=True)
    if pairs.empty or pairs.consensus_label.nunique() != 2:
        raise ValueError('Task A requires both binary consensus classes')
    curation = load_mechanistic_curation(curation_path)
    X = validate_task_a_features(build_task_a_features(pairs, curation))
    y = pairs.consensus_label.eq('interaction').to_numpy(dtype=int)
    return pairs, X, y, disputed


def task_a_splits(pairs, y, *, mode='herb', n_splits=5, repeats=5, seed=42):
    """Prevalidate all folds before any fit; each pair appears once per repeat.

    Task A has no literature inputs/citation clusters. Pair grouping is sufficient
    here, and must not be reused as Task B's citation-cluster split protocol.
    """
    if mode not in {'herb', 'drug', 'pair'}:
        raise ValueError('mode must be herb, drug, or pair')
    if n_splits < 2 or repeats < 1:
        raise ValueError('Require n_splits >= 2 and repeats >= 1')
    keys = (list(zip(pairs.herb_id, pairs.drug_id)) if mode == 'pair'
            else pairs[f'{mode}_id'].tolist())
    # Integer groups avoid tuple-array shape ambiguities and identifier encoding in X.
    mapping = {key: i for i, key in enumerate(dict.fromkeys(keys))}
    groups = np.array([mapping[key] for key in keys])
    if len(mapping) < n_splits:
        raise ValueError(f'Insufficient {mode} groups for {n_splits} folds')
    folds = []
    for repeat in range(repeats):
        splitter = StratifiedGroupKFold(n_splits=n_splits, shuffle=True,
                                        random_state=seed + repeat)
        for fold, (train, test) in enumerate(splitter.split(pairs, y, groups)):
            if set(groups[train]) & set(groups[test]):
                raise ValueError('Task A group overlap')
            if len(np.unique(y[train])) != 2 or len(np.unique(y[test])) != 2:
                raise ValueError(f'{mode} repeat {repeat} fold {fold} lacks both classes; '
                                 'curate more data or choose fewer folds (no silent skipping)')
            folds.append((repeat, fold, train, test))
    return folds


def run_task_a(consensus_path, curation_path, *, mode='herb', n_splits=5,
               repeats=5, seed=42, model='lr', check_only=False):
    if model not in {'lr', 'hgb'}:
        raise ValueError('model must be lr or hgb')
    pairs, X, y, disputed = prepare_task_a(consensus_path, curation_path)
    folds = task_a_splits(pairs, y, mode=mode, n_splits=n_splits,
                          repeats=repeats, seed=seed)
    result = dict(task='A', target='external_consensus', model=model, split=mode,
                  n_splits=n_splits, repeats=repeats, seed=seed,
                  feature_names=list(TASK_A_FEATURES), binary_pairs=len(pairs),
                  disputed_pairs_excluded=disputed,
                  status='validated_without_fitting' if check_only else 'uncalibrated_baseline_cv',
                  folds=[], predictions=[])
    if check_only:
        return result
    for repeat, fold, train, test in folds:
        estimator = (LogisticRegression(max_iter=1000, random_state=seed)
                     if model == 'lr' else HistGradientBoostingClassifier(random_state=seed))
        pipeline = task_a_training_pipeline(estimator)
        pipeline.fit(X.iloc[train], y[train])
        p = pipeline.predict_proba(X.iloc[test])[:, 1]
        auc = float(roc_auc_score(y[test], p))
        if auc > .98:
            raise ValueError(f'Task A AUC integrity alert: repeat={repeat}, fold={fold}, AUC={auc}')
        result['folds'].append(dict(repeat=repeat, fold=fold, roc_auc=auc,
                                    brier=float(brier_score_loss(y[test], p))))
        for index, probability in zip(test, p):
            result['predictions'].append(dict(repeat=repeat, fold=fold,
                herb_id=pairs.iloc[index].herb_id, drug_id=pairs.iloc[index].drug_id,
                target=int(y[index]), probability=float(probability)))
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--consensus', default='data/benchmark/consensus_labels.csv')
    parser.add_argument('--curation', default='data/curation/mechanistic_features.csv')
    parser.add_argument('--split', choices=['herb', 'drug', 'pair'], default='herb')
    parser.add_argument('--model', choices=['lr', 'hgb'], default='lr')
    parser.add_argument('--folds', type=int, default=5)
    parser.add_argument('--repeats', type=int, default=5)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--check-only', action='store_true')
    parser.add_argument('--output', type=Path)
    args = parser.parse_args(argv)
    try:
        result = run_task_a(args.consensus, args.curation, mode=args.split,
                            model=args.model, n_splits=args.folds, repeats=args.repeats,
                            seed=args.seed, check_only=args.check_only)
    except (ValueError, RuntimeError, OSError) as exc:
        parser.exit(2, f'Task A blocked: {exc}\n')
    payload = json.dumps(result, indent=2, allow_nan=False) + '\n'
    if args.output:
        args.output.write_text(payload)
    else:
        print(payload, end='')


if __name__ == '__main__':
    main()
