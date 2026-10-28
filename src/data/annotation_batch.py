"""Identifier-only, seeded coverage selection for human annotation (no labels)."""
import argparse
from collections import Counter
from hashlib import sha256
import json
from pathlib import Path

import pandas as pd

from src.data.consensus_annotation import KEYS, digest, pair_keys


def select_pairs(frame, size=24, seed=20261029):
    """Greedily cover herbs, then drugs, then balance representation.

    Lexicographic priorities are fixed before review. SHA-256(seed, pair) breaks
    ties independently of input order. This is a coverage pilot, not a random
    prevalence sample or a guarantee of globally optimal joint coverage.
    """
    pairs = pair_keys(frame)
    if not 1 <= size <= len(pairs):
        raise ValueError('Batch size must be between 1 and the unique pair count')
    remaining = set(map(tuple, pairs[KEYS].values))
    herbs, drugs = Counter(), Counter()
    chosen = []
    while len(chosen) < size:
        def priority(pair):
            herb, drug = pair
            tie = sha256(json.dumps([seed, herb, drug]).encode()).hexdigest()
            return (herbs[herb] > 0, drugs[drug] > 0,
                    herbs[herb] + drugs[drug], tie, herb, drug)
        herb, drug = min(remaining, key=priority)
        chosen.append((herb, drug))
        remaining.remove((herb, drug))
        herbs[herb] += 1
        drugs[drug] += 1
    return pd.DataFrame(chosen, columns=KEYS)


def prepare(input_path, output, size=24, seed=20261029):
    # Read only identifier columns, even if input contains outcomes or scores.
    candidates = pair_keys(pd.read_csv(input_path, usecols=KEYS, dtype=str,
                                      keep_default_na=False))
    selected = select_pairs(candidates, size, seed)
    manifest = {
        'method_version': 1, 'seed': seed, 'batch_size': size,
        'method': 'Greedy lexicographic: unseen herb first, unseen drug second, '
                  'lowest combined selected herb/drug counts third; '
                  'SHA256(JSON([seed, herb_id, drug_id])) tie-break; '
                  'identifier normalization, deduplication and sorting before selection.',
        'limitations': 'Coverage pilot, not proportional random sampling or '
                       'globally optimal joint coverage; no outcome/class quotas.',
        'selection_inputs': KEYS,
        'source_path': str(input_path),
        'candidate_identifier_sha256': sha256(candidates.to_csv(index=False).encode()).hexdigest(),
        'candidate_pair_count': len(candidates),
        'candidate_herb_count': candidates.herb_id.nunique(),
        'candidate_drug_count': candidates.drug_id.nunique(),
        'selected_herb_count': selected.herb_id.nunique(),
        'selected_drug_count': selected.drug_id.nunique(),
        'herb_counts': dict(sorted(Counter(selected.herb_id).items())),
        'drug_counts': dict(sorted(Counter(selected.drug_id).items())),
        'selected_identifiers': selected.to_dict('records'),
    }
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    selected.to_csv(output / 'selected_pairs.csv', index=False)
    manifest['selected_pairs_sha256'] = digest(output / 'selected_pairs.csv')
    (output / 'selection_manifest.json').write_text(json.dumps(manifest, indent=2) + '\n')
    return manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--pairs', required=True, type=Path)
    parser.add_argument('--output', required=True, type=Path)
    parser.add_argument('--size', type=int, default=24)
    parser.add_argument('--seed', type=int, default=20261029)
    args = parser.parse_args()
    manifest = prepare(args.pairs, args.output, args.size, args.seed)
    print(f"Selected {manifest['batch_size']} unresolved candidates: "
          f"{manifest['selected_herb_count']} herbs, {manifest['selected_drug_count']} drugs.")


if __name__ == '__main__':
    main()
