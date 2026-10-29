"""Structural planning diagnostics only; never release eligibility or performance."""
import argparse
from hashlib import sha256
from pathlib import Path
import json

from src.data.consensus_annotation import pair_keys
from src.data.review_packet import _table
from src.training.evaluation_protocol import _REQUIRED


def audit(pairs_path, consensus_path, *, protocol='full'):
    if protocol not in _REQUIRED:
        raise ValueError('Unknown evaluation protocol')
    raw_pairs = Path(pairs_path).read_bytes()
    raw_labels = Path(consensus_path).read_bytes()
    pairs = pair_keys(_table(raw_pairs))
    labels = _table(raw_labels)
    if not {'herb_id', 'drug_id', 'consensus_label'} <= set(labels):
        raise ValueError('Consensus table lacks required identity/label columns')
    # Reuse the complete consensus schema; never infer decisions from study findings.
    from src.data.evidence_schema import validate_consensus_labels_table
    valid, errors = validate_consensus_labels_table(labels)
    if not valid:
        raise ValueError('; '.join(errors))
    if labels.duplicated(['herb_id', 'drug_id']).any():
        raise ValueError('Duplicate consensus identities')
    joined = pairs.merge(labels[['herb_id', 'drug_id', 'consensus_label']],
                         on=['herb_id', 'drug_id'], how='left', validate='one_to_one')
    binary = joined[joined.consensus_label.isin(['interaction', 'no_interaction'])]
    roles = len(_REQUIRED[protocol])
    capacity = {}
    for regime, count in [('pair', len(binary)), ('herb', binary.herb_id.nunique()),
                          ('drug', binary.drug_id.nunique())]:
        capacity[regime] = {'independent_group_upper_bound': int(count),
                           'required_nonempty_roles': roles,
                           'necessary_group_count_met': bool(count >= roles)}
    return {'scope': 'necessary_structural_conditions_only', 'eligible': False,
            'protocol': protocol, 'candidate_pairs': len(pairs),
            'binary_reference_pairs': len(binary),
            'missing_reference_pairs': int(joined.consensus_label.isna().sum()),
            'disputed_reference_pairs': int(joined.consensus_label.eq('disputed').sum()),
            'class_counts': {str(k): int(v) for k, v in binary.consensus_label.value_counts().items()},
            'capacity': capacity,
            'input_sha256': {'pairs': sha256(raw_pairs).hexdigest(),
                             'consensus': sha256(raw_labels).hexdigest()},
            'blocked': ['No release authorization is issued by this planning diagnostic.',
                        'Group counts cannot establish class support or statistical adequacy.',
                        'Task B needs verified evidence, complete provenance topology and component isolation.',
                        'Scientific review, mechanism verification and release preflight remain mandatory.']}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--pairs', required=True, type=Path)
    parser.add_argument('--consensus', required=True, type=Path)
    parser.add_argument('--protocol', choices=sorted(_REQUIRED), default='full')
    args = parser.parse_args()
    print(json.dumps(audit(args.pairs, args.consensus, protocol=args.protocol), indent=2))


if __name__ == '__main__':
    main()
