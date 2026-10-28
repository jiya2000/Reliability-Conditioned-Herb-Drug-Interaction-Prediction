"""Human-only consensus annotation staging; never derives labels from studies."""
import argparse
from hashlib import sha256
import json
from pathlib import Path
from shutil import copyfile

import pandas as pd

from src.data.evidence_schema import (
    CONSENSUS_REQUIRED_COLUMNS, ConsensusPairRecord, VALID_CONSENSUS_LABELS,
    validate_consensus_labels_table,
)

KEYS = ['herb_id', 'drug_id']
COLUMNS = [*CONSENSUS_REQUIRED_COLUMNS, 'reviewer_id', 'rationale',
           'blinded_to_predictions', 'external_consensus_only', 'circularity_notes']
UNRESOLVED = 'unresolved'  # Staging status only; never a training target.


def read_csv(path):
    return pd.read_csv(path, dtype=str, keep_default_na=False)


def pair_keys(frame):
    """Project identifiers only, discarding outcomes, names and caller labels."""
    if not set(KEYS) <= set(frame):
        raise ValueError('Candidate input requires herb_id and drug_id')
    pairs = frame[KEYS].copy()
    for column in KEYS:
        pairs[column] = pairs[column].map(lambda v: str(v).strip())
    for row in pairs.to_dict('records'):
        # Reuse identifier rules without treating these dummy fields as labels.
        valid, errors = ConsensusPairRecord(**row, consensus_label='interaction',
            source_name='identifier validation only', source_url_or_citation='identifier validation only',
            retrieval_date='2000-01-01').validate()
        if not valid:
            raise ValueError('; '.join(errors))
    return pairs.drop_duplicates().sort_values(KEYS).reset_index(drop=True)


def digest(path):
    return sha256(Path(path).read_bytes()).hexdigest()


def initialize(inputs, output):
    """Create new, separate blinded workbooks; refuse to overwrite any directory."""
    pairs = pair_keys(pd.concat([pair_keys(read_csv(p)) for p in inputs], ignore_index=True))
    if pairs.empty:
        raise ValueError('No candidate pairs')
    template = pairs.reindex(columns=COLUMNS, fill_value='')
    template['consensus_label'] = UNRESOLVED
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    for name in ['reviewer_a.csv', 'reviewer_b.csv', 'adjudication.csv']:
        template.to_csv(output / name, index=False)
    pairs.to_csv(output / 'pairs.csv', index=False)
    (output / 'manifest.json').write_text(json.dumps({
        'workflow_version': 1, 'pair_count': len(pairs),
        'pairs_sha256': digest(output / 'pairs.csv'),
        'inputs': [{'path': str(p), 'sha256': digest(p)} for p in inputs],
        'policy': 'All pairs assigned to two independent reviewers; explicit adjudication required.'
    }, indent=2) + '\n')
    return len(pairs)


def workbook(path, pairs):
    frame = read_csv(path)
    if list(frame.columns) != COLUMNS:
        raise ValueError(f'{path}: expected annotation columns {COLUMNS}')
    frame = frame.apply(lambda column: column.str.strip())
    if frame.duplicated(KEYS).any():
        raise ValueError(f'{path}: duplicate pair')
    if set(map(tuple, frame[KEYS].values)) != set(map(tuple, pairs[KEYS].values)):
        raise ValueError(f'{path}: pair set differs from frozen assignment')
    if not frame.consensus_label.isin(VALID_CONSENSUS_LABELS | {UNRESOLVED}).all():
        raise ValueError(f'{path}: unsupported label (use unresolved for unknown cases)')
    for row in frame.to_dict('records'):
        if row['consensus_label'] != UNRESOLVED:
            validate_decision(row)
    return frame.set_index(KEYS)


def validate_decision(row):
    for name in ['reviewer_id', 'rationale', 'circularity_notes']:
        if not row[name] or row[name].lower() in {'tbd', 'todo', 'unknown', 'unverified', 'n/a'}:
            raise ValueError(f'Decision requires {name}')
    for name in ['blinded_to_predictions', 'external_consensus_only']:
        if row[name] != 'yes':
            raise ValueError(f'Decision requires {name}=yes')
    if row['consensus_label'] != UNRESOLVED:
        valid, errors = ConsensusPairRecord(**{c: row[c] for c in CONSENSUS_REQUIRED_COLUMNS}).validate()
        if not valid:
            raise ValueError('; '.join(errors))


def finalize(directory, output):
    """Export explicitly adjudicated decisions plus hashes; never majority-vote labels.

    All pairs have two review slots. A resolved export requires documented independent
    responses from both, even if a reviewer explicitly leaves the case unresolved.
    """
    directory, output = Path(directory), Path(output)
    manifest = json.loads((directory / 'manifest.json').read_text())
    if digest(directory / 'pairs.csv') != manifest['pairs_sha256']:
        raise ValueError('Frozen pair assignment changed')
    pairs = pair_keys(read_csv(directory / 'pairs.csv'))
    reviews = [workbook(directory / name, pairs) for name in
               ['reviewer_a.csv', 'reviewer_b.csv', 'adjudication.csv']]
    rows, pending, disagreements = [], [], []
    double_reviewed = 0
    for key in map(tuple, pairs.values):
        a, b, decision = [dict(zip(KEYS, key), **frame.loc[key].to_dict()) for frame in reviews]
        completed = True
        for review in [a, b]:
            try:
                validate_decision(review)
            except ValueError:
                completed = False
        completed = completed and a['reviewer_id'] != b['reviewer_id']
        double_reviewed += int(completed)
        if completed and a['consensus_label'] != b['consensus_label']:
            disagreements.append(dict(zip(KEYS, key)))
        if decision['consensus_label'] == UNRESOLVED:
            pending.append(dict(zip(KEYS, key)))
            continue
        for row in [a, b, decision]:
            validate_decision(row)
        if len({a['reviewer_id'], b['reviewer_id'], decision['reviewer_id']}) != 3:
            raise ValueError('Two independent reviewers and a separate adjudicator are required')
        rows.append(dict(zip(KEYS, key), **{c: decision[c] for c in COLUMNS if c not in KEYS}))
    export = pd.DataFrame(rows, columns=COLUMNS)
    valid, errors = validate_consensus_labels_table(export)
    if not valid:
        raise ValueError('; '.join(errors))
    # New output directory only: never edits the canonical consensus table.
    output.mkdir(parents=True, exist_ok=False)
    snapshot = output / 'annotation_snapshot'
    snapshot.mkdir()
    for name in ['manifest.json', 'pairs.csv', 'reviewer_a.csv', 'reviewer_b.csv', 'adjudication.csv']:
        copyfile(directory / name, snapshot / name)
    export.to_csv(output / 'consensus_labels.csv', index=False)
    pd.DataFrame(pending, columns=KEYS).to_csv(output / 'unresolved_pairs.csv', index=False)
    audit = {'exported_labels': len(export), 'unresolved_pairs': len(pending),
             'double_reviewed_pairs': double_reviewed,
             'reviewer_disagreements': disagreements,
             'inputs': {name: digest(directory / name) for name in
                       ['manifest.json', 'pairs.csv', 'reviewer_a.csv', 'reviewer_b.csv', 'adjudication.csv']},
             'export_sha256': digest(output / 'consensus_labels.csv'),
             'limitations': 'Structural checks and reviewer attestations do not authenticate sources or independence. '
                            'No scientific validation or agreement statistic is inferred.'}
    (output / 'audit.json').write_text(json.dumps(audit, indent=2) + '\n')
    return audit


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest='command', required=True)
    init = commands.add_parser('init')
    init.add_argument('--pairs', nargs='+', required=True, type=Path)
    init.add_argument('--output', required=True, type=Path)
    finish = commands.add_parser('finalize')
    finish.add_argument('--directory', required=True, type=Path)
    finish.add_argument('--output', required=True, type=Path)
    args = parser.parse_args()
    try:
        if args.command == 'init':
            print(f'Created {initialize(args.pairs, args.output)} unresolved pair assignments; no labels assigned.')
        else:
            audit = finalize(args.directory, args.output)
            print(f"Exported {audit['exported_labels']} adjudicated labels; {audit['unresolved_pairs']} unresolved pairs.")
    except (ValueError, OSError, KeyError) as exc:
        parser.exit(1, f'Annotation workflow failed: {exc}\n')


if __name__ == '__main__':
    main()
