"""Prepare human-review work, without producing scientific decisions or eligibility."""
import argparse
import csv
from hashlib import sha256
from io import BytesIO, StringIO
import json
from pathlib import Path

import pandas as pd

from src.data.consensus_annotation import initialize, pair_keys
from src.data.mechanistic_curation import CURATION_COLUMNS, _canonical_identifier

REVIEW_COLUMNS = ['entity', 'target', 'review_status', 'reviewer_id',
                  'source_identifier', 'supporting_passage_location',
                  'experimental_context', 'units_checked', 'direction_checked',
                  'exposure_relevance', 'rationale', 'review_date']
SNAPSHOTS = ('pairs.csv', 'mechanisms.csv')


def _table(raw):
    header = next(csv.reader(StringIO(raw.decode('utf-8-sig'))), [])
    if not header or len(header) != len(set(header)):
        raise ValueError('Missing or duplicate CSV headers')
    return pd.read_csv(BytesIO(raw), dtype=str, keep_default_na=False)


def prepare(pairs_path, mechanisms_path, output):
    """Read each explicit input once; freeze those bytes before deriving assignments.

    Mechanism records are candidates, not validated evidence. No canonical data,
    registry entry, reviewer attestation or production release is updated.
    """
    raw = {'pairs.csv': Path(pairs_path).read_bytes(),
           'mechanisms.csv': Path(mechanisms_path).read_bytes()}
    pairs = pair_keys(_table(raw['pairs.csv']))
    if pairs.empty:
        raise ValueError('No candidate pairs')
    mechanisms = _table(raw['mechanisms.csv'])
    if not set(CURATION_COLUMNS) <= set(mechanisms.columns):
        raise ValueError('Mechanism input requires the existing curation columns')
    keys = mechanisms[['entity', 'target']].copy()
    for column in keys:
        keys[column] = keys[column].map(
            lambda value: _canonical_identifier(value, kind=column))
    if keys.duplicated().any():
        raise ValueError('Duplicate entity-target candidates require resolution')
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    snapshots = output / 'coordinator_snapshots'
    snapshots.mkdir()
    for name, contents in raw.items():
        (snapshots / name).write_bytes(contents)
    # Only identifiers enter blinded consensus assignments; never supplied labels.
    pairs.to_csv(output / 'assignment_pairs.csv', index=False)
    initialize([output / 'assignment_pairs.csv'], output / 'consensus_review')
    review = keys.reindex(columns=REVIEW_COLUMNS, fill_value='')
    review['review_status'] = 'pending'
    review.to_csv(output / 'mechanism_review.csv', index=False)
    protected = {f'coordinator_snapshots/{name}': sha256(raw[name]).hexdigest()
                 for name in SNAPSHOTS}
    protected['assignment_pairs.csv'] = sha256((output / 'assignment_pairs.csv').read_bytes()).hexdigest()
    manifest = {'version': 1, 'purpose': 'human_review_preparation_only',
                'pair_count': len(pairs), 'mechanism_candidates': len(keys),
                'protected_sha256': protected,
                'limitations': 'Hashes detect accidental changes, not malicious manifest replacement. '
                'Pending workbooks and source records are not verified evidence. '
                'This packet cannot authorize training or certify reviewer independence.'}
    (output / 'packet_manifest.json').write_text(json.dumps(manifest, indent=2) + '\n')
    return manifest


def verify(directory):
    """Check frozen preparation inputs, leaving editable review workbooks alone."""
    directory = Path(directory)
    manifest = json.loads((directory / 'packet_manifest.json').read_text())
    expected = {f'coordinator_snapshots/{name}' for name in SNAPSHOTS} | {'assignment_pairs.csv'}
    if (manifest.get('version') != 1 or
            manifest.get('purpose') != 'human_review_preparation_only' or
            set(manifest.get('protected_sha256', {})) != expected):
        raise ValueError('Unsupported review packet manifest')
    for name in sorted(expected):
        path = directory / name
        if path.is_symlink() or sha256(path.read_bytes()).hexdigest() != manifest['protected_sha256'][name]:
            raise ValueError(f'Frozen review input changed: {name}')
    return manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest='command', required=True)
    create = commands.add_parser('prepare')
    create.add_argument('--pairs', required=True, type=Path)
    create.add_argument('--mechanisms', required=True, type=Path)
    create.add_argument('--output', required=True, type=Path)
    check = commands.add_parser('verify')
    check.add_argument('--directory', required=True, type=Path)
    args = parser.parse_args()
    try:
        result = (prepare(args.pairs, args.mechanisms, args.output)
                  if args.command == 'prepare' else verify(args.directory))
        print(json.dumps(result, indent=2))
    except (ValueError, OSError, KeyError) as exc:
        parser.exit(1, f'Review preparation failed: {exc}\n')


if __name__ == '__main__':
    main()
