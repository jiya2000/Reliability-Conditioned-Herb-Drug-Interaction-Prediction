"""Versioned expanded curation scope; not an approved model feature contract."""
from collections import Counter
from pathlib import Path
import argparse
import json

from src.data.review_packet import _table
from src.data.mechanistic_curation import CURATION_COLUMNS, validate_curation_dataframe

PANEL_VERSION = 'expanded_curation_v1'
FDA_SOURCE = 'https://www.fda.gov/drugs/drug-interactions-labeling/drug-development-and-drug-interactions-table-substrates-inhibitors-and-inducers'
# FDA Table 1-1 and Table 3-1. This is a curation priority list, not exhaustive biology.
ENZYMES = ('CYP1A2', 'CYP2B6', 'CYP2C8', 'CYP2C9', 'CYP2C19', 'CYP2D6', 'CYP3A4')
TRANSPORTERS = ('ABCB1', 'ABCG2', 'SLCO1B1', 'SLCO1B3', 'SLC22A6',
                'SLC22A8', 'SLC22A2', 'SLC47A1', 'SLC47A2')
PANEL_TARGETS = ENZYMES + TRANSPORTERS
ALIASES = dict(zip(TRANSPORTERS, ('P-gp', 'BCRP', 'OATP1B1', 'OATP1B3',
                                'OAT1', 'OAT3', 'OCT2', 'MATE1', 'MATE2-K')))


def prepare(curation_path, output):
    """Add unmeasured review candidates for existing entities, never inferred values.

    Existing curation is copied for audit only, not promoted to verified evidence.
    New records deliberately carry UNVERIFIED provenance and blank measurements.
    The primary extractor/allowlist and canonical dataset are never changed.
    """
    raw = Path(curation_path).read_bytes()
    curation = _table(raw)
    validate_curation_dataframe(curation)
    entities = sorted(set(curation.entity.str.strip()))
    existing = set(zip(curation.entity.str.strip(), curation.target.str.strip()))
    rows = []
    for entity in entities:
        for target in PANEL_TARGETS:
            if (entity, target) not in existing:
                row = dict.fromkeys(CURATION_COLUMNS, '')
                row.update(entity=entity, target=target, status='not_tested',
                           source='UNVERIFIED', citation='UNVERIFIED',
                           notes='Expanded curation candidate only; no evidence or effect inferred.')
                rows.append(row)
    import pandas as pd
    candidates = pd.DataFrame(rows, columns=CURATION_COLUMNS)
    validate_curation_dataframe(candidates)
    coverage = []
    for target in PANEL_TARGETS:
        counts = Counter(curation.loc[curation.target.str.strip().eq(target), 'status'])
        coverage.append(dict(target=target, kind='enzyme' if target in ENZYMES else 'transporter',
                             alias=ALIASES.get(target, target), existing_status_counts=dict(counts),
                             missing_entity_rows=sum((e, target) not in existing for e in entities)))
    from hashlib import sha256
    report = dict(panel_version=PANEL_VERSION, source_url=FDA_SOURCE,
                  source_scope='General CYP/transporter relevance, not herb-specific evidence or model approval.',
                  input_sha256=sha256(raw).hexdigest(), entities=len(entities),
                  new_unverified_rows=len(candidates), coverage=coverage,
                  training_eligible=False,
                  limitations='Presence/status counts do not verify sources. Unknown is not no effect. '
                  'Renal/uptake/efflux effects need compartment-specific interpretation; '
                  'no feature expansion or clinical exposure conversion is authorized.')
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    (output/'input_snapshot.csv').write_bytes(raw)
    candidates.to_csv(output/'additional_candidates.csv', index=False)
    (output/'coverage.json').write_text(json.dumps(report, indent=2)+'\n')
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--curation', required=True, type=Path)
    parser.add_argument('--output', required=True, type=Path)
    args = parser.parse_args()
    print(json.dumps(prepare(args.curation, args.output), indent=2))


if __name__ == '__main__':
    main()
