"""Quarantined discussion-signal intake; not Task B scientific evidence."""
import argparse
from datetime import date
from hashlib import sha256
import json
from pathlib import Path
from urllib.parse import urlsplit, urlunsplit

from src.data.mechanistic_curation import _canonical_identifier

FIELDS = {'record_id', 'platform', 'source_url', 'herb_id', 'drug_id',
          'event_term', 'retrieval_date', 'thread_id', 'repost_of',
          'collection_permission_ref', 'privacy_review_ref'}


def quarantine(records):
    """Validate minimal metadata; no scraping, causal labels, weights or promotion.

    Permission/privacy references are caller declarations requiring external review.
    Raw narratives and user identifiers are intentionally outside this schema.
    Same-thread and explicitly linked reposts form review components, not counts
    of independent patients. Unlinked duplicates cannot be detected reliably.
    """
    if not isinstance(records, list):
        raise ValueError('Expected a list of discussion metadata records')
    clean, seen = [], set()
    for row in records:
        if not isinstance(row, dict) or set(row) != FIELDS:
            raise ValueError('Discussion fields missing or unsupported; no labels, raw text or user identifiers allowed')
        if any(not isinstance(v, str) or v != v.strip() for v in row.values()):
            raise ValueError('Discussion fields must be canonical strings')
        if any(not row[k] for k in FIELDS - {'repost_of'}):
            raise ValueError('Incomplete discussion metadata')
        for field in ('herb_id', 'drug_id'):
            _canonical_identifier(row[field], kind='HERB_' if field=='herb_id' else 'DRUG_')
        date.fromisoformat(row['retrieval_date'])
        url = urlsplit(row['source_url'])
        if url.scheme != 'https' or not url.hostname or url.username or url.password or url.query or url.fragment:
            raise ValueError('Use an HTTPS source URL without credentials, query or fragment')
        canonical_url = urlunsplit(('https', url.netloc.lower(), url.path.rstrip('/'), '', ''))
        if row['record_id'] in seen:
            raise ValueError('Duplicate discussion record_id')
        seen.add(row['record_id'])
        clean.append(dict(row, source_url=canonical_url))
    parent = {row['record_id']: row['record_id'] for row in clean}
    def find(key):
        while parent[key] != key:
            key = parent[key]
        return key
    def union(a, b):
        a, b = find(a), find(b)
        parent[max(a, b)] = min(a, b)
    groups = {}
    for row in clean:
        if row['repost_of']:
            if row['repost_of'] not in parent:
                raise ValueError('Unresolved repost provenance')
            union(row['record_id'], row['repost_of'])
        for key in [('thread', row['platform'], row['thread_id']), ('url', row['source_url'])]:
            previous = groups.setdefault(key, row['record_id'])
            union(previous, row['record_id'])
    return [dict(row, signal_component=sha256(find(row['record_id']).encode()).hexdigest(),
                 evidence_type='social_signal', review_status='quarantined',
                 task_b_eligible=False, consensus_label=None,
                 limitations='Unverified self-report metadata; no causal inference, reliability weight or patient count.')
            for row in clean]


def prepare(input_path, output):
    raw = Path(input_path).read_bytes()
    def unique(pairs):
        result = {}
        for key, value in pairs:
            if key in result:
                raise ValueError('Duplicate JSON field')
            result[key] = value
        return result
    records = json.loads(raw, object_pairs_hook=unique)
    result = quarantine(records)
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    # No input snapshot: potentially sensitive rejected/raw fields are never copied.
    (output/'quarantined_signals.json').write_text(json.dumps(result, indent=2)+'\n')
    report = dict(scope='signal_triage_only', input_sha256=sha256(raw).hexdigest(),
                  records=len(result), task_b_eligible=False,
                  source_authenticity='not_verified', permissions='external_review_required')
    (output/'intake_manifest.json').write_text(json.dumps(report, indent=2)+'\n')
    return report


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input', required=True, type=Path)
    parser.add_argument('--output', required=True, type=Path)
    args=parser.parse_args()
    print(json.dumps(prepare(args.input,args.output), indent=2))


if __name__ == '__main__':
    main()
