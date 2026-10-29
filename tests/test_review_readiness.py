"""Synthetic structural feasibility is never empirical approval."""
import pandas as pd
import pytest

from src.data.review_readiness import audit
from src.data.evidence_schema import CONSENSUS_REQUIRED_COLUMNS


def tables(tmp_path, label=None):
    pairs = tmp_path / 'pairs.csv'
    labels = tmp_path / 'labels.csv'
    pd.DataFrame([{'herb_id': 'HERB_SYNTHETIC', 'drug_id': 'DRUG_SYNTHETIC'}]).to_csv(pairs, index=False)
    rows = [] if label is None else [dict(herb_id='HERB_SYNTHETIC', drug_id='DRUG_SYNTHETIC',
        consensus_label=label, source_name='SYNTHETIC ONLY',
        source_url_or_citation='SYNTHETIC FIXTURE NOT EVIDENCE', retrieval_date='2026-10-30')]
    pd.DataFrame(rows, columns=CONSENSUS_REQUIRED_COLUMNS).to_csv(labels, index=False)
    return pairs, labels


def test_empty_reference_blocks_every_regime(tmp_path):
    result = audit(*tables(tmp_path))
    assert result['eligible'] is False
    assert result['missing_reference_pairs'] == 1
    assert result['binary_reference_pairs'] == 0
    assert all(not value['necessary_group_count_met'] for value in result['capacity'].values())


def test_disputed_is_not_negative(tmp_path):
    result = audit(*tables(tmp_path, 'disputed'))
    assert result['disputed_reference_pairs'] == 1
    assert result['binary_reference_pairs'] == 0
    assert result['class_counts'] == {}


def test_binary_decision_still_does_not_authorize(tmp_path):
    result = audit(*tables(tmp_path, 'interaction'), protocol='baseline')
    assert result['eligible'] is False
    assert result['class_counts'] == {'interaction': 1}
    assert result['capacity']['pair']['required_nonempty_roles'] == 2


def test_unknown_protocol_rejected(tmp_path):
    with pytest.raises(ValueError, match='protocol'):
        audit(*tables(tmp_path), protocol='relaxed')


def test_invalid_label_rejected(tmp_path):
    with pytest.raises(ValueError):
        audit(*tables(tmp_path, 'unknown'))
