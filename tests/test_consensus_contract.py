"""Synthetic contract fixtures only; no clinical labels or real citations are curated."""
import pandas as pd
import pytest

from src.data.evidence_schema import (
    CONSENSUS_REQUIRED_COLUMNS, ConsensusLabelsNotPopulatedError,
    ConsensusPairRecord, load_consensus_labels_table, validate_consensus_labels_table,
)
from src.models.aggregation import load_consensus_labels, train_reliability_model_entrypoint
from src.models.aggregation_model import load_consensus_labels as alternate_loader


def record(**changes):
    row = dict(herb_id='HERB_SYNTHETIC', drug_id='DRUG_SYNTHETIC',
               consensus_label='interaction', source_name='SYNTHETIC FIXTURE ONLY',
               source_url_or_citation='SYNTHETIC bibliography fixture; not clinical evidence',
               retrieval_date='2026-10-28')
    return dict(row, **changes)


@pytest.mark.parametrize('label', ['interaction', 'no_interaction', 'disputed'])
def test_valid_records(label, tmp_path):
    row = record(consensus_label=label)
    assert ConsensusPairRecord(**row).validate() == (True, [])
    path = tmp_path / 'consensus.csv'
    pd.DataFrame([row]).to_csv(path, index=False)
    for loader in (load_consensus_labels, alternate_loader):
        assert loader(path).iloc[0]['consensus_label'] == label


@pytest.mark.parametrize('changes', [
    {'consensus_label': 'positive'}, {'consensus_label': 'unknown'},
    {'consensus_label': 0}, {'herb_id': 'DRUG_SYNTHETIC'},
    {'herb_id': 'HERB_'}, {'drug_id': 'drug_synthetic'},
    {'drug_id': 'DRUG_BAD ID'}, {'source_name': 'UNVERIFIED'},
    {'source_url_or_citation': 'TBD'}, {'source_url_or_citation': 'https://'},
    {'retrieval_date': '2026-02-30'},
    {'source_url_or_citation': 'https://[broken'},
    {'source_url_or_citation': 'https://fixture.invalid:bad'},
    {'source_url_or_citation': 'PMID: invalid'},
    {'source_url_or_citation': 'doi: invalid'},
])
def test_invalid_records(changes, tmp_path):
    frame = pd.DataFrame([record(**changes)])
    assert not validate_consensus_labels_table(frame)[0]
    path = tmp_path / 'consensus.csv'
    frame.to_csv(path, index=False)
    for loader in (load_consensus_labels, alternate_loader):
        with pytest.raises(ValueError):
            loader(path)


@pytest.mark.parametrize('column', CONSENSUS_REQUIRED_COLUMNS)
@pytest.mark.parametrize('missing', [None, float('nan'), pd.NA, '', '  ', 'NaN', 'None'])
def test_missing_values(column, missing):
    assert not validate_consensus_labels_table(pd.DataFrame([record(**{column: missing})]))[0]


@pytest.mark.parametrize('column', CONSENSUS_REQUIRED_COLUMNS)
def test_required_columns_even_when_empty(column):
    frame = pd.DataFrame(columns=CONSENSUS_REQUIRED_COLUMNS).drop(columns=column)
    assert not validate_consensus_labels_table(frame)[0]


@pytest.mark.parametrize('second', [record(), record(consensus_label='no_interaction'),
                                     record(herb_id=' HERB_SYNTHETIC ')])
def test_duplicate_pairs(second):
    valid, errors = validate_consensus_labels_table(pd.DataFrame([record(), second]))
    assert not valid
    assert any('Duplicate' in error for error in errors)


def test_empty_table_and_label_consumers(tmp_path):
    frame = pd.DataFrame(columns=CONSENSUS_REQUIRED_COLUMNS)
    assert validate_consensus_labels_table(frame) == (True, [])
    path = tmp_path / 'empty.csv'
    frame.to_csv(path, index=False)
    loaded, unavailable = load_consensus_labels_table(path)
    assert loaded.empty and unavailable
    with pytest.raises(ConsensusLabelsNotPopulatedError):
        load_consensus_labels_table(path, require_labels=True)
    for loader in (load_consensus_labels, alternate_loader, train_reliability_model_entrypoint):
        with pytest.raises(ConsensusLabelsNotPopulatedError):
            loader(path)


def test_unlabeled_candidates_are_rejected_not_dropped_or_made_negative(tmp_path):
    frame = pd.DataFrame([record(), record(herb_id='HERB_UNSTUDIED', consensus_label='')])
    path = tmp_path / 'candidates.csv'
    frame.to_csv(path, index=False)
    for loader in (load_consensus_labels, alternate_loader):
        with pytest.raises(ValueError, match='consensus_label'):
            loader(path)
    assert frame.iloc[1]['consensus_label'] == ''


def test_missing_and_headerless_files_are_not_empty_consensus(tmp_path):
    with pytest.raises(FileNotFoundError):
        load_consensus_labels_table(tmp_path / 'missing.csv')
    path = tmp_path / 'blank.csv'
    path.write_text('')
    with pytest.raises(pd.errors.EmptyDataError):
        load_consensus_labels_table(path)


def test_normalization_and_optional_metadata(tmp_path):
    row = record(herb_id=' HERB_SYNTHETIC ', consensus_label=' DISPUTED ',
                 notes=None, severity=None, direction=None)
    path = tmp_path / 'normalized.csv'
    pd.DataFrame([row]).to_csv(path, index=False)
    loaded, unavailable = load_consensus_labels_table(path, require_labels=True)
    assert not unavailable
    assert loaded.iloc[0]['herb_id'] == 'HERB_SYNTHETIC'
    assert loaded.iloc[0]['consensus_label'] == 'disputed'


def test_duplicate_columns_rejected():
    frame = pd.DataFrame(columns=[*CONSENSUS_REQUIRED_COLUMNS, 'herb_id'])
    assert not validate_consensus_labels_table(frame)[0]
