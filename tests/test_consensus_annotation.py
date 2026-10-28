"""Annotation workflow checks use synthetic pair identities and citations only."""
import json

import pandas as pd
import pytest

from src.data.consensus_annotation import initialize, finalize, read_csv
from src.data.evidence_schema import load_consensus_labels_table


@pytest.fixture
def assignment(tmp_path):
    path = tmp_path / 'synthetic.csv'
    pd.DataFrame([dict(herb_id='HERB_SYNTHETIC', drug_id='DRUG_SYNTHETIC',
                       outcome='interaction_present', consensus_label='interaction'),
                  dict(herb_id='HERB_SYNTHETIC', drug_id='DRUG_SYNTHETIC',
                       outcome='no_interaction_observed', consensus_label='no_interaction')]).to_csv(path, index=False)
    directory = tmp_path / 'assignment'
    assert initialize([path], directory) == 1
    return directory


def complete(directory, filename, reviewer, label):
    path = directory / filename
    frame = read_csv(path)
    frame.loc[0, ['consensus_label', 'source_name', 'source_url_or_citation',
                  'retrieval_date', 'reviewer_id', 'rationale', 'blinded_to_predictions',
                  'external_consensus_only', 'circularity_notes']] = [
        label, 'SYNTHETIC FIXTURE ONLY', 'Synthetic bibliography; not clinical evidence',
        '2026-10-29', reviewer, 'Synthetic rationale only', 'yes', 'yes',
        'Synthetic circularity review only']
    frame.to_csv(path, index=False)


def ready(directory, label='interaction'):
    complete(directory, 'reviewer_a.csv', 'synthetic_a', 'interaction')
    complete(directory, 'reviewer_b.csv', 'synthetic_b', 'no_interaction')
    complete(directory, 'adjudication.csv', 'synthetic_c', label)


def test_initialization_discards_findings_and_caller_labels(assignment, tmp_path):
    for name in ['reviewer_a.csv', 'reviewer_b.csv', 'adjudication.csv']:
        frame = read_csv(assignment / name)
        assert frame.consensus_label.tolist() == ['unresolved']
        assert frame.reviewer_id.tolist() == ['']
        assert frame.source_url_or_citation.tolist() == ['']
        assert 'outcome' not in frame
    audit = finalize(assignment, tmp_path / 'export')
    assert audit['exported_labels'] == 0 and audit['unresolved_pairs'] == 1
    labels, empty = load_consensus_labels_table(tmp_path / 'export/consensus_labels.csv')
    assert empty and labels.empty
    with pytest.raises(FileExistsError):
        initialize([assignment / 'pairs.csv'], assignment)


@pytest.mark.parametrize('label', ['interaction', 'no_interaction', 'disputed'])
def test_explicit_adjudication_exports_only_decision(assignment, tmp_path, label):
    ready(assignment, label)
    audit = finalize(assignment, tmp_path / 'export')
    labels, _ = load_consensus_labels_table(tmp_path / 'export/consensus_labels.csv', require_labels=True)
    assert labels.consensus_label.tolist() == [label]
    assert len(audit['reviewer_disagreements']) == 1
    assert audit['unresolved_pairs'] == 0
    assert json.loads((tmp_path / 'export/audit.json').read_text())['export_sha256']
    assert audit['double_reviewed_pairs'] == 1
    for name in ['manifest.json', 'pairs.csv', 'reviewer_a.csv', 'reviewer_b.csv', 'adjudication.csv']:
        assert (tmp_path / 'export/annotation_snapshot' / name).read_bytes() == (assignment / name).read_bytes()
    with pytest.raises(FileExistsError):
        finalize(assignment, tmp_path / 'export')


def test_disagreement_never_automatically_becomes_disputed(assignment, tmp_path):
    ready(assignment, 'unresolved')
    audit = finalize(assignment, tmp_path / 'export')
    assert audit['exported_labels'] == 0
    assert audit['unresolved_pairs'] == 1


@pytest.mark.parametrize('column,value', [
    ('reviewer_id', 'synthetic_a'), ('rationale', ''), ('circularity_notes', ''),
    ('blinded_to_predictions', 'no'), ('external_consensus_only', ''),
    ('source_url_or_citation', 'UNVERIFIED'), ('retrieval_date', '2026-02-30'),
    ('consensus_label', 'unknown'), ('herb_id', 'HERB_OTHER'),
])
def test_invalid_or_unreviewed_decisions_fail_closed(assignment, tmp_path, column, value):
    ready(assignment)
    path = assignment / 'adjudication.csv'
    frame = read_csv(path)
    frame.loc[0, column] = value
    frame.to_csv(path, index=False)
    with pytest.raises(ValueError):
        finalize(assignment, tmp_path / 'export')
    assert not (tmp_path / 'export').exists()


def test_missing_second_review_blocks_export(assignment, tmp_path):
    complete(assignment, 'reviewer_a.csv', 'synthetic_a', 'interaction')
    complete(assignment, 'adjudication.csv', 'synthetic_c', 'interaction')
    with pytest.raises(ValueError, match='reviewer_id'):
        finalize(assignment, tmp_path / 'export')


def test_frozen_assignment_tampering_rejected(assignment, tmp_path):
    with (assignment / 'pairs.csv').open('a') as stream:
        stream.write('HERB_OTHER,DRUG_OTHER\n')
    with pytest.raises(ValueError, match='assignment changed'):
        finalize(assignment, tmp_path / 'export')


def test_duplicate_review_rows_rejected(assignment, tmp_path):
    frame = read_csv(assignment / 'reviewer_a.csv')
    pd.concat([frame, frame]).to_csv(assignment / 'reviewer_a.csv', index=False)
    with pytest.raises(ValueError, match='duplicate'):
        finalize(assignment, tmp_path / 'export')


def test_incomplete_reviews_are_not_counted_as_disagreement(assignment, tmp_path):
    complete(assignment, 'reviewer_a.csv', 'synthetic_a', 'interaction')
    audit = finalize(assignment, tmp_path / 'export')
    assert audit['double_reviewed_pairs'] == 0
    assert audit['reviewer_disagreements'] == []


def test_documented_unresolved_review_does_not_need_invented_citation(assignment, tmp_path):
    ready(assignment)
    path = assignment / 'reviewer_b.csv'
    frame = read_csv(path)
    frame.loc[0, 'consensus_label'] = 'unresolved'
    frame.loc[0, ['source_name', 'source_url_or_citation', 'retrieval_date']] = ''
    frame.to_csv(path, index=False)
    audit = finalize(assignment, tmp_path / 'export')
    assert audit['exported_labels'] == 1
    assert audit['double_reviewed_pairs'] == 1


def test_cli_synthetic_round_trip(assignment, tmp_path, monkeypatch, capsys):
    from src.data.consensus_annotation import main
    output = tmp_path / 'cli_export'
    monkeypatch.setattr('sys.argv', ['annotation', 'finalize', '--directory', str(assignment),
                                    '--output', str(output)])
    main()
    assert 'Exported 0 adjudicated labels; 1 unresolved pairs.' in capsys.readouterr().out
