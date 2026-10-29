"""Synthetic preparation tests; no real evidence or production execution."""
import json

import pandas as pd
import pytest

from src.data.mechanistic_curation import CURATION_COLUMNS
from src.data.review_packet import prepare, verify


@pytest.fixture
def inputs(tmp_path):
    pairs = tmp_path / 'pairs.csv'
    pd.DataFrame([{'herb_id': 'HERB_SYNTHETIC', 'drug_id': 'DRUG_SYNTHETIC',
                   'consensus_label': 'interaction', 'prediction': '1'}]).to_csv(pairs, index=False)
    mechanisms = tmp_path / 'mechanisms.csv'
    row = dict.fromkeys(CURATION_COLUMNS, '')
    row.update(entity='HERB_SYNTHETIC', target='CYP3A4', status='not_tested')
    pd.DataFrame([row]).to_csv(mechanisms, index=False)
    return pairs, mechanisms


def test_pending_blinded_packet_and_exact_snapshots(inputs, tmp_path):
    pairs, mechanisms = inputs
    output = tmp_path / 'packet'
    result = prepare(pairs, mechanisms, output)
    assert verify(output) == result
    assert result['purpose'] == 'human_review_preparation_only'
    assert 'eligible' not in result
    for path in inputs:
        assert (output / 'coordinator_snapshots' / path.name).read_bytes() == path.read_bytes()
    review = pd.read_csv(output / 'consensus_review/reviewer_a.csv', keep_default_na=False)
    assert review.consensus_label.tolist() == ['unresolved']
    assert review.reviewer_id.tolist() == ['']
    assert 'prediction' not in review
    assert pd.read_csv(output / 'mechanism_review.csv').review_status.tolist() == ['pending']
    # Originals can change after preparation; verification uses the frozen bytes.
    pairs.write_text('changed')
    assert verify(output) == result
    with pytest.raises(FileExistsError):
        prepare(output / 'coordinator_snapshots/pairs.csv', mechanisms, output)


@pytest.mark.parametrize('name', ['coordinator_snapshots/pairs.csv',
                                 'coordinator_snapshots/mechanisms.csv', 'assignment_pairs.csv'])
def test_frozen_input_mutation_detected(inputs, tmp_path, name):
    output = tmp_path / 'packet'
    prepare(*inputs, output)
    with (output / name).open('a') as handle:
        handle.write('\n')
    with pytest.raises(ValueError, match='changed'):
        verify(output)


def test_manifest_cannot_redirect_verification(inputs, tmp_path):
    output = tmp_path / 'packet'
    prepare(*inputs, output)
    path = output / 'packet_manifest.json'
    manifest = json.loads(path.read_text())
    manifest['protected_sha256'] = {'../outside': 'fake'}
    path.write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match='manifest'):
        verify(output)


@pytest.mark.parametrize('bad', ['duplicate_header', 'missing_column', 'duplicate_entity'])
def test_malformed_candidates_do_not_create_output(inputs, tmp_path, bad):
    pairs, mechanisms = inputs
    if bad == 'duplicate_header':
        pairs.write_text('herb_id,drug_id,drug_id\nHERB_SYNTHETIC,DRUG_SYNTHETIC,DRUG_SYNTHETIC\n')
    elif bad == 'missing_column':
        mechanisms.write_text('entity,target\nHERB_SYNTHETIC,CYP3A4\n')
    else:
        frame = pd.read_csv(mechanisms)
        pd.concat([frame, frame]).to_csv(mechanisms, index=False)
    output = tmp_path / 'packet'
    with pytest.raises(ValueError):
        prepare(pairs, mechanisms, output)
    assert not output.exists()
