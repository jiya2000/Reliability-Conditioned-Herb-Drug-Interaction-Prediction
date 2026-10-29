"""Synthetic discussion records must never become scientific evidence."""
import copy
import json
from pathlib import Path
import pandas as pd
import pytest
from src.data.discussion_signals import quarantine, prepare
from src.data.evidence_schema import VALID_EVIDENCE_TYPES
from src.training.task_b import validate_studies


def record(key='SYNTHETIC_A'):
    return dict(record_id=key,platform='SYNTHETIC',source_url='https://example.invalid/posts/'+key,
        herb_id='HERB_SYNTHETIC',drug_id='DRUG_SYNTHETIC',event_term='synthetic_event',
        retrieval_date='2026-10-30',thread_id='SYNTHETIC_THREAD',repost_of='',
        collection_permission_ref='SYNTHETIC NOT APPROVAL',privacy_review_ref='SYNTHETIC NOT APPROVAL')


def test_quarantine_never_becomes_task_b_or_reference():
    rows=[record(),record('SYNTHETIC_B')]
    original=copy.deepcopy(rows)
    result=quarantine(rows)
    assert rows==original
    assert result[0]['signal_component']==result[1]['signal_component']
    assert all(r['task_b_eligible'] is False and r['consensus_label'] is None for r in result)
    assert all(r['evidence_type'] not in VALID_EVIDENCE_TYPES for r in result)
    with pytest.raises(ValueError):
        validate_studies(pd.DataFrame(result))


@pytest.mark.parametrize('field,value', [('source_url','http://example.invalid/a'),
    ('source_url','https://user:password@example.invalid/a'),
    ('source_url','https://example.invalid/a?user=private'),('retrieval_date','2026-02-30'),
    ('collection_permission_ref',''),('privacy_review_ref',''),('repost_of','MISSING'),
    ('herb_id','DRUG_SYNTHETIC')])
def test_invalid_metadata_fails(field,value):
    row=record();row[field]=value
    with pytest.raises(ValueError): quarantine([row])


@pytest.mark.parametrize('extra',['raw_text','username','study_finding','reliability_score','eligible'])
def test_no_personal_fields_or_forged_authority(extra):
    row=record();row[extra]='forged'
    with pytest.raises(ValueError): quarantine([row])


def test_duplicate_and_repost_components():
    a,b=record(),record('SYNTHETIC_B')
    with pytest.raises(ValueError): quarantine([a,a])
    b['platform']='OTHER_SYNTHETIC';b['thread_id']='OTHER_THREAD';b['repost_of']=a['record_id']
    result=quarantine([b,a])
    assert result[0]['signal_component']==result[1]['signal_component']


def test_explicit_file_intake_and_no_overwrite(tmp_path):
    source=tmp_path/'input.json';source.write_text(json.dumps([record()]))
    output=tmp_path/'output'
    assert prepare(source,output)['task_b_eligible'] is False
    with pytest.raises(FileExistsError):prepare(source,output)
    source.write_text('[{"record_id":"x","record_id":"y"}]')
    with pytest.raises(ValueError,match='Duplicate'): prepare(source,tmp_path/'bad')
    assert not (tmp_path/'bad').exists()


def test_previous_public_artifacts_still_integrate():
    from src.data.review_packet import verify
    from src.data.review_readiness import audit
    from src.data.mechanistic_curation import load_mechanistic_curation,is_curation_verified
    from src.features.task_a_contract import TASK_A_TARGETS
    root=Path(__file__).resolve().parents[1]
    batch=root/'docs/review_batches/public_pilot_20261030'
    assert verify(batch/'packet')['pair_count']==3
    assert audit(batch/'candidate_pairs.csv',root/'data/benchmark/consensus_labels.csv')==json.loads((batch/'readiness.json').read_text())
    panel=root/'docs/review_batches/expanded_panel_20261030'
    candidates=load_mechanistic_curation(panel/'additional_candidates.csv')
    assert is_curation_verified(candidates)[0] is False
    coverage=json.loads((panel/'coverage.json').read_text())
    from hashlib import sha256
    assert coverage['input_sha256']==sha256((panel/'input_snapshot.csv').read_bytes()).hexdigest()
    assert len(TASK_A_TARGETS)==6
