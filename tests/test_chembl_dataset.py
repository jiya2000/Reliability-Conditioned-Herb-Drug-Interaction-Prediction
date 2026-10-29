"""Synthetic API captures: no fitting, fabricated measurements or release grants."""
from pathlib import Path
import json
import pytest
from src.data.chembl_dataset import Capture, TARGETS, select_targets, normalize, trusted_url


def target_rows():
    return [dict(organism='Homo sapiens',target_type='SINGLE PROTEIN',
                 target_components=[dict(accession=a)],target_chembl_id='SYNTHETIC_'+n)
            for n,a in TARGETS.items()]


def test_target_mapping_requires_exact_human_single_protein():
    rows=target_rows()
    assert set(select_targets(rows))==set(TARGETS)
    with pytest.raises(ValueError):select_targets(rows[:-1])
    with pytest.raises(ValueError):select_targets(rows+[rows[0]])
    rows[0]['organism']='Mus musculus'
    with pytest.raises(ValueError):select_targets(rows)


@pytest.mark.parametrize('url',['http://www.ebi.ac.uk/chembl/api/data/activity.json',
    'https://evil.invalid/chembl/api/data/activity.json','https://www.ebi.ac.uk/other',
    'https://www.ebi.ac.uk@evil.invalid/chembl/api/data/'])
def test_pagination_cannot_leave_official_api(url):
    with pytest.raises(ValueError):trusted_url(url)


def test_all_pages_required_and_counts_stable(tmp_path,monkeypatch):
    capture=Capture(tmp_path)
    pages=iter([dict(page_meta=dict(total_count=2,next='/chembl/api/data/activity.json?offset=1'),activities=[{'id':1}]),
                dict(page_meta=dict(total_count=2,next=None),activities=[{'id':2}])])
    monkeypatch.setattr(capture,'get',lambda url:next(pages))
    assert len(capture.pages('activity',{}))==2
    monkeypatch.setattr(capture,'get',lambda url:dict(page_meta=dict(total_count=2,next=None),activities=[{'id':1}]))
    with pytest.raises(ValueError,match='Incomplete'):capture.pages('activity',{})


def test_document_structure_and_provenance_do_not_authorize():
    activity=dict(activity_id=1,molecule_chembl_id='SYNTHETIC',target_chembl_id='SYNTHETIC_TARGET',
        assay_chembl_id='SYNTHETIC_ASSAY',document_chembl_id='SYNTHETIC_DOC',standard_type='IC50',
        target_organism='Homo sapiens',standard_relation='=',standard_value='1000',standard_units='nM',
        canonical_smiles='CC',pchembl_value='6')
    docs={'SYNTHETIC_DOC':dict(doi='SYNTHETIC_NOT_EVIDENCE',pubmed_id=None)}
    row=normalize([activity],{'CYP3A4':'SYNTHETIC_TARGET'},docs)[0]
    assert row['exact_value_uM']==1 and row['structurally_usable_for_assay_research']
    assert row['task_a_eligible'] is False and 'consensus_label' not in row
    activity['assay_variant_mutation']='SYNTHETIC_MUTATION'
    assert not normalize([activity],{'CYP3A4':'SYNTHETIC_TARGET'},docs)[0]['structurally_usable_for_assay_research']
    with pytest.raises(ValueError,match='document'):normalize([activity],{'CYP3A4':'SYNTHETIC_TARGET'},{})


def test_capture_change_detected_before_use(tmp_path):
    from hashlib import sha256
    from src.data.chembl_dataset import verify_dataset
    (tmp_path/'raw').mkdir()
    (tmp_path/'raw/response_0000.json').write_text('{}')
    import pandas as pd
    pd.DataFrame([dict(activity_id=1)]).to_csv(tmp_path/'assay_records.csv',index=False)
    pd.DataFrame([dict(activity_id=1)]).to_csv(tmp_path/'assay_research_candidates.csv',index=False)
    manifest=dict(schema='hdi.chembl-assay-source/v1',complete_for_declared_queries=True,
        task_a_eligible=False,task_b_eligible=False,records=1,structurally_usable_records=1,
        captures=[dict(path='raw/response_0000.json',url='https://www.ebi.ac.uk/chembl/api/data/status.json',
                       sha256=sha256(b'{}').hexdigest())],
        outputs={name:sha256((tmp_path/name).read_bytes()).hexdigest() for name in
                 ['assay_records.csv','assay_research_candidates.csv']})
    (tmp_path/'manifest.json').write_text(json.dumps(manifest))
    assert verify_dataset(tmp_path)==manifest
    (tmp_path/'raw/response_0000.json').write_text('{"changed":true}')
    with pytest.raises(ValueError,match='changed'):verify_dataset(tmp_path)
