import pytest
from src.data.chembl_assay_review import review_rows


def activity(**changes):
    row=dict(activity_id=1,molecule_chembl_id='SYNTHETIC',target_chembl_id='SYNTHETIC',
        assay_chembl_id='SYNTHETIC',document_chembl_id='SYNTHETIC',standard_type='IC50',
        target_organism='Homo sapiens',standard_relation='=',standard_value='1000',standard_units='nM')
    row.update(changes)
    return row


def test_conversion_is_not_feature_approval():
    row=review_rows({'activities':[activity()]})[0]
    assert row['exact_value_uM']==1
    assert row['task_a_eligible'] is False
    assert 'entity' not in row


@pytest.mark.parametrize('change',[dict(standard_relation='>'),dict(standard_units='mg/ml'),
    dict(standard_value='NaN'),dict(standard_value='-1'),dict(target_organism='Mus musculus'),
    dict(standard_type='Kd'),dict(document_chembl_id=None),dict(potential_duplicate=1)])
def test_unsupported_or_flagged_records_not_exact_features(change):
    row=review_rows({'activities':[activity(**change)]})[0]
    assert row['exact_value_uM'] is None and row['review_flags']
    assert row['task_a_eligible'] is False


def test_duplicate_assay_identity_rejected():
    with pytest.raises(ValueError):review_rows({'activities':[activity(),activity()]})
