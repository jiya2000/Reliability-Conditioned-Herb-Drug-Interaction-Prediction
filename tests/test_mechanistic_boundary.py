"""Synthetic fixtures only: provenance strings below are not scientific citations."""
import numpy as np
import pandas as pd
import pytest
from src.data.mechanistic_curation import (
    CANONICAL_TARGETS, CURATION_COLUMNS, UnverifiedCurationError,
    extract_curated_feature_matrix, load_mechanistic_curation,
)


def record(entity, target='CYP3A4', status='measured', **values):
    row = dict.fromkeys(CURATION_COLUMNS, '')
    row.update(entity=entity, target=target, status=status,
               source='SYNTHETIC_FIXTURE', citation='SYNTHETIC_FIXTURE')
    row.update(values)
    return row


def pairs():
    return pd.DataFrame([{'herb_id': 'HERB_FIXTURE', 'drug_id': 'DRUG_FIXTURE'}])


def extract(*rows, **kwargs):
    return extract_curated_feature_matrix(pairs(), pd.DataFrame(rows), **kwargs)


def test_valid_extraction_order_dtype_and_summary():
    rows = []
    for target in CANONICAL_TARGETS:
        rows += [record('HERB_FIXTURE', target, 'tested_no_effect'),
                 record('DRUG_FIXTURE', target, value_substrate_flag=1)]
    rows[0] = record('HERB_FIXTURE', value_ic50_uM=2)
    rows[1] = record('DRUG_FIXTURE', value_fm=0.4, value_substrate_flag=1)
    result = extract(*rows)
    assert result.shape == (1, 10)
    assert result.dtype == np.float32
    np.testing.assert_allclose(result[0], [-0.4] + [0] * 8 + [0.4])


@pytest.mark.parametrize('measurement,expected', [
    ({'value_ic50_uM': 2}, -0.5), ({'value_ki_uM': 3}, -0.5),
    ({'value_fold_induction': 2}, 0.5), ({'value_fold_induction': 1}, 0),
])
def test_polarity(measurement, expected):
    result = extract(record('HERB_FIXTURE', **measurement),
                     record('DRUG_FIXTURE', value_fm=0.5))
    assert result[0, 0] == expected
    assert np.isnan(result[0, 1:]).all()


def test_substrate_fallback_and_missingness():
    assert extract(record('HERB_FIXTURE', value_ki_uM=2),
                   record('DRUG_FIXTURE', value_substrate_flag=1))[0, 0] == -1
    assert np.isnan(extract(record('HERB_FIXTURE', status='not_tested'),
                            record('DRUG_FIXTURE', value_fm=0.5))[0, 0])
    assert extract(record('HERB_FIXTURE', status='not_tested'),
                   record('DRUG_FIXTURE', value_fm=0))[0, 0] == 0
    assert extract(record('HERB_FIXTURE', status='tested_no_effect'),
                   record('DRUG_FIXTURE', status='not_tested'))[0, 0] == 0


@pytest.mark.parametrize('column', CURATION_COLUMNS)
def test_missing_schema_columns(column):
    frame = pd.DataFrame([record('HERB_FIXTURE', value_ki_uM=2)]).drop(columns=column)
    with pytest.raises(ValueError, match='missing required column'):
        extract_curated_feature_matrix(pairs(), frame)


@pytest.mark.parametrize('values', [
    {'value_ic50_uM': 0}, {'value_ki_uM': -1}, {'value_fold_induction': 0.5},
    {'value_ic50_uM': 'bad'}, {'value_ic50_uM': float('inf')},
    {'value_ic50_uM': 'NaN'}, {'exposure_conc_uM': -1},
    {'value_ki_uM': 2, 'value_fold_induction': 2},
    {'value_fm': 1.1}, {'value_substrate_flag': 0.5},
])
def test_invalid_values(values):
    with pytest.raises(ValueError):
        extract(record('HERB_FIXTURE', **values))


def test_duplicate_status_role_and_pair_validation():
    row = record('HERB_FIXTURE', value_ki_uM=2)
    for rows in [(row, row), (record('HERB_FIXTURE', status='unknown'),),
                 (record('HERB_FIXTURE', status='tested_no_effect', value_ki_uM=2),),
                 (record('DRUG_FIXTURE', value_ki_uM=2),)]:
        with pytest.raises(ValueError):
            extract(*rows)
    with pytest.raises(ValueError, match='Pair table'):
        extract_curated_feature_matrix(pd.DataFrame(), pd.DataFrame([row]))


@pytest.mark.parametrize('allow', [False, True])
def test_real_unverified_rows_cannot_bypass_guard(allow):
    real = load_mechanistic_curation()
    with pytest.raises(UnverifiedCurationError):
        extract_curated_feature_matrix(pairs(), real, allow_unverified=allow)


def test_unsourced_and_empty_measured_rows_blocked():
    for row in [record('HERB_FIXTURE'), record('HERB_FIXTURE', value_ki_uM=2, citation='')]:
        with pytest.raises(UnverifiedCurationError):
            extract(row)


def test_empty_pair_table_has_stable_shape():
    result = extract_curated_feature_matrix(pairs().iloc[:0], pd.DataFrame([
        record('HERB_FIXTURE', value_ki_uM=2)]))
    assert result.shape == (0, 10)


@pytest.mark.parametrize('reverse', [False, True])
@pytest.mark.parametrize('column', ['entity', 'target'])
def test_normalized_conflicting_duplicates_rejected(column, reverse):
    inhibition = record('HERB_FIXTURE', value_ki_uM=2)
    induction = record('HERB_FIXTURE', value_fold_induction=2)
    induction[column] = '\t' + induction[column] + ' '
    drug = record('DRUG_FIXTURE', value_fm=0.5)
    # Each record alone reproduces the opposing signs formerly selected by order.
    assert extract(inhibition, drug)[0, 0] == -0.5
    assert extract(induction, drug)[0, 0] == 0.5
    rows = [inhibition, induction, drug]
    with pytest.raises(ValueError, match='Duplicate entity-target'):
        extract(*(rows[::-1] if reverse else rows))


@pytest.mark.parametrize('column,value', [
    ('entity', 'HERB_'), ('entity', 'HERB_BAD ID'), ('entity', 123),
    ('entity', None), ('target', ''), ('target', 'CYP 3A4'),
    ('target', 123), ('target', None),
])
def test_malformed_curation_identifiers_rejected(column, value):
    row = record('HERB_FIXTURE', value_ki_uM=2)
    row[column] = value
    with pytest.raises(ValueError, match='identifier'):
        extract(row)


@pytest.mark.parametrize('column,value', [
    ('herb_id', 'HERB_'), ('herb_id', 'HERB_BAD ID'),
    ('drug_id', 'DRUG_'), ('drug_id', 123), ('drug_id', None),
])
def test_malformed_pair_identifiers_rejected(column, value):
    frame = pairs()
    frame[column] = value
    with pytest.raises(ValueError, match='identifier'):
        extract_curated_feature_matrix(frame, pd.DataFrame([
            record('HERB_FIXTURE', value_ki_uM=2)]))


def test_boundary_whitespace_is_accepted_without_mutation():
    frame = pairs().map(lambda v: ' ' + v + '\t')
    table = pd.DataFrame([record(' HERB_FIXTURE ', target=' CYP3A4\t', value_ki_uM=2),
                          record('DRUG_FIXTURE', value_fm=0.5)])
    original = table.copy(deep=True)
    result = extract_curated_feature_matrix(frame, table)
    assert result[0, 0] == -0.5
    assert np.isnan(result[0, 1:]).all()
    pd.testing.assert_frame_equal(table, original)


@pytest.mark.parametrize('kind', ['curation', 'pairs', 'csv'])
def test_duplicate_input_columns_rejected(kind, tmp_path):
    table = pd.DataFrame([record('HERB_FIXTURE', value_ki_uM=2)])
    frame = pairs()
    if kind in ('curation', 'csv'):
        table = pd.concat([table, table[['entity']]], axis=1)
    else:
        frame = pd.concat([frame, frame[['herb_id']]], axis=1)
    with pytest.raises(ValueError, match='Duplicate'):
        if kind == 'csv':
            path = tmp_path / 'duplicate.csv'
            table.to_csv(path, index=False)
            load_mechanistic_curation(path)
        else:
            extract_curated_feature_matrix(frame, table)
