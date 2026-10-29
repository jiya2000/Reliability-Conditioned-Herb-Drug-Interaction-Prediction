"""Synthetic expanded curation tests, not biological validation."""
import pandas as pd
import pytest
from src.data.mechanistic_curation import CURATION_COLUMNS, is_curation_verified
from src.data.mechanistic_panel import PANEL_TARGETS, prepare
from src.features.task_a_contract import TASK_A_TARGETS


def test_expansion_preserves_unknown_and_primary_contract(tmp_path):
    original = tmp_path/'original.csv'
    row = dict.fromkeys(CURATION_COLUMNS, '')
    row.update(entity='HERB_SYNTHETIC', target='CYP3A4', status='not_tested')
    pd.DataFrame([row]).to_csv(original, index=False)
    raw = original.read_bytes()
    output = tmp_path/'panel'
    report = prepare(original, output)
    candidates = pd.read_csv(output/'additional_candidates.csv', keep_default_na=False)
    assert len(PANEL_TARGETS) == 16 and len(candidates) == 15
    assert {'CYP2B6','CYP2C8','ABCG2','SLC47A2'} <= set(candidates.target)
    assert candidates.status.eq('not_tested').all()
    assert candidates.source.eq('UNVERIFIED').all()
    assert candidates.value_ic50_uM.eq('').all()
    assert is_curation_verified(candidates) == (False, 0, 15)
    assert report['training_eligible'] is False
    assert original.read_bytes() == raw == (output/'input_snapshot.csv').read_bytes()
    assert len(TASK_A_TARGETS) == 6
    with pytest.raises(FileExistsError):
        prepare(original, output)


def test_invalid_input_cannot_generate_panel(tmp_path):
    original=tmp_path/'bad.csv'
    original.write_text('entity,target\nHERB_SYNTHETIC,CYP3A4\n')
    with pytest.raises(ValueError):
        prepare(original,tmp_path/'output')
    assert not (tmp_path/'output').exists()
