"""Selection tests never assign scientific decisions or fit models."""
import json
from pathlib import Path

import pandas as pd
import pytest

from src.data.annotation_batch import prepare, select_pairs
from src.data.consensus_annotation import initialize, read_csv


@pytest.fixture
def candidates():
    return pd.DataFrame([(f'HERB_SYNTHETIC_{h}', f'DRUG_SYNTHETIC_{d}')
                         for h in range(18) for d in range(40)],
                        columns=['herb_id', 'drug_id'])


def test_reproducible_order_independent_and_seeded(candidates):
    first = select_pairs(candidates)
    pd.testing.assert_frame_equal(first, select_pairs(candidates.sample(frac=1, random_state=9)))
    assert not first.equals(select_pairs(candidates, seed=7))


def test_identifier_coverage(candidates):
    selected = select_pairs(candidates)
    assert len(selected) == len(selected.drop_duplicates()) == 24
    assert selected.herb_id.nunique() == 18
    assert selected.drug_id.nunique() == 24
    assert set(map(tuple, selected.values)) <= set(map(tuple, candidates.values))


def test_outcome_blindness_including_manifest(candidates, tmp_path):
    path = tmp_path / 'input.csv'
    candidates.to_csv(path, index=False)
    first = prepare(path, tmp_path / 'first')
    poisoned = pd.concat([candidates, candidates.iloc[:3]], ignore_index=True)
    for column in ['study_finding', 'consensus_label', 'model_score', 'prediction', 'notes']:
        poisoned[column] = ['interaction' if i % 2 else 'no_interaction'
                            for i in range(len(poisoned))]
    poisoned.sample(frac=1, random_state=4).to_csv(path, index=False)
    second = prepare(path, tmp_path / 'second')
    assert first == second
    assert (tmp_path / 'first/selected_pairs.csv').read_bytes() == (tmp_path / 'second/selected_pairs.csv').read_bytes()
    with pytest.raises(FileExistsError):
        prepare(path, tmp_path / 'first')


def test_real_identifier_pool_and_unresolved_initialization(tmp_path):
    source = Path(__file__).resolve().parents[1] / 'data/benchmark/consensus_labels_pairs_to_label.csv'
    manifest = prepare(source, tmp_path / 'selection')
    assert manifest['candidate_pair_count'] == 120
    assert manifest['candidate_herb_count'] == 18
    assert manifest['candidate_drug_count'] == 40
    assert manifest['selected_herb_count'] == 18
    assert manifest['selected_drug_count'] >= 18
    assert initialize([tmp_path / 'selection/selected_pairs.csv'], tmp_path / 'reviews') == 24
    for name in ['reviewer_a.csv', 'reviewer_b.csv', 'adjudication.csv']:
        frame = read_csv(tmp_path / 'reviews' / name)
        assert frame.consensus_label.eq('unresolved').all()
        metadata = frame.drop(columns=['herb_id', 'drug_id', 'consensus_label'])
        assert metadata.eq('').all().all()
    stored = json.loads((tmp_path / 'selection/selection_manifest.json').read_text())
    assert stored == manifest


@pytest.mark.parametrize('size', [0, 721])
def test_invalid_size(candidates, size):
    with pytest.raises(ValueError):
        select_pairs(candidates, size=size)
