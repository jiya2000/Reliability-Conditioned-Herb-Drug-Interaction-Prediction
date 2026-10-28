"""Synthetic export-to-runner integration; no scientific annotations."""
import pytest
from tests.test_task_a_training import tables
from tests.test_consensus_annotation import complete
from src.data.consensus_annotation import initialize, finalize
from src.data.evidence_schema import ConsensusLabelsNotPopulatedError, load_consensus_labels_table
from src.training.task_a import run_task_a


def test_export_consumed_by_task_a(tables, tmp_path):
    labels_path, curation_path, labels, _ = tables
    assignment = tmp_path / 'assignment'
    initialize([labels_path], assignment)
    # Write synthetic decisions for every assigned pair, retaining workbook schema.
    import pandas as pd
    for filename, reviewer in [('reviewer_a.csv', 'a'), ('reviewer_b.csv', 'b'),
                               ('adjudication.csv', 'c')]:
        complete(assignment, filename, reviewer, 'interaction')
        frame = pd.read_csv(assignment / filename, keep_default_na=False)
        for column in frame.columns:
            if column not in ['herb_id', 'drug_id']:
                frame[column] = frame.loc[0, column]
        lookup = labels.set_index(['herb_id', 'drug_id']).consensus_label
        frame['consensus_label'] = [lookup.loc[(r.herb_id, r.drug_id)] for r in frame.itertuples()]
        frame.to_csv(assignment / filename, index=False)
    finalize(assignment, tmp_path / 'export')
    exported = tmp_path / 'export/consensus_labels.csv'
    assert len(load_consensus_labels_table(exported, require_labels=True)[0]) == 24
    result = run_task_a(exported, curation_path, n_splits=3, repeats=1)
    assert len(result['predictions']) == 24


def test_unresolved_export_blocks_runner_before_mechanisms(tables, tmp_path):
    labels_path, _, _, _ = tables
    assignment = tmp_path / 'assignment'
    initialize([labels_path], assignment)
    finalize(assignment, tmp_path / 'export')
    with pytest.raises(ConsensusLabelsNotPopulatedError):
        run_task_a(tmp_path / 'export/consensus_labels.csv', tmp_path / 'absent.csv')
    with pytest.raises(ValueError):
        run_task_a(assignment / 'reviewer_a.csv', tmp_path / 'absent.csv')
