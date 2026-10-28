"""Evaluation integrity regressions; constructed records below are synthetic fixtures."""
from dataclasses import replace

import numpy as np
import pandas as pd
import pytest
import torch

from benchmarks.external_validation import analyze_abstention_mechanisms
from src.data.evidence_schema import EvidenceRecord
from src.data.evidence_lineage import EvidenceLineageAnalyzer
from src.models.conformal_abstention import ConformalTriStateAbstainer
from src.models.reliability_scorer import ReliabilityScorer
from src.training import evaluation


def test_quality_metadata_invariant_to_findings_and_outcome_text():
    records = [EvidenceRecord(
        herb_id="HERB_FIXTURE", drug_id="DRUG_FIXTURE",
        outcome="interaction_present", evidence_type="clinical_trial",
        source_id=f"FIXTURE_{i}", study_id=f"FIXTURE_STUDY_{i}",
        cited_study_ids=["FIXTURE_STUDY_0"] if i else [],
        species="human", preparation="extract", route="oral",
    ) for i in range(2)]
    changed = [records[0], replace(records[1], outcome="no_interaction_observed",
                                  original_text="No interaction with FIXTURE entities")]
    analyzer = EvidenceLineageAnalyzer()
    a = analyzer.aggregate_pair_evidence("HERB_FIXTURE", "DRUG_FIXTURE", records)
    b = analyzer.aggregate_pair_evidence("HERB_FIXTURE", "DRUG_FIXTURE", changed)
    assert a.contradiction_strength != b.contradiction_strength
    assert a.citation_echo_ratio == b.citation_echo_ratio == 0.5
    for mode in [None, "no_study_dependence", "no_context_conflict"]:
        np.testing.assert_array_equal(a.to_feature_vector(mode=mode), b.to_feature_vector(mode=mode))
    assert not np.array_equal(a.to_feature_group("full"), b.to_feature_group("full"))


@pytest.mark.parametrize("width,conflict_col", [(7, 3), (8, 5), (10, 5), (12, 5)])
def test_quality_reliability_ignores_conflict_even_in_legacy_layout(width, conflict_col):
    scorer = ReliabilityScorer(hidden_dim=16, feature_group="quality_only")
    scorer.eval()
    a = torch.full((2, width), 0.4)
    b = a.clone()
    b[:, conflict_col] = 0.99
    with torch.no_grad():
        np.testing.assert_array_equal(scorer(a)[0].numpy(), scorer(b)[0].numpy())
        torch.manual_seed(17)
        mean_a, uncertainty_a, _ = scorer.forward_with_uncertainty(a)
        torch.manual_seed(17)
        mean_b, uncertainty_b, _ = scorer.forward_with_uncertainty(b)
        torch.testing.assert_close(mean_a, mean_b, rtol=0, atol=0)
        torch.testing.assert_close(uncertainty_a, uncertainty_b, rtol=0, atol=0)


def test_external_diagnostics_use_fitted_sets_and_do_not_invent_uncertainty():
    abstainer = ConformalTriStateAbstainer()
    records = [dict(herb_id="HERB_FIXTURE", drug_id="DRUG_FIXTURE",
                    predicted_prob=p, reliability_R=0.7, contradiction_C=0.0)
               for p in [0.2, 0.5, 0.9]]
    with pytest.raises(ValueError, match="fitted"):
        analyze_abstention_mechanisms(records, abstainer)
    abstainer.calibrate([0.8, 0.2, 0.6, 0.4], [0, 1, 1, 0])
    result = analyze_abstention_mechanisms(records, abstainer)
    for record, analysis in zip(records, result["per_record_analysis"]):
        expected = abstainer.predict_single(record["predicted_prob"], reliability_score=0.7)
        assert analysis["conformal_set"] == expected.conformal_set
        assert analysis["epistemic_uncertainty_u"] is None
    assert result["summary_trigger_distribution"]["conformal_set_size_2"] == 2


def test_outer_predictions_are_pooled_without_assessment_label_refit(monkeypatch):
    # Real records exercise the legacy evaluator; this is an isolation check,
    # not evidence of clinical validity or consensus-supervised performance.
    df = pd.read_csv("data/benchmark/verified_botanical_hdi.csv")
    expected_validation_labels = []
    original_split = evaluation.LeakageSafeSplitter.split_train_val
    original_fit = evaluation.TemperatureScaler.fit
    original_metrics = evaluation.compute_comprehensive_metrics
    fold_probabilities = []
    fit_count = 0

    def split(self, outer_train, *args, **kwargs):
        train, val = original_split(self, outer_train, *args, **kwargs)
        assert set(train.study_id).isdisjoint(val.study_id)
        expected_validation_labels.append(np.array([
            1.0 if str(v).lower() in {"interaction_present", "positive", "1", "true"}
            else 0.0 if str(v).lower() in {"no_interaction_observed", "negative", "0", "false"}
            else 0.5 for v in val.outcome]))
        return train, val

    def fit(self, logits, labels, **kwargs):
        nonlocal fit_count
        assert fit_count < len(expected_validation_labels), "Unexpected assessment-label fit"
        np.testing.assert_array_equal(np.asarray(labels).flatten(), expected_validation_labels[fit_count])
        fit_count += 1
        return original_fit(self, logits, labels, **kwargs)

    def metrics(*args, **kwargs):
        if kwargs["split_name"].startswith("pooled_"):
            np.testing.assert_array_equal(kwargs["y_prob"], np.concatenate(fold_probabilities))
        else:
            fold_probabilities.append(kwargs["y_prob"].copy())
        return original_metrics(*args, **kwargs)

    monkeypatch.setattr(evaluation.LeakageSafeSplitter, "split_train_val", split)
    monkeypatch.setattr(evaluation.TemperatureScaler, "fit", fit)
    monkeypatch.setattr(evaluation, "compute_comprehensive_metrics", metrics)
    evaluation.run_repeated_nested_cv(df, model_variants=["standard_fusion"],
                                     n_repeats=1, n_splits=2, epochs=1,
                                     run_significance_tests=False)
    assert fit_count == 2
