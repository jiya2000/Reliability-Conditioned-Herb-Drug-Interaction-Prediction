"""Task B infrastructure: synthetic fixtures only, never scientific results."""
import numpy as np
import pandas as pd
import pytest

from src.data.evidence_schema import ConsensusLabelsNotPopulatedError
from src.data.mechanistic_curation import CURATION_COLUMNS, UnverifiedCurationError
from src.features.task_a_contract import TASK_A_TARGETS, TaskAFeatureContractError, build_task_a_features
from src.models.aggregation import ResidualEvidenceAggregator, train_reliability_model_entrypoint
from src.training.task_b import (fit_task_b, reliability_features, SupervisedStudyReliability,
                                RELIABILITY_FEATURES, assert_disjoint)


def test_signed_mass_math_and_echo_discount():
    agg = ResidualEvidenceAggregator(alpha=.25, evidence_scale=2, kappa=.5)
    out = agg.aggregate(.7, [1, -1, 1, 0], [.8, .6, .4, 1], [1, 1, 0, 1])
    assert out['S_pos'] == pytest.approx(.9)
    assert out['S_neg'] == pytest.approx(.6)
    assert out['net_evidence'] == pytest.approx(.3 / 2)
    assert out['logit'] == pytest.approx(1.0)
    assert out['contradiction'] == pytest.approx(1.2 / 1.5)
    assert out['conflicting_evidence'] and out['n_inconclusive'] == 1


@pytest.mark.parametrize('prior', [-100., -.123456789, 0., 17.3456])
def test_exact_zero_study_and_inconclusive_fallback(prior):
    agg = ResidualEvidenceAggregator(evidence_scale=100)
    assert agg.aggregate(prior, [], [], [])['logit'] == prior
    info = agg.aggregate(prior, [0, 0], [1, 1], [1, 0])
    assert info['logit'] == prior
    assert info['no_directional_evidence']
    assert info['contradiction'] == 0


def test_balanced_conflict_is_prior_with_high_contradiction():
    info = ResidualEvidenceAggregator().aggregate(-.4, [1, -1], [.8, .8], [1, 1])
    assert info['logit'] == -.4
    assert info['contradiction'] == pytest.approx(1)
    assert not info['no_directional_evidence']


@pytest.mark.parametrize('kwargs', [{'alpha': -1}, {'alpha': np.nan}, {'evidence_scale': -1}, {'kappa': 0}])
def test_invalid_aggregator_parameters(kwargs):
    with pytest.raises(ValueError):
        ResidualEvidenceAggregator(**kwargs)


@pytest.mark.parametrize('o,r,p', [([2], [1], [1]), ([1], [np.nan], [1]),
                                  ([1], [1.1], [1]), ([1], [1], [2]), ([1], [], [])])
def test_invalid_study_vectors(o, r, p):
    with pytest.raises(ValueError):
        ResidualEvidenceAggregator().aggregate(0, o, r, p)


@pytest.fixture
def fixture(tmp_path, monkeypatch):
    # Mock publication verification for explicit test-only references; never register
    # synthetic publications in the real registry or change its production gate.
    def synthetic_verifier(source_id, reference, *, require_registered):
        assert require_registered is True
        return (source_id.startswith('SYNTHETIC_SOURCE_') and reference == 'SYNTHETIC_FIXTURE',
                'test-only source verifier')
    monkeypatch.setattr('src.data.source_registry.DEFAULT_SOURCE_REGISTRY.verify_reference', synthetic_verifier)
    labels = pd.DataFrame([dict(herb_id=f'HERB_{i}', drug_id=f'DRUG_{i}',
        consensus_label='interaction' if i % 2 else 'no_interaction',
        source_name='SYNTHETIC_FIXTURE', source_url_or_citation='SYNTHETIC_FIXTURE',
        retrieval_date='2026-10-28') for i in range(12)])
    labels.loc[10, 'consensus_label'] = 'disputed'
    studies = []
    for i, pair in labels.iterrows():
        # High-quality primary agrees; animal primary disagrees; echo disagrees.
        for j in range(3):
            studies.append(dict(herb_id=pair.herb_id, drug_id=pair.drug_id,
                source_id=f'SYNTHETIC_SOURCE_{i}_{j}', source_url_or_doi='SYNTHETIC_FIXTURE', study_id=f'SYNTHETIC_STUDY_{i}_{j}',
                cited_study_ids=[f'SYNTHETIC_STUDY_{i}_0'] if j == 2 else [],
                study_finding='positive' if (i % 2 == 1) == (j == 0) else 'negative',
                species='human' if j == 0 else 'rat', evidence_type='clinical' if j == 0 else 'animal',
                route='oral', preparation='standardized_extract', dose='reported',
                sample_size=100 if j == 0 else 3, is_primary=int(j != 2)))
    curation = []
    for entity in list(labels.herb_id) + list(labels.drug_id):
        for target in TASK_A_TARGETS:
            row = dict.fromkeys(CURATION_COLUMNS, '')
            row.update(entity=entity, target=target, status='tested_no_effect',
                       source='SYNTHETIC_FIXTURE', citation='SYNTHETIC_FIXTURE')
            curation.append(row)
    lp, cp = tmp_path / 'labels.csv', tmp_path / 'curation.csv'
    labels.to_csv(lp, index=False)
    curation = pd.DataFrame(curation)
    curation.to_csv(cp, index=False)
    train = labels.iloc[:6][['herb_id', 'drug_id']]
    tune = labels.iloc[6:9][['herb_id', 'drug_id']]
    return dict(studies=pd.DataFrame(studies), train_pairs=train, tuning_pairs=tune,
                consensus_path=lp, curation_path=cp), labels, curation


def test_fit_predict_prior_fallback_and_unlabeled_disputed_prediction(fixture):
    args, labels, _ = fixture
    model = train_reliability_model_entrypoint(**args)
    heldout = labels.iloc[9:][['herb_id', 'drug_id']]
    # No study for pair 11; disputed pair 10 is predicted without using its label.
    evidence = args['studies'].loc[args['studies'].herb_id != 'HERB_11']
    output = model.predict(heldout, evidence)
    expected = model.prior_.decision_function(build_task_a_features(heldout, model.curation_))
    assert output.iloc[2].logit == expected[2]
    assert output.iloc[2].n_studies == 0
    assert len(output) == 3
    assert 0 <= model.aggregator_.alpha < .15  # harmful echoes are learned down from initialization
    assert model.aggregator_.evidence_scale >= 0
    # Planted agreement structure is learned; this is not a scientific score.
    X = reliability_features(args['studies'].iloc[:3])
    r = model.reliability_.predict(X)
    assert r[0] > r[1]
    # Source metadata, finding text and pair names never become model coefficients.
    assert model.reliability_.pipeline_[0].n_features_in_ == len(RELIABILITY_FEATURES)


def test_reliability_ignores_findings_and_raw_text(fixture):
    args, _, _ = fixture
    studies = args['studies']
    X = reliability_features(studies)
    altered = studies.copy()
    altered['study_finding'] = 'inconclusive'
    altered['original_text'] = 'positive outcome HERB_identity'
    altered['outcome'] = 999
    pd.testing.assert_frame_equal(X, reliability_features(altered))
    r = SupervisedStudyReliability().fit(X, np.tile([1, 0, 0], 12))
    with pytest.raises(ValueError, match='allowlist'):
        r.fit(X.assign(outcome=1), np.tile([1, 0, 0], 12))


def test_tuning_changes_do_not_fit_prior_or_reliability(fixture):
    args, labels, _ = fixture
    first = fit_task_b(**args)
    labels.loc[6:8, 'consensus_label'] = ['interaction', 'no_interaction', 'interaction']
    labels.to_csv(args['consensus_path'], index=False)
    modified = dict(args, studies=args['studies'].copy())
    mask = modified['studies'].herb_id.isin(['HERB_6', 'HERB_7', 'HERB_8'])
    modified['studies'].loc[mask, 'sample_size'] = 1e6
    second = fit_task_b(**modified)
    np.testing.assert_array_equal(first.prior_[-1].coef_, second.prior_[-1].coef_)
    for a, b in zip(first.reliability_.pipeline_.steps, second.reliability_.pipeline_.steps):
        attribute = 'mean_' if a[0] == 'standardscaler' else 'coef_'
        np.testing.assert_array_equal(getattr(a[1], attribute), getattr(b[1], attribute))


def test_pair_and_transitive_citation_overlap_rejected(fixture):
    args, _, _ = fixture
    with pytest.raises(ValueError, match='pair overlap'):
        fit_task_b(**dict(args, tuning_pairs=args['train_pairs']))
    studies = args['studies'].copy(deep=True)
    # Two partitions cite an absent common ancestor: indirect overlap still blocked.
    studies.at[0, 'cited_study_ids'] = ['ABSENT_COMMON_ANCESTOR']
    studies.at[18, 'cited_study_ids'] = ['ABSENT_COMMON_ANCESTOR']
    with pytest.raises(ValueError, match='cluster overlap'):
        fit_task_b(**dict(args, studies=studies))


def test_assessment_reuse_rejected(fixture):
    args, labels, _ = fixture
    model = fit_task_b(**args)
    with pytest.raises(ValueError, match='pair overlap'):
        model.predict(args['train_pairs'], args['studies'])
    studies = args['studies'].copy(deep=True)
    studies.at[27, 'cited_study_ids'] = ['SYNTHETIC_STUDY_0_0']
    with pytest.raises(ValueError, match='cluster overlap'):
        model.predict(labels.iloc[9:][['herb_id', 'drug_id']], studies)


def test_disputed_and_inconclusive_excluded_from_targets(fixture):
    args, labels, _ = fixture
    # Move disputed pair to fitting; it must not enter either binary target.
    args['train_pairs'] = pd.concat([args['train_pairs'], labels.iloc[10:11][['herb_id', 'drug_id']]])
    args['studies'].loc[0, 'study_finding'] = 'inconclusive'
    model = fit_task_b(**args)
    assert model.fit_summary_['disputed_pairs_excluded'] == 1
    assert model.fit_summary_['train_pairs'] == 6
    assert model.fit_summary_['agreement_studies'] == 17
    assert model.fit_summary_['inconclusive_train_studies'] == 1


def test_empty_consensus_and_unverified_mechanisms_block_before_fit(fixture, monkeypatch):
    args, labels, curation = fixture
    monkeypatch.setattr(SupervisedStudyReliability, 'fit', lambda *a: pytest.fail('fit before guard'))
    labels.iloc[:0].to_csv(args['consensus_path'], index=False)
    with pytest.raises(ConsensusLabelsNotPopulatedError):
        fit_task_b(**args)
    labels.to_csv(args['consensus_path'], index=False)
    curation.loc[0, 'source'] = 'UNVERIFIED'
    curation.to_csv(args['curation_path'], index=False)
    with pytest.raises(UnverifiedCurationError):
        fit_task_b(**args)


def test_real_entrypoint_blocked():
    with pytest.raises(ConsensusLabelsNotPopulatedError):
        train_reliability_model_entrypoint()


def test_unregistered_publication_blocks_before_fitting(fixture, monkeypatch):
    args, _, _ = fixture
    args['studies'].loc[0, 'source_url_or_doi'] = 'UNVERIFIED'
    monkeypatch.setattr(SupervisedStudyReliability, 'fit', lambda *a: pytest.fail('fit before source guard'))
    with pytest.raises(ValueError, match='Unverified Task B publication'):
        fit_task_b(**args)


def test_missing_mechanism_blocks_before_fitting(fixture, monkeypatch):
    args, _, curation = fixture
    curation.loc[~(curation.entity.isin(['HERB_0', 'DRUG_0']) &
                   curation.target.eq(TASK_A_TARGETS[0]))].to_csv(args['curation_path'], index=False)
    monkeypatch.setattr(SupervisedStudyReliability, 'fit', lambda *a: pytest.fail('fit before mechanism guard'))
    with pytest.raises(TaskAFeatureContractError, match='missing mechanistic'):
        fit_task_b(**args)


def test_shared_source_and_intermediate_lineage_overlap(fixture):
    args, labels, _ = fixture
    studies = args['studies'].copy(deep=True)
    studies.loc[18, 'source_id'] = studies.loc[0, 'source_id']
    with pytest.raises(ValueError, match='cluster overlap'):
        fit_task_b(**dict(args, studies=studies))
    model = fit_task_b(**args)
    studies = args['studies'].copy(deep=True)
    # A third pair bridges a fitting study and an assessment study.
    studies.at[33, 'cited_study_ids'] = ['SYNTHETIC_STUDY_0_0', 'SYNTHETIC_STUDY_9_0']
    with pytest.raises(ValueError, match='cluster overlap'):
        model.predict(labels.iloc[9:10][['herb_id', 'drug_id']], studies)


def test_all_zero_study_prediction_is_exact_mechanistic_prior(fixture):
    args, labels, _ = fixture
    model = fit_task_b(**args)
    pairs = labels.iloc[9:][['herb_id', 'drug_id']]
    output = model.predict(pairs, args['studies'].iloc[:0])
    expected = model.prior_.decision_function(build_task_a_features(pairs, model.curation_))
    np.testing.assert_array_equal(output.logit.to_numpy(), expected)
    assert output.no_directional_evidence.all()
    assert (output.n_studies == 0).all()


def test_inconclusive_only_prediction_has_no_signed_mass(fixture):
    args, labels, _ = fixture
    model = fit_task_b(**args)
    evidence = args['studies'].copy()
    evidence['study_finding'] = 'inconclusive'
    output = model.predict(labels.iloc[9:][['herb_id', 'drug_id']], evidence)
    assert (output.logit == output.prior_logit).all()
    assert (output.n_inconclusive == 3).all()
    assert (output.contradiction == 0).all()


def test_ambiguous_design_and_unknown_finding_fail_closed(fixture):
    args, _, _ = fixture
    evidence = args['studies'].copy()
    evidence.loc[0, 'study_finding'] = 'maybe'
    with pytest.raises(ValueError, match='study_finding'):
        reliability_features(evidence)
    evidence.loc[0, 'study_finding'] = 'inconclusive'
    evidence.loc[0, 'route'] = 'UNREVIEWED_ROUTE'
    with pytest.raises(ValueError, match='Unsupported route'):
        reliability_features(evidence)


def test_single_agreement_class_blocks_instead_of_inventing_targets(fixture):
    args, labels, _ = fixture
    for i, row in args['studies'].iterrows():
        verdict = labels.loc[labels.herb_id.eq(row.herb_id), 'consensus_label'].iloc[0]
        args['studies'].loc[i, 'study_finding'] = 'positive' if verdict == 'interaction' else 'negative'
    with pytest.raises(ValueError, match='both agreement classes'):
        fit_task_b(**args)
