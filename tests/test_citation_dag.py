"""
Tests for Novelty 1: Citation-Derivation DAG and Echo Chamber Attenuation (CDA).
"""

import pytest
import torch
from src.models.citation_dag import (
    CitationDAG,
    CitationEchoAnalyzer,
    CitationDAGEchoAttenuator,
    CitationDAGResult,
)


def test_empty_citation_dag():
    dag = CitationDAG()
    res = dag.analyze()
    assert res.total_papers == 0
    assert res.effective_independent_count == 0.0
    assert res.echo_chamber_index == 0.0


def test_single_paper_root():
    dag = CitationDAG(echo_discount_alpha=0.15)
    dag.add_paper("P1", year=2015, is_empirical=True)
    res = dag.analyze()
    assert res.total_papers == 1
    assert res.num_roots == 1
    assert res.effective_independent_count == 1.0
    assert res.echo_chamber_index == 0.0


def test_echo_chamber_chain_attenuation():
    """
    Simulate an echo chamber:
    P1 (original clinical trial in 2010)
    P2 cites P1 (review in 2014)
    P3 cites P2 (meta-analysis in 2018 citing the review)
    P4 cites P3 (guideline in 2022 citing the meta-analysis)
    Total raw count = 4 papers.
    Effective independent count should be strictly < 4 because of echo discounting!
    """
    dag = CitationDAG(echo_discount_alpha=0.20)
    dag.add_paper("P1", year=2010)
    dag.add_paper("P2", year=2014)
    dag.add_paper("P3", year=2018)
    dag.add_paper("P4", year=2022)

    dag.add_citation(citing_id="P2", cited_id="P1")
    dag.add_citation(citing_id="P3", cited_id="P2")
    dag.add_citation(citing_id="P4", cited_id="P3")

    res = dag.analyze()
    assert res.total_papers == 4
    assert res.num_roots == 1
    assert res.num_derivatives == 3
    assert res.max_depth == 3

    # Theoretical expected count with alpha=0.20:
    # d=0: 1.0
    # d=1: 0.8
    # d=2: 0.64
    # d=3: 0.512
    # sum = 2.952
    expected = 1.0 + 0.8 + 0.64 + 0.512
    assert pytest.approx(res.effective_independent_count, rel=1e-3) == expected
    assert res.echo_chamber_index > 0.20


def test_multiple_independent_roots():
    """Two completely independent lab studies without citations between them."""
    dag = CitationDAG(echo_discount_alpha=0.15)
    dag.add_paper("StudyA", year=2015)
    dag.add_paper("StudyB", year=2019)

    res = dag.analyze()
    assert res.total_papers == 2
    assert res.num_roots == 2
    assert res.effective_independent_count == 2.0
    assert res.echo_chamber_index == 0.0


def test_citation_echo_analyzer_from_records():
    records = [
        {"pmid": "1001", "year": 2012, "references": []},
        {"pmid": "1002", "year": 2016, "references": ["1001"]},
        {"pmid": "1003", "year": 2020, "references": ["1002"]},
    ]
    res = CitationEchoAnalyzer.analyze_records(records, alpha=0.15)
    assert res.total_papers == 3
    assert res.num_roots == 1
    assert res.effective_independent_count < 3.0


def test_differentiable_echo_attenuator_layer():
    attenuator = CitationDAGEchoAttenuator(initial_beta=0.75, learnable_beta=True)
    corroboration = torch.tensor([[10.0], [5.0], [20.0]])
    echo_ratio = torch.tensor([[0.80], [0.10], [0.00]])

    attenuated, damp = attenuator(corroboration, echo_ratio)

    assert attenuated.shape == (3, 1)
    assert damp.shape == (3, 1)

    # When echo_ratio is high (0.80), attenuation should be strong (value much lower than 10.0)
    assert attenuated[0].item() < corroboration[0].item() * 0.6
    # When echo_ratio is zero, dampening should be 1.0 (no attenuation)
    assert pytest.approx(damp[2].item(), rel=1e-3) == 1.0
    assert pytest.approx(attenuated[2].item(), rel=1e-3) == 20.0

    # Check backpropagation through beta parameter
    loss = attenuated.sum()
    loss.backward()
    assert attenuator.beta_param.grad is not None
