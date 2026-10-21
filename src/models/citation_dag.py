"""
Citation-Derivation DAG and Echo Chamber Attenuation Module (CDA).

NOVELTY CONTRIBUTION 1:
Constructs a Directed Acyclic Graph (DAG) of citation relationships for candidate
herb-drug evidence, identifies root empirical studies vs. derivative review echoes,
and applies depth-discounted attenuation to prevent citation echo chambers from
artificially inflating evidence corroboration.

Core Components:
1. CitationDAG: Represents paper citation chains as a DAG, extracts weakly
   connected components, identifies primary root studies, and computes topological
   depth of derivative citations.
2. CitationEchoAnalyzer: Quantifies the Echo Chamber Index (CEI) and calculates
   effective independent evidence count E_eff.
3. CitationDAGEchoAttenuator (PyTorch nn.Module): Differentiable neural layer that
   attenuates corroboration and metadata features using the echo attenuation factor.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Set, Tuple
import math
import networkx as nx
import torch
import torch.nn as nn
from loguru import logger


@dataclass
class PaperNode:
    """Represents a single paper in the citation graph."""
    paper_id: str
    year: int
    is_empirical: bool = True  # True if primary trial/experiment, False if review
    study_type: str = "in_vivo"  # "clinical", "in_vivo", "in_vitro", "review"
    doi: Optional[str] = None
    title: Optional[str] = None


@dataclass
class CitationDAGResult:
    """Summary of citation DAG analysis for a specific candidate pair."""
    total_papers: int
    num_roots: int
    num_derivatives: int
    effective_independent_count: float
    echo_chamber_index: float  # In [0, 1], 0 = completely independent, 1 = extreme echo
    max_depth: int
    component_depths: Dict[str, int]
    root_papers: List[str]
    derivative_papers: List[str]


class CitationDAG:
    """
    Constructs and analyzes the Citation-Derivation DAG from literature evidence.

    Reference convention:
    An edge (P_i -> P_j) indicates that paper P_i cites / references paper P_j.
    Therefore, earlier empirical papers are targets of citations (in-degree in citation net,
    or sinks/roots depending on traversal direction). Here, we direct edges from DERIVATIVE
    to SOURCE (P_cite -> P_source) or from ROOT to DERIVATIVE (P_root -> P_derivative).

    Standard convention adopted here:
    Edges are directed ROOT -> DERIVATIVE:
        Root (original trial) -> Derivative (review citing root) -> 2nd-degree citation.
    This creates a forward causal DAG where depth = distance from empirical origin.
    """

    def __init__(self, echo_discount_alpha: float = 0.15):
        """
        Args:
            echo_discount_alpha: Discount rate per citation hop.
                Weight at depth d is (1 - alpha)^d. Default alpha = 0.15.
        """
        self.alpha = float(echo_discount_alpha)
        self.graph = nx.DiGraph()
        self.node_metadata: Dict[str, PaperNode] = {}

    def add_paper(
        self,
        paper_id: str,
        year: int = 2020,
        is_empirical: bool = True,
        study_type: str = "in_vivo",
        doi: Optional[str] = None,
        title: Optional[str] = None,
    ) -> None:
        """Add a paper node to the citation graph."""
        self.graph.add_node(paper_id)
        self.node_metadata[paper_id] = PaperNode(
            paper_id=paper_id,
            year=year,
            is_empirical=is_empirical,
            study_type=study_type,
            doi=doi,
            title=title,
        )

    def add_citation(self, citing_id: str, cited_id: str) -> None:
        """
        Record that citing_id cites cited_id.
        In the forward derivation DAG, cited_id (earlier/root) points to citing_id (derivative).
        """
        if not self.graph.has_node(cited_id):
            self.add_paper(cited_id)
        if not self.graph.has_node(citing_id):
            self.add_paper(citing_id)
        # Direct edge: cited (origin) -> citing (derivative)
        self.graph.add_edge(cited_id, citing_id)

    def analyze(self) -> CitationDAGResult:
        """
        Performs topological analysis and echo chamber attenuation.

        Algorithm:
        1. Break graph into weakly connected components (WCC).
        2. In each component, find root nodes (in-degree == 0, earliest publication year).
        3. Break any cycles if present (making it a strict DAG via Minimum Feedback Arc Set).
        4. Compute shortest path depth d from component root for each node.
        5. Assign weight w = (1 - alpha)^d.
        6. Compute E_eff = sum(w) and Echo Chamber Index = 1 - (E_eff / N).
        """
        total_nodes = self.graph.number_of_nodes()
        if total_nodes == 0:
            return CitationDAGResult(
                total_papers=0,
                num_roots=0,
                num_derivatives=0,
                effective_independent_count=0.0,
                echo_chamber_index=0.0,
                max_depth=0,
                component_depths={},
                root_papers=[],
                derivative_papers=[],
            )

        # Ensure DAG by breaking cycles if any exist
        dag = self.graph.copy()
        if not nx.is_directed_acyclic_graph(dag):
            try:
                cycles = list(nx.simple_cycles(dag))
                for cycle in cycles:
                    if len(cycle) >= 2:
                        dag.remove_edge(cycle[-1], cycle[0])
            except Exception as e:
                logger.warning(f"Cycle resolution fallback in citation DAG: {e}")

        # Find weakly connected components
        wccs = list(nx.weakly_connected_components(dag))
        root_nodes: List[str] = []
        derivative_nodes: List[str] = []
        depth_dict: Dict[str, int] = {}
        total_effective_count = 0.0
        max_depth = 0

        for wcc in wccs:
            subgraph = dag.subgraph(wcc)
            # Find candidate roots (in-degree == 0 in subgraph)
            comp_roots = [n for n in subgraph.nodes() if subgraph.in_degree(n) == 0]

            if not comp_roots:
                # If tied, pick node with earliest publication year
                comp_roots = [
                    min(
                        subgraph.nodes(),
                        key=lambda n: self.node_metadata.get(n, PaperNode(n, 2025)).year,
                    )
                ]

            primary_root = comp_roots[0]
            root_nodes.append(primary_root)
            depth_dict[primary_root] = 0
            total_effective_count += 1.0  # Root has full weight 1.0

            # Compute depth for remaining nodes in component
            for node in subgraph.nodes():
                if node == primary_root:
                    continue
                try:
                    if nx.has_path(subgraph, primary_root, node):
                        depth = nx.shortest_path_length(subgraph, primary_root, node)
                    else:
                        # Connected to another root or undirected branch
                        depth = 1
                except Exception:
                    depth = 1

                depth_dict[node] = depth
                max_depth = max(max_depth, depth)
                derivative_nodes.append(node)

                # Depth-discounted evidence weight: w = (1 - alpha)^depth
                weight = math.pow(1.0 - self.alpha, depth)
                total_effective_count += weight

        # Echo chamber index: fraction of apparent evidence that is redundant echo
        echo_index = 0.0
        if total_nodes > 1:
            echo_index = max(0.0, min(1.0, 1.0 - (total_effective_count / float(total_nodes))))

        return CitationDAGResult(
            total_papers=total_nodes,
            num_roots=len(root_nodes),
            num_derivatives=len(derivative_nodes),
            effective_independent_count=round(total_effective_count, 4),
            echo_chamber_index=round(echo_index, 4),
            max_depth=max_depth,
            component_depths=depth_dict,
            root_papers=root_nodes,
            derivative_papers=derivative_nodes,
        )


class CitationEchoAnalyzer:
    """High-level utility to process raw literature records for a candidate pair."""

    @staticmethod
    def analyze_records(
        records: List[Dict[str, Any]],
        alpha: float = 0.15,
    ) -> CitationDAGResult:
        """
        Builds a CitationDAG from a list of evidence records with citations.

        Args:
            records: List of dicts, each containing:
                - 'id' or 'pmid': Paper identifier
                - 'year': Publication year
                - 'citations' or 'references': List of cited paper IDs
                - 'is_empirical' or 'study_type': Study attributes
            alpha: Echo discount factor.

        Returns:
            CitationDAGResult with discounted counts and echo metrics.
        """
        dag = CitationDAG(echo_discount_alpha=alpha)

        for rec in records:
            pid = str(rec.get("id", rec.get("pmid", rec.get("paper_id", "unknown"))))
            year = int(rec.get("year", 2020))
            st = str(rec.get("study_type", "in_vivo"))
            emp = bool(rec.get("is_empirical", st not in {"review", "meta_analysis"}))
            dag.add_paper(paper_id=pid, year=year, is_empirical=emp, study_type=st)

        # Second pass: add citation edges
        for rec in records:
            pid = str(rec.get("id", rec.get("pmid", rec.get("paper_id", "unknown"))))
            cites = rec.get("citations", rec.get("references", rec.get("cites", [])))
            for cited in cites:
                dag.add_citation(citing_id=pid, cited_id=str(cited))

        return dag.analyze()


class CitationDAGEchoAttenuator(nn.Module):
    """
    Differentiable Neural Layer for Citation Echo Attenuation.

    Takes raw corroboration signals C and echo metrics, and produces an
    attenuated corroboration score C_eff that prevents echo chambers from
    misleading downstream reliability scoring and cross-attention gating.

    Formula:
        C_attenuated = C * (1.0 - beta * echo_ratio)
        where beta in [0, 1] is a learnable or calibrated dampening parameter.
    """

    def __init__(self, initial_beta: float = 0.70, learnable_beta: bool = True):
        super().__init__()
        if learnable_beta:
            # Parametrized via sigmoid to strictly remain in [0, 1]
            # Inverse sigmoid of initial_beta: log(b / (1 - b))
            clamped_init = max(0.01, min(0.99, initial_beta))
            init_logit = math.log(clamped_init / (1.0 - clamped_init))
            self.beta_param = nn.Parameter(torch.tensor([init_logit], dtype=torch.float32))
        else:
            self.register_buffer("beta_param", torch.tensor([initial_beta], dtype=torch.float32))

        self.learnable = learnable_beta

    @property
    def beta(self) -> torch.Tensor:
        """Returns the dampening strength in [0, 1]."""
        if self.learnable:
            return torch.sigmoid(self.beta_param)
        return self.beta_param

    def forward(
        self,
        corroboration: torch.Tensor,
        echo_ratio: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Applies echo attenuation.

        Args:
            corroboration: (batch, 1) or (batch,) Raw corroboration counts or embedding
            echo_ratio: (batch, 1) or (batch,) Echo Chamber Index in [0, 1]

        Returns:
            attenuated_corroboration: C * (1 - beta * echo_ratio)
            damping_factor: (1 - beta * echo_ratio) in [0, 1]
        """
        damping_factor = torch.clamp(1.0 - self.beta * echo_ratio, min=0.05, max=1.0)
        attenuated = corroboration * damping_factor
        return attenuated, damping_factor
