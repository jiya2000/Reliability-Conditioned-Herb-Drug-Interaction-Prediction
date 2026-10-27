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
import numpy as np
import torch
import torch.nn as nn
from loguru import logger


DEFAULT_STUDY_TYPE_MULTIPLIERS: Dict[str, float] = {
    "review": 1.80,         # Derivative reviews are heavily discounted echoes
    "meta_analysis": 1.50,  # Meta-analyses aggregate existing studies
    "case_report": 0.70,    # Case reports bring independent observational findings
    "clinical": 0.50,       # Clinical trials have higher independent evidentiary value
    "rct": 0.30,            # High-quality RCT trials have minimal echo discounting
    "in_vivo": 0.80,        # Animal models
    "in_vitro": 1.00,       # Standard mechanistic studies
}


def compute_jaccard_text_reuse(text_a: Optional[str], text_b: Optional[str]) -> float:
    """Computes word-level Jaccard similarity between two texts."""
    if not text_a or not text_b:
        return 0.0
    import re
    words_a = set(re.findall(r"\b[a-z]{3,}\b", str(text_a).lower()))
    words_b = set(re.findall(r"\b[a-z]{3,}\b", str(text_b).lower()))
    if not words_a or not words_b:
        return 0.0
    intersection = len(words_a & words_b)
    union = len(words_a | words_b)
    return float(intersection) / float(union) if union > 0 else 0.0


@dataclass
class PaperNode:
    """Represents a single paper in the citation graph."""
    paper_id: str
    year: int
    is_empirical: bool = True  # True if primary trial/experiment, False if review
    study_type: str = "in_vivo"  # "clinical", "rct", "in_vivo", "in_vitro", "case_report", "review"
    doi: Optional[str] = None
    title: Optional[str] = None
    outcome: Optional[str] = None  # e.g., "interaction_present", "no_interaction_observed"
    text: Optional[str] = None


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
    conflicting_derivatives: List[Tuple[str, str]] = field(default_factory=list)
    has_conflicting_derivatives: bool = False
    study_type_scaled: bool = False
    mean_text_reuse: float = 0.0


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

    def __init__(
        self,
        echo_discount_alpha: float = 0.15,
        scale_by_study_type: bool = False,
        study_type_multipliers: Optional[Dict[str, float]] = None,
        use_text_reuse: bool = False,
        handle_contradictions: bool = True,
    ):
        """
        Args:
            echo_discount_alpha: Discount rate per citation hop.
                Weight at depth d is (1 - alpha)^d. Default alpha = 0.15.
            scale_by_study_type: If True, scales alpha by study type (RCTs discounted less, reviews discounted more).
            study_type_multipliers: Optional custom multipliers on alpha per study type.
            use_text_reuse: If True, scales alpha by text-reuse similarity between derivative and primary studies.
            handle_contradictions: If True, derivative studies contradicting the primary study are NOT discounted as echoes.
        """
        self.alpha = float(echo_discount_alpha)
        self.scale_by_study_type = scale_by_study_type
        self.study_type_multipliers = (
            study_type_multipliers if study_type_multipliers is not None else DEFAULT_STUDY_TYPE_MULTIPLIERS
        )
        self.use_text_reuse = use_text_reuse
        self.handle_contradictions = handle_contradictions

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
        outcome: Optional[str] = None,
        text: Optional[str] = None,
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
            outcome=outcome,
            text=text,
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
        5. Check whether derivative refutes or contradicts root empirical finding.
        6. Compute effective alpha scaling by study type and text-reuse similarity.
        7. Assign weight w = (1 - alpha_eff)^d (or w = 1.0 for contradicting refutations).
        8. Compute E_eff = sum(w) and Echo Chamber Index = 1 - (E_eff / N).
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
                conflicting_derivatives=[],
                has_conflicting_derivatives=False,
                study_type_scaled=self.scale_by_study_type,
                mean_text_reuse=0.0,
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
        conflicting_derivatives: List[Tuple[str, str]] = []
        text_reuse_list: List[float] = []
        depth_dict: Dict[str, int] = {}
        total_effective_count = 0.0
        max_depth = 0

        pos_keys = {"interaction_present", "positive", "1", "true"}
        neg_keys = {"no_interaction_observed", "negative", "0", "false"}

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
            root_obj = self.node_metadata.get(primary_root)

            # Compute depth for remaining nodes in component
            for node in subgraph.nodes():
                if node == primary_root:
                    continue
                try:
                    if nx.has_path(subgraph, primary_root, node):
                        depth = nx.shortest_path_length(subgraph, primary_root, node)
                    else:
                        depth = 1
                except Exception:
                    depth = 1

                depth_dict[node] = depth
                max_depth = max(max_depth, depth)
                derivative_nodes.append(node)
                node_obj = self.node_metadata.get(node)

                # Check for contradiction between root and derivative
                is_contradiction = False
                if self.handle_contradictions and root_obj and node_obj:
                    r_out = str(root_obj.outcome).lower() if root_obj.outcome else ""
                    n_out = str(node_obj.outcome).lower() if node_obj.outcome else ""
                    if (r_out in pos_keys and n_out in neg_keys) or (r_out in neg_keys and n_out in pos_keys):
                        is_contradiction = True
                        conflicting_derivatives.append((primary_root, node))

                # Compute effective alpha
                eff_alpha = self.alpha
                if self.scale_by_study_type and node_obj:
                    st_norm = str(node_obj.study_type).lower()
                    mult = self.study_type_multipliers.get(st_norm, 1.0)
                    eff_alpha = min(0.95, eff_alpha * mult)

                # Text reuse similarity discount
                if self.use_text_reuse and root_obj and node_obj:
                    reuse = compute_jaccard_text_reuse(root_obj.text, node_obj.text)
                    text_reuse_list.append(reuse)
                    eff_alpha = min(0.95, eff_alpha * (1.0 + reuse))

                # If derivative refutes the primary root study, it is an independent refutation, NOT an echo!
                if is_contradiction:
                    weight = 1.0
                else:
                    weight = math.pow(max(0.01, 1.0 - eff_alpha), depth)

                total_effective_count += weight

        # Echo chamber index: fraction of apparent evidence that is redundant echo
        echo_index = 0.0
        if total_nodes > 1:
            echo_index = max(0.0, min(1.0, 1.0 - (total_effective_count / float(total_nodes))))

        mean_reuse = float(np.mean(text_reuse_list)) if text_reuse_list else 0.0

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
            conflicting_derivatives=conflicting_derivatives,
            has_conflicting_derivatives=len(conflicting_derivatives) > 0,
            study_type_scaled=self.scale_by_study_type,
            mean_text_reuse=round(mean_reuse, 4),
        )


class CitationEchoAnalyzer:
    """High-level utility to process raw literature records for a candidate pair."""

    @staticmethod
    def analyze_records(
        records: List[Dict[str, Any]],
        alpha: float = 0.15,
        scale_by_study_type: bool = False,
        use_text_reuse: bool = False,
        handle_contradictions: bool = True,
    ) -> CitationDAGResult:
        """
        Builds a CitationDAG from a list of evidence records with citations.

        Args:
            records: List of dicts, each containing:
                - 'id' or 'pmid': Paper identifier
                - 'year': Publication year
                - 'citations' or 'references': List of cited paper IDs
                - 'is_empirical' or 'study_type': Study attributes
                - 'outcome': Optional outcome string
                - 'text': Optional abstract/text excerpt
            alpha: Echo discount factor.
            scale_by_study_type: If True, scales alpha by study type.
            use_text_reuse: If True, scales alpha by text similarity.
            handle_contradictions: If True, refutations retain full weight.

        Returns:
            CitationDAGResult with discounted counts and echo metrics.
        """
        dag = CitationDAG(
            echo_discount_alpha=alpha,
            scale_by_study_type=scale_by_study_type,
            use_text_reuse=use_text_reuse,
            handle_contradictions=handle_contradictions,
        )

        for rec in records:
            pid = str(rec.get("id", rec.get("pmid", rec.get("paper_id", "unknown"))))
            year = int(rec.get("year", 2020))
            st = str(rec.get("study_type", "in_vivo"))
            emp = bool(rec.get("is_empirical", st not in {"review", "meta_analysis"}))
            out = rec.get("outcome", None)
            txt = rec.get("text", rec.get("original_text", None))
            dag.add_paper(
                paper_id=pid,
                year=year,
                is_empirical=emp,
                study_type=st,
                outcome=out,
                text=txt,
            )

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

    def __init__(
        self,
        initial_beta: float = 0.70,
        learnable_beta: bool = True,
        alpha: float = 0.15,
        learnable_alpha: bool = False,
    ):
        super().__init__()
        self.learnable = learnable_beta
        self.learnable_alpha = learnable_alpha

        if learnable_beta:
            clamped_init = max(0.01, min(0.99, initial_beta))
            init_logit = math.log(clamped_init / (1.0 - clamped_init))
            self.beta_param = nn.Parameter(torch.tensor([init_logit], dtype=torch.float32))
        else:
            self.register_buffer("beta_param", torch.tensor([initial_beta], dtype=torch.float32))

        clamped_alpha = max(0.01, min(0.99, alpha))
        alpha_logit = math.log(clamped_alpha / (1.0 - clamped_alpha))
        if learnable_alpha:
            self.alpha_param = nn.Parameter(torch.tensor([alpha_logit], dtype=torch.float32))
        else:
            self.register_buffer("alpha_param", torch.tensor([alpha_logit], dtype=torch.float32))

    @property
    def beta(self) -> torch.Tensor:
        """Returns the dampening strength in [0, 1]."""
        if self.learnable:
            return torch.sigmoid(self.beta_param)
        return self.beta_param

    @property
    def alpha(self) -> torch.Tensor:
        """Returns the echo discount factor alpha in (0, 1)."""
        return torch.sigmoid(self.alpha_param)

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


class LearnableEchoAttenuator(nn.Module):
    """
    Standalone learnable echo attenuator for citation depth discount.
    Assigns weight w(d) = (1 - alpha)^d, with alpha bounded via sigmoid parameterization in (0, 1).
    """

    def __init__(self, initial_alpha: float = 0.15, learnable: bool = True):
        super().__init__()
        self.learnable = learnable
        clamped = max(0.01, min(0.99, initial_alpha))
        init_logit = math.log(clamped / (1.0 - clamped))
        if learnable:
            self.alpha_param = nn.Parameter(torch.tensor([init_logit], dtype=torch.float32))
        else:
            self.register_buffer("alpha_param", torch.tensor([init_logit], dtype=torch.float32))

    @property
    def alpha(self) -> torch.Tensor:
        """Bounded alpha in (0, 1)."""
        return torch.sigmoid(self.alpha_param)

    def forward(self, depth: torch.Tensor) -> torch.Tensor:
        """Computes depth-attenuated weight w = (1 - alpha)^depth."""
        base = torch.clamp(1.0 - self.alpha, min=0.01, max=0.99)
        return torch.pow(base, depth)

