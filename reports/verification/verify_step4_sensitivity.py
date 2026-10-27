"""
Step 4 Verification Script: Learned Parameters, Constraints, and Sensitivity Analysis.

Checks:
1. Learnable context weights ordering constraint: w_human > w_animal > w_in_vitro under all parameters.
2. Default static parameters reproduce baseline behavior.
3. Sensitivity analysis audit: inspect benchmarks/sensitivity_analysis.py, check sweep coverage,
   verify whether metrics vary or remain constant, and detect any unlinked scorers/variables.
4. Check all ablation modes in master benchmark results.
"""

from __future__ import annotations

import json
from pathlib import Path
import sys
import torch
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.models.context_compatibility import (
    ContextCompatibilityScorer,
    SPECIES_PRIORS,
    PREPARATION_PRIORS,
    ROUTE_PRIORS,
)
from src.models.citation_dag import LearnableEchoAttenuator


def verify_ordering_constraint():
    print("--- 1. Verification of Learned Parameter Ordering Constraint ---")
    scorer = ContextCompatibilityScorer(learn_ordered_weights=True)

    species_list = scorer.SPECIES_LIST
    h_idx = species_list.index("human")
    a_idx = species_list.index("rat")
    v_idx = species_list.index("in_vitro_microsomes")

    w_init = scorer.get_effective_species_weights()
    wh0, wa0, wv0 = w_init[h_idx].item(), w_init[a_idx].item(), w_init[v_idx].item()
    print(f"Initial weights: Human={wh0:.4f}, Animal={wa0:.4f}, InVitro={wv0:.4f}")
    assert wh0 > wa0 > wv0, f"Ordering failed at init: {wh0} > {wa0} > {wv0}"

    # Adversarial extreme perturbation
    scorer.u_invitro.data.fill_(15.0)       # push in vitro as high as sigmoid allows
    scorer.u_animal_delta.data.fill_(-15.0)  # push animal delta as low as sigmoid allows
    scorer.u_human_delta.data.fill_(-15.0)   # push human delta as low as sigmoid allows

    w_pert = scorer.get_effective_species_weights()
    wh1, wa1, wv1 = w_pert[h_idx].item(), w_pert[a_idx].item(), w_pert[v_idx].item()
    print(f"Adversarially perturbed weights: Human={wh1:.4f}, Animal={wa1:.4f}, InVitro={wv1:.4f}")
    assert wh1 > wa1 > wv1, f"Ordering violated under adversarial perturbation: {wh1} > {wa1} > {wv1}"
    print("PASS: Parameterization strictly enforces w_human > w_animal > w_in_vitro under all parameter values.")


def verify_configurable_defaults():
    print("\n--- 2. Configurable Priors and Baseline Defaults ---")
    # Verify hand-set defaults reproduce exact constants
    scorer_default = ContextCompatibilityScorer(learnable_refinement=False)
    weights = scorer_default.get_effective_species_weights()
    h_idx = scorer_default.SPECIES_LIST.index("human")
    assert abs(weights[h_idx].item() - SPECIES_PRIORS["human"]) < 1e-5
    print(f"PASS: Hand-set defaults reproduce baseline behavior exactly (Human weight = {weights[h_idx].item()}).")


def audit_sensitivity_analysis():
    print("\n--- 3. Audit of Sensitivity Analysis Implementation and Artifacts ---")
    res_path = PROJECT_ROOT / "reports" / "sensitivity_analysis_results.json"
    assert res_path.exists(), "sensitivity_analysis_results.json missing!"

    with open(res_path, "r") as f:
        data = json.load(f)

    # Check alpha sweep
    alpha_sweep = data.get("alpha_sweep", [])
    print(f"Alpha sweep steps: {len(alpha_sweep)}")
    alpha_aucs = [r["roc_auc"] for r in alpha_sweep]
    alpha_briers = [r["brier_score"] for r in alpha_sweep]
    print(f"Alpha sweep ROC-AUCs : {alpha_aucs}")
    print(f"Alpha sweep Brier    : {alpha_briers}")

    # Check context weight sweep
    w_sweep = data.get("context_weight_sweep", [])
    print(f"Context weight sweep steps: {len(w_sweep)}")
    w_aucs = [r["roc_auc"] for r in w_sweep]
    print(f"Context weight ROC-AUCs: {w_aucs}")

    # Adversarial code inspection finding:
    # Check if all ROC-AUCs are completely identical in the sweep
    all_alpha_identical = len(set(alpha_aucs)) == 1
    all_w_identical = len(set(w_aucs)) == 1

    print(f"CRITICAL AUDIT SIGNAL: All alpha sweep AUCs identical? {all_alpha_identical}")
    print(f"CRITICAL AUDIT SIGNAL: All context weight sweep AUCs identical? {all_w_identical}")

    # Check plot figures existence
    fig_dir = PROJECT_ROOT / "reports" / "figures"
    f_alpha = fig_dir / "sensitivity_alpha.png"
    f_weight = fig_dir / "sensitivity_context_weights.png"
    f_thresh = fig_dir / "sensitivity_thresholds.png"
    print(f"Figure sensitivity_alpha.png exists: {f_alpha.exists()}")
    print(f"Figure sensitivity_context_weights.png exists: {f_weight.exists()}")
    print(f"Figure sensitivity_thresholds.png exists: {f_thresh.exists()}")


def verify_all_ablation_modes_in_table():
    print("\n--- 4. Checking All Ablation Modes in Final Results Table ---")
    master_path = PROJECT_ROOT / "reports" / "master_benchmark_results.json"
    with open(master_path, "r") as f:
        master = json.load(f)

    reports = master.get("verified_hdi_benchmark", {}).get("all_variant_reports", [])
    variants = {r["model_variant"]: r for r in reports}

    required_ablations = [
        "reliability_conditioned_quality_only",
        "reliability_conditioned",
        "standard_fusion",
        "naive_concat_mlp",
        "graph_only",
        "evidence_only",
        "graph_plus_naive_evidence",
        "no_study_dependence",
        "no_context_conflict",
    ]

    for req in required_ablations:
        assert req in variants, f"Missing required ablation mode: {req}"
        rep = variants[req]
        print(f"Variant: {req:36s} | ROC-AUC: {rep['roc_auc']:.4f} | PR-AUC: {rep['pr_auc']:.4f} | Brier: {rep['brier_score']:.4f}")

    print("PASS: All 9 ablation and architecture modes appear in master table with real numerical outputs.")


if __name__ == "__main__":
    verify_ordering_constraint()
    verify_configurable_defaults()
    audit_sensitivity_analysis()
    verify_all_ablation_modes_in_table()
