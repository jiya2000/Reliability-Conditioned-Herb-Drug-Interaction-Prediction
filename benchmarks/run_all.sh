#!/usr/bin/env bash
set -euo pipefail

# ==============================================================================
# Master Reproducibility Script: Reliability-Conditioned HDI Prediction
# Executes all evaluation and benchmark components sequentially:
# 1. Leakage Probe
# 2. Hyperparameter Sensitivity Analysis
# 3. Verified Botanical HDI Benchmark (All Baselines & Ablations)
# 4. Citation DAG Audit
# 5. External Validation (Zero-Refitting on NaPDI Template)
# 6. Auditing Verification Weak Spots & Statistical Hardening
# 7. Master Results Aggregation
# 8. Cross-Deliverable Consistency Audit
# ==============================================================================

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${ROOT_DIR}"
export PYTHONPATH="${ROOT_DIR}:${PYTHONPATH:-}"

PYTHON_BIN="${ROOT_DIR}/venv/bin/python"
if [ ! -f "${PYTHON_BIN}" ]; then
    PYTHON_BIN="python"
fi

echo "========================================================================"
echo "REPRODUCING ALL BENCHMARKS & EVALUATIONS"
echo "Root Directory: ${ROOT_DIR}"
echo "Python Executable: ${PYTHON_BIN}"
echo "========================================================================"

echo ""
echo ">>> [1/8] Running Leakage Probe (Phase 1)..."
"${PYTHON_BIN}" benchmarks/leakage_probe.py

echo ""
echo ">>> [2/8] Running Hyperparameter Sensitivity Analysis (Phase 4)..."
"${PYTHON_BIN}" benchmarks/sensitivity_analysis.py

echo ""
echo ">>> [3/8] Running Verified Botanical HDI Benchmark (Phases 1-6)..."
"${PYTHON_BIN}" benchmarks/run_verified_hdi_benchmark.py

echo ""
echo ">>> [4/8] Auditing Citation DAG Quality & Refutations (Phase 9)..."
"${PYTHON_BIN}" scripts/audit_citation_dag.py

echo ""
echo ">>> [5/8] Running External Validation without Refitting (Phase 10)..."
"${PYTHON_BIN}" benchmarks/external_validation.py

echo ""
echo ">>> [6/8] Auditing Verification Weak Spots & Statistical Hardening..."
"${PYTHON_BIN}" benchmarks/audit_weak_spots.py

echo ""
echo ">>> [7/8] Aggregating Master Benchmark Results..."
"${PYTHON_BIN}" benchmarks/aggregate_master_results.py

echo ""
echo ">>> [8/8] Auditing Cross-Deliverable Metric Consistency..."
"${PYTHON_BIN}" scripts/check_consistency.py

echo ""
echo "========================================================================"
echo "✅ MASTER REPRODUCTION COMPLETE."
echo "Master report written to: reports/master_benchmark_results.json"
echo "All cross-deliverable consistency checks passed."
echo "========================================================================"
