#!/usr/bin/env python3
"""
Cross-deliverable Metric and Consistency Verifier.

Audits paper/, patent/, app/, README.md, notebooks/, and docs/ to verify:
1. Zero occurrence of obsolete/leaky metrics (0.9762, 0.9707, 0.9167, 0.0846, 92.31%, T* = 1.48).
2. Headline metrics match reports/master_benchmark_results.json.
3. Exits with non-zero exit code if any discrepancies are found.
"""

import sys
import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

OBSOLETE_TOKENS = [
    "0.9762",
    "0.9707",
    "0.9167",
    "0.0846",
    "92.31%",
    "T* = 1.48",
    "T*\\approx 1.48",
    "T = 1.48",
]

# Paths to audit for obsolete tokens
AUDIT_DIRS = ["paper", "patent", "app", "notebooks", "docs"]
AUDIT_FILES = ["README.md"]

# Deliverables that must explicitly contain the canonical leakage-free metrics
CANONICAL_CHECK_FILES = [
    ROOT / "README.md",
    ROOT / "paper" / "main.tex",
    ROOT / "paper" / "manuscript.md",
    ROOT / "patent" / "patent_specification.md",
    ROOT / "patent" / "VIT_Invention_Disclosure_Form_IDF_B.md",
]


def load_master_metrics():
    master_path = ROOT / "reports" / "master_benchmark_results.json"
    if not master_path.exists():
        raise FileNotFoundError(f"Master benchmark results not found at {master_path}")
    with open(master_path, "r") as f:
        data = json.load(f)
    headline = data["verified_hdi_benchmark"]["headline_metrics"]
    return {
        "roc_auc": f"{headline['roc_auc']:.4f}",  # '0.8512'
        "pr_auc": f"{headline['pr_auc']:.4f}",    # '0.8794'
        "f1": f"{headline['f1']:.4f}",            # '0.7826'
        "brier": f"{headline['brier_score']:.4f}",# '0.1554'
        "acc_pct": f"{headline['accuracy'] * 100:.2f}%",  # '80.77%'
    }


def audit_obsolete_tokens():
    errors = []
    files_to_check = [ROOT / f for f in AUDIT_FILES]
    for d in AUDIT_DIRS:
        dp = ROOT / d
        if dp.exists():
            for p in dp.rglob("*"):
                if p.is_file() and p.suffix in [".md", ".tex", ".py", ".ipynb", ".txt"]:
                    files_to_check.append(p)

    for fp in files_to_check:
        if not fp.exists():
            continue
        try:
            content = fp.read_text(encoding="utf-8", errors="ignore")
        except Exception as e:
            errors.append(f"Could not read {fp}: {e}")
            continue

        for token in OBSOLETE_TOKENS:
            if token in content:
                # Find matching lines
                lines = content.splitlines()
                for line_idx, line in enumerate(lines, 1):
                    if token in line:
                        errors.append(
                            f"Obsolete token '{token}' found in {fp.relative_to(ROOT)}:{line_idx}: {line.strip()[:100]}"
                        )
    return errors


def audit_canonical_metrics(metrics):
    errors = []
    for fp in CANONICAL_CHECK_FILES:
        if not fp.exists():
            errors.append(f"Required deliverable file does not exist: {fp.relative_to(ROOT)}")
            continue
        content = fp.read_text(encoding="utf-8", errors="ignore")

        # Check ROC-AUC (0.8512)
        if metrics["roc_auc"] not in content:
            errors.append(
                f"Missing canonical ROC-AUC {metrics['roc_auc']} in {fp.relative_to(ROOT)}"
            )
        # Check PR-AUC (0.8794)
        if metrics["pr_auc"] not in content:
            errors.append(
                f"Missing canonical PR-AUC {metrics['pr_auc']} in {fp.relative_to(ROOT)}"
            )
        # Check Brier (0.1554)
        if metrics["brier"] not in content:
            errors.append(
                f"Missing canonical Brier score {metrics['brier']} in {fp.relative_to(ROOT)}"
            )

    return errors


def main():
    print("=" * 70)
    print("RUNNING CROSS-DELIVERABLE CONSISTENCY CHECK")
    print("=" * 70)

    try:
        metrics = load_master_metrics()
        print(f"Loaded master benchmark headline metrics:")
        print(f"  ROC-AUC:  {metrics['roc_auc']}")
        print(f"  PR-AUC:   {metrics['pr_auc']}")
        print(f"  F1-Score: {metrics['f1']}")
        print(f"  Brier:    {metrics['brier']}")
        print(f"  Accuracy: {metrics['acc_pct']}")
    except Exception as e:
        print(f"FATAL: Failed to load master metrics: {e}")
        sys.exit(1)

    print("\n1. Scanning for obsolete tokens across paper, patent, app, notebooks, docs, README...")
    obsolete_errors = audit_obsolete_tokens()
    if obsolete_errors:
        print(f"❌ Found {len(obsolete_errors)} obsolete token violations:")
        for err in obsolete_errors:
            print(f"   - {err}")
    else:
        print("✅ Zero obsolete tokens found in audited deliverables.")

    print("\n2. Verifying canonical headline metrics in key deliverable files...")
    canonical_errors = audit_canonical_metrics(metrics)
    if canonical_errors:
        print(f"❌ Found {len(canonical_errors)} missing canonical metric violations:")
        for err in canonical_errors:
            print(f"   - {err}")
    else:
        print("✅ Canonical headline metrics present across all key deliverable files.")

    total_errors = len(obsolete_errors) + len(canonical_errors)
    print("\n" + "=" * 70)
    if total_errors == 0:
        print("CONSISTENCY AUDIT PASSED: All deliverables match master benchmark.")
        print("=" * 70)
        sys.exit(0)
    else:
        print(f"CONSISTENCY AUDIT FAILED: {total_errors} issue(s) detected.")
        print("=" * 70)
        sys.exit(1)


if __name__ == "__main__":
    main()
