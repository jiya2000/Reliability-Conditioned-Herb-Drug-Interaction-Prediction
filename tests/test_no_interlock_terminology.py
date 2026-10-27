"""Unit test verifying that prohibited 'interlock' terminology is absent from the repository.

Mandated by TASK 3:
Add a test or CI grep that fails if "interlock" (case-insensitive)
appears anywhere in tracked files except reports/revision_log.md and the verification reports.
"""

from pathlib import Path
import subprocess
import re
import pytest

BINARY_EXTENSIONS = {
    ".png", ".jpg", ".jpeg", ".pdf", ".pt", ".pth", ".pkl", ".pyc", ".ico", ".svg"
}

ALLOWED_EXEMPT_PATHS = {
    "reports/revision_log.md",
    "reports/fix_summary.md",
    "tests/test_no_interlock_terminology.py",
}


def test_no_prohibited_interlock_terminology():
    """Fail if 'interlock' (case-insensitive) appears in tracked files outside exempted reports."""
    repo_root = Path(__file__).resolve().parent.parent

    # Retrieve all git-tracked files
    res = subprocess.run(
        ["git", "ls-files"],
        cwd=repo_root,
        capture_output=True,
        text=True,
        check=True,
    )
    tracked_files = [f.strip() for f in res.stdout.splitlines() if f.strip()]

    # Also check tracked + untracked project files in docs, src, scripts, patent, paper, app, benchmarks
    additional_paths = []
    for search_dir in ["docs", "src", "scripts", "patent", "paper", "app", "benchmarks"]:
        p = repo_root / search_dir
        if p.exists():
            for item in p.rglob("*"):
                if item.is_file():
                    rel = str(item.relative_to(repo_root))
                    if rel not in tracked_files:
                        additional_paths.append(rel)

    all_files_to_check = set(tracked_files + additional_paths)

    violations = []
    pattern = re.compile(r"\binterlock(?:s|ed|ing)?\b|interlock", re.IGNORECASE)

    for rel_path in sorted(all_files_to_check):
        # Exempted files
        if rel_path in ALLOWED_EXEMPT_PATHS or rel_path.startswith("reports/verification"):
            continue

        file_path = repo_root / rel_path
        if not file_path.exists():
            continue

        # Skip binaries
        if file_path.suffix.lower() in BINARY_EXTENSIONS:
            continue

        try:
            content = file_path.read_text(encoding="utf-8", errors="ignore")
        except Exception:
            continue

        for line_no, line in enumerate(content.splitlines(), start=1):
            if pattern.search(line):
                violations.append(f"{rel_path}:{line_no}: {line.strip()}")

    assert not violations, (
        f"Found {len(violations)} prohibited occurrence(s) of 'interlock':\n"
        + "\n".join(violations[:20])
    )
