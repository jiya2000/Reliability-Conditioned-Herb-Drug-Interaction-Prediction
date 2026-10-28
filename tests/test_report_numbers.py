"""
Test Report Numbers Integrity.

Enforces Rule:
No number in any report may be typed by hand.
Every number with 3+ decimals in reports/*.md or docs/*.md must be found
in results/*.json (rounded to 3 or 4 decimals), or the test fails.
"""

import glob
import re
import json
import pytest
from pathlib import Path


def load_all_json_numbers():
    json_files = glob.glob("results/*.json")
    all_numbers = set()

    def extract_from_obj(obj):
        if isinstance(obj, (int, float)):
            f = float(obj)
            all_numbers.add(f)
            all_numbers.add(-f)
            all_numbers.add(round(f, 4))
            all_numbers.add(round(-f, 4))
            all_numbers.add(round(f, 3))
            all_numbers.add(round(-f, 3))
        elif isinstance(obj, str):
            # parse numbers inside strings e.g. '< 0.0010', '0.8572'
            for m in re.findall(r"[-+]?\d*\.?\d+", obj):
                try:
                    f = float(m)
                    all_numbers.add(f)
                    all_numbers.add(-f)
                    all_numbers.add(round(f, 4))
                    all_numbers.add(round(-f, 4))
                    all_numbers.add(round(f, 3))
                    all_numbers.add(round(-f, 3))
                except ValueError:
                    pass
        elif isinstance(obj, dict):
            for v in obj.values():
                extract_from_obj(v)
        elif isinstance(obj, list):
            for v in obj:
                extract_from_obj(v)

    for jf in json_files:
        try:
            with open(jf, "r") as f:
                d = json.load(f)
                extract_from_obj(d)
        except Exception as e:
            pytest.fail(f"Could not load JSON file {jf}: {e}")

    return all_numbers


@pytest.fixture(scope="module")
def json_numbers():
    return load_all_json_numbers()


def test_shortcut_report_numbers_match_json(json_numbers):
    """
    Assert that every number with 3+ decimals in reports/shortcut_tests.md
    originates strictly from results/*.json.
    """
    path = Path("reports/shortcut_tests.md")
    assert path.exists(), "reports/shortcut_tests.md does not exist"
    
    content = path.read_text()
    dec_re = re.compile(r"[-+]?\b\d+\.\d{3,}\b")
    matches = dec_re.findall(content)
    
    missing = []
    for m in matches:
        val = float(m)
        found = any(abs(jnum - val) < 1e-4 for jnum in json_numbers)
        if not found:
            missing.append(m)
            
    assert not missing, f"Found ungrounded numbers in {path}: {set(missing)}"


def test_all_reports_and_docs_numbers_match_json(json_numbers):
    """
    Assert that all active numbers with 3+ decimals across reports/*.md and docs/*.md
    are found in results/*.json (rounded).
    """
    md_files = glob.glob("reports/*.md") + glob.glob("docs/*.md")
    dec_re = re.compile(r"[-+]?\b\d+\.\d{3,}\b")
    
    missing_by_file = {}
    for mf in md_files:
        with open(mf, "r") as f:
            content = f.read()
            
        # Ignore strike-through text (superseded claims per Item 1)
        content_clean = re.sub(r"~~.*?~~", "", content, flags=re.DOTALL)
        # Ignore URLs and DOIs
        content_clean = re.sub(r"https?://\S+", "", content_clean)
        content_clean = re.sub(r"10\.\d{4,}/\S+", "", content_clean)
        
        matches = dec_re.findall(content_clean)
        missing = []
        for m in matches:
            val = float(m)
            found = any(abs(jnum - val) < 1e-4 for jnum in json_numbers)
            if not found:
                missing.append(m)
                
        if missing:
            missing_by_file[mf] = list(set(missing))
            
    assert not missing_by_file, f"Found ungrounded 3+ decimal numbers in reports/docs: {missing_by_file}"


def test_permutation_p_value_format_in_json():
    """
    Assert that permutation test p-values when zero null iterations reach true AUC
    are set to string '< 1/(n_perm+1)' in results/shortcut_tests.json.
    """
    with open("results/shortcut_tests.json", "r") as f:
        res = json.load(f)
        
    perm = res.get("permutation_test", {})
    assert len(perm) > 0, "No permutation_test found in results/shortcut_tests.json"
    
    for mode, models in perm.items():
        for model_name, p_dict in models.items():
            p_val = p_dict.get("p_value")
            assert p_val in ["< 1/(n_perm+1)", "<1/(n_perm+1)"], (
                f"Expected p_value '< 1/(n_perm+1)' or '<1/(n_perm+1)' for {mode}/{model_name}, got {p_val}"
            )
