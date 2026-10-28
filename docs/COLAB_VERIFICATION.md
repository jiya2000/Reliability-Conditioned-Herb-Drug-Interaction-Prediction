# Colab verification

Use a disposable Colab runtime to verify a **reviewed snapshot of the current working tree**. This is test verification, not a scientific training run. Do not regenerate labels, mechanisms or historical scores to obtain a green suite.

## 1. Transfer a clean snapshot

On the local machine, read `git status --short`. Current safeguards include modified and untracked files: `git archive HEAD` or cloning the existing remote alone would omit them unless they have first been committed to the reviewed revision.

Create a separate staging directory and copy only reviewed project files, retaining relative paths:

- `src/`, `tests/`, `benchmarks/`, `scripts/`, `configs/`, `docs/`; root `pyproject.toml`, both requirements files, `README.md`, and **`.gitignore`**. Include other tracked source/documentation directories if verifying the whole repository inventory.
- Tracked `data/benchmark/*.csv`, `data/curation/*.csv`, `data/audit/*.csv`, `data/external/*.csv`, `data/manifest.yaml`, and **`data/processed/drugbank_parsed.json`**. The last file is tracked despite its parent being ignored.
- Preserve tracked `reports/` and `results/` JSON, Markdown and CSV fixtures. Tests read historical metrics, predictions and report numbers; these are required fixtures, not disposable output. In particular retain both copies of `verified_hdi_benchmark_results.json`, `results/phase5_task_a.json`, `results/shortcut_tests.json`, `reports/master_benchmark_results.json`, `reports/benchmark_test_predictions.json`, and the audit/sensitivity JSON/Markdown files.
- Keep these required images under `reports/figures/`: `abstention_risk_coverage.png`, `sensitivity_alpha.png`, `sensitivity_context_weights.png`, `sensitivity_thresholds.png`. Other figures, checkpoints, training runs, prediction archives and notebook outputs need not be transferred for the current tests.

Use `git ls-files` to identify tracked fixtures, and `git ls-files --others --exclude-standard` to identify intended new code/docs/tests. Review file contents as well as names. Do **not** recursively ZIP the original working directory or use `git add -f` to gather ignored files. Exclude `.git/` and its remotes/configuration, `.env*`, credentials/keys, `.aws/`, personal agent/editor settings, private data, `venv/`, `.venv/`, `env/`, caches, bytecode, build/install outputs, logs, checkpoints and run directories. Do not export environment-variable dumps or authentication tokens. Remove symlinks from staging rather than following them into unrelated directories.

From the **reviewed staging directory only**, create an archive outside it:

```sh
python3 -m zipfile -c /tmp/hdi-colab.zip .
python3 -m zipfile -l /tmp/hdi-colab.zip
```

Review that inventory before upload. Record the source commit, working-tree change list and archive SHA-256 separately; a commit ID alone does not describe an uncommitted snapshot. Upload the ZIP with Colab's Files pane and extract into a new `/content/hdi` directory. Do not upload the original repository's Git metadata or reuse an old runtime's outputs.

**Private prerequisite:** `tests/test_phase4_scaffolding.py::test_private_stratification_metadata_is_gitignored` requires `data/benchmark/private/pair_strata.csv`, which is intentionally Git-ignored and absent from an ordinary Git export. If authorized for this runtime, transfer that existing file separately into the exact path after extraction; keep it ignored and out of shared archives/notebooks/logs. Otherwise record the missing prerequisite and its test failure. Do not create dummy contents, publish the file or skip the test to conceal its absence.

## 2. Create the interpreter and install dependencies

The inspected Colab archive (`hdi-verification.zip`) contains exit code **1**, Python **3.13.18**, an empty `packages.txt`, and only `/content/hdi/venv/bin/python: No module named pytest` in `pytest.log`. **No tests ran.** Empty package output does not prove why installation was missed; the log establishes missing pytest, not a scientific/test regression or a Python 3.13 incompatibility.

`pyproject.toml` declares **Python >=3.10**, not exactly 3.11. CI explicitly selects **3.11** and the recorded local verification uses **3.11.16**. Use 3.11 for this comparison instead of assuming the Colab kernel's interpreter is compatible with the entire dependency stack. The commands below install a separate managed 3.11 interpreter without replacing Colab's notebook kernel. They follow [uv's Python installation guide](https://docs.astral.sh/uv/guides/install-python/) and [environment creation guide](https://docs.astral.sh/uv/pip/environments/).

Dependency scope:

- `requirements_core.txt` supplies Torch, NumPy, pandas, scikit-learn, SciPy, Transformers, Matplotlib, PyYAML, Loguru and tqdm. Install it explicitly: `pip install -e .` alone does **not** install them because `pyproject.toml` has no runtime dependency list.
- Install **pytest, pytest-cov and NetworkX** explicitly. NetworkX is used by the tests and Task B lineage path but is missing from the core requirements. CPU Torch/Torchvision follows CI's installation recipe and uses the [official CPU wheel index](https://docs.pytorch.org/get-started/locally/).
- The broad `requirements.txt` also specifies optional application, database, chemistry, scientific NLP, tracking and compiled graph packages. Those integrations are not required by the inspected test suite; the graph builder has an optional PyG import and current tests do not exercise its PyG export. This is the complete dependency recipe for the current verification scope, not installation/verification of every optional integration. Do not blindly install the broad file and attempt to build Torch extensions to fix a missing pytest import.
- `uv.lock` only lists the editable project; it is **not** a resolved dependency lock. Requirement lower bounds do not prove compatibility of every future release. Pip resolves releases for Python 3.11; import/API smoke checks, `pip check` and collection below must pass before testing. Record the actual versions. This recipe is not an exact pinned reproduction of the local environment.

After extracting the reviewed snapshot to `/content/hdi`, run the following **whole cell**, including `%%bash`. Use a fresh runtime for another setup attempt; the cell refuses to overwrite its environment or log directory. It never assumes `/content/hdi/venv` exists or uses that potentially incomplete environment. Each Bash cell sets its own directory/environment; notebook `!export` commands do not persist reliably between cells.

```bash
%%bash
set -Eeuo pipefail
cd /content/hdi
test -f pyproject.toml
test -f requirements_core.txt
test -d tests
if [[ -e /content/hdi-verify-venv || -e /content/hdi-verification-rerun ]]; then
  echo 'Use a fresh runtime, re-extract the reviewed snapshot, then rerun setup.' >&2
  exit 1
fi
mkdir /content/hdi-verification-rerun
exec > >(tee /content/hdi-verification-rerun/setup.log) 2>&1
trap 'setup_status=$?; printf "%s\n" "$setup_status" > /content/hdi-verification-rerun/setup-exit-code.txt' EXIT
export PYTHONPATH=/content/hdi MPLBACKEND=Agg CUDA_VISIBLE_DEVICES=""

# Bootstrap uv with the available Colab interpreter; tests will use managed 3.11.
python3 --version
python3 -m pip --version
python3 -m pip install --upgrade uv
python3 -m uv --version
python3 -m uv python install 3.11
python3 -m uv venv --python 3.11 --seed /content/hdi-verify-venv

# From here, every installation and check targets the SAME new interpreter.
/content/hdi-verify-venv/bin/python --version
/content/hdi-verify-venv/bin/python -m pip install --upgrade pip setuptools wheel
/content/hdi-verify-venv/bin/python -m pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
/content/hdi-verify-venv/bin/python -m pip install -r requirements_core.txt
/content/hdi-verify-venv/bin/python -m pip install 'pytest>=7.4.0' 'pytest-cov>=4.1.0' 'networkx>=3.2.0'
/content/hdi-verify-venv/bin/python -m pip install -e .
/content/hdi-verify-venv/bin/python -m pip check
/content/hdi-verify-venv/bin/python -m pip freeze > /content/hdi-verification-rerun/packages.txt
test -s /content/hdi-verification-rerun/packages.txt
/content/hdi-verify-venv/bin/python --version > /content/hdi-verification-rerun/python.txt

/content/hdi-verify-venv/bin/python - <<'PY'
import importlib
import importlib.metadata as metadata
import sys
from pathlib import Path
import tomllib

assert sys.version_info[:2] == (3, 11), sys.version
assert Path(sys.prefix) == Path('/content/hdi-verify-venv'), sys.prefix
assert sys.prefix != sys.base_prefix, 'Expected an isolated virtual environment'
print('executable:', sys.executable)
print('prefix:', sys.prefix)
print('project Python requirement:', tomllib.loads(Path('pyproject.toml').read_text())['project']['requires-python'])
for module, distribution in [
    ('pip', 'pip'), ('torch', 'torch'), ('torchvision', 'torchvision'),
    ('numpy', 'numpy'), ('pandas', 'pandas'), ('sklearn', 'scikit-learn'),
    ('scipy', 'scipy'), ('transformers', 'transformers'),
    ('matplotlib', 'matplotlib'), ('yaml', 'PyYAML'), ('loguru', 'loguru'),
    ('tqdm', 'tqdm'), ('networkx', 'networkx'),
    ('pytest', 'pytest'), ('pytest_cov', 'pytest-cov'),
]:
    loaded = importlib.import_module(module)
    print(distribution, metadata.version(distribution), loaded.__file__)
assert not importlib.import_module('torch').cuda.is_available(), 'Expected CPU verification'
from transformers import AutoModel, AutoTokenizer
from sklearn.preprocessing import StandardScaler
import pandas as pd
StandardScaler().set_output(transform='pandas').fit_transform(pd.DataFrame({'x': [0., 1.]}))
from src.training.task_a import run_task_a
from src.training.task_b import fit_task_b
print('Interpreter, dependency imports and project API smoke checks passed')
PY
/content/hdi-verify-venv/bin/python -m pytest --version

# A test invokes git ls-files: create a NEW local-only snapshot index, no remote.
# The source archive must already have excluded original Git metadata/secrets.
if [[ ! -d .git ]]; then
  git init -q
fi
git add --all
test -f data/processed/drugbank_parsed.json
if [[ -z "$(git ls-files -- data/processed/drugbank_parsed.json)" ]]; then
  git add -f -- data/processed/drugbank_parsed.json
fi
# Do not force-add any private file. See the private prerequisite in section 1.

/content/hdi-verify-venv/bin/python - <<'PY'
from pathlib import Path
import hashlib
p = Path('results/verified_hdi_benchmark_results.json')
actual = hashlib.sha256(p.read_bytes()).hexdigest()
expected = '67e072a5fc06c4fa2b138a51522b300bb53685b6aba24acd3f9cc35f546c2f60'
assert actual == expected, f'Historical fixture differs: {actual}'
Path('/content/hdi-verification-rerun/historical-result-sha256.txt').write_text(actual + '\n')
PY
# Collect the entire suite to catch missing imports/plugins before execution.
/content/hdi-verify-venv/bin/python -m pytest --collect-only -q -p no:cacheprovider > /content/hdi-verification-rerun/collection.log 2>&1
tail -n 5 /content/hdi-verification-rerun/collection.log
touch /content/hdi-verification-rerun/setup-ok
```

If any line fails, **stop before the test cell**. Read `setup.log`, `setup-exit-code.txt` and (if present) `collection.log`. The `setup-ok` marker is written only after installation, interpreter/import checks, fixture integrity and full collection succeed. If pip itself is missing, setup fails at its explicit check; it cannot silently continue to pytest. A failed bootstrap/download/resolution is an environment/setup issue, not the historical AUC alert.

No project/API/database secrets are needed. The full suite includes a `slow` BiomedBERT test that can download the public `microsoft/BiomedNLP-BiomedBERT-base-uncased-abstract` model. Allow network/disk/time. Its deterministic fallback can conceal unavailable pretrained weights, so inspect warnings and report fallback coverage explicitly. Do not add `-m 'not slow'`, offline flags, xfails or other exclusions and call that a full-suite verification.

## 3. Run tests and capture the actual exit code

Only after setup succeeds, run this whole cell. It uses absolute paths and repeats the required environment settings. The extra JUnit output and summary options capture evidence; they do not exclude tests or change assertions. Run the test cell once per fresh setup to avoid overwriting evidence.

```bash
%%bash
set -Eeuo pipefail
cd /content/hdi
export PYTHONPATH=/content/hdi MPLBACKEND=Agg CUDA_VISIBLE_DEVICES=""
test -f /content/hdi-verification-rerun/setup-ok
test "$(cat /content/hdi-verification-rerun/setup-exit-code.txt)" = 0
test ! -e /content/hdi-verification-rerun/exit-code.txt
/content/hdi-verify-venv/bin/python -m pip check
/content/hdi-verify-venv/bin/python -m pytest --version

# Disable errexit only for pytest so failures still produce logs and exit-code.txt.
set +e
/content/hdi-verify-venv/bin/python -m pytest -q -p no:cacheprovider -ra \
  --junitxml=/content/hdi-verification-rerun/junit.xml \
  2>&1 | tee /content/hdi-verification-rerun/pytest.log
pipeline_status=("${PIPESTATUS[@]}")
set -e
pytest_status=${pipeline_status[0]}
tee_status=${pipeline_status[1]}
printf '%s\n' "$pytest_status" > /content/hdi-verification-rerun/exit-code.txt
printf '%s\n' "$tee_status" > /content/hdi-verification-rerun/logging-exit-code.txt
tail -n 50 /content/hdi-verification-rerun/pytest.log > /content/hdi-verification-rerun/test-summary.txt
cat /content/hdi-verification-rerun/test-summary.txt
# Preserve pytest's exit status; logging failures are separately recorded.
exit "$pytest_status"
```

A red cell can be expected if the preserved historical alert fires. **Do not infer the reason from exit code 1 alone.** Pytest exit 2 (interrupted/collection problem), 3 (internal error), 4 (usage error) or 5 (no tests collected) requires investigation. Setup/collection errors, missing fixtures, failed downloads and additional test failures are separate findings. `logging-exit-code.txt` must also be zero before treating the log as complete.

## 4. Separate the historical alert from new findings

The one known historical failure is exactly:

```text
tests/test_ablation_leakage.py::test_no_actual_benchmark_model_exceeds_threshold
Non-reference model 'no_context_conflict' has suspicious ROC-AUC 1.0000 > 0.98!
```

The setup hash check verifies the preserved result fixture before testing. Do not raise the threshold, mark the test xfail, rename the variant or regenerate the fixture to suppress it. The prior forensic investigation established the saved perfect ranking and identified outcome-derived count/strength leakage in the historical pipeline. Additional failures must be investigated independently.

After the test cell, run this cell **even if pytest returned nonzero**. It creates a structured summary with passing, failing, error and skipped counts, preserving all additional failures and skip reasons. A missing/malformed JUnit report means no complete structured result is available, not a test pass. JUnit counts represent report entries and may include collection errors; compare them with the collected/executed totals in `pytest.log`. The check recognizes the historical failure only when both its test identity and exact alert text match. Review any skips before calling them expected.

```bash
%%bash
set -Eeuo pipefail
/content/hdi-verify-venv/bin/python - <<'PY'
import json
from pathlib import Path
import xml.etree.ElementTree as ET

out = Path('/content/hdi-verification-rerun')
report = out / 'junit.xml'
assert report.is_file(), 'Missing JUnit report: inspect setup and pytest logs'
root = ET.parse(report).getroot()
summary = dict(pytest_exit_code=int((out / 'exit-code.txt').read_text()),
               passed=0, failed=0, errors=0, skipped=0,
               known_historical_alert=[], additional_failures=[], skips=[])
alert = "Non-reference model 'no_context_conflict' has suspicious ROC-AUC 1.0000 > 0.98!"
for case in root.iter('testcase'):
    node = case.get('classname', '') + '::' + case.get('name', '')
    failure, error, skip = case.find('failure'), case.find('error'), case.find('skipped')
    if failure is not None or error is not None:
        item = failure if failure is not None else error
        summary['failed' if failure is not None else 'errors'] += 1
        text = item.get('message', '') + '\n' + (item.text or '')
        historical = (failure is not None
            and case.get('classname', '') == 'tests.test_ablation_leakage'
            and case.get('name') == 'test_no_actual_benchmark_model_exceeds_threshold'
            and alert in text)
        summary['known_historical_alert' if historical else 'additional_failures'].append(
            dict(test=node, message=text))
    elif skip is not None:
        summary['skipped'] += 1
        summary['skips'].append(dict(test=node, message=skip.get('message', '')))
    else:
        summary['passed'] += 1
summary['only_known_failure_observed'] = (
    summary['pytest_exit_code'] == 1
    and len(summary['known_historical_alert']) == 1
    and not summary['additional_failures'])
payload = json.dumps(summary, indent=2)
(out / 'verification-summary.json').write_text(payload + '\n')
print(payload)
print('Review pytest.log for collection counts, warnings, download fallbacks and interruptions.')
PY
```

`only_known_failure_observed` is a classification of failures, **not** a full-suite pass or confirmation of pretrained-model coverage. Missing private `pair_strata.csv` is an additional prerequisite failure, not this historical alert. Local focused Task A/B contract runs do not establish the full-suite count in Colab.

Download the evidence in a final cell (logs/version information only; no private dataset):

```python
import shutil
from google.colab import files
archive = shutil.make_archive('/content/hdi-verification-rerun', 'zip',
                              root_dir='/content', base_dir='hdi-verification-rerun')
files.download(archive)
```

Keep the original source ZIP and its commit/working-tree manifest. Tests can write artifacts in the disposable checkout; do not copy regenerated reports/results over local history. These instructions have been inspected locally; no successful Colab installation or full-suite run is claimed until the new evidence is reviewed.
