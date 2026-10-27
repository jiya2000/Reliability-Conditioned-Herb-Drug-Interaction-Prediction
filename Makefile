.PHONY: reproduce test test-fast bench app audit-dag external-val help

VENV_PYTHON ?= ./venv/bin/python
VENV_PYTEST ?= ./venv/bin/pytest
VENV_STREAMLIT ?= ./venv/bin/streamlit

help:
	@echo "Available commands for Reliability-Conditioned HDI Prediction:"
	@echo "  make reproduce     - Run full benchmark suite and aggregate master results"
	@echo "  make test          - Run full pytest suite with verbose output"
	@echo "  make test-fast     - Run full pytest suite quietly"
	@echo "  make bench         - Run verified botanical HDI benchmark across all baselines"
	@echo "  make app           - Launch interactive Streamlit decision support app"
	@echo "  make audit-dag     - Audit citation DAG structure and study-type discounting"
	@echo "  make external-val  - Run zero-refitting external validation on NaPDI cohort"

reproduce:
	@bash benchmarks/run_all.sh

test:
	@$(VENV_PYTEST) -v

test-fast:
	@$(VENV_PYTEST) -q

bench:
	@$(VENV_PYTHON) benchmarks/run_verified_hdi_benchmark.py

app:
	@$(VENV_STREAMLIT) run app/interactive_app.py

audit-dag:
	@$(VENV_PYTHON) scripts/audit_citation_dag.py

external-val:
	@$(VENV_PYTHON) benchmarks/external_validation.py
