# Reliability-Conditioned Herb–Drug Interaction Prediction

An auditable research pipeline for studying how sourced mechanisms and the reliability of published evidence can support herb–drug interaction prediction.

> **Current release: empirical evaluation is blocked.** Independent consensus labels are empty, mechanistic records remain unverified, and Task B evidence approval/statistical support are incomplete. Public production APIs issue no execution authority. No clinical readiness, predictive superiority or nominal coverage guarantee is claimed.

## Start here

- [Beginner-friendly explanation, architecture and professor pitches](explanation.md)
- [Research-project website: launch and navigation](docs/website.md)
- [Research-paper draft](docs/research_paper_draft.md) — methodology/engineering draft, not an empirical paper
- [IDF-B technical draft](docs/IDF-B_draft.md) — follows the official repository form; institutional fields require review
- [Exact verification commands, results and limitations](docs/PROGRESS.md)

## Prepare independent evidence review

[Evidence review protocol and packet commands](docs/evidence_review_protocol.md) connect the existing blinded consensus workflow to frozen candidate inputs and a pending mechanism-review worksheet. This preparation does not assign labels, verify biological claims, or authorize evaluation.

[Public literature pilot and readiness report](docs/review_batches/public_pilot_20261030/README.md) provide actual primary-source candidates with unresolved review workbooks. They are preparation artifacts, not an independently labeled dataset.

[Expanded biological curation panel](docs/mechanistic_panel_expansion.md) explains the current five-enzyme/one-transporter baseline and prepares a broader 16-target evidence-collection scope without claiming predictive improvement.

[Task B source inventory and discussion-signal intake](docs/task_b_sources_and_discussions.md) explains the current paper candidates and the separate quarantined metadata workflow for potential discussion signals.

## Research approach

**Task A** studies mechanism-only prediction using the approved six pathway effects and their absolute sum, with existing logistic regression and histogram gradient boosting families. Publication findings, raw text and entity encodings are excluded from its features. Independent consensus supplies the intended target.

**Task B** investigates evidence adjustment of a mechanistic prior. Study direction is distinct from study reliability: design/lineage descriptors support agreement supervision against external consensus, while directional findings enter explicit aggregation. Its snapshot adapter keeps prior/reliability fitting and residual tuning blocked pending a reviewed evidence-release contract.

Fit, tuning, probability calibration, conformal calibration and assessment have separate roles. Pair-, herb-, drug- and Task B source-component constraints are validated without silently changing the requested regime. Snapshot binding and registered fit provenance detect state changes. Private algorithms are exercised with synthetic fixtures and cannot authorize public execution.

## Run the existing Streamlit website

From the repository root, use the existing environment:

```sh
venv/bin/python -m streamlit run app/interactive_app.py
```

The default **Project presentation** view explains the problem, architecture, tasks, safeguards, status and next steps. Its document buttons download local Markdown files. Select **Evidence explorer** to retain the existing read-only source/curation inspection. Both entry points stop before legacy prototype code. No startup training, prediction API or fake chart is used. `app/frontend.py` remains an alternate Streamlit entry.

For a new environment:

```sh
python3 -m venv venv
venv/bin/python -m pip install -r requirements_core.txt
venv/bin/python -m pip install 'networkx>=3.2' 'pytest>=7' 'streamlit>=1.29' 'plotly>=5.18'
venv/bin/python -m pip install --no-deps -e .
```

The broader `requirements.txt` includes optional graph/NLP/database packages; installation may require platform-specific work. No new package is required by the presentation. See [explanation.md](explanation.md#11-environment-commands-and-website-navigation) for roles and setup details.

## Focused verification

```sh
PYTHONDONTWRITEBYTECODE=1 venv/bin/python -m pytest tests/test_pipeline_components.py tests/test_evaluation_orchestrator.py tests/test_release_preflight.py tests/test_evaluation_protocol.py tests/test_task_a_training.py tests/test_task_b_training.py tests/test_task_a_feature_contract.py tests/test_redesigned_leakage_regression.py tests/test_calibration_audit.py tests/test_conformal_abstention.py tests/test_conformal_overhaul.py tests/test_interaction_model_and_calibration.py tests/test_trainer_conformal.py tests/test_app_research_status.py tests/test_project_presentation.py -q -p no:cacheprovider
```

These are engineering tests, including synthetic algorithm checks. A passing count is not predictive performance. The website has no separate Node/JavaScript build; Python compilation and real Streamlit AppTest rendering are its relevant checks. No full-suite, benchmark reproduction or Colab run is needed to view this project.

## Implementation and documentation map

| Area | Starting point |
|---|---|
| Source/consensus schema | [evidence schema](docs/evidence_schema.md), `src/data/evidence_schema.py` |
| Mechanistic input and approved feature boundary | [mechanisms](docs/mechanistic_features.md), [Task A contract](docs/task_a_feature_contract.md) |
| Explicit role and group isolation | `src/training/evaluation_protocol.py` |
| Dataset eligibility | [release requirements](docs/dataset_release.md), `src/training/release_preflight.py` |
| Public execution boundary / private test algorithms | [evaluation contract](docs/evaluation_orchestrator.md) |
| Presentation / existing explorer | `app/project_presentation.py`, `app/research_status.py` |
| Research intent | [project specification](PROJECT_SPEC.md), [task definitions](docs/task_definition.md) — narrower current contracts govern supported behavior |

## Integrity and intended use

Historical benchmark outputs, prior patent drafts and older graph/text prototypes are retained unchanged for provenance. They can contain superseded claims and must not be presented as empirical validation of the redesigned tasks. Do not run reproduction scripts to overwrite them merely to obtain a current headline metric.

The public Phase 3.2 facade deliberately raises `PermissionError` for execution, including caller-supplied `eligible=True` records. Private synthetic code is not an empirical workaround or an interpreter security sandbox. Current release eligibility remains false, and a reviewed orchestrator-owned authorization issuer does not yet exist.

Research software only: no recommendation to use or avoid a herb–drug combination, no co-administration clearance, and no substitute for clinical judgment. Source traceability and software tests cannot establish clinical validity or legal/patent status. The private annotation workspace is excluded from inspection and publication.

Licensed under the [MIT License](LICENSE).
