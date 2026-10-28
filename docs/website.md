# Research-project website

The existing **Streamlit** frontend is preserved. No duplicate site, new frontend framework, Node build or external font service is introduced.

## Launch

From the repository root, in the existing virtual environment:

```sh
venv/bin/python -m streamlit run app/interactive_app.py
```

Open the localhost URL printed by Streamlit. `make app` uses the same entry point. The alternate existing entry is `venv/bin/python -m streamlit run app/frontend.py`. The first view is the research presentation; both entry points stop before inactive legacy prototype code.

Install Streamlit in the same Python environment if necessary. The [README](../README.md) describes the core setup. The app uses existing dependencies only. The new presentation uses standard HTML/CSS embedded into Streamlit, with a fallback for older Streamlit versions lacking `st.html`.

## Navigation

- **Project presentation:** use the section links for problem, architecture, tasks, safeguards, status and next steps. Open safeguard disclosures with Enter/Space. Use document buttons below the presentation to download the actual local Markdown files.
- **Evidence explorer:** select it in the View selector. Existing identifiers, recorded publication findings, consensus status and mechanism records remain inspectable. This is not model inference.
- Document destinations: [explanation](../explanation.md), [paper draft](research_paper_draft.md), [IDF-B draft](IDF-B_draft.md), [release contract](dataset_release.md), [evaluation contract](evaluation_orchestrator.md), [feature contract](task_a_feature_contract.md).

The site prominently states that consensus is empty, mechanisms remain unverified and empirical evaluation is blocked. There are no performance charts, manufactured predictions, user submissions or external service calls. The disabled explorer prediction control is intentional.

## Design and accessibility

A restrained light paper/teal palette, system fonts and serif headings create an academic presentation. The architecture is an ordered, text-accessible flow rather than a decorative image. Cards collapse to one column on narrow screens; status rows and navigation wrap. Native details/summary and Streamlit controls support keyboard operation. Links have visible focus outlines. Text contrasts with explicit surfaces; no color-only status is required. Reduced-motion CSS disables animation/transition; no animation is needed to understand the content.

Document downloads read only a fixed allowlist of local files. HTML contains no scripts, untrusted interpolation or remote assets. The view does not train, load fitted artifacts or call a prediction API. The existing explorer retains its fail-closed data handling.

## Verification

```sh
PYTHONDONTWRITEBYTECODE=1 venv/bin/python -m pytest tests/test_project_presentation.py tests/test_app_research_status.py -q -p no:cacheprovider
```

Real Streamlit AppTest checks render both entry points, their presentation/download controls and the evidence-explorer view. Static checks validate section targets and document availability. There is no separate frontend build command: compile the Python entry points and run these rendering checks. Exact current-session commands/results and any browser-review limitations are recorded in [PROGRESS](PROGRESS.md). Automated rendering does not constitute a full accessibility audit or clinical validation.
