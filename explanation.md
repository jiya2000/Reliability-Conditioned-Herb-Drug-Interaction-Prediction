# Reliability-Conditioned Herb–Drug Interaction Prediction

## Read this first

This project investigates how to combine sourced biological mechanisms with an assessment of published evidence when studying herb–drug interactions. Its central idea is to keep a study's reported finding separate from the independently curated answer used to judge a model. The repository contains model components, evidence checks, leakage controls and a read-only presentation/explorer. **It does not currently contain an eligible empirical evaluation: the independent consensus dataset is empty, mechanistic records remain unverified, and production execution is blocked.** Passing software tests is not evidence that the model predicts real interactions accurately.

## 1. The real-world problem

A person may use a plant preparation and a medicine together. Researchers want to understand whether that combination changes the medicine's exposure or effects. The evidence is not a simple vote: two articles may describe the same underlying study, an animal experiment may have different relevance from a human experiment, and studies may disagree because preparations, doses or populations differ.

There is also a machine-learning trap. Suppose an abstract says “an interaction was observed,” and that same sentence is used both as the model's input and as the reason its target is positive. A model can learn to repeat the sentence rather than learn anything that generalizes to another pair. The redesigned project therefore separates evidence inputs from external consensus targets.

**Illustrative example, not a real clinical claim:** imagine Plant A and Medicine B. One experiment reports an effect, a review repeats it, and another experiment is inconclusive. That is not automatically two positive independent studies, nor is the inconclusive experiment a negative observation. The project aims to preserve those distinctions. It cannot tell a person whether to start or stop either product.

## 2. Essential terms in everyday language

| Term | Meaning here |
|---|---|
| Herb | A botanical preparation or plant-derived product. Its identity alone does not describe its composition, preparation or dose. |
| Drug | A medicinal substance. Different routes and amounts can matter to interpreting evidence. |
| Interaction | A change associated with using substances together. The research label is a curated category, not a prescription instruction. |
| Mechanistic evidence | Sourced information about a possible biological pathway, such as an enzyme effect or a drug's metabolic fraction. A possible mechanism is not proof of a clinical outcome. |
| Consensus | An external, independently curated decision about a pair, with provenance. It is not a majority vote over the input articles. |
| Reliability | In the proposed Task B model, learned agreement between study direction and independent consensus using study-design/lineage descriptors. It is not an authenticated measure of truth. |
| Calibration | Adjusting predicted probabilities using a separate dataset so that their interpretation can later be assessed. It does not automatically make a model correct. |
| Abstention | Withholding a definitive model answer when its decision rule is uncertain or unsuitable. The app's current lack of predictions is a readiness block, not a measured model abstention outcome. |
| Data leakage | Letting information from the answer or assessment data influence learning, selection or prediction. It makes evaluation misleading. |
| Evaluation | Measuring a fixed system against independent targets under a declared protocol. A software unit test is a different kind of check. |

## 3. Research question, contributions and novelty

The main research question is: **can a mechanistic prior and agreement-supervised study reliability support useful evidence aggregation when targets and evaluation groups are independent?** Related questions ask how performance changes for previously unseen herbs or drugs and whether probability calibration or selective prediction is useful under a justified design.

The intended contribution is an auditable combination: separate evidence and targets, restrict mechanism features, learn evidence agreement, account for citation/source dependence, and bind execution to explicit release contracts. The completed engineering contribution is the implementation and testing of many of those contracts. Improved predictive performance remains a hypothesis.

Logistic regression, gradient boosting, scaling, grouped evaluation, temperature scaling and conformal scoring are established techniques, not inventions claimed by this document. The combination's scientific originality and any protectable novelty require a verified literature/prior-art review. Earlier repository GNN/text prototypes are not the approved primary Task A feature route. The current method must not be described simply as a clinically validated graph neural network.

## 4. Architecture: three kinds of data, one controlled workflow

```text
Study table                 Consensus table              Mechanistic table
What a publication reports  Independent pair decisions    Sourced pathway records
         │                           │                            │
         └──────────── schemas, provenance, immutable bytes ──────┘
                                     │
                    explicit roles + disjointness constraints
                                     │
                       Phase 2 release preflight
                                     │
                  current outcome: eligible=False → STOP
                                     │ future reviewed authorization only
               Task A mechanism model / Task B evidence adjustment
                                     │
                probability calibration → conformal calibration
                                     │
                          held-out assessment
```

The diagram describes the intended flow, not an active empirical run. Source checks, partition validation and release reports can operate without model fitting. The website uses that distinction to present information without silently training a model.

A **snapshot** is the exact captured file bytes. Hashing and parsing must consume the same bytes rather than reopening a path that might have changed. Captures are per file, not an atomic multi-file transaction; approved manifest hashes are needed to detect an inconsistent collection. A **plan** identifies tasks, requested generalization regime and explicit role assignments. Neither a plan nor a caller-written manifest is scientific approval.

## 5. Task A: the mechanism-only question

**Purpose:** study what can be predicted about a pair from approved mechanism information, without feeding the model that pair's publication outcomes.

**Inputs:** herb/drug join keys and curated mechanism records. The approved builder constructs six signed pathway effects: CYP3A4, CYP2C9, CYP2C19, CYP2D6, CYP1A2 and ABCB1. Its seventh feature is the sum of their absolute values. The sign describes target modulation; it must not be casually translated into a universal increase/decrease in human drug exposure.

The current extractor combines herb modulation with a drug pathway weight, using sourced metabolism fraction where available or a substrate flag fallback defined by the extractor. Missing mechanisms remain missing and fitting rejects them. A cited, explicitly tested null is different from “we do not know.” Future PK/exposure or other pathway features cannot enter by adding arbitrary columns.

**Targets:** independently curated `interaction` and `no_interaction` decisions. `disputed` stays in the pair universe and predictions but is excluded from binary optimization. Unresolved/inconclusive consensus strings are rejected; they are never mapped to zero.

**Components and intended outputs:** the existing logistic regression (LR) and histogram gradient boosting (HGB) families produce binary probabilities within the private synthetic algorithms. Feature-contract checks run before and after scaling. The scaler is fitted only on the fit role. Tuning compares explicit candidate configurations on the tuning role without refitting on those rows.

**Assumptions and limitations:** mechanism curation must be sourced, verified and complete; consensus must be external and independent. The current release satisfies neither requirement. No deployable, validated Task A predictor is supplied. Pair prediction is an experimental question, not clearance for co-administration.

## 6. Task B: the evidence-aggregation question

**Purpose:** study how a body of published findings should adjust a mechanistic starting point. Unlike Task A, study findings are intended inputs, but they still cannot become the external consensus target.

**Inputs:** a mechanistic prior, study direction, study-design descriptors, primary/echo annotations, registered source references and citation/source relationships. The existing reliability feature vocabulary uses methodological categories and structural information; raw abstracts and publication direction do not become reliability-quality predictors. Direction enters the explicit aggregation step.

The intended reliability target asks whether a directional study agrees with independent consensus. Inconclusive studies have no agreement target and no directional evidence mass. Disputed consensus pairs are excluded from binary reliability/prior optimization. Directional positive and negative support remain distinct; contradictory evidence is not “fixed” by relabeling it.

**Model components:** existing Task B infrastructure uses a mechanistic logistic prior, a supervised study-reliability model and a residual aggregator. Prior/reliability fitting is conceptually separate from residual parameter tuning. Primary/echo treatment is parameterized rather than counted as unlimited independent confirmation. With no directional evidence, the existing aggregator preserves the prior; that fallback is algebra, not validation that the prior is correct.

**Current limitation:** the snapshot adapter validates actual immutable study rows, registered references and exact agreement with declared topology, but its separate fitting and residual-tuning interfaces both remain blocked. No reviewed actual-evidence-release/independence contract exists. Older Task B runners remain in the repository; they are not a way to authorize the new production pipeline.

## 7. What each control does

| Component | Job and boundary |
|---|---|
| `src/data/evidence_schema.py` | Checks the shape and vocabulary of evidence and consensus tables. Schema validity cannot authenticate a clinical claim. |
| `src/data/source_registry.py` | Checks source/reference registration. A traceable publication is not automatically an independent or correct observation. |
| `src/data/mechanistic_curation.py` | Validates statuses, measurements and provenance fields; retains missing values. Software checks do not replace expert source review. |
| `src/features/task_a_contract.py` | Rebuilds exactly the approved feature columns from mechanisms and guards the training boundary. Identifiers only join rows. |
| `src/training/evaluation_protocol.py` | Validates explicit pair/entity/source-component assignments; fails on infeasible overlap instead of downgrading the regime. |
| `src/training/release_preflight.py` | Checks snapshots, recorded approvals, policies, provenance, feature completeness and role support. Current eligibility is always false because scientific support is not established. |
| `src/training/evaluation_orchestrator.py` | Validates the request, captures each allowed input once and returns a blocked report. It currently issues no execution authorization. |
| `src/training/pipeline_components.py` | Public execution facade: fitting, prediction, scoring and calibration fail closed. A caller's `eligible=True` dictionary cannot unlock it. |
| `src/training/_pipeline_algorithms.py` | Private synthetic test algorithms. Fit-role registration binds immutable configuration/feature/plan/snapshot/table/estimator provenance. These records do not authorize production execution. |

Three reviewed defects were addressed: public component execution was closed; arbitrary fitted-estimator wrapping was rejected in favor of factory-issued fit records; and snapshot rebinding/cached-table changes were added to state checks. Public snapshots are read-only, returned tables are copies, and changed bindings invalidate use. Python private modules are an API convention, not protection against arbitrary code rewriting the interpreter.

## 8. Why the split regime matters

- **Pair-disjoint:** the same herb–drug pair cannot appear in different roles. This tests new combinations but can still contain herbs and drugs seen in other combinations.
- **Herb-disjoint:** an herb cannot cross roles. This asks whether the system generalizes beyond herbs represented in fitting data.
- **Drug-disjoint:** a drug cannot cross roles. This asks the corresponding question for medicines.
- **Source-component-disjoint:** Task B additionally keeps connected source/citation components together. Shared sources, indirect citation ancestors and pairs bridging components can otherwise carry the same evidence into multiple roles.

These constraints can make a requested assignment impossible. The correct response is an explicit error or more appropriate independently collected data, not silently changing herb-disjoint to pair-disjoint. Structural separation also does not prove statistical independence, adequate sample size or human reviewer independence.

## 9. Five roles, five different jobs

| Role | Allowed job | Must not influence |
|---|---|---|
| Fit | Learn scaling and model parameters from fitting targets | Other roles must not enter fitting |
| Tuning | Select among declared configurations or residual parameters | Must not refit the base predictor on tuning rows |
| Probability calibration | Fit a probability adjustment for a fixed predictor | Must not choose/refit its base predictor |
| Conformal calibration | Compute score thresholds for a fixed predictor and optional fixed calibrator | Must not alter that predictor/calibrator |
| Assessment | Measure final fixed predictions against independent targets | Must not affect preprocessing, selection, fitting, calibration or predictions |

The full protocol has all five roles. Baseline, tuned, probability-calibrated and conformal protocols use their declared required subset. Missing required roles are errors; the system does not merge them silently.

Temperature scaling has explicit failure states and no silent identity-calibration substitute. Conformal scoring uses rank `ceil((n+1)*(1-alpha))`; rank `n+1` gives conservative all-label sets. The arithmetic can be tested with synthetic fixtures, but no nominal coverage is certified because the sampling/exchangeability design is unresolved. Non-singleton sets can motivate future abstention; the current UI has no integrated calibrated predictor.

Strict scoring checks exact ordered pair identity, probability finiteness/range and label vocabulary. An undefined metric is reported unavailable with a reason. Average precision and trapezoidal PR-AUC are different quantities. Existing high-AUC integrity alerts are retained. No confidence interval or significance result is inferred from unit tests or from disjointness alone.

## 10. How to navigate the repository

Start at [README.md](README.md), then [the website guide](docs/website.md). `app/interactive_app.py` is the standard Streamlit entry point; `app/frontend.py` retains an alternate entry. Both stop before inactive legacy prediction code. `app/project_presentation.py` renders the new overview and document downloads; `app/research_status.py` remains the read-only evidence explorer.

The `src/data`, `src/features`, `src/models` and `src/training` directories separate validation, approved features, model building blocks and evaluation controls. `tests/test_pipeline_components.py` includes public denial and private synthetic regression cases. The `docs` directory contains the release and evaluation contracts, progress log, [paper draft](docs/research_paper_draft.md) and [IDF-B draft](docs/IDF-B_draft.md).

`PROJECT_SPEC.md` explains research intent; the narrower feature/release contracts describe current supported behavior. Historical `reports`, `results`, benchmark scripts and patent drafts are retained for provenance and may contain superseded claims. Do not treat them as the redesigned pipeline's empirical evidence. Do not run reproduction scripts to overwrite historical outputs. The private annotation directory is out of scope and must remain untouched.

## 11. Environment, commands and website navigation

Python 3.10+ is declared; the recorded focused checks use a Python 3.11 virtual environment. If the supplied `venv` exists, use it. For a new environment:

```sh
python3 -m venv venv
venv/bin/python -m pip install -r requirements_core.txt
venv/bin/python -m pip install 'networkx>=3.2' 'pytest>=7' 'streamlit>=1.29' 'plotly>=5.18'
venv/bin/python -m pip install --no-deps -e .
```

Core requirements include the scientific/PyTorch stack and transformers; networkx and pytest are explicitly added for the focused checks. Plotly supports existing calibration/conformal tests. The broader `requirements.txt` includes optional graph/NLP/database packages and can require platform-specific installation work. It is not a platform-independent lock. No GPU, source credentials or downloaded model weights are needed for the presentation or its static content. The explorer's evidence validation uses local files.

Launch the existing application:

```sh
venv/bin/python -m streamlit run app/interactive_app.py
# Alternate existing entry: venv/bin/python -m streamlit run app/frontend.py
```

Open the localhost URL printed by Streamlit. The default **Project presentation** view has section navigation and expandable safeguard explanations. Scroll to document downloads for local Markdown files. Choose **Evidence explorer** in the View selector to inspect source records and readiness blockers. Neither view trains or calls a prediction API. The disabled prediction control is intentional.

Run the focused scientific engineering checks, not a full historical reproduction:

```sh
PYTHONDONTWRITEBYTECODE=1 venv/bin/python -m pytest tests/test_pipeline_components.py tests/test_evaluation_orchestrator.py tests/test_release_preflight.py tests/test_evaluation_protocol.py tests/test_task_a_training.py tests/test_task_b_training.py tests/test_task_a_feature_contract.py tests/test_redesigned_leakage_regression.py tests/test_calibration_audit.py tests/test_conformal_abstention.py tests/test_conformal_overhaul.py tests/test_interaction_model_and_calibration.py tests/test_trainer_conformal.py -q -p no:cacheprovider
```

Frontend checks are in `tests/test_project_presentation.py` and `tests/test_app_research_status.py`. Streamlit has no separate JavaScript build in this repository; Python syntax checks and real Streamlit AppTest rendering provide its application checks. Exact session results and warnings belong in [PROGRESS.md](docs/PROGRESS.md); the earlier 480 passing cases were engineering verification, not accuracy.

## 12. Status and scientific limitations

**Implemented:** table/schema/source checks; approved mechanism feature contracts; explicit partition validation; snapshot/table binding; private fitted-role provenance; strict scoring and calibration/conformal components; fail-closed public execution; read-only presentation and explorer.

**Tested synthetically:** private fitting/tuning, calibration, conformal ranks, mutation rejection, public authorization denial and assessment-label independence. Informative synthetic features are deliberately used for regression checks, not presented as medical observations.

**Blocked:** populated independent consensus, verified complete mechanisms, a reviewed actual Task B study-release contract, statistical adequacy, justified uncertainty inference, an authorized empirical execution backend and deployment bundle. Public methods remain disabled even for a caller claiming eligibility.

**Future:** curate and audit data, verify prior art, decide statistical support/inference policies, review authorization issuance, and only then compare methods on eligible independent data. More tests cannot fill the missing evidence. The website is a presentation of a research prototype, not a clinical tool.

## 13. Common problems and troubleshooting

| Symptom | What it means and what to do |
|---|---|
| `PermissionError` from a public component | Expected fail-closed behavior. Do not supply a fabricated approval, change the gate or run private code as an empirical workaround. |
| Consensus empty / both classes required | Independent decisions are missing or a role lacks binary support. Curate independent targets; do not relabel disputed/unknown records. |
| Mechanisms unverified or missing features | Complete and review source-backed measurements; do not replace missing biology with zeros. |
| Infeasible connected component | Requested roles overlap under entity/source constraints. Revise assignments or collect independent groups; do not downgrade the declared regime silently. |
| Snapshot/fit-provenance changed | Cached data, bytes, plan or predictor state changed. Revalidate from approved captured inputs rather than updating a hash to suppress the check. |
| Calibration optimization fails | No valid fitted calibrator exists. Stop dependent steps; do not silently use identity scaling. |
| AUC integrity alert | Investigate leakage/identity/target contamination. Do not weaken the guard or replace historical results. |
| Download missing | Launch from the full checkout; ensure the named Markdown file exists. The UI reports unavailable documents instead of linking to a fake page. |
| Streamlit/module import error | Use the same virtual environment for launching and installing packages. Prefer repository-root commands. |
| joblib core-count warning | The environment may fall back to logical cores. This warning is not evidence of prediction failure or scientific validity. |

## 14. Explaining the project aloud

### Nontechnical version

“We are building a way to assess evidence about plant products and medicines without counting repeated claims as new discoveries. The software separates what papers report from independent reference decisions, and it refuses to produce research results until the required data are ready. We have tested the software controls; we have not proved real-world predictive performance.”

### Technical version

“The redesign separates independent pair consensus from publication-level directional evidence and fixed mechanistic inputs. Task A uses an allowlisted seven-feature LR/HGB route. Task B investigates agreement-supervised reliability and residual aggregation. Explicit entity/source-component-disjoint roles, immutable snapshots and registered fit provenance prevent common contamination routes. Production execution remains closed pending release eligibility and an approved authorization contract.”

### Concise professor pitch

“This project asks whether reliability-aware evidence aggregation can improve herb–drug interaction research without leaking publication outcomes into ground truth. The deliverable today is a tested, auditable pipeline and an honest blocked status. The next research milestone is independent annotation and verified mechanism curation, not another accuracy claim.”

### Two-minute pitch

“Herb–drug evidence can be hard to interpret because experiments differ, papers repeat older findings, and disagreement is real. A machine-learning benchmark can make this worse if the model sees an article's conclusion and is then scored against a label obtained from that same conclusion.

Our redesign uses three separate tables: published studies, independently curated pair consensus and sourced mechanistic measurements. Task A asks what a small mechanism-only model can predict without publication outcomes. Task B asks how a body of directional studies should modify that prior, using study-design and lineage descriptors to learn agreement with independent consensus.

We make the evaluation assumptions explicit. Fit, tuning, probability calibration, conformal calibration and assessment are separated. Pair-, herb- or drug-disjoint regimes answer different generalization questions; Task B also isolates shared-source and citation components. Snapshot and fitted-role provenance checks detect changes, and public execution is blocked rather than accepting caller-created approval records.

Today, the important limitation is that consensus is empty and mechanisms remain unverified. We have engineering and synthetic regression evidence, not a defensible empirical comparison, coverage guarantee or clinical recommendation. The immediate work is expert-backed annotation, source/mechanism verification, statistical design and release review. Only after those steps could we evaluate whether the proposed reliability approach actually helps.”

### Likely professor questions

| Question | Truthful suggested answer |
|---|---|
| What is new? | The candidate contribution is the integrated evidence/target separation and provenance-controlled reliability pipeline. Novelty over prior work still needs verified review. |
| Why not use an LLM/GNN for everything? | The approved primary route deliberately limits features to reduce shortcuts and keep the starting comparison interpretable. Larger models would require separate justified contracts and evaluation. |
| What is your accuracy? | No current eligible empirical accuracy exists. Test pass counts measure software behavior, not predictive success. |
| Why is consensus empty? | Independent annotation has not been completed in the eligible release. Publication outcomes cannot substitute for it. |
| Does a registered DOI prove quality? | No. Registration supports traceability; study validity and independent consensus need additional review. |
| Is abstention clinically safe? | That has not been established. No clinical safety or nominal coverage guarantee is claimed. |
| Is it patented? | No patent grant or patentability claim is established by this repository or draft form. Institutional review is pending. |
| What would make the project empirically complete? | Independent labels, verified mechanisms, approved actual evidence, adequate independent groups, a justified inference design, a reviewed authorization issuer and a predeclared empirical comparison. |

## 15. A future eligible evaluation, step by step

This is an intended workflow requiring new approvals and engineering, **not a command sequence currently able to execute production models**.

1. Define the target population, task, positive/negative/disputed policy, requested regime and intended estimands before looking at assessment results.
2. Independently curate consensus with recorded reviewer/adjudicator provenance and evidence independence. Retain unresolved cases outside the binary training target.
3. Verify complete mechanism measurements and actual study sources, findings, primary/echo status and lineage; preserve disagreement.
4. Record approved release hashes and capture immutable input bytes. Hash and parse those exact snapshots.
5. Supply explicit role assignments. Validate required roles, entity/source-component separation, binary class support and absence of contradictory provenance.
6. Establish and approve statistical support/inference criteria. Current counts and disjointness alone do not justify them.
7. Run actual Phase 2 preflight. A failed or false eligibility outcome stops the workflow. Even a future true outcome needs a reviewed orchestrator-owned authorization issuer; none exists today.
8. Fit approved preprocessing and model components using only fit-role targets. Register immutable fitting provenance. Task B needs its separately approved evidence-release route.
9. Select declared candidates/residual parameters using tuning only, without silently retraining on additional roles.
10. Freeze the predictor; fit probability calibration if required; freeze it; then perform separate conformal calibration if required. Failure stops dependent steps.
11. Generate assessment predictions and any sets before reading assessment targets. Score exact identities with explicit unavailable metrics and retained AUC safeguards.
12. Report the declared design, provenance and limitations. Compute uncertainty or significance only if a justified inference design was approved. Never interpret an engineering pass or an unexplained high score as clinical readiness.

## 16. Quick glossary

**LR:** logistic regression, a binary probability model. **HGB:** histogram gradient boosting, a tree-based model family. **CYP:** an enzyme family involved in drug metabolism. **ABCB1/P-gp:** a transporter represented in the approved pathways. **Prior:** the mechanism-based starting score before evidence adjustment. **Residual:** the evidence-driven change to that score. **Echo:** a derivative report repeating underlying evidence. **Topology:** the represented citation/source connections. **Provenance:** where a record or fitted result came from. **Hash:** a fingerprint of exact bytes/state, not proof of truth. **BCE/log loss:** a probability-sensitive binary prediction loss. **ROC-AUC:** a ranking metric requiring both classes under this scoring policy. **Average precision / PR-AUC:** different summaries of precision–recall behavior; the implementation keeps them distinct. **Exchangeability:** a sampling symmetry assumption relevant to conformal guarantees, not something proved by a split validator. **Fail closed:** stop rather than invent missing evidence or silently relax a contract.
