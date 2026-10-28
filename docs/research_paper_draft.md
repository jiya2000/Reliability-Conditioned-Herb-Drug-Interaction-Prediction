# Reliability-Conditioned Herb–Drug Interaction Prediction

**Research-paper draft — engineering and methodology report, not a completed empirical paper.**<br>
Authors, affiliations, corresponding author and submission venue: **[REQUIRES STUDENT/ADVISOR INPUT]**.

## Abstract

Herb–drug interaction research must distinguish independently observed evidence from repeated citations and separate publication findings from evaluation targets. We describe an implemented research-pipeline architecture that separates sourced mechanistic inputs, publication-level study records and independently curated pair consensus. Task A uses a fixed mechanism-only feature contract with logistic regression or histogram gradient boosting. Task B investigates agreement-supervised study reliability and residual adjustment of a mechanistic prior, with citation/source-aware provenance constraints. Explicit role assignments separate fitting, tuning, probability calibration, conformal calibration and assessment. Immutable input snapshots and registered fitted-role provenance support integrity checks, while the public production boundary currently refuses execution. **The independent consensus dataset is empty, mechanistic records remain unverified, and empirical evaluation is blocked.** Private algorithms have synthetic regression validation, but no predictive-performance, statistical-superiority, clinical-safety or nominal conformal-coverage claim is supported. The contribution reported here is an auditable implementation and evaluation contract; its scientific utility remains a future empirical question requiring independent annotation, verified evidence and a justified statistical design.

**Keywords:** herb–drug interaction; evidence provenance; mechanistic features; study reliability; data leakage; calibration; selective prediction; research software.

## 1. Introduction and motivation

Published evidence about botanical–pharmaceutical combinations is heterogeneous in experimental setting, preparation, route and methodological detail. Multiple reports can also depend on the same source or underlying study. A prediction pipeline should therefore preserve both what a report says and how that report relates to other evidence.

A further methodological concern arises when publication conclusions serve simultaneously as inputs and evaluation labels. A model can reproduce conclusion language without demonstrating generalization to independent pair-level decisions. This project redesigns its primary research route around the separation of evidence from external consensus targets. It also separates mechanism-only prediction from aggregation of available literature.

The repository previously explored graph/text models and historical benchmarks. Those artifacts are retained for provenance, but their numerical results are not evidence for the redesigned tasks described here. This draft makes no prevalence, effect-size or comparative-superiority claim based on those artifacts.

The intended contributions are: (i) a bounded mechanism-only feature route; (ii) a candidate reliability-conditioned evidence-adjustment formulation; (iii) explicit entity/source-component-disjoint evaluation roles; and (iv) snapshot, fitted-role and fail-closed release controls. Only their implementation and scoped software validation are reported as completed work.

## 2. Problem definition and research questions

Let a pair consist of canonical herb and drug identifiers. Let its external consensus state be `interaction`, `no_interaction` or `disputed`. For binary fitting, only the first two states have targets. Missing, unresolved and disputed decisions are not manufactured negative examples. Study findings, in contrast, are publication-level inputs and can be directional or inconclusive.

**Task A** asks whether sourced mechanism features can predict external consensus without using publication findings as features. **Task B** asks whether directional studies, weighted by agreement-supervised reliability and citation/source information, can usefully adjust a mechanistic starting score.

Research questions remain open:

1. Does the proposed evidence adjustment improve a predeclared metric relative to appropriate mechanism and evidence baselines under independent targets?
2. How does generalization differ under pair-, herb- and drug-disjoint assignments?
3. How sensitive is evidence adjustment to primary/echo annotations, shared-source components and disputed decisions?
4. Under a justified sampling design, do separate probability calibration and selective prediction improve the intended probability/decision criteria?

These are hypotheses, not findings. No experimental thresholds, sample-size requirements or significance levels are invented in this draft.

## 3. Related work — literature review pending

**External bibliographic verification is required before submission.** Existing repository drafts mention prior systems and patents, but their bibliographic details and claimed distinctions have not been independently verified for this draft. They are not reproduced as established comparisons.

The literature review should examine mechanistic/tabular interaction prediction, graph-based interaction models, study-quality/evidence synthesis, citation dependence, calibrated prediction and conformal/selective prediction. For each verified source, record the actual target, dataset provenance, split unit, supported uncertainty claims and the precise relationship to the proposed method. A title or familiar model name is not enough to establish relevance or novelty.

Logistic regression, gradient boosting, standard scaling, grouped evaluation, temperature scaling and conformal scoring are established components, not individual inventions claimed here. Whether the integration materially differs from prior work requires a claim-level literature and prior-art comparison. No fabricated citation or unsupported assertion of absence in prior systems is supplied.

## 4. Proposed methodology and architecture

The architecture uses three separate data routes:

```text
Published study records ── schema + registered-source + lineage checks ─┐
Independent consensus ─── decision provenance + independence checks ───┼─ exact snapshots
Mechanistic measurements ─ status + value + provenance checks ────────┘
                                   │
                   release preflight + explicit partition validation
                                   │
                          current eligibility: FALSE
                                   │ future reviewed authorization only
           mechanism features → Task A / study evidence → Task B adjustment
                                   │
              probability calibration → conformal score calibration
                                   │
                        fixed held-out assessment
```

Snapshots preserve exact bytes for parsing and hashing rather than reopening original paths after validation. The validated plan records assignments, protocol, task and requested regime. Captures are per-file; they do not form an atomic filesystem transaction. Approved manifest hashes are necessary to bind a coherent input release.

The production orchestrator currently produces a blocked report and issues no execution authorization. Public component methods cannot be enabled with a caller-supplied eligibility report or synthetic flag. Private algorithms are a synthetic verification surface, not a production or empirical authorization mechanism.

## 5. Dataset, annotation, provenance and evidence contracts

### 5.1 External consensus

The intended target table requires canonical pair identifiers, consensus vocabulary and source metadata. Recorded review/adjudication and independence attestations are release prerequisites. Structural validators cannot authenticate reviewer independence or scientific correctness. Consensus must be external to the findings used for Task B; agreement supervision would otherwise be circular.

**Current status: the consensus dataset is empty.** Consequently, no target-dependent empirical training or assessment of these redesigned tasks is available. Annotation effort, review process, adjudication and permissible source access remain to be completed and reviewed. No new clinical labels were created for this draft.

### 5.2 Mechanistic curation

Mechanistic records carry entity/target keys, statuses and sourced quantitative or tested-null information. Validation rejects malformed measurements, conflicting incompatible fields and duplicate entity/target records. Missing information remains unknown. A recorded `tested_no_effect` status requires appropriate provenance; a missing entry is not an observed null.

**Current status: mechanistic records remain unverified.** Presence of a citation string is not independent authentication. Complete, externally reviewed measurements are required before primary empirical fitting.

### 5.3 Publication evidence and source lineage

Actual Task B study snapshots must contain validated study rows, parsed cited-study identifiers, explicit findings and primary/echo annotations. Registered-source verification is required. The snapshot adapter compares study/source/citation topology with declared topology and role constraints. Extra intermediate study records remain conservatively unsupported until an approved evidence-release representation defines them.

Topology and registration alone do not establish a complete, independently approved evidence release. The actual-evidence-release/independence contract is unresolved, and Task B snapshot fitting remains blocked. Historical publication records must not be reinterpreted as independent consensus decisions.

## 6. Task A: mechanism-only prediction

The primary builder computes effects for the fixed targets CYP3A4, CYP2C9, CYP2C19, CYP2D6, CYP1A2 and ABCB1 and recomputes the sum of their absolute values. For a target, the extractor combines herb modulation with a drug pathway weight using its defined sourced metabolism-fraction/substrate rules. The sign is target modulation; interpreting it as human exposure direction requires further pharmacological context.

Herb/drug identifiers only join measurements and define partitions. Raw abstracts, publication outcomes, source identifiers, study-design metadata and arbitrary embeddings are excluded from the approved Task A feature matrix. Unapproved columns, malformed numeric values and missing mechanisms fail before fitting; no imputation policy is assumed.

Existing LR/HGB families are preserved. Feature-contract validation is applied both before scaling and at the estimator boundary. Scaling and model fitting use only fit-role binary targets. Explicit configuration candidates are evaluated on tuning BCE, with no silent refit on tuning. Disputed pairs remain outside binary optimization while their identity and state remain available for reporting.

These computations are exercised in private synthetic tests. No eligible empirical Task A model is claimed.

## 7. Task B: reliability-conditioned evidence adjustment

Task B infrastructure combines a mechanistic logistic prior, an agreement-supervised study-reliability model and an existing residual aggregator. Reliability predictors use a fixed vocabulary of design/lineage descriptors, including species/assay/route/preparation categories, dose/sample-size indicators, primary status and structural component size. Publication direction supplies aggregation evidence, not a reliability-quality input.

For directional studies with independent binary consensus, agreement is the intended supervision target. Inconclusive studies have neither an agreement target nor signed evidence mass. Disputed consensus pairs are excluded from binary optimization. This is a measure of agreement under a defined target, not proof that a study or reference is true.

The existing residual aggregation family keeps positive and negative weighted support separate and adjusts the prior using their normalized directional balance. Primary/echo attenuation and evidence-scale parameters must be fitted/tuned in their declared stages rather than inferred from assessment outcomes. With no directional evidence, the algebra preserves the prior; this does not validate that prior.

Prior/reliability fitting and residual tuning have separate snapshot-adapter interfaces. Both deliberately remain blocked until the actual evidence-release contract is approved. Legacy model code is not automatically migrated into the new authorized evaluation path.

## 8. Leakage-safe evaluation protocol

The partition validator accepts explicit assignments rather than silently generating or repairing a requested design. Required roles depend on the declared protocol; the full protocol uses fit, tuning, probability calibration, conformal calibration and assessment, each nonempty and distinct.

Pair-disjoint evaluation excludes repeated pairs across roles but can retain familiar entities. Herb- and drug-disjoint evaluation ask stronger, different generalization questions. Task B additionally keeps connected citation/shared-source components together, including transitive relationships and pairs that bridge components. Infeasible assignments fail; a requested regime is never silently downgraded.

Assessment targets must not affect preprocessing, feature selection, fitting, hyperparameter/residual selection, calibration or predictions. The private composition constructs predictions and sets before requesting assessment labels. Informative synthetic regression cases alter assessment labels and compare fitted estimator state, selected configuration, calibration temperature and predictions.

Strict scoring requires exact ordered identity alignment, valid labels and finite in-range probabilities. ROC-AUC, average precision and trapezoidal PR-AUC require both binary classes under the declared scoring policy; unavailable quantities receive explicit reasons. Average precision is not relabeled as trapezoidal area. Existing AUC integrity alerts remain unchanged. Disjointness and class support do not establish statistical sample adequacy.

## 9. Calibration, abstention and uncertainty methodology

Private temperature scaling operates on a registered fixed predictor with a separate probability-calibration role. Interior fitting probabilities and both binary classes are required. Optimization failure, nonfinite state and numerical boundary solutions fail explicitly; identity calibration is not substituted after failure. Numerical log-temperature bounds are implementation limits, not scientific thresholds.

Private binary conformal calibration uses nonconformity `1 − p(true class)` on its own role, after predictor/calibrator freezing. The threshold rank is `ceil((n+1)(1−alpha))`; rank `n+1` yields conservative all-label sets. Alpha must be supplied explicitly. Missing or inconsistent fit provenance and changed predictor/calibrator identities prevent use.

**Coverage certification remains blocked.** No exchangeability or finite-sample sampling design has been established for the intended grouped biomedical data. Excluding disputed targets also changes the population on which score calibration is defined. A non-singleton set may inform a future selective-prediction rule, but no current clinical abstention safety claim follows. The app's withholding of predictions is a release/readiness safeguard, not an observed selective-risk result.

No uncertainty intervals, significance tests or superiority claims are reported. A future inference design must justify the independent sampling units and account for the declared grouping before selecting an interval or comparison procedure.

## 10. Implementation details

The research code uses Python with NumPy/pandas/SciPy/scikit-learn and existing PyTorch-era components. Streamlit provides a read-only project presentation and evidence explorer. No new web framework or JavaScript build is required.

`SnapshotTables` preserves bounded immutable input bytes. `RoleData` accepts the exact snapshot type, copies derived tables and validates the explicit plan. Its read-only snapshot property and state checks detect rebinding or mutation before use. The private fit factory registers predictors only after fitting on verified fit-role data. Frozen records bind role, configuration, feature contract, plan, snapshots, derived tables and estimator state. Copying an estimator/record or manually constructing a wrapper does not create a registration.

The public execution facade has no active backend dispatch. The orchestrator currently issues no grants, even if a caller presents a purported successful report. Future production execution requires a reviewed orchestrator-owned issuance design after literal actual Phase 2 eligibility and all prerequisites pass. Python module privacy and state hashes are not security isolation against arbitrary interpreter code, nor proof of evidence authenticity.

Important implementation contracts are linked in the references below. Existing runners and historical artifacts are retained unchanged; preservation should not be mistaken for endorsement of superseded claims.

## 11. Current validation and engineering results

**Empirical results: unavailable.** The independent consensus dataset is empty, mechanistic records remain unverified and the actual Task B evidence-release/statistical-support prerequisites are incomplete. There is no eligible redesigned-task performance table, confidence interval or clinical validation.

The prior recorded focused run passed 480 software checks with one environment-specific joblib core-detection warning. These cases cover synthetic algorithms, release/partition contracts, public authorization denial, malformed inputs, calibration failure and leakage regression behavior. They are engineering test counts, not observations about predictive accuracy or generalization. The current-session focused run passed 499 checks, including 16 existing app checks and three presentation checks, with the same joblib warning. Exact commands and frontend verification are recorded in [PROGRESS.md](PROGRESS.md).

The website presents architecture and status rather than simulated performance charts. The existing evidence explorer remains read-only and displays missing consensus and unverified mechanisms without fitting or manufacturing probabilities.

## 12. Limitations and threats to validity

- **Missing targets and verified biology:** without independent consensus and sourced complete mechanisms, the intended scientific comparisons cannot run.
- **Reference bias:** even future consensus may reflect source overlap, incomplete review, disputed interpretation or selection effects. Attestations require human verification.
- **Dependence and adequacy:** graph separation encodes declared relationships but cannot discover every hidden shared cohort/source, establish independence or justify a sample size.
- **Feature scope:** the approved mechanism representation is deliberately narrow and does not capture all clinically important pathways, preparations, doses or patient contexts.
- **Reliability target:** agreement with a reference is not the same as scientific truth. Case selection and imperfect consensus can bias the learned weighting.
- **Synthetic validation:** fixtures can detect violations of software invariants but cannot establish external validity, calibration benefit or useful abstention.
- **Release/authorization gap:** production is intentionally disabled until statistical/evidence and authorization contracts are reviewed. No deployed model bundle is claimed.
- **Related-work gap:** the external literature and patent comparisons require verification; neither scientific novelty nor protectability is established here.
- **Reproducibility limits:** snapshots are per-file captures, fitted-state hashes depend on software versions, and private Python modules are not an interpreter security sandbox.

## 13. Ethics, clinical safety and intended use

The software is for research-method development and evidence inspection. It cannot safely recommend or prohibit herb–drug combinations, replace a healthcare professional, or provide co-administration clearance. An empty evidence table, unknown mechanism or model abstention must never be interpreted as safety.

Future curation requires appropriate source licensing, attribution, reviewer procedures and institutional review of applicable ethics/data-use requirements. No patient study, clinical approval, regulatory classification, inventor signature or institutional endorsement is asserted. The private annotation workspace remains excluded from inspection and publication.

## 14. Future work

The next milestones are independent consensus annotation/adjudication, mechanism/source verification, a reviewed actual study-release contract and a statistical-support/inference design. A future authorization issuer must bind the exact validated release, plan and fitted-role provenance. Only then should predeclared baseline/ablation comparisons and uncertainty analyses be conducted on eligible independent assessment data.

Larger models, additional pathway/PK features, deployed abstention and clinical integration remain separate research/validation decisions. A Streamlit presentation is not evidence of clinical deployment readiness.

## References

### Verified local implementation/design sources

1. [Task definition](task_definition.md) — research intent; the narrower current contracts govern supported features/execution.
2. [Task A feature contract](task_a_feature_contract.md) — approved feature boundary and exclusions.
3. [Dataset release contract](dataset_release.md) — recorded prerequisites and current eligibility limitations.
4. [Evaluation/orchestration contract](evaluation_orchestrator.md) — public fail-closed boundary, private algorithms and provenance.
5. [Evidence schema](evidence_schema.md) and [mechanistic feature documentation](mechanistic_features.md).
6. [Progress log](PROGRESS.md) — exact scoped engineering verification, warnings and superseded historical entries.
7. [Source implementation](../src/training/pipeline_components.py), [partition validator](../src/training/evaluation_protocol.py) and [private algorithm implementation](../src/training/_pipeline_algorithms.py).

### External scholarly references

[REQUIRES STUDENT/ADVISOR INPUT]: add only bibliographically verified papers and patent records after literature review. No external citations have been fabricated or inferred from the older comparison tables.
