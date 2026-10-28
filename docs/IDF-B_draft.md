# Invention Disclosure Format (IDF)-B — completed technical draft for review

**DRAFT — NOT SUBMITTED OR APPROVED.** Structure and field wording follow the two-page official form in [Patent_Form_Details.pdf](Patent_Form_Details.pdf). This Markdown transcription preserves its ten numbered fields and TRL choices; it is not the signed institutional form. Transfer the reviewed answers into that official form. The older [repository patent draft](../patent/VIT_Invention_Disclosure_Form_IDF_B.md) contains historical claims that are not adopted here.

©VIT IPR&TTCELL<br>
**Document No.** 02-IPR-R003<br>
**Invention Disclosure Format (IDF)-B**<br>
**Issue No/Date** 2 / 1 February 2024<br>
**Amd. No/Date** 0/00.00.0000

## 1. Title of the invention:

Reliability-Conditioned Herb–Drug Interaction Prediction

## 2. Field /Area of invention:

Computational evidence assessment, biomedical machine learning, herb–drug interaction research and provenance-aware model evaluation. This draft concerns research software and data-processing contracts, not a clinically validated treatment or prescribing system. Classification codes, inventorship and patent eligibility require institutional review; none is asserted here.

## 3. Prior Patents and Publications from literature (provide a table summarizing the prior art)

| Prior art category to investigate | Relevance to the proposed work | Verification required |
|---|---|---|
| Mechanistic and tabular interaction prediction | Comparison with the fixed mechanistic LR/HGB baseline | [REQUIRES STUDENT/ADVISOR INPUT]: verified publications, full citations and relevant claims |
| Study-quality and evidence aggregation methods | Comparison with agreement-supervised study reliability and directional aggregation | [REQUIRES STUDENT/ADVISOR INPUT]: documented search and claim-level comparison |
| Citation-dependence and source-provenance methods | Comparison with shared-source/citation component isolation | [REQUIRES STUDENT/ADVISOR INPUT]: verified prior patents and publications |
| Calibration, conformal prediction and selective prediction | Established techniques used as components, not claimed as inventions individually | [REQUIRES STUDENT/ADVISOR INPUT]: verified foundational sources and applicability |

No named patent/publication is represented as verified in this draft. Existing repository comparison tables are not sufficient proof of bibliographic correctness or novelty. A prior-art search and institutional IP review remain necessary.

## 4. Summary and background of the invention (Address the gap / Novelty)

Published herb–drug findings can differ in experimental context and can repeat evidence from shared sources. A model evaluated against labels derived from those same findings can appear successful by reproducing its inputs. The research separates publication findings from independently curated consensus targets and from sourced mechanistic measurements.

The proposed contribution is the integration of a mechanistic starting model with agreement-supervised study reliability, source/citation-aware aggregation and leakage-controlled evaluation. The current engineering contribution is a set of explicit data, feature, partition, snapshot and execution contracts. Public execution fails closed while release prerequisites are unmet. Whether the combination is scientifically better or legally novel is unresolved; no patentability or empirical superiority is claimed.

## 5. Objective(s) of Invention

1. Maintain distinct tables for study findings, external consensus decisions and mechanistic curation.
2. Preserve a fixed mechanistic feature boundary for Task A using existing logistic regression and histogram gradient boosting families.
3. Investigate Task B agreement-supervised reliability and directional residual evidence aggregation without using publication findings as consensus targets.
4. Prevent pair/entity/source-component overlap across explicitly assigned evaluation roles.
5. Bind predictor state to fitting-role, configuration, feature, plan and snapshot provenance.
6. Support separately fitted calibration and score-based uncertainty components while blocking unsupported coverage certification.
7. Conduct future independent empirical evaluation only after evidence, annotations, sample support and authorization contracts are reviewed and approved.

## 6. Working principle of the invent (in brief)

Capture immutable input bytes; validate schemas, hashes, recorded provenance and explicit partition assignments; build only approved features; and check release eligibility. Task A is intended to predict from mechanisms alone. Task B is intended to adjust a mechanistic prior using study direction weighted by learned agreement reliability and primary/echo status. Fit, tuning, probability calibration, conformal calibration and assessment have distinct roles. At present the production orchestrator issues no execution authority, so the workflow stops with a blocked report rather than producing empirical predictions.

## 7. Description of the invention in detail (Include drawing and or photograph as needed)

### Architecture

```text
Study evidence ── source/citation + finding/design validation ─┐
External consensus ── independent decision/provenance checks ─┼─ immutable snapshots
Mechanistic curation ── verified, complete measurements ──────┘
                                      │
                          release + partition contracts
                                      │
                           actual eligibility gate
                         BLOCKED in the current release
                                      │ future approved execution only
                  Task A mechanism prior / Task B evidence adjustment
                                      │
                  separate calibration roles → held-out assessment
```

The approved Task A builder outputs effects for CYP3A4, CYP2C9, CYP2C19, CYP2D6, CYP1A2 and ABCB1 plus their absolute sum. Identifiers are join/grouping keys, not features. Missing mechanisms are not silently imputed. Disputed decisions are retained separately and excluded from binary fitting; unresolved labels are not treated as negatives.

Private synthetic algorithms register a predictor only after fit-role fitting. Frozen fit records bind role data, configuration, features, validated plan, snapshots, derived tables and estimator state. Rebinding or mutation invalidates use. Public APIs cannot be unlocked by these private records or by caller-created eligibility mappings. This is an API boundary, not an interpreter sandbox.

Task B validates immutable study bytes, registered references and correspondence to declared citation/source topology. Its snapshot adapter separately exposes prior/reliability fitting and residual tuning, both blocked pending a reviewed actual-evidence-release contract. Calibration and conformal algorithms are available only for private synthetic component checks; no coverage or clinical safety certification is supplied.

### Resources and proposed milestones

Existing local Python/Streamlit tooling supports engineering checks. Future work needs appropriately licensed source access, independent reviewers/adjudication, mechanism verification and statistical supervision. Budget, participants, dates and resource commitments: **[REQUIRES STUDENT/ADVISOR INPUT]**.

Proposed order, not an approved timetable: independent annotation → mechanism/source audit → evidence-release and statistical-support review → authorization design → eligible empirical comparison → scientific write-up. Owners, durations and deadlines: **[REQUIRES STUDENT/ADVISOR INPUT]**.

### Expected outcomes

A reproducible evidence-assessment research pipeline, an independently annotated release if curation succeeds, and an honest empirical comparison if eligibility is eventually established. Improved accuracy, calibrated probabilities and useful abstention are hypotheses to test, not completed outcomes.

## 8. Experimental validation results:

**No eligible empirical validation is available for the redesigned tasks.** The independent consensus dataset is empty, mechanistic records remain unverified, and full Task B evidence approval/statistical support are incomplete. Production execution remains blocked.

The latest prior engineering validation recorded in `docs/PROGRESS.md` was 480 focused passing tests. Those checks test software behavior and synthetic components; they are not predictive-performance results. The current session passed 499 focused checks, including app/presentation checks; exact verification is recorded in that progress log. No AUC, accuracy, clinical effect, significance or nominal coverage claim is made here. Historical benchmark artifacts are preserved but are not used to establish the redesigned pipeline's validity.

## 9. What aspect(s) of the invention need(s) protection?

For institutional discussion only: the specific integration of evidence/target separation, source-component isolation, agreement-supervised evidence adjustment and provenance-bound fail-closed execution may be assessed as a candidate technical contribution. Established individual methods such as logistic regression, gradient boosting, scaling, calibration and conformal scoring are not claimed as inventions here.

**[REQUIRES STUDENT/ADVISOR INPUT]**: inventors and contributions, ownership, prior disclosures, prior-art search, proposed protectable scope, and IPR-cell assessment. No patent claim, grant, freedom-to-operate opinion or protection entitlement is asserted.

## 10. What is Technology readiness level of your invention? (Tick the appropriate TRL)

**[REQUIRES STUDENT/ADVISOR INPUT]**: select the appropriate level with supporting evidence. No TRL is ticked by this draft; passing engineering tests does not establish laboratory or clinical validation of the research hypothesis.

| Research | Research | Research | Development | Development | Development | Deployment | Deployment | Deployment |
|---|---|---|---|---|---|---|---|---|
| TRL 1 | TRL 2 | TRL 3 | TRL 4 | TRL 5 | TRL 6 | TRL 7 | TRL 8 | TRL 9 |
| Basic Principles observed | Technology concept formulated | Experimental proof of concept | Technology validated in a lab | Technology validated in a relevant environment (industrially relevant in case of key enabling technologies) | Technology demonstrated in a relevant environment (industrially relevant in case of key enabling technologies) | System prototype demonstration in an operational environment | System complete and qualified | Actual system proven in an operational environment (competitive manufacturing in case of key enabling technologies, or in space) |
| Not ticked | Not ticked | Not ticked | Not ticked | Not ticked | Not ticked | Not ticked | Not ticked | Not ticked |

----------------------END OF THE DOCUMENT-----------------------------

### Draft review notes (not additional official form fields)

**[REQUIRES STUDENT/ADVISOR INPUT]**: confirm current official template/version, student/advisor identities and affiliations, inventorship, any required signatures/submission dates, data licences/access permissions, annotation/ethics requirements and institutional endorsements. Do not copy names, dates or TRL ticks from the older repository draft without confirmation. Transfer the reviewed content into the official form; this draft is not a submission or approval.
