# Expanding the biological scope responsibly

## What the current baseline actually uses

The primary Task A contract has **five enzymes and one transporter**, not six
human metabolic enzymes plus a transporter: CYP3A4, CYP2C9, CYP2C19, CYP2D6,
CYP1A2 and ABCB1/P-gp. These produce six target effects and one absolute-sum
feature. This was a narrow project-specification choice, not a finding that other
mechanisms do not matter, nor an FDA-approved herb prediction panel.

The general extractor separately knows nine targets, including CYP2E1, SLCO1B1
and NR1I2. Their presence in curation does not admit them to the primary model.
NR1I2/PXR is a regulatory receptor, not a drug-metabolizing enzyme. Adding it to
a sum alongside its downstream CYP effects risks double counting.

“Biology only before reading a paper” is misleading. The biology must be sourced
from papers or suitable databases. The boundary excludes publication outcomes,
abstract text and entity encodings from predictive inputs. It does not remove
the need to read mechanistic studies. Inhibition yields a negative target-activity
polarity in the current equation, not an automatic increase in a calibrated risk
score. Clinical exposure, toxicity and active metabolites need separate reasoning.

## Added curation panel

The versioned `expanded_curation_v1` panel in `src/data/mechanistic_panel.py`
contains seven enzyme targets and nine transporter targets:

| Family | Targets |
| --- | --- |
| CYP enzymes | CYP1A2, CYP2B6, CYP2C8, CYP2C9, CYP2C19, CYP2D6, CYP3A4 |
| Efflux transporters | ABCB1/P-gp, ABCG2/BCRP |
| Uptake transporters | SLCO1B1/OATP1B1, SLCO1B3/OATP1B3, SLC22A6/OAT1, SLC22A8/OAT3, SLC22A2/OCT2 |
| MATE transporters | SLC47A1/MATE1, SLC47A2/MATE2-K |

The MATE2-K entry uses the SLC47A2 gene identifier for curation indexing; evidence
must specify the relevant isoform. FDA lists CYP3A4/5 together; this panel retains
the existing CYP3A4 identifier and does not pretend to resolve CYP3A5-specific
measurements. Existing CYP2E1 and NR1I2 research rows are retained outside this
particular expansion panel, not deleted or declared unimportant.

These priorities are supported by the [FDA substrate/inhibitor/inducer tables](https://www.fda.gov/drugs/drug-interactions-labeling/drug-development-and-drug-interactions-table-substrates-inhibitors-and-inducers),
particularly Table 1-1 and Table 3-1, checked on 30 October 2026. The
[ICH M12 guideline](https://www.ema.europa.eu/en/documents/scientific-guideline/ich-m12-guideline-drug-interaction-studies-step-5_en.pdf)
also identifies enzyme/transporter interaction assessment. These are authoritative
sources for general interaction-study scope; they do not establish herb-specific
measurements, validate this model, or cover all pharmacodynamic interactions.

## Implemented expansion and reproducibility

```sh
venv/bin/python -m src.data.mechanistic_panel \
  --curation data/curation/mechanistic_features.csv \
  --output /path/to/new_expanded_panel
```

The checked-in preparation is `docs/review_batches/expanded_panel_20261030/`.
It snapshots the existing curation bytes, records coverage/status counts and
creates 522 missing entity-target candidate rows across 58 entities. All new
rows are `not_tested`, with blank measurements and UNVERIFIED provenance. This
status means pending measurement/search, not that a systematic search found no
evidence. The candidate file must fail verification. No values or sources were
invented. Existing measurements and the fixed primary feature contract are not
silently changed. A source URL explaining target relevance is not evidence that
a particular herb inhibits that target.

This implements broader evidence-collection scope. It does **not** implement a
16-target predictive model. Such a model needs a distinct reviewed feature
contract, verified measurements for herb and drug, compartment-aware interpretation
of renal/uptake/efflux pathways and an explicit missingness policy. Canonical
training eligibility remains false.

## Does a smaller panel reduce reliability?

It limits biological coverage and can miss interactions mediated by other
pathways. However, adding unsupported or noisy inputs does not establish better
accuracy. A larger panel may increase missingness, correlated effects and
estimation difficulty. Neither the small nor expanded panel has demonstrated
predictive reliability on independent reference labels.

After eligibility is established, prespecify a comparison of the original and
expanded feature contracts using the same independently labeled assessment pairs
and disjoint roles. Choose features and tune only within development roles;
keep assessment labels untouched. Report coverage of known measurements as well
as predictive discrimination, calibration and abstention, with justified inference.
Do not choose the panel after inspecting assessment performance. Scientific
validation requires this empirical comparison; additional unit tests alone
cannot answer it. Current tests verify structural safeguards only.
