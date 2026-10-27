# Evidence Schema Documentation & Canonical Data Dictionary

This document specifies the schema, typing, validation rules, and pharmacokinetic metadata definitions for the canonical herb–drug interaction (HDI) literature evidence database in `data/raw/canonical_evidence.csv`.

## Overview
The canonical evidence table adheres to strict zero-synthetic provenance requirements. Every record corresponds to an authentic biomedical study identified by persistent citation IDs (PMID, PMCID, or DOI), audited extraction confidence, study methodology categorization, and standardized clinical findings.

---

## Field Specifications

| Column Name | Type | Mandatory? | Allowed Values / Format | Description |
| :--- | :--- | :--- | :--- | :--- |
| `herb_id` | string | **Yes** | e.g., `St_Johns_Wort`, `Ginkgo_biloba` | Normalized canonical identifier for the herbal botanical entity. |
| `drug_id` | string | **Yes** | e.g., `Cyclosporine`, `Warfarin` | Normalized canonical identifier for the co-administered drug. |
| `outcome` | string | **Yes** | `interaction_present`, `no_interaction_observed`, `conflicting`, `unknown` | Empirical clinical/biological outcome observed in the study. |
| `effect_direction` | string | Optional | `decreased_exposure`, `increased_exposure`, `increased_toxicity`, `decreased_effect`, `no_effect`, `unspecified` | Directional pharmacodynamic or pharmacokinetic mechanism. |
| `clinical_significance` | string | Optional | `major`, `moderate`, `minor`, `none`, `uncertain` | Clinical severity grading for patient safety decision support. |
| `evidence_type` | string | **Yes** | `rct`, `clinical`, `case_report`, `observational`, `animal`, `in_vitro`, `mechanistic`, `in_silico` | Hierarchy of scientific evidence level. |
| `source_id` | string | **Yes** | e.g., `PMID:10755392`, `DOI:10.1016/...` | Permanent verifiable bibliographic citation identifier. |
| `study_id` | string | Optional | string | Specific study cluster identifier if distinct from source. |
| `cited_study_ids` | string/JSON | Optional | JSON array or comma-separated list of PMIDs/DOIs | Outgoing citation references used to construct the Citation DAG. |
| `extraction_confidence` | float | **Yes** | $[0.0, 1.0]$ | Confidence score of biomedical extraction and curation. |
| `dose` | string/float | Optional | e.g., `300 mg`, `900 mg/day`, `9 mg` | Administered botanical dose in clinical or preclinical experiment. |
| `extract_standardization` | string/float | Optional | e.g., `3% hyperforin`, `24% ginkgoflavone glycosides`, `0.03` | Percentage or active constituent standardization of botanical extract. |
| `treatment_duration` | string | Optional | e.g., `14 days`, `single dose`, `28 days` | Duration of co-exposure or pre-treatment regimen. |
| `interaction_direction` | string | Optional | `induction`, `inhibition`, `reversible_inhibition`, `tdi` | Specific molecular kinetic interaction category. |
| `time_dependence` | string | Optional | `time_dependent`, `reversible`, `steady_state` | Whether inhibition/induction requires repeated administration. |
| `species` | string | Optional | `human`, `rat`, `mouse`, `dog`, `microsomes`, `caco-2` | Biological experimental model system. |
| `preparation` | string | Optional | `standardized_extract`, `isolated_constituent`, `crude_decoction`, `microsomes`, `hepatocytes`, `caco2` | Physical or chemical botanical preparation. |
| `route` | string | Optional | `oral`, `parenteral_iv`, `intraperitoneal`, `in_vitro` | Route of administration. |
| `population` | string | Optional | e.g., `healthy volunteers`, `renal transplant patients` | Clinical trial cohort characteristics. |
| `publication_date` | string | Optional | YYYY or YYYY-MM-DD | Date of formal peer-reviewed publication. |
| `original_text` | string | Optional | Text | Verbatim excerpt or findings summary from the publication. |

---

## Pharmacokinetic Reachability Integration (Phase 3)
In vitro findings (`evidence_type == 'in_vitro'`) are screened for physiological plausibility via `src/models/pk_reachability.py`:
1. **Hepatic/Systemic Route:** Evaluated using unbound human plasma concentration:
   $$\rho_{\text{reach}} = \min\left(1.0, \frac{[I]_{u,\max}}{\text{IC}_{50} \text{ or } \text{EC}_{50}}\right)$$
2. **Intestinal CYP3A4 / P-gp Route:** Evaluated using gut lumen concentration per FDA Guidance:
   $$I_{\text{gut}} = \frac{\text{dose}}{250\text{ mL}} = \frac{\text{dose\_mg} \times 4000.0}{\text{MW}}\,\mu\text{M}$$
   $$\rho_{\text{reach}} = \min\left(1.0, \frac{I_{\text{gut}}}{\text{potency}}\right)$$
   Where `dose_mg` and `MW` inputs are required, and missing values fall back gracefully with an auditable missingness flag.
