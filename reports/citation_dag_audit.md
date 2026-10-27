# Citation-Derivation DAG Audit Report (Phase 9)

## 1. Overview and Provenance
- **Dataset:** `data/benchmark/verified_botanical_hdi.csv` (141 authentic clinical/pharmacological studies, 0 synthetic records).
- **Audit Target File:** `data/audit/dag_audit_samples.csv` (146 records audited).
- **Total Botanical-Drug Pairs:** 120.
- **Total Roots Identified:** 124.
- **Total Derivative Echoes:** 18.
- **Multi-Study Citation Clusters:** 15.
- **Conflicting Derivative Refutations:** 1.

---

## 2. Echo Chamber Attenuation Comparison: Flat vs Study-Type Scaled

| Metric | Raw Unfiltered Count | Flat Discount (alpha = 0.15) | Study-Type + Text-Reuse Scaled |
| :--- | :--- | :--- | :--- |
| **Total Evidence Units** | 141.0 | 139.45 | 140.30 |
| **Echo Chamber Reduction** | 0.0% | 1.1% | 0.5% |
| **Independent Roots** | 124 | 124 | 124 |

### Key Findings:
1. **Redundancy Reduction:** Citation echo attenuation dampens raw citation inflation by **0.5%**, ensuring that reviews reiterating primary trials do not artificially inflate algorithmic confidence.
2. **Study-Type Nuance:** Scaling alpha by study methodology prevents high-quality clinical replications from being excessively penalized while heavily discounting review summaries.
3. **Contradiction Preservation:** When a derivative study refutes or contradicts the primary root trial, it retains full evidence weight ($w = 1.0$) rather than being discounted as an echo, correctly signaling clinical controversy.

---

## 3. Sample Audited Components

```csv
component_id,herb_id,drug_id,study_id,source_pmid,is_root,primary_root,depth,study_type,outcome,is_contradiction,text_reuse_score,flat_weight,study_type_scaled_weight
COMP_HERB_BLACKCOHOSH_DRUG_DEXTROMETHORPHAN_STUDY_GURLEY_2005E,HERB_BLACKCOHOSH,DRUG_DEXTROMETHORPHAN,STUDY_GURLEY_2005E,STUDY_GURLEY_2005E,True,STUDY_GURLEY_2005E,0,clinical,no_interaction_observed,False,1.0,1.0,1.0
COMP_HERB_BLACKCOHOSH_DRUG_DIGOXIN_STUDY_GURLEY_2005B,HERB_BLACKCOHOSH,DRUG_DIGOXIN,STUDY_GURLEY_2005B,STUDY_GURLEY_2005B,True,STUDY_GURLEY_2005B,0,clinical,no_interaction_observed,False,1.0,1.0,1.0
COMP_HERB_BLACKCOHOSH_DRUG_MIDAZOLAM_STUDY_GURLEY_2008C,HERB_BLACKCOHOSH,DRUG_MIDAZOLAM,STUDY_GURLEY_2008C,STUDY_GURLEY_2008C,True,STUDY_GURLEY_2008C,0,clinical,no_interaction_observed,False,1.0,1.0,1.0
COMP_HERB_BLACKCOHOSH_DRUG_THEOPHYLLINE_STUDY_DONOVAN_2004F,HERB_BLACKCOHOSH,DRUG_THEOPHYLLINE,STUDY_DONOVAN_2004F,STUDY_DONOVAN_2004F,True,STUDY_DONOVAN_2004F,0,clinical,no_interaction_observed,False,1.0,1.0,1.0
COMP_HERB_BLACKCOHOSH_DRUG_WARFARIN_STUDY_GURLEY_2008G,HERB_BLACKCOHOSH,DRUG_WARFARIN,STUDY_GURLEY_2008G,STUDY_GURLEY_2008G,True,STUDY_GURLEY_2008G,0,clinical,no_interaction_observed,False,1.0,1.0,1.0
COMP_HERB_CHAMOMILE_DRUG_DEXTROMETHORPHAN_STUDY_GURLEY_2008I,HERB_CHAMOMILE,DRUG_DEXTROMETHORPHAN,STUDY_GURLEY_2008I,STUDY_GURLEY_2008I,True,STUDY_GURLEY_2008I,0,clinical,no_interaction_observed,False,1.0,1.0,1.0
COMP_HERB_CHAMOMILE_DRUG_THEOPHYLLINE_STUDY_GANZERA_2006,HERB_CHAMOMILE,DRUG_THEOPHYLLINE,STUDY_GANZERA_2006,STUDY_GANZERA_2006,True,STUDY_GANZERA_2006,0,clinical,interaction_present,False,1.0,1.0,1.0
COMP_HERB_CHAMOMILE_DRUG_WARFARIN_STUDY_SEGAL_2006,HERB_CHAMOMILE,DRUG_WARFARIN,STUDY_SEGAL_2006,STUDY_SEGAL_2006,True,STUDY_SEGAL_2006,0,case_report,interaction_present,False,1.0,1.0,1.0
COMP_HERB_CRANBERRY_DRUG_DIGOXIN_STUDY_NGO_2010,HERB_CRANBERRY,DRUG_DIGOXIN,STUDY_NGO_2010,STUDY_NGO_2010,True,STUDY_NGO_2010,0,clinical,no_interaction_observed,False,1.0,1.0,1.0
COMP_HERB_CRANBERRY_DRUG_MIDAZOLAM_STUDY_LILJA_2007,HERB_CRANBERRY,DRUG_MIDAZOLAM,STUDY_LILJA_2007,STUDY_LILJA_2007,True,STUDY_LILJA_2007,0,clinical,no_interaction_observed,False,1.0,1.0,1.0

```
