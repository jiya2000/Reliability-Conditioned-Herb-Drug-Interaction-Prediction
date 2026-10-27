# Mechanistic Feature Blind-Curation To-Do List
## Reliability-Conditioned Herb–Drug Interaction (HDI) Benchmark

**Document Identifier:** `REP-CURATION-TODO-2026-10-27`  
**Target File:** `data/curation/mechanistic_features.csv`  
**Current Status:** **722 rows generated, 0 verified, 722 UNVERIFIED** (522 PK rows + 200 PD block rows)  
**Integrity Rule:** In accordance with Phase 3 scientific protocol, models will **NOT** run on new mechanistic features until human curators supply external, verifiable citations (DOI / PMID / database ID) and measured values into `data/curation/mechanistic_features.csv`. No values may be inferred from labels or legacy dictionaries.

***Blind Protocol:*** Curator must **not see herb/drug labels or study outcomes** (interaction/no-interaction) during curation. The curator will be presented with entity-target pairs anonymized by ID only. Measured values must be recorded **regardless of expected result** — do not suppress or alter values based on anticipated interaction direction. This protocol eliminates confirmation bias and ensures the curation dataset reflects the true evidence base.

---

## 1. Curation Schema Specifications

Every row in `data/curation/mechanistic_features.csv` contains the following fields:

| Field | Type | Description | Verification Requirement |
| :--- | :--- | :--- | :--- |
| **`entity`** | String | Botanical ID (`HERB_*`) or Prescription Drug ID (`DRUG_*`) | Must match canonical benchmark identifier |
| **`target`** | String | Pharmacokinetic or pharmacodynamic target | One of canonical PK targets (CYP3A4, CYP2D6, CYP2C9, CYP2C19, CYP1A2, CYP2E1, ABCB1, SLCO1B1, NR1I2) **or** PD targets (VKORC1, PTGS1, PTGS2, HSD11B2, SLCO1A2) |
| **`status`** | String | Curation status: `measured` | Tested and measured value obtained; `tested_no_effect` | Actively tested and found no measurable effect; `not_tested` | No experimental data found for this entity-target pair. Must be set based on evidence, not assumed. |
| **`value_ic50_uM`** | Float | Half-maximal inhibitory concentration in $\mu	ext{M}$ | For inhibition targets. Enter measured $IC_{50}$ value. Leave empty if `status` is not `measured`. |
| **`value_ki_uM`** | Float | Inhibition constant in $\mu	ext{M}$ | For inhibition targets. Enter measured $K_i$ value. Leave empty if `status` is not `measured`. |
| **`value_fold_induction`** | Float | Fold induction of enzyme expression/activity | For induction targets (e.g., CYP3A4, CYP2B6). Enter fold-induction value (e.g., 2.5 for 2-fold induction). Leave empty if `status` is not `measured`. |
| **`value_fm`** | Float | Fraction metabolized via target | For drugs only. Enter $f_m$ value (0.0–1.0). Leave empty if `status` is not `measured` or if the entity is not a drug. |
| **`value_substrate_flag`** | Integer | Substrate designation (1 = substrate, 0 = not a substrate) | For drugs only. Enter `1` if the drug is a known substrate for this target, `0` if not a substrate, or leave empty if unknown. Do not use for botanical entities. |
| **`exposure_conc_uM`** | float | Exposure concentration in $\mu	ext{M}$ for static-model R values | Enter steady-state or $C_{	ext{max}}$ concentration in $\mu	ext{M}$. Enables static PK/PD modeling. Leave empty if not available. |
| **`exposure_source`** | String | Source of exposure data (FDA label, Pharmacometrics, literature) | Replace `"UNVERIFIED"` with actual source name (`FDA_DDI_Table`, `DrugBank`, `NaPDI`, primary literature). |
| **`assay_or_unit`** | String | Measurement context (e.g., `microsomal_IC50_uM`, `in_vivo_clearance_fm`, `Ki_uM`, `fold_induction`) | Must describe measurement assay type. |
| **`citation`** | String | Primary literature citation or persistent identifier | Valid DOI (e.g., `10.1016/j.clpt.2003.08.006`) or PubMed ID (`PMID_...`). Replace `"UNVERIFIED"` with actual citation. |
| **`retrieval_date`** | String | Date of accession (`YYYY-MM-DD`) | Date verified by human curator. Replace `"UNVERIFIED"` with actual date. |
| **`notes`** | String | Specific experimental context (cell line, human liver microsomes, healthy volunteers, species, preparation) | Relevant methodological notes. Record facts without interpretation of interaction outcome. |

---

## 2. Canonical Target Definitions

### 9 Pharmacokinetic (PK) Targets
The 9 canonical PK targets represent the core metabolic and transport pathways governing herb–drug interaction:

1. **`CYP3A4`**: Major hepatic and intestinal cytochrome P450 enzyme responsible for >50% of prescription drug metabolism.
2. **`CYP2D6`**: Polymorphic phase I metabolic enzyme clearing cardiovascular and neuropsychiatric agents.
3. **`CYP2C9`**: Primary metabolic clearance pathway for S-warfarin, phenytoin, and NSAIDs.
4. **`CYP2C19`**: Polymorphic enzyme clearing proton pump inhibitors (omeprazole) and clopidogrel.
5. **`CYP1A2`**: Phase I enzyme clearing theophylline, caffeine, and clozapine.
6. **`CYP2E1`**: Ethanol- and chlorzoxazone-metabolizing cytochrome P450 enzyme.
7. **`ABCB1` (P-glycoprotein / MDR1)**: Major intestinal and canalicular ATP-binding cassette efflux transporter.
8. **`SLCO1B1` (OATP1B1)**: Hepatic uptake transporter clearing statins (rosuvastatin, atorvastatin).
9. **`NR1I2` (PXR - Pregnane X Receptor)**: Master nuclear xenobiotic receptor driving transcriptional induction of CYP3A4 and ABCB1.

### 5 Pharmacodynamic (PD) Targets (Optional Block)
Added as a separate block for Task A (PK-only) and PK+PD task modes:

10. **`VKORC1`**: Vitamin K epoxide reductase complex subunit 1 — direct pharmacodynamic target of warfarin and analogs.
11. **`PTGS1` (COX-1)**: Cyclooxygenase-1 — antiplatelet target (aspirin), platelet aggregation.
12. **`PTGS2` (COX-2)**: Cyclooxygenase-2 — inflammatory prostaglandin target, inflammation and analgesia.
13. **`HSD11B2`**: 11beta-hydroxysteroid dehydrogenase 2 — cortisol–cortisone conversion target; mineralocorticoid activity risk.
14. **`SLCO1A2` (OATP1A2)**: Hepatic/intestinal uptake transporter — substrate for nadolol, fexofenadine, fluoroquinolones.

---

## 3. Entity-Target Coverage Matrix

The curation table covers all entity–target pairs across the 141-record benchmark panel:

### 3.1 PK Rows: 18 Botanical Entities × 9 Targets = 162 Entries
Each herb × each of the 9 canonical PK targets has one row. Status starts as `not_tested`. Curators fill `status = measured` with supporting literature.

**Priority entities** (per suggested source: NaPDI, FDA DDI Table, DrugBank, NPASS):
- `HERB_HYPERICUM`: targets needed `NR1I2`, `CYP3A4`, `ABCB1`, `CYP2C9`, `CYP1A2`, `CYP2E1`
- `HERB_GOLDENSEAL`: targets needed `CYP3A4`, `CYP2D6`, `ABCB1`
- `HERB_GREENTEA`: targets needed `SLCO1B1`, `ABCB1`, `CYP3A4`, `CYP2C9`
- `HERB_CURCUMA`: targets needed `CYP3A4`, `CYP2C9`, `ABCB1`, `SLCO1B1`
- `HERB_GARLIC`: targets needed `CYP3A4`, `CYP2C9`, `CYP2E1`, `ABCB1`
- `HERB_GINKGO`: targets needed `CYP3A4`, `CYP2C9`, `CYP2C19`, `ABCB1`
- `HERB_GINSENG`: targets needed `CYP3A4`, `CYP2D6`, `ABCB1`
- `HERB_KAVA`: targets needed `CYP1A2`, `CYP2E1`, `CYP3A4`, `CYP2C9`
- `HERB_LICORICE`: targets needed `CYP3A4`, `CYP2C9`, `ABCB1`
- `HERB_GRAPEFRUIT`: targets needed `CYP3A4`, `ABCB1`, `SLCO1B1`
- `HERB_ECHINACEA`: targets needed `CYP3A4`, `CYP1A2`
- `HERB_CRANBERRY`: targets needed `CYP2C9`
- `HERB_CHAMOMILE`: targets needed `CYP1A2`, `CYP2C9`, `CYP3A4`
- `HERB_PEPPERMINT`: targets needed `CYP3A4`
- Negative control botanicals: `HERB_MILKTHISTLE`, `HERB_VALERIAN`, `HERB_SAWPALMETTO`, `HERB_BLACKCOHOSH` —  (per blind protocol: record measured values regardless of expected null result)
- `HERB_KAVA`: also has `GABRA1` target (GABA-A receptor modulation)

### 3.2 PK Rows: 40 Drug Entities × 9 Targets = 360 Entries
Each prescription drug × each of the 9 canonical PK targets has one row. Status starts as `not_tested`. Curators fill based on DrugBank, FDA DDI Table, Flockhart Table, and NPASS.

**Priority clinical substrates** (per suggested source: FDA DDI Table, Flockhart Table, DrugBank):
- `DRUG_WARFARIN`: `CYP2C9` (primary substrate), `VKORC1` (PD block)
- `DRUG_MIDAZOLAM`: `CYP3A4` (sensitive index substrate)
- `DRUG_THEOPHYLLINE`: `CYP1A2` (sensitive index substrate)
- `DRUG_CHLORZOXAZONE`: `CYP2E1` (probe substrate)
- `DRUG_DEXTROMETHORPHAN`: `CYP2D6` (probe substrate)
- `DRUG_FELODIPINE`, `DRUG_NIFEDIPINE`, `DRUG_DILTIAZEM`: `CYP3A4` substrates
- `DRUG_ATORVASTATIN`, `DRUG_SIMVASTATIN`: `CYP3A4` substrates, `SLCO1B1` substrates
- `DRUG_FEXOFENADINE`: `ABCB1`, `SLCO1A2` substrates
- `DRUG_OMEPRAZOLE`: `CYP2C19` (primary), `CYP3A4` (secondary)
- `DRUG_WARFARIN`: `VKORC1` (PD block)

### 3.3 PD Block Rows: Drug-PD Target Pairs (Optional)
Additional rows for drug–PD target pairs not covered by the PK substrate framework. These are flagged for optional use in PK+PD task mode.

**PD target coverage per drug:**
- `DRUG_WARFARIN`: `VKORC1` (direct PD target, warfarin mechanism)
- `DRUG_ASPIRIN`: `PTGS1` (irreversible COX-1 inhibitor), `PTGS2` (COX-2 inhibition)
- `DRUG_PREDNISOLONE`: `HSD11B2` (11β-HSD2 substrate)
- `DRUG_HYDROCHLOROTHIAZIDE`: `HSD11B2` (mineralocorticoid additive risk)
- Other drugs may have PD entries as evidence emerges.

---

## 4. Human Curator Instructions

### Blind Curation Protocol
1. **Anonymization:** The curator will not see herb/drug labels (`HERB_*` / `DRUG_*`), outcome labels (`interaction_present` / `no_interaction_observed`), or study outcomes (positive/negative results). Each row presents only: `entity_id` and `target_id`. This prevents confirmation bias.
2. **Value Recording:** For each entity-target pair, the curator locates peer-reviewed experimental literature or database records. Measured values **must be recorded regardless of expected result** — do not suppress, round, or alter values based on anticipated interaction direction (induction vs. inhibition, positive vs. negative interaction).
3. **Status Assignment:**
   - `measured`: A quantifiable value was found in the literature (e.g., $IC_{50} = 10\ \mu	ext{M}$, fold induction = 3.2, $f_m = 0.85$). Record the value in the appropriate column.
   - `tested_no_effect`: The entity-target pair was actively tested and no measurable effect was detected (e.g., "no significant CYP3A4 inhibition observed at 100 $\mu$M"). Record `status = tested_no_effect` and leave value columns empty or note "no effect observed."
   - `not_tested`: No experimental data found for this pair. Leave status as `not_tested` and all value columns empty.
4. **Filling the Form:**
   - Enter the measured parameter into the appropriate value column (`value_ic50_uM`, `value_ki_uM`, `value_fold_induction`, `value_fm`, `value_substrate_flag`).
   - Enter the measurement type in `assay_or_unit`.
   - Replace `"UNVERIFIED"` in `exposure_source` with the actual database or source name (`FDA_DDI_Table`, `DrugBank`, `NaPDI`).
   - Replace `"UNVERIFIED"` in `citation` with the exact DOI or PubMed ID (e.g., `10.1016/j.clpt.2003.08.006` or `PUBMED_14631289`).
   - Fill `retrieval_date` with today's date (`YYYY-MM-DD`).
   - Record methodological details in `notes` without interpreting the interaction outcome.
5. **Save the file.**
6. **Run verification:** Run `pytest tests/test_phase3_and_ci_checks.py` to verify schema conformance and check whether all rows have been completed.

---

## 5. Verification and Guard

The `UnverifiedCurationError` guard remains in effect. Models will **strictly block** execution on `feature_set="new"` until all rows in `data/curation/mechanistic_features.csv` have human-verified sources and citations. The verification function `is_curation_verified()` checks:

- All `source` fields are not `"UNVERIFIED"` (or have corresponding citations entered)
- All `citation` fields are not `"UNVERIFIED"` (valid DOI or PubMed ID)
- All rows have a status of `measured` or `tested_no_effect` (i.e., no `not_tested` rows remain unverified)

**Important:** The blind protocol ensures that `not_tested` rows are expected and acceptable — the guard only fails if verified rows are missing, not if some rows remain `not_tested` pending external data.

---

## 6. Suggested External Sources for Curation

### NaPDI (Center of Excellence for Natural Product-Drug Interaction Research)
- **URL:** https://napdi.org
- **Best For:** Standardized botanical extracts, in vitro enzyme inhibition ($IC_{50}$, $K_i$), and clinical dynamic pharmacokinetic interaction parameters.

### FDA DDI Table
- **URL:** https://www.fda.gov/drugs/drug-interactions-labeling/drug-development-and-drug-interactions-table-substrates-inhibitors-and-inducers
- **Best For:** Clinical index substrates, probe drugs, and validated substrate dependency fractions ($f_m$).

### DrugBank Online
- **URL:** https://go.drugbank.com/
- **Best For:** Standardized drug pharmacology summaries, UniProt target accession numbers, pathway enzymes, and transporter bindings.

### Flockhart Cytochrome P450 Table
- **URL:** https://drug-interactions.medicine.iu.edu/
- **Best For:** Clinical in vivo substrate sensitivity tiers, strong/moderate/weak inhibitor classifications, and in vivo interaction outcomes.

### NPASS (Natural Product Activity and Species Source Database)
- **URL:** https://bidd.group/NPASS/
- **Best For:** Isolated phytochemical constituent affinities and in vitro $K_i$ / $IC_{50}$ values for specific active molecules.

---

## 7. Example Curated Row

| entity | target | status | value_ic50_uM | value_ki_uM | value_fold_induction | value_fm | value_substrate_flag | exposure_conc_uM | exposure_source | assay_or_unit | citation | retrieval_date | notes |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| DRUG_MIDAZOLAM | CYP3A4 | measured | 1.8 | | | 0.95 | 1 | | FDA_DDI_Table | microsomal_IC50_uM | 10.1016/s0091-2721(00)00145-5 | 2026-01-15 | Gold standard FDA clinical index substrate; $f_{m,CYP3A4} > 0.90$ |
| HERB_HYPERICUM | CYP3A4 | measured | 0.5 | | | | 1 | | NaPDI | microsomal_IC50_uM | 10.1016/j.clpt.2005.05.012 | 2026-01-15 | Hyperforin-mediated PXR activation, strong CYP3A4 induction |
| DRUG_WARFARIN | VKORC1 | measured | | | | | | 5.0 | FDA_DDI_Table | | 2026-01-15 | Warfarin's direct pharmacodynamic target; vitamin K epoxide reductase |

### Why the PD Block Has 200 Rows

The pharmacodynamic (PD) block contains 200 rows representing drug–PD target pairs that are not covered by the PK substrate framework. These include:
- `DRUG_WARFARIN` + `VKORC1` (direct pharmacodynamic target)
- `DRUG_ASPIRIN` + `PTGS1` and `PTGS2` (COX enzyme targets)
- `DRUG_PREDNISOLONE` + `HSD11B2` (11β-HSD2 substrate)
- `DRUG_HYDROCHLOROTHIAZIDE` + `HSD11B2` (mineralocorticoid additive risk)
These rows are optional and only used when running the model in PK+PD task mode. They are flagged with `status = not_tested` by default and should be curated separately from the PK rows.
### Status: `not_tested` as Missing

The `status` field uses three values: `measured`, `tested_no_effect`, and `not_tested`. `not_tested` indicates that no experimental data was found for this entity-target pair; it should be treated as **missing data**, never as zero. Rows with `status = not_tested` have all value columns left empty. The `measured_count` curation-only feature (below) counts only rows with `status = measured`.
### Measured Count Curation-Only Feature

A new curation-only feature `measured_count` can be computed from the CSV: the number of entity-target pairs with `status = measured`. Once verified data exists, this can be reported alongside the overlap AUC:

| Feature | overlap_6d AUC | measured_count AUC |
| :--- | :---: | ---: |
| Train-only | 0.8669 | TBD |
| Full panel | 0.8572 | TBD |

This feature is strictly for curation progress tracking and should not be used for model training until all entries have `status = measured`.