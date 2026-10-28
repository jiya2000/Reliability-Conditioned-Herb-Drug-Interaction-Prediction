# Pharmacokinetic & Pharmacodynamic Target Curation Protocol (Phase 3d)
## Botanical and Drug Mechanistic Feature Curation Guide

**Document Identifier:** `REP-CURATION-TODO-2026-10-28`  
**Target File:** `data/curation/mechanistic_features.csv`  
**Current Status:** **722 rows generated, 0 verified, 722 UNVERIFIED** (522 PK rows + 200 PD block rows)  
**Integrity Rule:** In accordance with Phase 3 scientific protocol, models will **NOT** run on new mechanistic features until human curators supply external, verifiable citations (DOI / PMID / database ID) and measured values into `data/curation/mechanistic_features.csv`. No values may be inferred from labels or legacy dictionaries.

---

## 1. Blind Curation Protocol

To eliminate confirmation bias and ensure evidentiary decoupling:
1. **Blinded to Outcomes:** Curators are strictly **blind to benchmark interaction labels and study outcomes** (`interaction_present`, `no_interaction_observed`, clinical significance, or effect directions).
2. **Visible Entity Identifiers:** Entity identifiers (`HERB_*` botanical names and `DRUG_*` generic drug names) and target identifiers are visible so standard pharmacopeias and literature can be queried.
3. **Randomized Order:** Entity-target pairs in the curation workflow are presented in randomized order.
4. **Benchmark CSV Isolation:** Curators must **never open `data/benchmark/verified_botanical_hdi.csv`** or any other evaluation benchmark files during curation.
5. **Objective Recording:** Measured values and experimental parameters must be recorded objectively from literature/database records regardless of whether an interaction is anticipated.

---

## 2. Target Specifications

### 2.1 9 Canonical Pharmacokinetic (PK) Targets
The 9 canonical PK targets represent the core metabolic enzymes and transport proteins governing pharmacokinetic HDI:
1. **`CYP3A4`**: Major hepatic and intestinal cytochrome P450 enzyme responsible for clearance of >50% of prescription drugs.
2. **`CYP2D6`**: Polymorphic phase I enzyme clearing cardiovascular, psychiatric, and analgesic drugs.
3. **`CYP2C9`**: Primary metabolic clearance pathway for S-warfarin, phenytoin, and NSAIDs.
4. **`CYP2C19`**: Phase I enzyme metabolizing proton pump inhibitors, clopidogrel, and antiepileptics.
5. **`CYP1A2`**: Phase I enzyme clearing theophylline, caffeine, and clozapine.
6. **`CYP2E1`**: Ethanol- and chlorzoxazone-metabolizing cytochrome P450 enzyme.
7. **`ABCB1` (P-glycoprotein / MDR1)**: Major intestinal, renal, and blood-brain barrier ATP-binding cassette efflux transporter.
8. **`SLCO1B1` (OATP1B1)**: Hepatic basolateral uptake transporter clearing statins and endogenous bile acids.
9. **`NR1I2` (PXR - Pregnane X Receptor)**: Master nuclear receptor driving xenobiotic transcriptional induction of CYP3A4 and ABCB1.

### 2.2 5 Pharmacodynamic (PD) Targets (200-Row Drug Block)
The PD block contains 200 rows representing 40 prescription drugs $\times$ 5 pharmacodynamic targets:
10. **`VKORC1`**: Vitamin K epoxide reductase complex subunit 1 (direct molecular target of warfarin).
11. **`PTGS1` (COX-1)**: Cyclooxygenase-1 (antiplatelet target inhibited by aspirin).
12. **`PTGS2` (COX-2)**: Cyclooxygenase-2 (inflammatory prostaglandin synthesis target).
13. **`HSD11B2`**: 11$\beta$-hydroxysteroid dehydrogenase type 2 (cortisol-cortisone shuttle enzyme).
14. **`SLCO1A2` (OATP1A2)**: Intestinal and systemic organic anion uptake transporter.

**Computability of PD Overlap:**
Because the current PD block curates only prescription drug targets (40 drugs $\times$ 5 PD targets = 200 rows) and does not currently curate corresponding botanical pharmacodynamic targets (18 herbs $\times$ 5 PD targets = 90 rows), pair-level PD overlap is **not computable**. PD overlap evaluation is deferred until corresponding botanical PD targets are curated.

---

## 3. Curation Table Schema

The curation table (`data/curation/mechanistic_features.csv`) conforms to the following 15-column schema:

| Column | Type | Allowed Values / Format | Description |
| :--- | :--- | :--- | :--- |
| **`entity`** | String | `HERB_*` or `DRUG_*` | Canonical entity identifier |
| **`target`** | String | Canonical PK / PD target name | Target symbol (e.g. `CYP3A4`, `ABCB1`, `VKORC1`) |
| **`status`** | String | `measured`, `tested_no_effect`, `not_tested` | Evidence classification (see below) |
| **`value_ic50_uM`** | Float | Numeric ($\mu\text{M}$) or empty | Half-maximal inhibitory concentration |
| **`value_ki_uM`** | Float | Numeric ($\mu\text{M}$) or empty | Apparent or competitive inhibition constant |
| **`value_fold_induction`** | Float | Numeric ($\ge 1.0$) or empty | Expression or activity fold-induction |
| **`value_fm`** | Float | Numeric ($0.0 - 1.0$) or empty | Fraction of drug clearance via this target |
| **`value_substrate_flag`** | Integer | `0`, `1`, or empty | Substrate classification (drugs only) |
| **`exposure_conc_uM`** | Float | Numeric ($\mu\text{M}$) or empty | Steady-state or maximum clinical plasma concentration |
| **`exposure_source`** | String | Source name or `UNVERIFIED` | Origin of exposure concentration data |
| **`assay_or_unit`** | String | Context description | Assay methodology (e.g. `human_liver_microsomes_IC50`) |
| **`source`** | String | Database or compendium name | Authoritative repository (`NaPDI`, `FDA_DDI_Table`, `DrugBank`, `NPASS`) |
| **`citation`** | String | Valid DOI or PubMed PMID | External citation (`10.xxxx/...` or `PMID_...`) |
| **`retrieval_date`** | String | `YYYY-MM-DD` or empty | Date accessed by curator |
| **`notes`** | String | Factual experimental context | Assay conditions, species, extract standardization |

### Status Classifications:
- **`measured`**: A quantitative experimental value was reported in literature or authoritative databases. At least one numeric value column (`value_ic50_uM`, `value_ki_uM`, `value_fold_induction`, `value_fm`, or `value_substrate_flag`) must be populated.
- **`tested_no_effect`**: The entity-target interaction was actively tested in experimental studies and demonstrated no measurable inhibition, induction, or transport. Numeric value columns are left empty.
- **`not_tested`**: No experimental data exists in published literature or queried databases for this entity-target pair. A `not_tested` row is allowed **only if** accompanied by a citation or database lookup note verifying that a search was conducted and confirmed that no data exists.

---

## 4. Verification Guard Rules (`is_curation_verified`)

The programmatic execution guard in `src/data/mechanistic_curation.py` (`is_curation_verified()`) enforces strict criteria before any new-curation model may run:
1. **No `UNVERIFIED` Source:** Every row must replace `"UNVERIFIED"` in `source` with the actual database or literature source.
2. **No `UNVERIFIED` Citation:** Every row must replace `"UNVERIFIED"` in `citation` with a verifiable identifier (DOI or PubMed ID).
3. **Status Consistency:**
   - Rows marked `status = measured` must have at least one valid numeric value column populated.
   - Rows marked `status = tested_no_effect` must have a supporting citation documenting the negative finding.
   - Rows marked `status = not_tested` are allowed **only if** accompanied by a citation or database lookup note stating that external searches confirmed no experimental data was found.
4. **Curation Density Check:** All 18 botanical entities must have all 9 canonical PK targets verified (`status = measured` or `status = tested_no_effect` with verified citations/sources) before model execution.

---

## 5. Curation-Only Baseline (`measured_count`)

A curation-only baseline metric, **`measured_count`**, counts the number of targets with `status = measured` per entity. 
- Once human curation is complete and verified data exists in `data/curation/mechanistic_features.csv`, `measured_count` will be evaluated under grouped cross-validation to benchmark whether target measurement frequency alone predicts interaction outcomes.
- A permanent CI unit test (`test_permanent_e_curation_gap_check`) enforces that any new mechanistic model must significantly exceed curation density baselines ($|\Delta \text{AUC}| \ge 0.03$).

---

## 6. Example Row Format (Template Only)

| entity | target | status | value_ic50_uM | value_ki_uM | value_fold_induction | value_fm | value_substrate_flag | exposure_conc_uM | exposure_source | assay_or_unit | source | citation | retrieval_date | notes |
| :--- | :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :--- | :--- | :--- | :--- | :---: | :--- |
| `EXAMPLE_ONLY` | `CYP3A4` | `not_tested` | | | | | | | `UNVERIFIED` | `microsomal_IC50_uM` | `UNVERIFIED` | `UNVERIFIED` | | |