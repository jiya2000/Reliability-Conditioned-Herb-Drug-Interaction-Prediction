# Mechanistic Feature Blind-Curation To-Do List
## Reliability-Conditioned Herb–Drug Interaction (HDI) Benchmark

**Document Identifier:** `REP-CURATION-TODO-2026-10-27`  
**Target File:** `data/curation/mechanistic_features.csv`  
**Current Status:** **522 rows generated, 0 verified, 522 UNVERIFIED**  
**Integrity Rule:** In accordance with Phase 3 scientific protocol, models will **NOT** run on new mechanistic features until human curators supply external, verifiable citations (DOI / PMID / database ID) and measured values into `data/curation/mechanistic_features.csv`. No values may be inferred from labels or legacy dictionaries.

---

## 1. Curation Schema Specifications

Every row in `data/curation/mechanistic_features.csv` contains the following fields:

| Field | Type | Description | Verification Requirement |
| :--- | :--- | :--- | :--- |
| **`entity`** | String | Botanical ID (`HERB_*`) or Prescription Drug ID (`DRUG_*`) | Must match canonical benchmark identifier |
| **`target`** | String | Pharmacokinetic enzyme, transporter, or nuclear receptor | One of canonical 9 targets |
| **`value`** | Float | Quantitative or semi-quantitative modulatory parameter: (e.g. $IC_{50}$ in $\mu\text{M}$, clearance fraction $f_m$, or $+1.0/-1.0$ potency score) | **Leave empty if unsourced** |
| **`assay_or_unit`** | String | Measurement context (e.g., `microsomal_IC50_uM`, `in_vivo_clearance_fraction_fm`, `Ki_uM`, `fold_induction`) | Must describe measurement assay |
| **`source`** | String | Authoritative database or repository name | `NaPDI`, `FDA_DDI_Table`, `DrugBank`, `Flockhart_Table`, `NPASS` |
| **`citation`** | String | Primary literature citation or persistent identifier | Valid DOI (e.g., `10.1016/...`) or PubMed PMID (`PMID_...`) |
| **`retrieval_date`** | String | Date of accession (`YYYY-MM-DD`) | Date verified by human curator |
| **`notes`** | String | Specific experimental context (e.g., cell line, human liver microsomes, healthy volunteers) | Relevant methodological notes |

---

## 2. Canonical Target Definitions

The 9 canonical targets represent the core pharmacokinetic pathways governing metabolic and transport HDI:
1. **`CYP3A4`**: Major hepatic and intestinal cytochrome P450 enzyme responsible for >50% of prescription drug metabolism.
2. **`CYP2D6`**: Polymorphic phase I metabolic enzyme clearing cardiovascular and neuropsychiatric agents.
3. **`CYP2C9`**: Primary metabolic clearance pathway for S-warfarin, phenytoin, and NSAIDs.
4. **`CYP2C19`**: Polymorphic enzyme clearing proton pump inhibitors (omeprazole) and clopidogrel.
5. **`CYP1A2`**: Phase I enzyme clearing theophylline, caffeine, and clozapine.
6. **`CYP2E1`**: Ethanol- and chlorzoxazone-metabolizing cytochrome P450 enzyme.
7. **`ABCB1` (P-glycoprotein / MDR1)**: Major intestinal and canalicular ATP-binding cassette efflux transporter (digoxin, talinolol, cyclosporine substrate).
8. **`SLCO1B1` (OATP1B1)**: Hepatic uptake transporter clearing statins (rosuvastatin, atorvastatin).
9. **`NR1I2` (PXR - Pregnane X Receptor)**: Master nuclear xenobiotic receptor driving transcriptional induction of CYP3A4 and ABCB1.

---

## 3. Entities Requiring Human Curation Grouped by Suggested Source

### 3.1 Suggested Source: NaPDI (Center of Excellence for Natural Product-Drug Interaction Research)
**URL:** [https://napdi.org](https://napdi.org)  
**Best For:** Standardized botanical extracts, in vitro enzyme inhibition ($IC_{50}$, $K_i$), and clinical dynamic pharmacokinetic interaction parameters.

#### Priority Botanical Entities (18 Botanicals × 9 Targets = 162 Entries):
1. **`HERB_HYPERICUM` (St. John's Wort / *Hypericum perforatum*)**
   - Targets needed: `NR1I2` (PXR activation), `CYP3A4` (mRNA induction), `ABCB1` (P-gp induction), `CYP2C9`, `CYP1A2`, `CYP2E1`.
2. **`HERB_GOLDENSEAL` (*Hydrastis canadensis*)**
   - Targets needed: `CYP3A4` ($IC_{50}$ in HLMs), `CYP2D6` ($IC_{50}$), `ABCB1` (inhibition).
3. **`HERB_GREENTEA` (*Camellia sinensis* / EGCG)**
   - Targets needed: `SLCO1B1` (OATP inhibition), `ABCB1`, `CYP3A4`, `CYP2C9`.
4. **`HERB_CURCUMA` (*Curcuma longa* / Curcumin)**
   - Targets needed: `CYP3A4`, `CYP2C9`, `ABCB1`, `SLCO1B1`.
5. **`HERB_GARLIC` (*Allium sativum*)**
   - Targets needed: `CYP3A4`, `CYP2C9`, `CYP2E1`, `ABCB1`.
6. **`HERB_GINKGO` (*Ginkgo biloba*)**
   - Targets needed: `CYP3A4`, `CYP2C9`, `CYP2C19`, `ABCB1`.
7. **`HERB_GINSENG` (*Panax ginseng*)**
   - Targets needed: `CYP3A4`, `CYP2D6`, `ABCB1`.
8. **`HERB_KAVA` (*Piper methysticum*)**
   - Targets needed: `CYP1A2`, `CYP2E1`, `CYP3A4`, `CYP2C9`.
9. **`HERB_LICORICE` (*Glycyrrhiza glabra*)**
   - Targets needed: `CYP3A4`, `CYP2C9`, `ABCB1`.
10. **`HERB_GRAPEFRUIT` (*Citrus paradisi*)**
    - Targets needed: `CYP3A4` (irreversible intestinal mechanism-based inactivation), `ABCB1`, `SLCO1B1`.
11. **`HERB_ECHINACEA` (*Echinacea purpurea*)**
    - Targets needed: `CYP3A4` (hepatic vs intestinal differential modulation), `CYP1A2`.
12. **`HERB_CRANBERRY` (*Vaccinium macrocarpon*)**
    - Targets needed: `CYP2C9` (in vitro vs clinical in vivo clearance).
13. **`HERB_CHAMOMILE` (*Matricaria chamomilla*)**
    - Targets needed: `CYP1A2`, `CYP2C9`, `CYP3A4`.
14. **`HERB_PEPPERMINT` (*Mentha piperita*)**
    - Targets needed: `CYP3A4` (menthol modulatory action).
15. **Negative Control Botanicals (Critical for Zero-Leakage Negative Grounding):**
    - `HERB_MILKTHISTLE` (*Silybum marianum* / Silymarin): verify lack of clinical CYP3A4/P-gp modulation in humans.
    - `HERB_VALERIAN` (*Valeriana officinalis*): verify lack of clinical CYP3A4/2D6 modulation in humans.
    - `HERB_SAWPALMETTO` (*Serenoa repens*): verify lack of clinical CYP modulation.
    - `HERB_BLACKCOHOSH` (*Actaea racemosa*): verify lack of clinical CYP3A4/2D6 modulation.

---

### 3.2 Suggested Source: FDA Drug Development and Drug Interactions Guidance Tables
**URL:** [https://www.fda.gov/drugs/drug-interactions-labeling/drug-development-and-drug-interactions-table-substrates-inhibitors-and-inducers](https://www.fda.gov/drugs/drug-interactions-labeling/drug-development-and-drug-interactions-table-substrates-inhibitors-and-inducers)  
**Best For:** Clinical index substrates, probe drugs, and validated substrate dependency fractions ($f_m$).

#### Priority Clinical Substrates (40 Drugs × 9 Targets = 360 Entries):
1. **CYP3A4 Sensitive Clinical Index Substrates:**
   - `DRUG_MIDAZOLAM` (Sensitive index substrate; $f_{m,CYP3A4} > 0.90$)
   - `DRUG_CYCLOSPORINE` (Narrow therapeutic index; substrate of CYP3A4 and ABCB1)
   - `DRUG_TACROLIMUS` (Narrow therapeutic index; substrate of CYP3A4 and ABCB1)
   - `DRUG_INDINAVIR`, `DRUG_SAQUINAVIR`, `DRUG_RITONAVIR` (Protease inhibitors; CYP3A4 substrates/inhibitors)
   - `DRUG_SIMVASTATIN`, `DRUG_ATORVASTATIN` (HMG-CoA reductase inhibitors; CYP3A4 substrates)
   - `DRUG_FELODIPINE`, `DRUG_NIFEDIPINE`, `DRUG_DILTIAZEM` (Calcium channel blockers; CYP3A4 substrates)
   - `DRUG_DOCETAXEL`, `DRUG_PACLITAXEL` (Taxane antineoplastics; CYP3A4 substrates)
   - `DRUG_ALPRAZOLAM`, `DRUG_BUSPIRONE` (Anxiolytics; CYP3A4 substrates)
2. **P-glycoprotein (ABCB1) Clinical Index Substrates:**
   - `DRUG_DIGOXIN` (Index substrate; renal and biliary ABCB1 clearance)
   - `DRUG_DABIGATRAN` (Index substrate; non-CYP mediated ABCB1 substrate)
   - `DRUG_FEXOFENADINE` (Index substrate; ABCB1 and OATP substrate)
3. **CYP2C9 Sensitive Clinical Index Substrates:**
   - `DRUG_WARFARIN` (Narrow therapeutic index; S-warfarin cleared exclusively by CYP2C9)
   - `DRUG_PHENYTOIN` (Narrow therapeutic index; CYP2C9 and CYP2C19 substrate)
   - `DRUG_LOSARTAN` (Angiotensin receptor blocker; CYP2C9 metabolic activation)
   - `DRUG_IBUPROFEN` (NSAID; CYP2C9 substrate)
4. **CYP1A2 Sensitive Clinical Index Substrates:**
   - `DRUG_THEOPHYLLINE` (Narrow therapeutic index; $f_{m,CYP1A2} > 0.80$)
   - `DRUG_CAFFEINE` (Probe substrate; $f_{m,CYP1A2} > 0.95$)
5. **CYP2D6 Sensitive Clinical Index Substrates:**
   - `DRUG_DEXTROMETHORPHAN` (Index substrate for CYP2D6 activity)
   - `DRUG_BUPROPION`, `DRUG_SERTRALINE`, `DRUG_FLUVOXAMINE` (Antidepressants)
6. **CYP2E1 Clinical Probe Substrates:**
   - `DRUG_CHLORZOXAZONE` (Validated probe substrate for CYP2E1 metabolic clearance)
7. **Transporter / Non-CYP Cleared Drugs:**
   - `DRUG_METFORMIN` (Organic cation transporter OCT2/MATE substrate; 0% CYP clearance)
   - `DRUG_ROSUVASTATIN` (OATP1B1 and BCRP substrate; minimal CYP metabolism)
   - `DRUG_NADOLOL` (OATP1A2 substrate; renal excretion)
   - `DRUG_HYDROCHLOROTHIAZIDE`, `DRUG_LORAZEPAM`, `DRUG_PREDNISOLONE`, `DRUG_ASPIRIN`

---

### 3.3 Suggested Source: Flockhart Cytochrome P450 Drug Interaction Table
**URL:** [https://drug-interactions.medicine.iu.edu/](https://drug-interactions.medicine.iu.edu/)  
**Best For:** Clinical in vivo substrate sensitivity tiers, strong/moderate/weak inhibitor classifications, and in vivo interaction outcomes.

- Use to cross-verify clearance fraction $f_m$ for `DRUG_WARFARIN` (CYP2C9), `DRUG_MIDAZOLAM` (CYP3A4), `DRUG_THEOPHYLLINE` (CYP1A2), and `DRUG_CHLORZOXAZONE` (CYP2E1).

---

### 3.4 Suggested Source: DrugBank Online
**URL:** [https://go.drugbank.com/](https://go.drugbank.com/)  
**Best For:** Standardized drug pharmacology summaries, UniProt target accession numbers, pathway enzymes, and transporter bindings.

- Use for all 40 prescription drugs to record primary clearance enzymes, transporter affinities, and known therapeutic blood concentration ranges ($C_{\text{max}}$).

---

### 3.5 Suggested Source: NPASS (Natural Product Activity and Species Source Database)
**URL:** [https://bidd.group/NPASS/](https://bidd.group/NPASS/)  
**Best For:** Isolated phytochemical constituent affinities and in vitro $K_i$ / $IC_{50}$ values for specific active molecules:
- Hyperforin & Hypericin (from *Hypericum perforatum*)
- Berberine & Hydrastine (from *Hydrastis canadensis*)
- Curcumin, Demethoxycurcumin, Bisdemethoxycurcumin (from *Curcuma longa*)
- Epigallocatechin gallate (EGCG) (from *Camellia sinensis*)
- Silybin A, Silybin B, Silicristin (from *Silybum marianum*)
- Valerenic acid (from *Valeriana officinalis*)
- Ginkgolides A, B, C & Bilobalide (from *Ginkgo biloba*)
- Allicin & Ajoene (from *Allium sativum*)
- Kavalactones (Kawain, Methysticin, Yangonin from *Piper methysticum*)
- Glycyrrhizin & Glycyrrhetinic acid (from *Glycyrrhiza glabra*)
- Bergamottin & 6',7'-dihydroxybergamottin (from *Citrus paradisi*)

---

## 4. Human Curator Instructions

1. Open `data/curation/mechanistic_features.csv` in your spreadsheet editor or Python environment.
2. For each entity-target pair:
   - Locate the peer-reviewed experimental literature citation (e.g. from NaPDI or PubMed) or database record.
   - Enter the measured parameter into `value` (e.g., $IC_{50}$ in $\mu\text{M}$ for inhibition, or fold-induction).
   - Enter the measurement type in `assay_or_unit`.
   - Replace `"UNVERIFIED"` in `source` with the actual database or source name (`NaPDI`, `FDA_DDI_Table`, `DrugBank`, `NPASS`).
   - Replace `"UNVERIFIED"` in `citation` with the exact DOI or PubMed ID (e.g., `10.1016/j.clpt.2003.08.006` or `PUBMED_14631289`).
   - Fill `retrieval_date` with today's date (`YYYY-MM-DD`).
3. Save the file.
4. Run `pytest tests/test_phase3_and_ci_checks.py` to verify schema conformance and check whether all rows have been completed.
