# External Validation Cohort Overlap Diagnostics

## 1. Multi-Level Overlap Summary
This diagnostic audits entity and publication overlap between the external cohort (`data/external/napdi_template.csv`) and the source training benchmark (`data/benchmark/verified_botanical_hdi.csv`):

| Level | Overlapping Entities / Identifiers | External Count | Train Split Overlap | Status & Audit Impact |
| :--- | :--- | :---: | :---: | :--- |
| **Pair Level** | `HERB_CURCUMA + DRUG_MIDAZOLAM, HERB_GOLDENSEAL + DRUG_METFORMIN, HERB_GOLDENSEAL + DRUG_MIDAZOLAM, HERB_CURCUMA + DRUG_TALINOLOL` | 5 | 4/5 (80%) | 4 candidate pairs exist in train; 1 pair (`HERB_KRATOM + DRUG_OXYCODONE`) is novel. |
| **Herb Level** | `HERB_CURCUMA, HERB_GOLDENSEAL` | 3 | 2/3 (67%) | Curcuma and Goldenseal exist in train; Kratom is a completely novel botanical entity. |
| **Drug Level** | `DRUG_METFORMIN, DRUG_MIDAZOLAM, DRUG_OXYCODONE, DRUG_TALINOLOL` | 4 | 4/4 (100%) | Midazolam, Metformin, Oxycodone, Talinolol are known substrates. |
| **Study Level** | `None` (0 overlapping study keys) | 5 | **0/5 (0%)** | **100% Disjoint Studies**: Paine 2018, Nguyen 2021, Tanna 2020, Gurley 2003, Paoletti 2007. |
| **Source Level** | `None` (0 overlapping PMIDs/DOIs) | 5 | **0/5 (0%)** | **100% Disjoint PubMed Records**: Zero data leakage or shared primary publications. |

---

## 2. Per-Record Overlap Annotations

| Herb | Drug | External Study Key | External PMID | Pair in Train? | Herb in Train? | Drug in Train? | Study in Train? | Audit Classification |
| :--- | :--- | :--- | :--- | :---: | :---: | :---: | :---: | :--- |
| `HERB_CURCUMA` | `DRUG_MIDAZOLAM` | `STUDY_PAINE_2018` | `PUBMED_29774652` | ⚠️ Yes | ⚠️ Yes | ⚠️ Yes | ❌ Disjoint | Pair exists in training benchmark (distinct clinical study/cohort) |
| `HERB_GOLDENSEAL` | `DRUG_METFORMIN` | `STUDY_NGUYEN_2021` | `PUBMED_32749033` | ⚠️ Yes | ⚠️ Yes | ⚠️ Yes | ❌ Disjoint | Pair exists in training benchmark (distinct clinical study/cohort) |
| `HERB_KRATOM` | `DRUG_OXYCODONE` | `STUDY_TANNA_2020` | `PUBMED_32168541` | ✅ No | ✅ No (Novel) | ⚠️ Yes | ❌ Disjoint | Novel external pair (completely unobserved in training benchmark); De novo botanical entity (unseen herb) |
| `HERB_GOLDENSEAL` | `DRUG_MIDAZOLAM` | `STUDY_GURLEY_2003` | `PUBMED_14631289` | ⚠️ Yes | ⚠️ Yes | ⚠️ Yes | ❌ Disjoint | Pair exists in training benchmark (distinct clinical study/cohort) |
| `HERB_CURCUMA` | `DRUG_TALINOLOL` | `STUDY_PAOLETTI_2007` | `PUBMED_17563541` | ⚠️ Yes | ⚠️ Yes | ⚠️ Yes | ❌ Disjoint | Pair exists in training benchmark (distinct clinical study/cohort) |

---

## 3. Methodological Implications for External Validation
1. **Zero Publication/Study Leakage**: None of the 5 external studies or PubMed citations were ever seen during training or validation. The test evaluates true out-of-sample publication generalization.
2. **Pair Overlap Characterization**: Although 4 candidate pairs share herb-drug entity identifiers with training pairs, they represent distinct clinical cohorts and trials.
3. **Partitioned Reporting**: Performance is reported both for the **Full Cohort ($N=5$)**, the **Strict Pair-Overlap-Excluded Cohort ($N=1$, Kratom+Oxycodone)**, and the **Curcuma-Excluded Cohort ($N=3$)**.
