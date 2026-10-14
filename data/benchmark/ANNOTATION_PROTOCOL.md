# Botanical Herb–Drug Interaction (HDI) Annotation Protocol

## 1. Objective and Scope
This document specifies the scientific curation, taxonomy, evidentiary criteria, and verification protocol used to construct the benchmark dataset in `verified_botanical_hdi.csv`. 

Unlike naive binary interaction datasets that conflate pharmacokinetic decreases with lack of interaction, this protocol enforces:
1. **Decoupling Interaction Presence from Pharmacokinetic Direction**: Both exposure increases (e.g., clearance inhibition) and exposure decreases (e.g., CYP3A4/P-gp induction) represent genuine pharmacokinetic interactions ($Y=1$).
2. **Identification of True Negative Controls**: Rigorously performed clinical trials demonstrating lack of alteration in drug area under the curve (AUC), clearance, or pharmacodynamic endpoints represent true negative controls ($Y=0$).
3. **Explicit Handling of Scientific Contradiction**: Divergence between observational case reports and randomized controlled trials (or botanical species variations) is retained and evaluated rather than smoothed over or discarded.
4. **Mandatory DOI/PMID Provenance**: Every record maps to a primary published paper indexed in PubMed or CrossRef. Synthetic placeholders are strictly prohibited.

---

## 2. Scientific Label Taxonomy

### 2.1 Primary Binary Interaction Target ($Y \in \{0, 1\}$)
- **`interaction_present` ($Y=1$)**: Concomitant administration produces a statistically and clinically meaningful alteration in drug exposure (AUC, $C_{\max}$), clearance, target receptor sensitivity, or toxicity profile.
- **`no_interaction_observed` ($Y=0$)**: Dedicated clinical investigation (pharmacokinetic trial, controlled cohort) evaluated the combination under therapeutic doses and reported no statistically or clinically significant alteration in drug disposition or pharmacodynamics.
- **`conflicting` (Abstain / Evaluated for Uncertainty)**: Independent studies report opposing findings (e.g., case report suggests toxicity, RCT demonstrates safety) or route-dependent opposing effects.
- **`unknown` (Excluded from Supervised Loss)**: Insufficient or unverified evidence.

### 2.2 Pharmacological Effect Direction (`effect_direction`)
- **`decreased_exposure`**: Botanical constituent induces metabolic enzymes (e.g., CYP3A4, CYP1A2, CYP2C9) or efflux transporters (P-gp/ABCB1), or inhibits uptake transporters (OATP1A2, OATP1B1), resulting in reduced systemic drug concentration. *Clinical risk*: Subtherapeutic drug levels, loss of efficacy, organ transplant rejection, HIV treatment failure.
- **`increased_exposure`**: Botanical constituent inhibits metabolic enzymes or efflux transporters, elevating systemic drug concentrations. *Clinical risk*: Exaggerated drug response, dose-dependent adverse reactions, toxicity.
- **`increased_toxicity`**: Pharmacodynamic synergy or additive target toxicity (e.g., additive platelet aggregation inhibition with warfarin/aspirin leading to spontaneous hemorrhage).
- **`decreased_effect`**: Pharmacodynamic antagonism (reduction in target receptor response or functional efficacy, such as reduced warfarin INR).
- **`no_effect`**: Rigorous confirmation of absence of pharmacokinetic or pharmacodynamic alteration.
- **`contradictory`**: Divergent mechanisms reported across studies.

### 2.3 Clinical Significance (`clinical_significance`)
- **`major`**: High risk of life-threatening outcome, organ transplant rejection, therapeutic failure in critical illness, or severe hemorrhage. Concomitant use contraindicated or requires critical therapeutic drug monitoring.
- **`moderate`**: Measurable pharmacokinetic alteration typically requiring dosage adjustment, close clinical monitoring, or patient counseling.
- **`minor`**: Statistically detectable change without significant clinical consequence.
- **`none`**: Confirmed lack of interaction.

---

## 3. Evidentiary Hierarchy

Evidence sources are tiered and weighted according to standard clinical pharmacology principles:
1. **Meta-Analysis / Systematic Review of RCTs** (`meta_analysis`, weight: 1.00)
2. **Randomized Controlled Trial** (`rct`, weight: 0.95)
3. **Controlled Clinical Pharmacokinetic Study** (`clinical`, weight: 0.90)
4. **Well-Documented Human Case Series / Report with Rechallenge or Validated Assay** (`case_report`, weight: 0.65)
5. **In Vivo Mammalian Study** (`animal`, weight: 0.40)
6. **In Vitro Transporter / CYP Assay** (`in_vitro`, weight: 0.20)
7. **Inconclusive / Equivocal Report** (`inconclusive`, weight: 0.10)

---

## 4. Citation Lineage & Secondary Echo Discounting

When multiple publications describe the same underlying interaction:
- **Primary Clinical Discovery**: Assigned full evidentiary weight.
- **Derivative / Repeating Reports**: Secondary publications that cite the primary discovery without generating novel primary patient data are discounted via the citation echo factor:
  $$\text{Strength} = Q_{\text{primary}} + \sum_{k=1}^m \alpha_{\text{echo}} \cdot Q_{\text{secondary}, k}$$
  where $\alpha_{\text{echo}} = 0.15$.
- **Cluster Identification**: Evaluated using directed citation graphs (citing study $\to$ cited study).

---

## 5. Study-Disjoint Evaluation Protocol

To prevent evaluation leakage:
- All partitions (train, validation, test) are formed such that **no citation cluster is shared** between any splits.
- If Study B cites Study A, both Study A and Study B must reside in the same partition.
- Test metrics reflect performance on completely unseen clinical investigation clusters.
