# Dataset sources, actual contents and permitted use

Audit date: 30 October 2026. **No current dataset combination is an eligible
empirical release.** “Official provider,” “published research database,” “valid
schema,” and “independent reference truth” are different properties. None should
be inferred from filenames, registry membership or manifest booleans.

## 1. Independent consensus is missing, not panel-approved

`data/benchmark/consensus_labels.csv` has zero data rows. It is a locally created
schema for future adjudication, not an official clinical-panel dataset. There is
no evidence that a clinical panel has supplied any decisions. Calling it a gold
standard describes its intended role, not its current state. Keeping it empty
prevents invented targets, but is a limitation rather than demonstrated scientific
validation. Storing labels separately does not establish independence: references
must also avoid circular reuse of model-input studies.

I did not identify and verify a downloadable official consensus dataset matching
this project's 120 pairs, outcome definition and provenance requirements. That
is not a claim no such resource could exist. Authoritative clinical summaries
were retrieved instead, with no automatic label conversion:

- [NIH NCCIH: Herb–Drug Interactions—What the Science Says](https://www.nccih.nih.gov/health/providers/digest/herb-drug-interactions-science).
- [NIH NCI: Cancer Therapy Interactions With Foods and Dietary Supplements, PDQ](https://www.cancer.gov/about-cancer/treatment/cam/hp/dietary-interactions-pdq).

Both official HTML snapshots are downloaded at
`/private/tmp/hdi_official_reference_guidance_20261030/`. Exact URLs, timestamps
and hashes are in `source_audits/official_guidance_20261030.json`. These are
clinical-reference pages, not a complete labeled benchmark, and cannot be
translated into blanket negative labels. Their underlying citations may overlap
model evidence. NCI's scope is cancer treatment, not every herb–drug pair.

Task A needs independent consensus as its supervised outcome. Task B intends to
use external-consensus agreement for study reliability and pair-level supervision.
Neither can currently train empirically with this empty table.

## 2. Complete scoped inventory

The machine-readable `source_audits/dataset_inventory_20261030.json` records row
counts and hashes for the explicit tracked data tables below. Private annotation
workspaces were excluded entirely. Counts are engineering inventory, not verified
sample sizes or independent observations.

| Local input | Contents and origin status | Where/how used |
| --- | --- | --- |
| `data/benchmark/verified_botanical_hdi.csv` | 141 local evidence rows; direct PubMed audit finds extensive PMID/DOI inconsistencies. Not verified despite its name. | Intended study input for Task B and read-only explorer; study findings must not become reference labels or Task A features. Historical results inherit provenance limitations. |
| `data/curation/mechanistic_features.csv` | 722 locally generated placeholders; all measurements empty and provenance UNVERIFIED. | Intended mechanistic Task A inputs and Task B prior. Current curation guard blocks use. |
| `data/benchmark/consensus_labels.csv` | Header only; zero adjudicated labels. | Intended independent outcomes for both tasks; blocks supervision now. |
| `consensus_labels_pairs_to_label.csv` | 120 candidate pairs, all consensus decisions blank. | Annotation assignment only. |
| `consensus_labels_pairs_shuffled.csv` | 120 identifier-only assignments. | Blinding/order preparation, never predictive inputs or labels. |
| `consensus_labels_random_unstudied.csv` | 100 candidate pairs, all consensus decisions blank. | Annotation candidates; unstudied does not mean no interaction. |
| `data/external/napdi_template.csv` | Five locally supplied example records; all five DOI/PMID pairs mismatch current PubMed metadata. No authenticated NaPDI export provenance. | External-format template only until citations and records are checked. |
| `data/audit/dag_audit_samples.csv` | 146 derived audit rows. | Citation/lineage diagnostic, not independent new clinical evidence. |
| `data/processed/drugbank_parsed.json` | 6,821 top-level entries; original release, license and extraction chain not authenticated here. | Legacy auxiliary drug data; cannot substitute for herb–drug reference truth. |

The last three assignment CSVs above live under `data/benchmark/`. Their filenames
must not be interpreted as proof of scientific annotation. The historical
`ANNOTATION_PROTOCOL.md` contains heuristic weights; these are not clinical-panel
approvals or scientifically validated reliability probabilities.

The DrugBank parser references a historical local Kaggle cache and writes a dict,
whereas the tracked derivative is a list. That mismatch prevents assuming the
current parser proves this file's extraction history. Do not redistribute or use
it empirically without checking the original source and license.

## 3. Downloaded sources and new preparation artifacts

| Source/artifact | Verified scope | Intended use and restrictions |
| --- | --- | --- |
| [NCBI PubMed API](https://www.ncbi.nlm.nih.gov/books/NBK25499/) | Actual publisher metadata/abstract records downloaded for canonical PMIDs. Some returned papers are unrelated to claimed rows. | Audit and rebuild study evidence; no automatic outcome extraction or consensus. See `source_audits/README.md`. |
| Four-publication public pilot | All four pilot PMID/DOI pairs match current NCBI metadata. Full claim/context review pending. | Small review batch only; three pairs cannot support the full five-role protocol. |
| [EMBL-EBI ChEMBL](https://www.ebi.ac.uk/chembl/) | 361 actual human-target IC50/Ki records downloaded for quercetin, CHEMBL50. | Compound assay review, not all herbs; no mixture mapping, induction/fm inference or Task A authorization. See `source_audits/mechanistic_data_origin.md`. |
| Expanded 16-target panel | 522 new unverified candidate rows plus an input snapshot and coverage report. | Broader curation queue; not 522 measurements. FDA tables support general target relevance only. |
| Discussion-signal schema | Empty template and synthetic tests; zero real social posts collected. | Quarantined review triage, excluded from scientific evidence, consensus and prediction adjustment. |
| NIH NCCIH / NCI downloaded references | Official source pages captured with hashes. | Reference-review aid, not automatic gold labels or blanket clinical conclusions. |

All new review workbooks remain unresolved/pending. Integrity hashes detect changes
to captured bytes, not scientific truth or malicious replacement of a manifest.
Temporary downloads are not a durable archive and licensing/redistribution rights
must be checked before publication. No source has been silently merged into
canonical tables.

## 4. Declared external datasets are not loaded data

`data/manifest.yaml` lists the resources below. Presence of `license_verified: true`
in that manifest is a historical declaration, not a fresh verification. Do not
infer that a resource has been downloaded or its current terms approved.

| Provider and source link | Current audit status | Role and compatibility limits |
| --- | --- | --- |
| [HERB](http://herb.ac.cn/), [primary publication](https://pubmed.ncbi.nlm.nih.gov/33264402/) | Publication identity confirmed. Expected local HERB file checked and absent. Provider downloads/current terms not authenticated. | Proposed herb–compound–target graph; association edges are not assay magnitudes or consensus labels. |
| [Dryad TCM network](https://doi.org/10.5061/dryad.wh70rxwx9) | Provider landing page confirmed. Expected local herb-target file absent. Actual source files include `Herb_target.csv` and `herb_compound.csv`, unlike loader expectations. | Proposed graph topology. Needs an explicit source-preserving adapter; cannot rename files and assume schema compatibility or clinical evidence. |
| [InterPAD provider](https://bddg.hznu.edu.cn/interpad/), [publication](https://www.nature.com/articles/s41598-025-10240-6) | Research dataset/article identified; expected local canonical export absent. | Phytochemical–anticancer-drug interactions, not a direct whole-herb consensus benchmark. Preserve chemical and experimental contexts; publication curation does not prove independence from model evidence. |
| [NaPDI repository](https://repo.napdi.org/) | Public research source identified; direct browser access returned a gateway error. Local five-row template not verified as an export. | Potential natural-product assay/PK data; provider export, experiment identity and extraction require review. |
| [DrugBank](https://go.drugbank.com/) | Licensed provider; local derivative's source chain unverified. | Auxiliary drug data only, with permitted access and release/license provenance required. |
| [IUPHAR/BPS Guide to Pharmacology](https://www.guidetopharmacology.org/download.jsp) | Download page redirected to registration/login. No restricted download attempted. | Potential curated ligand–target data, not a ready-made herb–drug reference set. |

These graph sources are supported by optional/legacy loaders and proposed research
architecture; they are not silently consumed by the current fixed mechanistic
Task A builder. The fixed builder uses only approved named target effects from
curation. Public production execution remains disabled.

## 5. What remains before an official-source empirical release

1. Rebuild evidence from checked original citations and document every correction.
2. Map assay compounds to botanical constituents with formulation/context evidence;
   map drugs and targets separately, retaining source IDs and measurements.
3. Establish independent reference decisions using an agreed clinical endpoint,
   qualified reviewers and explicit adjudication. Abstain on insufficient evidence.
4. Audit shared studies/citation components between inputs and reference sources.
5. Freeze feasible disjoint roles, reviewed feature/missingness semantics, Task B
   contracts and statistically justified evaluation criteria.
6. Pass actual release prerequisites and complete reviewed execution authorization
   before fitting. No performance claim is justified until then.

An official source improves traceability; it cannot turn an incomplete, incorrectly
mapped or circular dataset into a valid benchmark by itself.

The separate five-row template audit is preserved in `source_audits/napdi_template_pubmed_20261030.json`; raw PubMed XML is in `/private/tmp/hdi_napdi_template_pubmed_20261030/`. All five PMIDs returned, but none of their supplied DOI values matched the returned article DOI. Do not treat this template as an authentic NaPDI export.
