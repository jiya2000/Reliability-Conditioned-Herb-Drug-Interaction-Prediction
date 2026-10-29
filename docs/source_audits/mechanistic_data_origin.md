# Mechanistic table origin and official assay download

## Correction to previous descriptions

`data/curation/mechanistic_features.csv` is **not a dataset of 722 laboratory
measurements**. Inspection found 722 rows, 58 entities and 14 distinct target
identifiers; every status is `not_tested`, every assay-value field is empty and
all source/citation fields say `UNVERIFIED`. Some targets belong to the additional
pharmacodynamic block rather than the nine-target general PK extractor.

Git history shows the table originated as an entity-target curation grid in
`c85753f` and was redesigned/expanded in `6ab6542`. Its notes explicitly await
human curation from NaPDI, NPASS or primary literature. Mentioning those resources
in placeholder notes does not mean measurements were downloaded from them.
The claim that this table already powers an empirically validated biological
baseline is incorrect. The pipeline blocks it, as intended.

## Downloaded authoritative source records

Retrieved data directly from the official [EMBL-EBI ChEMBL API](https://www.ebi.ac.uk/chembl/api/data/docs).
First resolved the exact preferred name QUERCETIN to CHEMBL50, then fetched all
human-target IC50/Ki records returned by that compound-specific query. The API
returned 361 records and no next page. This is complete for that query, not a
complete herb–drug or metabolic-target dataset. Many returned targets are outside
the current Task A allowlist.

[Repeat the official activity query](https://www.ebi.ac.uk/chembl/api/data/activity.json?molecule_chembl_id=CHEMBL50&target_organism=Homo%20sapiens&standard_type__in=IC50%2CKi&limit=1000).
The [ChEMBL service documentation](https://chembl.gitbook.io/chembl-interface-documentation/web-services/chembl-data-web-services)
describes its API. Records preserve activity, molecule, target, assay and source
document identifiers. These are real database-reported assay records, not values
invented to fill our table. They still require assay/document review, target and
entity mapping and interpretation. No claim of clinical reliability follows.

Local downloads:

- `/private/tmp/hdi_chembl_official_20261030/molecule_search.json`
- `/private/tmp/hdi_chembl_official_20261030/activities_page_0.json`
- `/private/tmp/hdi_chembl_official_20261030/assay_review.csv`
- `/private/tmp/hdi_chembl_official_20261030/download_manifest.json`

`chembl_quercetin_20261030.json` in this directory records the exact query,
timestamp, counts and hashes. Temporary local storage is not a permanent archive.
The raw records are kept separately from canonical data and are not automatically
committed or redistributed.

## Review adapter and integration boundaries

`src/data/chembl_assay_review.py` converts captured activity pages into a pending
review format. It preserves original values, units and inequality relations.
Only positive finite exact IC50/Ki values in nM/uM for human targets, without
flagged validity/duplication or missing provenance, receive an exact micromolar
conversion. This leaves 270 rows as exact quantitative review candidates; the
remaining 91 stay flagged. None is approved for Task A. Even an unflagged record
is not proof the assay is appropriate, selective or free of interference.

The adapter does not map quercetin to an entire herb, interpret a mixture as a
purified compound, aggregate conflicting experiments, infer induction from
inhibition, or manufacture substrate fractions. Assays involving a human target
are not necessarily human clinical studies. Whole-herb composition, formulation,
concentration/exposure, metabolites and drug-specific clearance evidence remain
needed. IC50, Ki and fold induction are distinct quantities.

The curated IUPHAR/BPS download page was also inspected, but currently redirects
to registration/login. No login bypass or restricted download was attempted.
ChEMBL provides an accessible authoritative source, not a ready-made replacement
for this project's missing measurements and independent consensus labels.
