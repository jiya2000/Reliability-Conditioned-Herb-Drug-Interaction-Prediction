# Official ChEMBL assay dataset

This is a separate real-data collection, not a modification of the historical
herb–drug tables. It supports compound–target assay research. It supplies neither
clinical interaction labels nor drug fraction-metabolized measurements.

## Source and reproducibility

Provider: [EMBL-EBI ChEMBL](https://www.ebi.ac.uk/chembl/).
[API documentation](https://chembl.gitbook.io/chembl-interface-documentation/web-services/chembl-data-web-services).

```sh
PYTHONDONTWRITEBYTECODE=1 venv/bin/python -m src.data.chembl_dataset \
  --output data/external/chembl_assays_20261030
```

The output must be new. This command resolves six human SINGLE PROTEIN targets
using exact UniProt accessions: CYP3A4/P08684, CYP2D6/P10635, CYP2C9/P11712,
CYP2C19/P33261, CYP1A2/P05177 and ABCB1/P08183. Ambiguous mappings fail.

The queries request human-target IC50 measurements, exact relations, nM units,
non-null pChEMBL values and source assay-confidence score 9. This is a declared
restricted assay dataset, not all measurements or a representative sample of
herbs. API filter strings and resolved target IDs are retained in the manifest.
Assay-confidence 9 describes target assignment, not clinical evidence quality.

All pages are fetched and counts checked; no first-page sample is passed off as
complete. Raw pages include available compound structures and assay descriptions.
Publication records are separately fetched by document ID. API status is captured
before and after retrieval, and source pages plus CSVs receive SHA-256 hashes.
Only the official HTTPS API is accepted for pagination. Request failures retry
three times; unresolved failures leave an incomplete directory without a final
manifest. Such a directory must not be treated as a valid dataset. Choose a new
output directory to retry; automatic resume is not currently implemented.

## Output files

- `raw/response_*.json`: exact source responses, including target and publication
  metadata. These have not been replaced by summaries or AI-generated values.
- `assay_records.csv`: all downloaded assay records with source quantities,
  compound structures, document DOI/PMID/title and explicit review flags.
- `assay_research_candidates.csv`: records passing conservative structural checks;
  these are research candidates, not approved clinical or primary Task A inputs.
- `manifest.json`: final completion marker, source version/status, filters,
  per-target counts, unique compound/document counts and integrity hashes.

Structural screening flags source validity comments, potential duplicates,
unsupported/nonfinite quantities, missing structure or document links and reported
mutant assays. Exact eligible nM values are converted to micromolar values;
original units, values and relations are retained. Do not convert an inequality
into an exact measurement. IC50 depends on assay conditions and is not automatically
interchangeable with Ki or clinical exposure.

## Integration boundaries

The current `chembl_assay_review` adapter is reused. Compound identities stay
ChEMBL identities. No herb IDs, induction values, substrate flags, clinical severity,
negative interaction labels or consensus judgments are inferred. Repeated activity
records are retained for explicit downstream aggregation and source-disjoint split
design, never described as independent patients. Both Task A and Task B eligibility
remain false in the final manifest.

This dataset can be the input to a separately implemented compound-assay evaluation.
It is not yet connected to a trained assay predictor or the original production
Task A/B execution path. Clinical herb–drug prediction still lacks reference labels
and verified whole-herb/drug mappings. Published assay data allow retrospective
software research without new laboratory experiments; they do not establish
clinical performance by themselves.

At the user's explicit request, this source snapshot and both CSVs are included in
the repository using explicitly selected paths despite the general ignored-data
policy. The audit summary under `docs/source_audits/` documents the retrieval.
Source attribution: EMBL-EBI ChEMBL, ChEMBL_37; exact API requests, capture dates
and original records are preserved in the manifest and raw responses. Consult
[ChEMBL's terms](https://www.ebi.ac.uk/chembl/) for applicable attribution and reuse
conditions. Inclusion here does not constitute scientific or clinical approval.

## Completed download

The completed local source is ChEMBL_37. The six declared queries returned
**18,833 records**, **10,554 distinct ChEMBL compounds** and **2,036 referenced
source documents**. The structural candidate table contains **16,018 rows**.

| Target | Downloaded IC50 records |
| --- | ---: |
| CYP3A4 | 6,642 |
| CYP2D6 | 3,349 |
| CYP2C9 | 3,107 |
| CYP2C19 | 1,737 |
| CYP1A2 | 1,764 |
| ABCB1 | 2,234 |

Actual source-to-CSV identity alignment, raw/output hashes, per-target counts,
finite positive converted quantities, structure presence in screened candidates
and continued exclusion from production all passed validation. The audit report
is `source_audits/chembl_assay_dataset_20261030.json`. Counts describe filtered
source records, not verified independent clinical evidence or model performance.

To verify local captured data without network access:

```sh
venv/bin/python - <<'PY'
from src.data.chembl_dataset import verify_dataset
report = verify_dataset('data/external/chembl_assays_20261030')
print(report['records'], report['structurally_usable_records'])
PY
```
