# Direct NCBI PubMed audit — 30 October 2026

**The historical canonical dataset cannot currently be treated as scientifically
verified.** Its internal registry and descriptive filenames are insufficient.
Direct NCBI retrieval reveals substantive identifier inconsistencies.

## What was downloaded

Using the official NCBI EFetch API, retrieved XML publication records and available
abstracts for the explicit PMID identifiers in the canonical CSV. The input CSV
and complete raw response were snapshotted and hashed. The audit compares each
row's supplied DOI with DOI identifiers in the returned PubMed record. It does
not validate the herb/drug finding or establish study independence.

- 139 distinct PMIDs requested; 131 returned; eight unique PMIDs not returned.
- Nine evidence rows reference those missing PMIDs.
- 130 rows have mismatched or uncomparable DOI metadata.
- One row has a matching DOI; one row is a non-PubMed source requiring other audit.

These are row counts, not causal validation outcomes. A nonreturned record is not
proof the paper never existed. Mismatches need review against both identifiers;
never automatically replace a citation with the paper returned for an incorrect
PMID. In the first row, PMID 10534248 returns a cervical-dystonia botulinum-toxin
paper rather than the claimed herb–cyclosporine study: at least some discrepancies
are substantial scientific mapping errors, not merely DOI formatting.

The four independently collected pilot citations were fetched separately: all
four PMIDs returned and all four DOIs matched. This confirms bibliographic
consistency, not complete scientific extraction, expert verification or reference
labels. The pilot still cannot authorize training.

## Files and reproducing retrieval

`canonical_pubmed_20261030.json` and `pilot_pubmed_20261030.json` preserve audit
findings and exact snapshot hashes. Raw XML and abstracts are local downloads,
not copied into tracked documentation:

- `/private/tmp/hdi_pubmed_official_20261030/`
- `/private/tmp/hdi_pubmed_pilot_official_20261030/`

Temporary downloads may be removed by the operating system. Preserve them in an
appropriate local research archive before relying on long-term availability.
Abstracts may have publisher copyright; retrieving them does not grant unrestricted
redistribution rights. Do not automatically publish raw downloads.

```sh
venv/bin/python -m src.data.pubmed_download \
  --input data/benchmark/verified_botanical_hdi.csv \
  --output /path/to/new_local_pubmed_snapshot
```

The downloader requires a new directory, uses one request for at most 200 distinct
explicit PMIDs, saves raw XML and derived metadata separately, and never generates
interaction labels or production eligibility. Missing, duplicate or unexpected
response identifiers cannot silently become successful matches. A changed NCBI
record may produce a different response hash on a future retrieval.

## Required next steps

Quarantine the old table from scientific use, preserve it historically, resolve
citations one by one against original publications, and re-extract claims with
supporting passages and qualified review. Rebuild source-registry entries from
those checked records rather than trusting matching internal assertions. A larger
trusted herb–drug dataset has not been acquired here: PubMed provides publications,
not an independently labeled herb–drug benchmark. Consensus labels, mechanisms and
all scientific release prerequisites remain unresolved. No historical artifact or
canonical data was rewritten and no model was fitted.
