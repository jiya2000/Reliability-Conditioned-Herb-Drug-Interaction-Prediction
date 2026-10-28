# Curated mechanistic feature boundary

The existing 15-column contract is documented in `reports/mechanistic_curation_todo.md`. There is no generic `value` column. Extraction uses these actual columns without interpreting free-text notes or borrowing legacy profile values.

| Existing column/status | Feature meaning |
|---|---|
| Herb `value_ic50_uM` or `value_ki_uM` | A positive finite inhibition measurement establishes target-activity polarity **−1**. Potency magnitude is retained in the source table; it is not converted into an invented clinical effect size. Both constants may coexist and are counted once. |
| Herb `value_fold_induction` | Finite fold above one establishes polarity **+1**; exactly one is neutral **0**. A fold below one is invalid under the existing induction schema. Concurrent inhibition and induction above one is ambiguous and rejected. |
| Drug `value_fm` | Clearance fraction in [0, 1], used as the target weight. |
| Drug `value_substrate_flag` | Binary 0/1 fallback weight only when the clearance fraction is absent; it does not estimate a clearance fraction. |
| `tested_no_effect` | Explicit measured null, represented by zero. Mechanistic numeric measurements must be empty for this status. |
| `not_tested`, absent row, absent relevant measurement | Unknown, represented by `NaN`, never assumed neutral. A known zero factor still establishes a zero pair contribution. |
| `exposure_conc_uM`, `exposure_source`, `assay_or_unit` | Retained metadata, not converted to reachability: the schema does not identify free concentration or tissue compartment. Concentration, when supplied, must be finite and nonnegative. |
| `source`, `citation`, remaining metadata | Provenance/curation context; not predictive values. Source and citation must pass the existing guard. No online verification is implied by nonempty metadata. |

For each target, the pair feature is herb target-activity polarity multiplied by the drug weight. Signs describe **target activity**, not direction of clinical drug exposure: uptake/efflux transport, receptor biology, and active metabolites prevent a universal exposure-sign interpretation. This direction-only boundary is not the completed Task A clinical prior. It intentionally does not invent potency scaling, NTI, compartment reachability, or unspecified numeric mappings.

Output remains float32 with ten columns: the nine entries of `src.data.mechanistic_curation.CANONICAL_TARGETS`, in that exact order, followed by the sum of their absolute values. The sum remains `NaN` if any component is unknown. Downstream training must explicitly address missingness; this phase does not silently impute or change consumers. Empty pair input returns shape `(0, 10)`.

Validation rejects missing schema columns, duplicate entity-target rows, malformed/nonfinite/out-of-range numbers, unknown statuses, wrong entity roles, and mechanistic measurements on non-measured rows. Duplicate experiments require an explicit future curation policy rather than last-row-wins behavior. Extra target rows remain in the table; only the declared canonical targets enter the output.

Unverified tables remain blocked, including when legacy callers pass `allow_unverified=True`; the argument remains solely for call compatibility. Synthetic test records are explicitly marked `SYNTHETIC_FIXTURE` and never written into real curation data or prediction artifacts. Scientific provenance still requires human verification. Consensus labels and verified mechanistic citations remain outstanding.
