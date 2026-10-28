# Primary Task A feature contract

This audit follows `PROJECT_SPEC.md` sections 2, 3 and 4.1. There is no complete primary consensus-supervised trainer in the checkout. The legacy neural benchmarks and `scripts/run_phase5_task_a.py` are exploratory diagnostic paths; their names do not establish primary-model validity. This phase adds a reusable feature/preprocessing boundary and prevents the curated switch from falling through to legacy fitting. It does not add a model or complete scientific evaluation.

## Approved model inputs

The explicit immutable `TASK_A_FEATURES` tuple in `src/features/task_a_contract.py` is the primary allowlist. It deliberately uses only the six pathways named in the specification's pair-feature equation. All seven current features are classified as **mechanistic**:

| Ordered model column | Raw measurement and derivation | Classification |
|---|---|---|
| `signed_effect_CYP3A4` | Herb CYP3A4 modulation polarity × drug CYP3A4 fraction/substrate weight | Mechanistic |
| `signed_effect_CYP2C9` | Herb CYP2C9 modulation polarity × drug CYP2C9 fraction/substrate weight | Mechanistic |
| `signed_effect_CYP2C19` | Herb CYP2C19 modulation polarity × drug CYP2C19 fraction/substrate weight | Mechanistic |
| `signed_effect_CYP2D6` | Herb CYP2D6 modulation polarity × drug CYP2D6 fraction/substrate weight | Mechanistic |
| `signed_effect_CYP1A2` | Herb CYP1A2 modulation polarity × drug CYP1A2 fraction/substrate weight | Mechanistic |
| `signed_effect_ABCB1` | Herb P-glycoprotein modulation polarity × drug fraction/substrate weight | Mechanistic |
| `sum_abs_signed_effect` | Sum of absolute values of the six columns above, recomputed after pathway selection | Mechanistic |

“Approved” here means structurally admitted by the feature policy, not that the actual measurements have been scientifically verified. Signs describe **target activity**, not clinical exposure direction. Positive IC50/Ki establishes inhibition −1; fold induction above one establishes +1, exactly one establishes zero. This reuses the existing extractor; no potency-strength or clinical exposure mapping is invented. Drug `value_fm` takes precedence; binary `value_substrate_flag` is used only as a fallback, never described as an estimated fraction. Explicit tested nulls are zero. Unknowns stay NaN, including the sum if any selected pathway is unknown.

The general mechanistic extractor remains unchanged: it emits nine target effects plus a nine-target absolute sum. The primary builder selects the six targets above by their declared names/order and **recomputes** the sum, rather than accidentally admitting CYP2E1, SLCO1B1/OATP1B1 or NR1I2 through the aggregate. Those three are mechanistic but outside the current primary allowlist. OATP1B1 is mentioned in the spec's knowledge table; extending its primary feature equation requires an explicit later contract decision. Max-absolute effect, NTI, tissue reachability, dose/exposure and other future sourced PK features are not yet produced by this approved builder and cannot enter by adding columns. No existing numerical schema for the general extractor changed.

## Raw-to-training trace

1. **Pair keys:** a pair table supplies `herb_id`, `drug_id`. `build_task_a_features` first projects exactly those two keys; study records, text and caller-provided feature columns never reach the extractor. IDs only choose rows in the mechanistic curation table. Consistently renaming join keys with identical measurements produces identical X. IDs, source/study keys and DataFrame index values are never converted to feature columns.
2. **Measurement table:** `load_mechanistic_curation`/`is_curation_verified` validate roles, statuses, ranges, duplicate rows and source/citation presence. `extract_curated_feature_matrix` repeats the provenance guard and computes per-target polarity × weight. This is the repaired existing extraction boundary. Its raw `entity` is an identity lookup key; `target` selects the fixed mechanistic pathway, not an entity encoding. Free-text `notes`, source/citation strings, abstract text and outcomes never define feature values. No legacy dictionary fallback exists in the curated builder.
3. **Named primary X:** the builder selects the six effects, recomputes the absolute sum and returns a numeric DataFrame with precisely `TASK_A_FEATURES`. Even identically named columns supplied on the raw pair/study table are ignored in favor of reconstructed measurements. Missing values are preserved for inspection. Nonunique pair-table column names are rejected to prevent ambiguous joins/projection.
4. **Preprocessing:** `task_a_training_pipeline(existing_lr_or_hgb)` first applies `TaskAFeatureContract`, requiring exactly the seven named, real numeric columns. Extras are rejected, not silently dropped. Duplicate names, missing names, positional/sparse arrays, booleans, complex values, numeric strings, infinities and NaNs are rejected for fitting. Unknown aliases, one-hot columns and embeddings fail regardless of whether already numeric or scaled. There is no dtype-based feature discovery, categorical/text encoder, remainder passthrough or automatic missingness indicator.
5. **Scaling:** the fixed `StandardScaler` fits only the partition passed to pipeline `.fit`; it retains the column names in pandas output. No transform on assessment data refits it. Scaling changes numeric units but does not introduce columns. No imputation or missingness handling was added.
6. **Training boundary:** a second independent contract check runs after preprocessing, immediately before the existing LR/HGB estimator. It rejects appended columns, changed names or unnamed matrices at that boundary. Tests instrument the actual estimator `.fit` to verify exactly seven named columns and no forbidden input. Prediction uses the same checks. This is a pipeline around the existing model classes, not a new model family.
7. **Existing runner safeguard:** `FeatureSwitchedBenchmark(feature_set='new').get_feature_matrix` now uses this builder and returns named X. `run_cv(new)` retains the real-data provenance error first and, even if measurements become verified, explicitly blocks before reading publication `outcome` or fitting the legacy LR/HGB/GAT loop. A future independent-consensus trainer must use the guarded pipeline; none is silently substituted here. `run_cv(legacy)` remains the pre-existing exploratory benchmark and is not a primary route.

## Excluded feature classification

The primary allowlist rejects **all** other columns, including seemingly benign study-design/lineage metadata. The following inventory covers the inputs in the existing candidate/legacy paths rather than treating quality-only metadata as mechanistic:

| Raw or derived feature family | Classification | Task A disposition |
|---|---|---|
| `herb_id`, `drug_id`, entity names, one-hot IDs, categorical encodings, hashes, ID-seeded vectors | Identity | Lookup/grouping only; never X |
| `study_id`, `source_id`, `cited_study_ids`, provenance keys | Lineage identifiers | Joins/audit only; never X |
| `species`, `evidence_type`/assay, `route`, `preparation`, `population`, reported study `dose`, sample size, extract standardization, treatment duration | Study-design | Excluded; study-dose metadata is not a sourced mechanistic exposure feature |
| `mean_source_quality`, `extraction_confidence`, `context_compatibility_score`, `metadata_missingness` | Study-design/quality | Excluded, even if useful to Task B reliability |
| `citation_echo_ratio`, total records, independent citation cluster size, primary/echo status | Lineage | Excluded; structural quality is not mechanism |
| `outcome`, `study_finding`, `raw_label`, `effect_direction`, `clinical_significance`, binary publication target | Outcome/text-derived | Excluded; never a Task A target fallback |
| `raw_positive_count`, `raw_negative_count`, raw conflicting/unknown counts, independent positive/negative cluster counts, `S_pos`, `S_neg`, net/raw evidence margin | Outcome/text-derived | Excluded, including encoded/scaled aliases |
| `contradiction_strength`, `undiscounted_contradiction_strength`, `C_hd`, polarity-partitioned lineage summaries | Outcome/text-derived | Excluded; “quality” naming does not make findings mechanistic |
| Raw abstracts, TF-IDF/BiomedBERT embeddings, masked text and clinical proxy text features | Outcome/text-derived (often also identity) | Excluded even after masking or projection |
| Regex flags such as human trial, randomization, control, blinding, dose presence, oral/IV route, sample-size presence | Study-design, text-derived where extracted from abstracts | Excluded; no design flag belongs in primary Task A X |
| Legacy entity profiles, graph embeddings, overlap scores and profile-density/curation-count features | Mechanistic-looking, identity-specific and label-adjacent legacy inputs | Excluded from curated primary X; cannot substitute for cited measurements |
| Existing lineage `rho_reach`, `f_m_CYP` defaults; heuristic dictionary PK values | Mechanistic-looking legacy/default fields | Excluded; matching a biological name is insufficient provenance |
| Curation IC50/Ki, fold induction, drug fraction/substrate flag and measured/null/unknown status | Mechanistic | Consumed only by the trusted builder to create the seven allowed outputs |
| Curation assay/unit and exposure metadata | Study-design / mechanistic context | Validation/context only; no new PK conversion |
| Curation source, citation, retrieval date | Lineage/provenance | Guard/audit only; never encoded as predictors |
| Curation notes and extraction method | Text-derived / study-design provenance | Not interpreted as predictors |
| Independent consensus label, severity/direction and provenance | Outcome target / external metadata | Separate target/audit table; never a predictor column |

The exploratory script's `X_overlap_plus_curation`, `X_overlap_plus_sanitized_design`, full/sanitized flags, identity matrices, raw/masked/design text and legacy LR/GBT/GAT outputs therefore do not become primary models by matching a Task A filename. `benchmarks/run_verified_hdi_benchmark.py` and nested legacy evaluation consume raw text/entity profiles; neither is called by the approved builder or pipeline. This audit does not change those diagnostics or their saved artifacts.

## Limits

Column contracts prevent accidental column admission, not deliberate scientific falsification. Renaming an outcome/identity vector to an approved measurement name, mutating a pipeline to remove its guards, or directly calling an estimator outside the supported pipeline cannot be authenticated by a feature-name check. The supported raw route reconstructs X from the provenance-checked curation table; source independence and measurement correctness still require blinded human verification. The fixed scaler is trusted to perform its declared transform; tests inject contaminants at its public output boundary to verify the following contract, rather than interpreting unnamed internal scaler arrays as independently traceable features.

All actual curation rows remain unverified and actual consensus labels remain empty. The scalar/provenance checks are not source authentication. No missingness policy, NTI or reachability feature, full consensus trainer, new CV/evaluation or primary scientific score is implemented by this phase. Historical leakage results and the >0.98 guard are preserved unchanged.
