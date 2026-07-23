# Final Implementation Plan: Reliability-Conditioned Herb-Drug Interaction Prediction

## 0. Core Invention Statement

A method for predicting novel herb-drug interactions via graph neural network link prediction, in which a learned reliability score — computed from source metadata (corroboration count, temporal recency, biomedical quality, molecular plausibility, source type) — dynamically gates the cross-attention weights fusing heterogeneous textual embeddings with molecular graph embeddings, resolving a specific multimodal alignment problem that none of the three closest 2026 prior-art systems solve.

**Three-way differentiation (verified against primary sources):**
- vs. **BELIEF** (arXiv:2605.17435): BELIEF performs Dempster-Shafer reliability-weighted fusion for closed-set biomedical QA. It does not do graph-based link prediction, and its fusion mechanism doesn't extend to open-ended heterogeneous graphs.
- vs. **NeuroGRIP** (arXiv:2607.14314): NeuroGRIP uses reliability scores as a post-hoc filter to prune edges in a homogeneous EEG spatial-temporal graph. It has no mechanism to fuse across modalities (text + molecular graph).
- vs. **DDI-AttendNet** (Frontiers in Pharmacology, Jan 2026): Uses cross-attention, but strictly for intra-domain molecular substructure alignment — not for bridging text and molecular graph modalities.

## 1. Non-Obviousness Argument (for IP cell review)

Naively stitching together the closest prior art fails for a specific technical reason, not just a domain difference:

BELIEF's Dempster-Shafer fusion operates over a finite answer space and does not scale to combinatorial graph link prediction. NeuroGRIP's reliability scoring is a post-hoc filter on an already-homogeneous graph — it presupposes the graph exists and all nodes share one modality; it has no fusion mechanism at all. DDI-AttendNet's cross-attention aligns two representations from the *same* modality (molecular substructures on both sides).

None of the three has a mechanism for the specific problem this system solves: fusing a **fixed-dimension molecular graph embedding** with a **noisy, variable-length, code-mixed textual embedding**, where the relative trust in the textual signal varies per-instance and must be learned rather than assumed. The inventive step is using the learned reliability score $R$ as a **direct, dynamic gate on the cross-attention weights themselves** — not a post-hoc filter (NeuroGRIP), not a voting weight in closed-set fusion (BELIEF), and not an intra-modal alignment signal (DDI-AttendNet).

**Evidentiary support required:** the Week 12 ablation (reliability-conditioned vs. unconditioned cross-attention, same architecture otherwise) is the empirical proof this argument needs. A meaningful AUC-ROC gap here is what turns this from an asserted argument into a demonstrated "unexpected result" — which carries real weight in a non-obviousness determination. Without that gap, this section of the disclosure has no evidentiary backing and should not be filed as-is.

## 2. Architecture

See the diagram above. Two parallel paths converge at the reliability-conditioned cross-attention layer:

**Text path:** formal + informal sources → extraction pipeline (NER + relation extraction) → candidate metadata instances (C, T, B, M, S) → reliability scorer → R

**Graph path:** chemical/target knowledge graphs (DrugBank, ChEMBL, IMPPAT) → heterogeneous GNN encoder → molecular embeddings

**Fusion:** reliability-conditioned cross-attention (R gates the text-to-molecule attention weights) → interaction link prediction with explainable risk score and evidence spans

Note: "candidate metadata instances" replaces the earlier "evidence objects" terminology specifically to avoid vocabulary overlap with BELIEF in the eventual filing — cosmetic, but worth doing since examiners and reviewers do pattern-match on terminology, even though the actual differentiation is mechanistic, not lexical.

## 3. Tech Stack (unchanged from original plan)

| Layer | Tools |
|---|---|
| NER / relation extraction | Hugging Face `transformers`, scispaCy/BioBERT (formal text), IndicBERT/MuRIL (code-mixed text) |
| Entity linking | UMLS Metathesaurus API, fuzzy matching against TKDL/IMPPAT via `rapidfuzz` + embedding similarity |
| Knowledge graph | Neo4j or NetworkX, PyTorch Geometric or DGL |
| GNN + reliability scorer + cross-attention | PyTorch |
| Serving/demo | FastAPI backend, Streamlit or React frontend |
| Experiment tracking | Weights & Biases or MLflow |

## 4. Datasets (unchanged — see prior plan for full access notes)

DrugBank, ChEMBL, PubChem, DDI Corpus, CADEC, PsyTAR, IMPPAT 2.0, UMLS Metathesaurus, self-collected code-mixed health-forum corpus. Apply for UMLS license and IMPPAT bulk access in Week 1 — these remain the only external-approval dependencies on the critical path.

## 5. Month-by-Month Roadmap

### Month 1 — Foundation & Data Pipeline (unchanged from original plan)
- Week 1: UMLS/IMPPAT access requests, repo setup, literature review of BELIEF/NeuroGRIP/DDI-AttendNet/kANNa
- Week 2: Ingest DrugBank/ChEMBL/PubChem into normalized entity tables
- Week 3: Collect and clean DDI Corpus, CADEC, PsyTAR; self-collect pilot code-mixed corpus
- Week 4: Baseline formal-text NER (PubMedBERT fine-tune on DDI Corpus), report F1

### Month 2 — Extraction Layer (unchanged from original plan)
- Week 5: Fine-tune informal-text NER/RE on CADEC/PsyTAR
- Week 6: Adapt IndicBERT/MuRIL for code-mixed text; build small manually-annotated eval set (100-200 sentences — a genuine standalone contribution)
- Week 7: Relation extraction with negation/uncertainty handling
- Week 8: Entity normalization and linking module

### Month 3 — Knowledge Graph & Reliability-Conditioned Fusion
- **Week 9:** Construct the heterogeneous knowledge graph (Neo4j/PyTorch Geometric). Populate using DrugBank/ChEMBL as the ground-truth anchor set.
- **Week 10:** Inject extracted edges from the Month 2 pipeline. Format every candidate as a **candidate metadata instance** containing (C, T, B, M, S).
- **Week 11:** Train the reliability scorer. **Primary approach:** end-to-end training — R has no direct label; it's learned by backpropagating the link-prediction loss on DrugBank anchor edges through the full pipeline (extraction → R → cross-attention → prediction). **Stretch goal (if ahead of schedule):** add explicit weak-supervision labeling functions (e.g., "high corroboration + peer-reviewed source → pseudo-label high reliability") to pretrain R before end-to-end fine-tuning — this strengthens the non-obviousness argument by giving R an interpretable inductive bias rather than being purely a learned gate.
- **Week 12 (critical ablation):** Implement the reliability-conditioned cross-attention layer. Run three variants: (a) GNN-only baseline, (b) unconditioned cross-attention (same architecture, R removed), (c) full reliability-conditioned model. The (b)-vs-(c) gap is the evidentiary core of both the paper and the patent disclosure — do not skip or compress this comparison.

### Month 4 — Explainability, Evaluation, Packaging
- **Week 13:** Explainable output layer — for every predicted edge, surface the R-score breakdown (which metadata dimension drove the trust decision) alongside evidence spans.
- **Week 14:** Full evaluation (precision/recall/AUC/MRR). Error analysis focused specifically on contradictory-evidence cases (e.g., older literature says safe, recent code-mixed forum corroboration says otherwise).
- **Week 15:** Minimal demo interface (query drug + herb → risk score, R-score breakdown, evidence spans). Keep scope minimal — no CDSS integration, no alerting system.
- **Week 16:** Finalize paper draft (target: BioNLP@ACL, AMIA, PSB, or JAMIA) and patent disclosure document for institutional IP cell review.

## 6. Patent Claim Direction

> A computer-implemented method for predicting novel compound interactions via graph neural network link prediction, comprising: extracting interaction candidate data from a plurality of heterogeneous formal and informal text sources; instantiating said candidate data as candidate metadata instances comprising source metadata; training a reliability scorer to generate a dynamic reliability score for each candidate metadata instance based on a link-prediction objective evaluated against a predefined anchor dataset; generating molecular node embeddings via a graph neural network; generating textual embeddings from the candidate data; and computing a novel interaction link probability utilizing a cross-attention layer, wherein attention weights mapping the textual embeddings to the molecular node embeddings are dynamically scaled by the generated reliability score prior to link prediction, to resolve a multimodal embedding misalignment between fixed-dimension molecular representations and variable-length heterogeneous textual representations.

**Note on claim language:** "based on a link-prediction objective evaluated against a predefined anchor dataset" replaces the earlier "predicting agreement" phrasing to accurately reflect the end-to-end training methodology in Week 11, rather than implying a direct-label weak-supervision scheme that isn't actually being built (unless the Week 11 stretch goal is completed, in which case the claim should be updated to describe the labeling-function pretraining explicitly). **Patent claims should describe what the system actually does, not what sounds more novel** — a claim that overstates the training methodology is a liability if the built system doesn't match it.

## 7. Key Risks & Mitigations (updated)

| Risk | Mitigation |
|---|---|
| UMLS/IMPPAT access delays | Apply week 1; DrugBank+ChEMBL as fallback ontology backbone |
| Code-mixed text is low-resource | Scope to 1-2 languages (e.g., Hindi-English); treat small annotated eval set as a standalone contribution |
| Reliability-conditioned ablation shows no meaningful gap | This is the single biggest risk to the patent argument — if Week 12 shows R-gating doesn't beat unconditioned attention, the non-obviousness argument in Section 1 has no evidentiary support. Budget extra time here; consider it the highest-priority week in the whole project |
| No domain expert available for validation | DrugBank known interactions as silent ground truth; one informal pharmacology review session significantly strengthens credibility |
| Scope creep into full CDSS product | Keep Week 15 demo to query-in, risk-score-out; no EHR integration, no alerting |
| Prior-art landscape continues to shift (this is an active 2026 research area) | Re-run the literature search in Week 16 before finalizing the disclosure — a fourth close-neighbor paper could appear before you file |

## 8. What "done" looks like at Week 16

- A trained reliability-conditioned model with a demonstrated, statistically meaningful improvement over an unconditioned cross-attention baseline (the evidentiary core of the whole strategy)
- A small but genuine standalone contribution: the annotated code-mixed health-text dataset
- A working end-to-end demo
- A paper draft with the three-way prior-art differentiation clearly stated
- A patent disclosure whose claim language accurately reflects the built system, with the non-obviousness argument backed by the Week 12 ablation result
