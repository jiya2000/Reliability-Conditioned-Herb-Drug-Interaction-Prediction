# Status & Progress — HDI Prediction Project

> **Last Updated**: 2026-07-23T23:40:00+05:30
> **Repository**: https://github.com/jiya2000/Reliability-Conditioned-Herb-Drug-Interaction-Prediction

---

## Overall Progress

| Component | Status | Files | Notes |
|-----------|--------|-------|-------|
| 1. Project Skeleton | ✅ Complete | .gitignore, requirements.txt, pyproject.toml, README.md, configs/default.yaml | All foundational files created |
| 2. Data Pipeline | ✅ Complete | src/data/*.py | Loaders for DrugBank, ChEMBL, IMPPAT, corpora |
| 3. NLP Extraction | ✅ Complete | src/extraction/*.py | NER, RE, entity linking, candidate metadata |
| 4. Knowledge Graph | ✅ Complete | src/graph/*.py | KG construction and utilities |
| 5. Model Architecture | ✅ Complete | src/models/*.py | GNN, text encoder, R-scorer, cross-attention |
| 6. Training & Eval | ✅ Complete | src/training/*.py | Trainer, evaluator, ablation runner |
| 7. Explainability | ✅ Complete | src/explainability/*.py | Evidence surfacing |
| 8. Demo App | ✅ Complete | app/*.py | FastAPI + Streamlit |
| 9. Tests | ✅ Complete | tests/*.py | Unit tests for all components |
| 10. CLI Entrypoint | ✅ Complete | main.py | `main.py` entrypoint |

---

## Detailed Log

### 2026-07-23 — Session 1: Project Initialization & Core Development

**23:15** — Started implementation based on approved plan
- Reviewed `HDI_Final_Implementation_Plan (1).md` and architecture diagram
- Initialized git repository
- Added remote: `https://github.com/jiya2000/Reliability-Conditioned-Herb-Drug-Interaction-Prediction`

**23:17** — Component 1: Project Skeleton ✅
- Created `.gitignore` — Python, data, models, IDE files excluded
- Created `requirements.txt` — 40+ dependencies (PyTorch, PyG, Transformers, scispaCy, etc.)
- Created `pyproject.toml` — Project metadata, setuptools config, pytest config
- Created `configs/default.yaml` — Full configuration for all 10+ modules
- Created `README.md` — Architecture overview, project structure, setup, usage

**23:25** — Components 2 & 3: Data Pipeline & NLP Extraction ✅
- Built `drugbank_loader.py`, `chembl_loader.py`, `imppat_loader.py`, `corpus_loader.py`, `dataset.py`
- Built `ner_model.py`, `relation_extractor.py`, `entity_linker.py`, `candidate_metadata.py`

**23:30** — Components 4 & 5: Knowledge Graph & Model Architecture ✅
- Implemented `kg_builder.py` integrating data sources into PyG HeteroData
- Built `gnn_encoder.py`, `text_encoder.py`, `reliability_scorer.py`, `link_predictor.py`
- Implemented the core invention: **`cross_attention.py`** with reliability gating.
- Built full `hdi_model.py` with ablation modes.

**23:38** — Components 6-10: Training, UI, Tests, Entrypoint ✅
- Built `trainer.py`, `evaluator.py`, `ablation.py`
- Implemented `EvidenceSurfacer` for interpretability.
- Created `app/backend.py` (FastAPI) and `app/frontend.py` (Streamlit).
- Wrote unit tests in `tests/`.
- Created `main.py` CLI runner.

---

## Architecture Reference

```
Formal & informal text ──→ Extraction Pipeline ──→ Candidate Metadata ──→ Reliability Scorer ──→ R
                                                                                                   ↘
Chemical & target KGs ──→ Heterogeneous GNN Encoder ──→ Molecular Embeddings ──────────────────→ R-Gated Cross-Attention ──→ Link Prediction
```

---

## Git Commit History

| Commit | Message | Files Changed |
|--------|---------|---------------|
| `d0f5c93` | Initial project skeleton & Data pipeline | skeleton files, src/data/*.py |
| `9f90248` | feat: NLP extraction + KG builder + core model | src/extraction/*, src/graph/*, src/models/* |
| (pending) | feat: Training, UI, tests, entrypoint | src/training/*, src/explainability/*, app/*, tests/*, main.py, STATUS_AND_PROGRESS.md |

---

## Key Decisions

1. **NetworkX + PyTorch Geometric** over Neo4j for initial development (simpler setup, no external DB)
2. **PubMedBERT** as primary formal text encoder (strong biomedical performance)
3. **IndicBERTv2** for code-mixed text (best available for Hindi-English health text)
4. **RGCN** as default GNN (well-established for heterogeneous graphs; HGT as upgrade path)
5. **Multiplicative gating** for reliability-conditioned cross-attention (simplest effective approach)
