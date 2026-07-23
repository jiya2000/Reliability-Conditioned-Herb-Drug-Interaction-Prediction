# Status & Progress — HDI Prediction Project

> **Last Updated**: 2026-07-23T23:17:00+05:30
> **Repository**: https://github.com/jiya2000/Reliability-Conditioned-Herb-Drug-Interaction-Prediction

---

## Overall Progress

| Component | Status | Files | Notes |
|-----------|--------|-------|-------|
| 1. Project Skeleton | ✅ Complete | .gitignore, requirements.txt, pyproject.toml, README.md, configs/default.yaml | All foundational files created |
| 2. Data Pipeline | 🔄 In Progress | src/data/*.py | Loaders for DrugBank, ChEMBL, IMPPAT, corpora |
| 3. NLP Extraction | ⏳ Pending | src/extraction/*.py | NER, RE, entity linking, candidate metadata |
| 4. Knowledge Graph | ⏳ Pending | src/graph/*.py | KG construction and utilities |
| 5. Model Architecture | ⏳ Pending | src/models/*.py | GNN, text encoder, R-scorer, cross-attention |
| 6. Training & Eval | ⏳ Pending | src/training/*.py | Trainer, evaluator, ablation runner |
| 7. Explainability | ⏳ Pending | src/explainability/*.py | Evidence surfacing |
| 8. Demo App | ⏳ Pending | app/*.py | FastAPI + Streamlit |
| 9. Tests | ⏳ Pending | tests/*.py | Unit tests for all components |
| 10. Git Pushes | 🔄 Ongoing | — | Frequent commits |

---

## Detailed Log

### 2026-07-23 — Session 1: Project Initialization

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
- Created `STATUS_AND_PROGRESS.md` — This file

**23:17** — Building Component 2: Data Pipeline...

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
| (pending) | Initial project skeleton | .gitignore, requirements.txt, pyproject.toml, README.md, configs/, STATUS_AND_PROGRESS.md |

---

## Key Decisions

1. **NetworkX + PyTorch Geometric** over Neo4j for initial development (simpler setup, no external DB)
2. **PubMedBERT** as primary formal text encoder (strong biomedical performance)
3. **IndicBERTv2** for code-mixed text (best available for Hindi-English health text)
4. **RGCN** as default GNN (well-established for heterogeneous graphs; HGT as upgrade path)
5. **Multiplicative gating** for reliability-conditioned cross-attention (simplest effective approach)
