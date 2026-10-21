# Reliability-Conditioned Herb-Drug Interaction Prediction

> **A patent-worthy method for predicting herb-drug interactions via reliability-conditioned graph neural networks, citation-derivation DAGs with echo attenuation, pharmacokinetic context compatibility, and calibrated tri-state abstention.**

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.1+-ee4c2c.svg)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

## 📖 Invention Overview

This project implements a **computer-implemented system and method for reliability-conditioned prediction of botanical–pharmaceutical interactions** using:

1. **Citation-Derivation DAG with Echo Chamber Attenuation** — Constructs directed acyclic graphs from literature evidence, identifies primary empirical studies, and applies depth-discounted attenuation (`alpha = 0.15`) to prevent circular citation inflation.

2. **Pharmacokinetic Context Compatibility** — Multi-factor scoring weighting biological species, botanical extract standardization, administration route, and in vitro concentration reachability per FDA/EMA guidelines.

3. **Neural Reliability Gating** — Dynamically balances biological graph embeddings against contextual literature embeddings via a continuous reliability coefficient `R ∈ [0, 1]` learned end-to-end.

4. **Temperature-Scaled Calibration** — Validation-set temperature scaling for calibrated posterior probabilities.

5. **Tri-State Clinical Decision Engine** — Full 3-condition abstention rule (epistemic uncertainty, scientific contradiction, low reliability with borderline confidence) routing to automated EHR alerts, safe co-administration clearance, or asynchronous clinical pharmacist review.

This integration of all five elements into a single HDI prediction framework has **no precedent** in peer-reviewed literature, preprint servers, or publicly searchable patent databases as of October 2026.

## 🆕 Novelty Components (New)

| Component | File | Key Innovation |
|---|---|---|
| **10D Evidence Lineage Vector** | `src/models/evidence_lineage.py` | Exact patent-specification vector: `[S_pos, S_neg, N_primary, N_echo, κ̄, C_h,d, q̄, Δ_time, N_raw_pos, N_raw_neg]` |
| **Temporal Decay Weighting** | `src/models/evidence_lineage.py` | Exponential recency decay with learnable half-life; `Delta_time ∈ [0,1]` |
| **PK Reachability Screening** | `src/models/pk_reachability.py` | FDA/EMA-grounded IVIVE: compares in vitro potency vs. human unbound Cmax; penalty 0.25/1.0 |
| **Differentiable Reachability Wrapper** | `src/models/pk_reachability.py` | Makes penalty factors end-to-end trainable while keeping core screen deterministic |
| **Full 3-Condition Tri-State Decision** | `src/models/patent_novelty_system.py` | Patent Eq.8: ABSTAIN if epistemic>0.25 OR (contradiction>0.5 AND margin<0.35) OR (reliability<0.1 AND margin<0.10) |

## 🏗️ Architecture

Three parallel paths converge at the **reliability-conditioned cross-attention layer**:

1. **Text Path**: Formal & informal text sources → NER + relation extraction → candidate metadata → reliability scorer → R
2. **Graph Path**: Chemical/target knowledge graphs (DrugBank, ChEMBL, IMPPAT) → heterogeneous GNN encoder → molecular embeddings  
3. **Evidence Lineage Path**: Literature evidence records → citation-derivation DAG → echo-attenuated strengths → 10D lineage vector → context compatibility → PK reachability penalty → reliability gating

The reliability score **R** acts as a **dynamic gate on cross-attention weights**, resolving the multimodal alignment problem between fixed-dimension molecular graph embeddings and noisy, variable-length textual embeddings.

Three additional specialized modules enhance clinical safety:

- **PK Reachability Screen**: Grounded in FDA/EMA in vitro-to-in vivo extrapolation; flags in vitro findings as clinically reachable (penalty 1.0) or unreachable (penalty 0.25, w_conc per Claim 15)
- **10D Evidence Lineage**: Structured metadata replacing ad-hoc feature engineering; all 10 components auditable from source records
- **Full Tri-State Decision**: Three clinically safe outcomes: `INTERACTION_PRESENT`, `NO_INTERACTION`, `ABSTAIN_CLINICAL_REVIEW` (routed to pharmacist review queue)

**Three-way differentiation from 2026 prior art:**
- vs. **BELIEF**: Dempster-Shafer fusion for closed-set QA — no graph link prediction
- vs. **NeuroGRIP**: Post-hoc reliability filter on homogeneous graphs — no cross-modal fusion  
- vs. **DDI-AttendNet**: Intra-modal molecular substructure attention — no text-molecule bridging vs. citation-derivation DAGs + PK reachability + tri-state abstention

## 🆕 Model Listings (Updated)

```
src/models/
├── gnn_encoder.py           # Heterogeneous GNN (RGCN/HGT)
├── text_encoder.py          # Biomedical text encoder (BioBERT)
├── reliability_scorer.py    # R-score from 5D metadata + 8D evidence lineage
├── cross_attention.py       # Reliability-conditioned cross-attention ★
├── link_predictor.py        # Link prediction head
├── hdi_model.py             # Full end-to-end model (3 ablation modes)
├── citation_dag.py          # Citation-derivation DAG + echo attenuation
├── context_compatibility.py # PK context compatibility scorer
├── conformal_abstention.py  # Split conformal tri-state engine
├── evidence_lineage.py      # **NEW**: 10D lineage vector + temporal decay
├── pk_reachability.py       # **NEW**: PK reachability screening + differentiable wrapper
└── patent_novelty_system.py # **NEW**: Unified orchestrator + 3-condition decision
```

## 🧪 Test Coverage

- **17 new unit tests** across `tests/test_evidence_lineage.py` and related tests
- All 3 abstention conditions verified: epistemic uncertainty >0.25, contradiction >0.5 with low confidence, reliability <0.1 with borderline confidence
- Clear positive/negative prediction states verified
- All existing modules remain backward-compatible

## 🚀 Setup

```bash
git clone https://github.com/jiya2000/Reliability-Conditioned-Herb-Drug-Interaction-Prediction.git
cd Reliability-Conditioned-Herb-Drug-Interaction-Prediction

python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

pip install -r requirements.txt
pip install -e .
```

## 📄 Configuration

All parameters are centralized in `configs/default.yaml`. Key sections:
- **data**: Paths to raw/processed datasets
- **extraction**: NLP model choices and hyperparameters
- **gnn**: Graph neural network architecture
- **reliability**: Metadata dimension sizes for the R-scorer
- **cross_attention**: Core fusion layer configuration
- **training**: Learning rate, epochs, negative sampling

## 📦 Usage

### Training
```bash
python -m src.training.trainer --config configs/default.yaml
```

### Ablation Study
```bash
python -m src.training.ablation --config configs/default.yaml
```

### Demo Application
```bash
# Start FastAPI backend
uvicorn app.backend:app --host 0.0.0.0 --port 8000

# Start Streamlit frontend (in another terminal)
streamlit run app/frontend.py
```

## 📊 Datasets

| Dataset | Purpose | Access |
|---------|---------|--------|
| DrugBank | Drug-drug interactions, drug metadata | Academic license |
| ChEMBL | Bioactivity data, molecular structures | Open access |
| IMPPAT 2.0 | Indian medicinal plant phytochemicals | Request access |
| DDI Corpus | Annotated DDI sentences | Open access |
| CADEC | Consumer adverse drug events | Open access |
| PsyTAR | Psychiatric drug reviews | Open access |
| UMLS | Biomedical concept normalization | NLM license |

## 🛠️ Tech Stack

| Layer | Tools |
|-------|-------|
| NER / RE | HuggingFace Transformers, scispaCy, BioBERT, IndicBERT/MuRIL |
| Entity Linking | UMLS API, rapidfuzz, embedding similarity |
| Knowledge Graph | NetworkX, PyTorch Geometric |
| GNN + Fusion | PyTorch, PyTorch Geometric |
| Serving | FastAPI, Streamlit |
| Tracking | Weights & Biases |

## 📜 License

MIT License — see [LICENSE](LICENSE) for details.