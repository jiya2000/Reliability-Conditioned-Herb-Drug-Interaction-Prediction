# Review-1: Progress Report

## Topic
**Reliability-Conditioned Prediction of Herb-Drug Interactions using Heterogeneous Graph Neural Networks**
The objective of this project is to build an AI-powered system that predicts potential adverse interactions between traditional herbs and pharmaceutical drugs. Unlike standard prediction models, our system computes a "Reliability Score" for its predictions by evaluating the quality, recency, and corroboration of the underlying evidence in the knowledge graph.

## Data
We are utilizing the global gold-standard pharmaceutical database for this project:
*   **Primary Dataset:** DrugBank XML Database (downloaded via Kaggle API `sergeguillemart/drugbank`).
*   **Scale:** The raw dataset is 1.5GB in size.
*   **Entities Extracted:** Successfully parsed over 5,000 distinct drugs/compounds and their structured metadata.
*   **Edges/Interactions:** Extracted over 450,000 ground-truth interaction pairs along with textual clinical descriptions (evidence spans).
*   **Phase 2 Expansion (Planned):** For the final research paper, the graph will be fused with traditional medicine datasets (IMPPAT and SymMap) to explicitly map Ayurvedic and Chinese herbal structures.

## Finalise Flow Pipeline and GANTT Chart

### System Flow Pipeline
1.  **Data Ingestion:** Large XML corpus downloaded from Kaggle is parsed iteratively.
2.  **Graph Construction:** Entities (Drugs/Herbs) are mapped to indices, and interactions are mapped to edge pairs to construct a PyTorch Geometric `HeteroData` graph.
3.  **Cross-Attention GNN:** The `HeterogeneousGNNEncoder` learns node embeddings by passing messages across the graph, utilizing a multi-head cross-attention mechanism to weigh evidence reliability (corroboration, temporal recency, biomedical quality).
4.  **Prediction:** The `LinkPredictor` outputs a probability of interaction and an explainable risk breakdown.
5.  **User Interface:** A Streamlit-based web dashboard queries a FastAPI backend to surface predictions and real clinical evidence directly from the parsed graph.

### GANTT Chart / Timeline
*   **Week 1-2:** Literature review, architectural design, and environment setup.
*   **Week 3-4 (Current):** Data acquisition (Kaggle DrugBank), iterative XML parsing, and data preprocessing into Graph JSON structures.
*   **Week 5-6:** Implementation of the PyTorch Geometric Heterogeneous GNN and Link Predictor. Model ablation studies.
*   **Week 7-8:** Integration of IMPPAT/SymMap for the herbal expansion. Model fine-tuning.
*   **Week 9-10:** Frontend dashboard finalization, evaluation metrics, and research paper drafting.

## Any Other Progress (Till Data Preprocessing)
*   **XML Parsing Engine:** Built a memory-efficient `xml.etree.ElementTree.iterparse` script that processes the 1.5GB DrugBank file without crashing the system memory.
*   **Graph Mapping:** Successfully converted the raw XML nodes into a structured `drugbank_parsed.json` file.
*   **Web Dashboard:** Developed and deployed a "pro-level" Glassmorphism UI (FastAPI + Streamlit) that currently queries the real preprocessed Kaggle data to demonstrate end-to-end connectivity.
*   **Dependency Management:** Migrated the entire environment to `uv` for lightning-fast, deterministic package resolution.
