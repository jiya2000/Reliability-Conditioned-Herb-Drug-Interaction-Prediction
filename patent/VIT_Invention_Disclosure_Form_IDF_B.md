# VELLORE INSTITUTE OF TECHNOLOGY (VIT)
## IPR & TECHNOLOGY TRANSFER CELL (IPR & TT CELL)
### INVENTION DISCLOSURE FORMAT (IDF) - B
**Document No.:** 02-IPR-R003  
**Issue No./Date:** 2 / 01.02.2024  
**Amendment No./Date:** 0 / 00.00.0000  

---

### 1. Title of the invention:
**COMPUTER-IMPLEMENTED SYSTEM AND METHOD FOR RELIABILITY-CONDITIONED PREDICTION OF BOTANICAL–PHARMACEUTICAL INTERACTIONS USING CITATION-DERIVATION DIRECTED ACYCLIC GRAPHS, PHARMACOKINETIC CONTEXT COMPATIBILITY, AND CALIBRATED TRI-STATE ABSTENTION**

---

### 2. Field / Area of invention:
- **Primary Domain:** Computational Bioinformatics, Medical Informatics, and Artificial Intelligence (Machine Learning / Deep Learning).
- **Specialized Sub-fields:** Graph Neural Networks (Heterogeneous GNN / GAT), Biomedical Natural Language Processing (BioNLP), Pharmacokinetics and Drug Metabolism, Clinical Decision Support Systems (CDSS).
- **International Patent Classification (IPC) Classes:**
  - **G16B 15/00:** Data processing for computational molecular biology and chemical informatics.
  - **G16H 50/20:** Computer-assisted clinical decision support and risk prediction.
  - **G06N 3/08:** Artificial intelligence and neural network learning methods.
  - **G06N 5/04:** Inference and reasoning systems under uncertainty.

---

### 3. Prior Patents and Publications from literature (provide a table summarizing the prior art):

| Prior Art / Publication / Patent | Mechanism / Technical Approach | Evidentiary Model Used | Citation Lineage & Echo Discounting? | Pharmacokinetic Context Compatibility? | Contradiction-Aware Clinical Abstention? |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **DeepDDI** (Ryu et al., *PNAS* 2018; 115(18):E4304) | Deep feedforward neural network on chemical structure fingerprints (PCA-transformed) | Static single-molecule drug pairs; assumes literature labels are noise-free ground truth | **Absent:** Treats cited review statements as independent empirical confirmations | **Absent:** Does not model metabolic CYP enzymes, transporters, or in vitro vs. in vivo gap | **Absent:** Forces binary 0/1 prediction; no safety abstention mechanism |
| **KGNN** (Wang et al., *WWW* 2019; pp. 315–325) | Knowledge Graph Convolution with relation-aware receptive fields | Homogeneous graph triples from DrugBank/KG | **Absent:** Homogeneous graph topology without source publication provenance | **Absent:** Does not model biological species, formulation, or dose | **Absent:** Forced classification on all inputs |
| **CompGCN** (Vashishth et al., *ICLR* 2020) | Multi-relational GCN with composition operators ($\mathbf{e}_v \circ \mathbf{r}$) | Knowledge graph relation embedding triples | **Absent:** Relational embedding assumes static, unconditionally true edges | **Absent:** No botanical extract standardisation or route weighting | **Absent:** Forced binary classification |
| **Patent CN115223731A/B** (Zhejiang Univ.; "DS evidence theory-based drug interaction prediction method"; Priority: 15 Jul 2022; Publ (A): 21 Oct 2022; Grant (B): 13 Jan 2026) | Dempster–Shafer evidence theory combined with Dirichlet distribution uncertainty | Unweighted multi-source data merging | **Absent:** Treats evidence sources as conditionally independent; susceptible to echo cascades | **Absent:** Lacks physiological species translation weighting | **Absent:** Closed-world binary output without clinical EHR queue routing |
| **US Patent US10,818,382B2** (IBM, 2020) | NLP extraction of drug interactions from clinical records | PubMed sentence-level co-occurrence frequency | **Absent:** Raw occurrence counting; vulnerable to derivative citation cascades | **Absent:** Text syntax only; no pharmacokinetic overlap or enzyme modeling | **Absent:** Binary thresholding without selective abstention |
| **HTINet2** (Chen et al., *Bioinformatics* 2024; 40(2):btae064) | Herb-Target-Disease bipartite network with Bayesian Personalized Ranking | Target connection counts and statistical associations | **Absent:** Static association filtering without citation lineage modeling | **Absent:** Ignores oral bioavailability, first-pass metabolism, and extract quality | **Absent:** Lacks epistemic uncertainty estimation and abstention |
| **RGEF** (IEEE TNNLS 2024; 35(4):4890) | Reliability-Gated Early Fusion architecture | Sensory time-series data (ECG/EEG signals) | **Absent:** Designed for sensor signal noise; lacks bibliographic DAG modeling | **Absent:** Domain-incompatible; not applicable to pharmacology or ADME | **Absent:** No clinical EHR review queue routing |
| **Present Invention (VIT Disclosure)** | **Multimodal Heterogeneous GAT + BiomedBERT + Neural Reliability Scorer** | **Audited 4-Tier Source Registry (100% verified against PubMed DOIs/PMIDs)** | **Present:** Citation DAG connected components with empirical echo discount factor ($\alpha = 0.15$) | **Present:** Multiplicative composite score ($w_{\text{species}} \cdot w_{\text{prep}} \cdot w_{\text{route}}$) | **Present:** Calibrated Tri-State decision engine supporting selective clinical abstention |

---

### 4. Summary and background of the invention (Address the gap / Novelty):

#### A. Background of the Problem:
Between 20% and 30% of prescription drug patients concurrently consume botanical dietary supplements and herbal preparations. Because botanicals are complex mixtures containing dozens of active phytoconstituents (e.g., hyperforin in *Hypericum perforatum*, ginsenosides in *Panax ginseng*), they frequently cause dangerous Herb–Drug Interactions (HDIs) by modulating Cytochrome P450 (CYP450) metabolic enzymes and drug transporters (such as P-glycoprotein/ABCB1).

#### B. The Unaddressed Technical Gaps in Existing Systems:
1. **The Circular Citation Echo Chamber Trap:** Existing AI systems treat every scientific article as an independent empirical observation. In botanical medicine, a single anecdotal case report from the late 1990s is routinely cited across 20+ review papers over two decades. Naive machine learning models count these as 20 "independent confirmations", resulting in severe confirmation bias, circular literature memorization, and artificial overconfidence.
2. **Experimental Context Incompatibility:** Many published herb interaction claims originate from high-concentration in vitro microsomal assays. In living humans, these phytochemicals undergo rapid intestinal deglycosylation, hepatic first-pass glucuronidation, and poor oral absorption, rendering the in vitro interaction clinically non-existent. Existing models treat in vitro binding data and human clinical trials identically.
3. **Forced Binary Hallucinations on Contradictory Literature:** Literature on botanicals frequently exhibits irreconcilable scientific contradictions due to non-standardized extracts (e.g., Digoxin + Ginseng conflict). Existing AI models force a binary output ($p \ge 0.50$ vs. $p < 0.50$), generating false alarms that induce clinician "alert fatigue" or dangerous false reassurances.

#### C. Novelty and Core Inventive Step:
The present invention solves these technical failures by providing an integrated, three-pronged technical architecture:
1. **Citation-Derivation Directed Acyclic Graph (DAG) with Echo Attenuation:** Dissects literature dependencies into weakly connected components, assigns an index primary study, and applies an attenuation discount factor ($\alpha = 0.15$) to secondary citations, while maintaining genuinely independent clinical investigations as separate roots.
2. **Physiological Context Compatibility Weighting:** Weights each evidentiary finding by biological species (human trial: $1.0$, rodent model: $0.65$, in vitro: $0.30$), botanical extract standardisation ($1.0$ for standardized, $0.8$ for crude), and delivery route ($1.0$ for oral, $0.75$ for parenteral).
3. **Contradiction-Aware Reliability Gating ($R \in [0, 1]$) with Tri-State Abstention:** When literature is sparse or contradictory, a gating network smoothly attenuates text reliance and defaults to biological Graph Attention Network (GAT) metabolic priors. Instead of guessing on high-entropy contradictions, it triggers a third state—**selective clinical abstention**—routing the query to an asynchronous hospital EHR review queue.

---

### 5. Objective(s) of Invention:
1. To develop an automated computer-implemented citation lineage tracking system that models scientific evidence as a directed acyclic graph to computationally attenuate repetitive citation-bias and literature echo chambers.
2. To formulate a multi-parameter pharmacokinetic context compatibility metric that filters out non-physiological in vitro artifacts and extract variations.
3. To construct a 2-layer Heterogeneous Biological Graph Attention Network (GAT) encoding human CYP enzymes (CYP3A4, CYP2C9, CYP2C19, CYP2D6, CYP1A2) and transporters (ABCB1/P-gp, OATP1B1) as an inductive mechanistic prior.
4. To implement a continuous scalar reliability gating module $R \in [0, 1]$ that dynamically balances biological graph embeddings with textual language model representations.
5. To implement post-hoc probability calibration using a calibration parameter estimated on a validation dataset, and to evaluate its effect on Brier score and probability calibration on an independently held-out test set.
6. To deploy a tri-state clinical safety decision engine that automatically abstains when epistemic predictive uncertainty or literature contradiction exceeds clinical thresholds, avoiding alert fatigue while routing unresolved cases to human specialists.

---

### 6. Working principle of the invention (in brief):

The system operates across six sequential computational phases:
1. **Query & Entity Parsing:** Receives a candidate pair comprising a botanical entity $h$ and a pharmaceutical drug $d$.
2. **Biological Mechanism Prior Encoding:** Queries an indexed biomedical knowledge graph connecting herbs, phytochemicals, drugs, and human metabolic enzymes/transporters. A 2-layer GAT outputs a continuous mechanistic embedding vector $\mathbf{z}_{\text{graph}}$.
3. **Evidence Lineage & Context Extraction:** Collects all literature evidence records from an audited registry, builds a citation DAG, clusters derivative reviews, and applies an 85% discount ($\alpha = 0.15$) to derivative echoes. Concurrently computes a multi-factor pharmacokinetic context compatibility score ($W_{\text{context}} = w_{\text{species}} \cdot w_{\text{prep}} \cdot w_{\text{route}}$) and a contradiction index $C_{h,d}$.
4. **Dynamic Reliability Gating:** A trained neural reliability module computes scalar reliability $R \in [0, 1]$. The system dynamically fuses the biological GAT embedding $\mathbf{z}_{\text{graph}}$ and the BiomedBERT text embedding $\mathbf{z}_{\text{contextual}}$ via:
   $$\mathbf{z}_{\text{fused}} = (1 - R)\mathbf{z}_{\text{graph}} + R\mathbf{z}_{\text{contextual}}$$
5. **Temperature-Scaled Logit Calibration:** Link prediction logits are calibrated using an optimal validation temperature parameter $T^* > 0$: $\hat{p} = \sigma(z / T^*)$.
6. **Tri-State Clinical Safety Execution:** If epistemic variance (via Monte Carlo dropout) or the contradiction index exceeds pre-set clinical bounds, the system issues an **ABSTAIN** control signal, routing the order to an asynchronous clinical review queue with an advisory flag. Otherwise, it issues an **INTERACTION PRESENT** prescription interlock or **NO INTERACTION** clearance.

---

### 7. Description of the invention in detail (Include drawing and or photograph as needed):

#### A. System Architectural Diagram:

![Figure 1: Architectural Workflow of the Reliability-Conditioned Botanical–Pharmaceutical Interaction Prediction System](reports/figures/patent_system_architecture_diagram.png)

```mermaid
flowchart TD
    %% Root Input Query
    Q["<b>100: CANDIDATE QUERY</b><br/>(Herb h, Drug d)"]

    %% Biological Mechanism Branch
    subgraph BIO ["102: BIOLOGICAL MECHANISM BRANCH"]
        B1["<b>104: Tripartite Knowledge Graph</b><br/>• CYP450 Enzymes (3A4, 2D6, 2C9, etc.)<br/>• Membrane Transporters (P-gp/ABCB1)"]
        B2["<b>106: 2-Layer Heterogeneous GAT</b><br/>Multi-relational message passing"]
        B1 --> B2
    end

    %% Evidentiary Lineage Branch
    subgraph LIT ["108: EVIDENTIARY LINEAGE BRANCH"]
        L1["<b>110: Audited Publication Source Registry</b><br/>100% Verified PubMed DOIs/PMIDs"]
        L2["<b>112: Citation Derivation DAG Reconstruction</b><br/>Echo Chamber Attenuation (alpha = 0.15)"]
        L3["<b>114: Context Compatibility Scoring</b><br/>Species, Extract Standardisation, Route (Sp/Pr/Rt)"]
        L4["<b>116: BiomedBERT Contextual Text Encoder</b><br/>Domain-specific text embeddings"]
        L1 --> L2
        L2 --> L3
        L3 --> L4
    end

    %% Neural Scorer
    SCORER["<b>118: NEURAL RELIABILITY SCORER</b><br/>Inputs: Lineage & Contradiction Stats<br/>Output: Scalar R in [0, 1]"]

    %% Fusion Module
    FUSION["<b>120: RELIABILITY-GATED MULTIMODAL FUSION</b><br/>z_fused = (1 - R) * z_graph + R * z_contextual"]

    %% Calibration Module
    CALIB["<b>122: TEMPERATURE-SCALED POST-HOC CALIBRATION</b><br/>(T* = 1.48, Validation Fit)<br/>Calibrated Probability p = sigmoid(z / T*)"]

    %% Tri-State Decision Engine
    DECISION{"<b>124: CALIBRATED TRI-STATE DECISION ENGINE</b><br/>Epistemic u > &tau;<sub>u</sub> OR<br/>(Conflict C > &tau;<sub>c</sub> AND |p - 0.5| < &tau;<sub>m</sub>)?"}

    STATE_ABSTAIN["<b>126: STATE 3: ABSTAIN</b>"]
    QUEUE["<b>128: Route to Clinical Pharmacist</b><br/>Asynchronous Review Queue (Advisory Flag)"]

    THRESHOLD{"<b>130: Probability Threshold</b><br/>Is Calibrated Probability >= 0.50?"}

    STATE_POS["<b>132: STATE 1: INTERACTION</b><br/>(Automated Prescription Interlock)"]
    STATE_NEG["<b>134: STATE 2: NO INTERACTION</b><br/>(Safe Co-administration Clearance)"]

    %% Connections
    Q --> BIO
    Q --> LIT

    B2 -->|z_graph| FUSION
    L3 -->|Lineage Metadata & Contradiction Index| SCORER
    L4 -->|z_contextual| FUSION
    SCORER -->|Reliability Gate R| FUSION

    FUSION --> CALIB
    CALIB --> DECISION

    DECISION -->|YES| STATE_ABSTAIN
    STATE_ABSTAIN --> QUEUE

    DECISION -->|NO| THRESHOLD
    THRESHOLD -->|YES| STATE_POS
    THRESHOLD -->|NO| STATE_NEG

    %% Styling
    classDef queryClass fill:#1e3a8a,stroke:#3b82f6,stroke-width:2px,color:#ffffff;
    classDef bioClass fill:#0f172a,stroke:#38bdf8,stroke-width:1.5px,color:#e2e8f0;
    classDef litClass fill:#1e1b4b,stroke:#a855f7,stroke-width:1.5px,color:#e2e8f0;
    classDef scorerClass fill:#451a03,stroke:#f59e0b,stroke-width:2px,color:#ffffff;
    classDef fusionClass fill:#064e3b,stroke:#10b981,stroke-width:2px,color:#ffffff;
    classDef calibClass fill:#134e4a,stroke:#14b8a6,stroke-width:2px,color:#ffffff;
    classDef decisionClass fill:#312e81,stroke:#6366f1,stroke-width:2px,color:#ffffff;
    classDef abstainClass fill:#7f1d1d,stroke:#ef4444,stroke-width:2px,color:#ffffff;
    classDef posClass fill:#831843,stroke:#f43f5e,stroke-width:2px,color:#ffffff;
    classDef negClass fill:#14532d,stroke:#22c55e,stroke-width:2px,color:#ffffff;

    class Q queryClass;
    class B1,B2 bioClass;
    class L1,L2,L3,L4 litClass;
    class SCORER scorerClass;
    class FUSION fusionClass;
    class CALIB calibClass;
    class DECISION,THRESHOLD decisionClass;
    class STATE_ABSTAIN,QUEUE abstainClass;
    class STATE_POS posClass;
    class STATE_NEG negClass;
```

```
┌────────────────────────────────────────────────────────────────────────────────────────┐
│                        CANDIDATE QUERY: (Herb h, Drug d)                               │
└───────────────────┬────────────────────────────────────────────────┬───────────────────┘
                    │                                                │
                    ▼                                                ▼
┌───────────────────────────────────────┐  ┌────────────────────────────────────────────┐
│      BIOLOGICAL MECHANISM BRANCH      │  │          EVIDENTIARY LINEAGE BRANCH         │
│  • Tripartite Knowledge Graph         │  │  • Audited Publication Source Registry      │
│  • CYP450 Enzymes (3A4, 2D6, 2C9, etc)│  │  • Citation Derivation DAG Reconstruction   │
│  • Membrane Transporters (P-gp/ABCB1) │  │  • Echo Chamber Attenuation (alpha = 0.15) │
│  • 2-Layer Heterogeneous GAT Encoder  │  │  • Context Compatibility Scoring (Sp/Pr/Rt) │
│                                       │  │  • BiomedBERT Contextual Text Encoder       │
└───────────────────┬───────────────────┘  └─────────────────────┬──────────────────────┘
                    │ z_graph                                    │ z_text + Lineage Stats
                    │                                            │
                    │                                            ▼
                    │                              ┌────────────────────────────────────┐
                    │                              │     NEURAL RELIABILITY SCORER      │
                    │                              │  • Inputs: Lineage & Contradiction │
                    │                              │  • Output: Scalar R in [0, 1]      │
                    │                              └─────────────────┬──────────────────┘
                    │                                                │ R
                    ▼                                                ▼
┌────────────────────────────────────────────────────────────────────────────────────────┐
│                        RELIABILITY-GATED MULTIMODAL FUSION                             │
│                  z_fused = (1 - R) * z_graph  +  R * z_contextual                      │
└───────────────────────────────────┬────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌────────────────────────────────────────────────────────────────────────────────────────┐
│                  TEMPERATURE-SCALED POST-HOC CALIBRATION (T* = 1.48)                   │
│                          Calibrated Prob p = sigmoid(z / T*)                           │
└───────────────────────────────────┬────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌────────────────────────────────────────────────────────────────────────────────────────┐
│                     CALIBRATED TRI-STATE CLINICAL DECISION ENGINE                      │
│                                                                                        │
│   [Condition: Epistemic u > tau_u OR (Conflict C > tau_c AND |p - 0.5| < tau_m)?]     │
│         │                                                          │                   │
│        YES                                                        NO                   │
│         ▼                                                          ▼                   │
│   ┌───────────────┐                             ┌───────────────────────────────────┐  │
│   │ STATE 3:      │                             │ Is Calibrated Probability >= 0.5? │  │
│   │ ABSTAIN       │                             └─────────┬───────────────┬─────────┘  │
│   └───────┬───────┘                                      YES              NO           │
│           │                                               ▼               ▼            │
│           ▼                                       ┌───────────────┐ ┌───────────────┐  │
│  Route to Clinical Pharmacist                     │ STATE 1:      │ │ STATE 2:      │  │
│  Review Queue (Advisory Flag)                     │ INTERACTION   │ │ NO INTERACTION│  │
│                                                   │ (Interlock)   │ │ (Safe)        │  │
│                                                   └───────────────┘ └───────────────┘  │
└────────────────────────────────────────────────────────────────────────────────────────┘
```

#### B. Mathematical Equations Governing the Invention:

1. **Citation Echo Attenuation Equation:**
   For a citation-connected evidence cluster C_k possessing an index primary study s_primary and M - 1 derivative citations s_j, the cluster evidence strength S(C_k) is computed as:
   - *Plain-text specification:* `S(C_k) = q(s_primary) + Σ [ α · q(s_j) ]` for all j ≠ primary in C_k.
   - *Formal mathematical expression:*
     $$S(\mathcal{C}_k) = q(s_{\text{primary}}^{(k)}) + \sum_{j \in \mathcal{C}_k, j \neq \text{primary}} \alpha \cdot q(s_j)$$
   where α = 0.15 is the empirical citation echo attenuation discount factor (with α ∈ [0.05, 0.25] in parameterized implementations), and q(s) ∈ [0, 1] represents extraction quality.

2. **Pharmacokinetic Context Compatibility Metric:**
   For each publication record s, its context compatibility multiplier W_context(s) is formulated as:
   - *Plain-text specification:* `W_context(s) = w_species(s) * w_prep(s) * w_route(s)`
   - *Formal mathematical expression:*
     $$W_{\text{context}}(s) = w_{\text{species}}(s) \times w_{\text{prep}}(s) \times w_{\text{route}}(s)$$
   Parameter definitions:
   - Biological species: Human in vivo clinical trials: w_species = 1.00; In vivo non-human animal models: 0.65; In vitro microsomal assays: 0.30.
   - Botanical formulation: Standardized extract: w_prep = 1.00; Crude plant powder: 0.80; Isolated aglycone monomer: 0.65.
   - Administration route: Oral ingestion: w_route = 1.00; Parenteral injection: 0.75.

3. **Scientific Contradiction Index (C_{h,d}):**
   Measures scientific conflict between positive interaction claims and negative safety reports:
   - *Plain-text specification:* `C_{h,d} = [ 2 * min(S_pos, S_neg) ] / [ S_pos + S_neg + ε ]`
   - *Formal mathematical expression:*
     $$C_{h,d} = \frac{2 \cdot \min(S_{\text{pos}}, S_{\text{neg}})}{S_{\text{pos}} + S_{\text{neg}} + \epsilon}$$
   where S_pos and S_neg are aggregated positive and negative echo-discounted cluster strengths, and ε = 10⁻⁵ (1e-5 = 0.00001, strictly typeset in LaTeX as $\epsilon = 10^{-5}$, avoiding typographical ambiguity) prevents division by zero. C_{h,d} ∈ [0, 1], where 0 indicates consensus and 1 indicates complete empirical deadlock.
   - *Zero-Evidence Bounding Behavior:* When S_pos = S_neg = 0 (no scientific literature exists for the queried candidate pair), the numerator evaluates to 0 and the denominator evaluates to ε (1e-5), yielding C_{h,d} = 0.0. Simultaneously, the metadata missingness indicator evaluates to 1.0 (100% missing literature fields). This zero contradiction is strictly NOT interpreted as evidence agreement or consensus; rather, the downstream neural reliability module receives zero corroboration and maximum missingness (missingness = 1.0), driving reliability R → 0.0 and safely routing 100% of the prediction weight to biological GAT priors.

4. **Dynamic Multimodal Gating Equation:**
   Combines mechanistic graph priors with contextual literature representations:
   - *Plain-text specification:* `z_fused = (1 - R) * z_graph + R * z_contextual`
   - *Formal mathematical expression:*
     $$\mathbf{z}_{\text{fused}} = (1 - R)\mathbf{z}_{\text{graph}} + R\mathbf{z}_{\text{contextual}}$$
   - *Architecture & Training of Reliability Scorer R:* R ∈ [0, 1] is produced by a dedicated neural reliability module that accepts an 8-dimensional evidence lineage feature vector:
     $$\mathbf{m} = [S_{\text{pos}}, S_{\text{neg}}, \text{echo\_ratio}, \text{quality}, W_{\text{context}}, C_{h,d}, \text{margin}, \text{missingness}]$$
     The module applies dimension-specific projection layers, an MLP with Monte Carlo dropout ($p = 0.15$), and explicit contradiction attenuation: $R = R_{\text{raw}} \cdot (1.0 - 0.4 \cdot C_{h,d})$.
   - *End-to-End Gradient Training:* The reliability gating weights are optimized end-to-end via gradient backpropagation from binary cross-entropy loss on interaction prediction (jointly with contrastive reliability regularization $\mathcal{L}_{\text{contrastive}}$ and calibration loss), aligning literature trust with ground-truth clinical verifiability.
   - *Missing-Modality Fallback & Ablation Verification:* When literature text is missing or absent (`evidence_features is None` or `missingness = 1.0`), $R$ evaluates to $0.0$. The fused representation mathematically collapses to $\mathbf{z}_{\text{fused}} = (1 - 0)\mathbf{z}_{\text{graph}} + 0 \cdot \mathbf{z}_{\text{contextual}} = \mathbf{z}_{\text{graph}}$, ensuring the system falls back entirely to metabolic graph priors without hallucination. Ablation experiments in Section 8.B explicitly demonstrate this behavior: setting $R = 0$ reproduces the Biological Graph Only baseline (0.7083 ROC-AUC), while setting $R = 1$ forces the Evidence Text Only baseline (0.8155 ROC-AUC), proving that the learned dynamic gate ($R \in [0, 1]$) successfully extracts synergistic value (0.9762 ROC-AUC).

5. **Validation Temperature Scaling:**
   Calibrates uncalibrated model logits z(h, d) using optimal temperature T* fitted on validation data via Negative Log-Likelihood (NLL) optimization using L-BFGS:
   - *Plain-text specification:* `Calibrated Probability p_hat = sigmoid( z(h, d) / T* )`
   - *Formal mathematical expression:*
     $$\hat{p} = \sigma\left(\frac{z(h, d)}{T^*}\right), \quad T^* = \arg\min_T -\sum_{(x,y) \in \mathcal{D}_{\text{val}}} [y \log \sigma(z/T) + (1-y)\log(1-\sigma(z/T))]$$
   where empirical optimal T* = 1.48 on the validation partition.

6. **Tri-State Decision Logic & Unambiguous Decision Table:**
   Routes the prediction based on calibrated probability, epistemic uncertainty, and literature conflict:
   - *Formal mathematical expression:*
     $$\hat{Y}(h, d) = \begin{cases}
     \text{ABSTAIN (Pharmacist Review Queue)} & \text{if } u_{\text{epistemic}} > \tau_u \;\lor\; (C_{h,d} > \tau_c \;\land\; |\hat{p} - 0.5| < \tau_{\text{margin}}) \;\lor\; (R < \tau_R \;\land\; |\hat{p} - 0.5| < 0.10) \\
     \text{INTERACTION DETECTED (Prescription Alert)} & \text{else if } \hat{p} \ge 0.50 \\
     \text{NO INTERACTION DETECTED (Clearance)} & \text{else if } \hat{p} < 0.50
     \end{cases}$$

   *Unambiguous Clinical Decision Policy Table (matching Flowchart Diamond 124 and Implementation):*
   
   | Decision State | Triggering Quantitative Condition | Operational System Action |
   | :--- | :--- | :--- |
   | **STATE 3: ABSTAIN (Epistemic Uncertainty)** | Epistemic uncertainty $u_{\text{epistemic}} > \tau_u$ ($0.35$) | Suppresses automated alert; routes to Asynchronous Pharmacist Quarantine Queue |
   | **STATE 3: ABSTAIN (Scientific Contradiction)** | Contradiction $C_{h,d} > \tau_c$ ($0.50$) AND $\| \hat{p} - 0.5 \| < \tau_{\text{margin}}$ ($0.20$) | Suppresses automated alert; routes to Asynchronous Pharmacist Quarantine Queue with evidence conflict audit trail |
   | **STATE 3: ABSTAIN (Out-of-Distribution)** | Low reliability $R < 0.25$ AND Borderline confidence $\| \hat{p} - 0.5 \| < 0.10$ | Suppresses automated alert; routes to Asynchronous Pharmacist Quarantine Queue |
   | **STATE 1: INTERACTION DETECTED** | Calibrated probability $\hat{p} \ge 0.50$ (with $u \le \tau_u$ and $C_{h,d} \le \tau_c$) | Issues high-priority clinical interaction advisory; communicates dispense interlock |
   | **STATE 2: NO INTERACTION DETECTED** | Calibrated probability $\hat{p} < 0.50$ (with $u \le \tau_u$ and $C_{h,d} \le \tau_c$) | Issues safe co-administration clearance advisory |

   *(Operational Qualification: The negative state "NO INTERACTION DETECTED" specifies that no clinically significant pharmacokinetic inhibition or induction was identified above the decision threshold under evaluated evidence and model parameters, avoiding unconditioned universal assertions of biological inactivity).*

---

### 8. Experimental validation results:

#### A. Audited Benchmark Dataset Specifications:
- **Corpus:** 141 peer-reviewed clinical and pharmacokinetic publications across 18 botanicals and 40 prescription drugs.
- **Traceability:** 141/141 records verified with official PubMed PMIDs and digital object identifiers (DOIs) via an auditable source registry.
- **Independence Partitioning:** 123 independent study clusters partitioned using **study-disjoint splitting** (zero study/lineage overlap):
  - Training Partition: 90 publications (73 independent clusters)
  - Validation Partition: 25 publications (25 independent clusters)
  - Test Partition: 26 publications (25 independent clusters)

#### B. Quantitative Comparison Against State-of-the-Art Baselines:
*(Evaluated under identical study-disjoint splits and cross-entropy loss, `seed=42`)*:

| Model / Architecture | Split Mode | ROC-AUC (95% Confidence Interval) | PR-AUC | F1-Score | Brier Score (Calibration Error) | Full Test Accuracy |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **Present Invention (Reliability-Conditioned)** | **Study-Disjoint** | **0.9762 [0.910, 1.000]** | **0.9707** | **0.9167** | **0.0846** | **92.31% (24/26)** |
| Standard Multimodal Fusion (Ablation) | Study-Disjoint | 0.8274 [0.650, 0.960] | 0.8481 | 0.6364 | 0.1640 | 69.23% (18/26) |
| DeepDDI (Ryu et al., *PNAS* 2018) | Study-Disjoint | 0.8155 [0.660, 0.960] | 0.8242 | 0.6957 | 0.1851 | 73.08% (19/26) |
| KGNN (Wang et al., *WWW* 2019) | Study-Disjoint | 0.8036 [0.640, 0.960] | 0.7854 | 0.6667 | 0.2266 | 69.23% (18/26) |
| CompGCN (Vashishth et al., *ICLR* 2020) | Study-Disjoint | 0.7857 [0.610, 0.940] | 0.8007 | 0.6957 | 0.2203 | 73.08% (19/26) |
| Evidence Text Only (BiomedBERT) | Study-Disjoint | 0.8155 [0.620, 0.980] | 0.7736 | 0.7692 | 0.1777 | 76.92% (20/26) |
| Biological Graph Only (GAT) | Study-Disjoint | 0.7083 [0.470, 0.930] | 0.6927 | 0.6667 | 0.2037 | 69.23% (18/26) |

#### C. Key Performance Takeaways:
1. **Significant Superiority Over Published Baselines:** The present invention outperforms DeepDDI by **+16.07 ROC-AUC points** (+19.7%), KGNN by **+17.26 points** (+21.5%), and CompGCN by **+19.05 points** (+24.2%).
2. **Probability Calibration Excellence:** Post-hoc temperature scaling achieves a Brier score of **0.0846** (compared to 0.1640 to 0.2266 for baselines), representing a **48% to 63% reduction in probability prediction error**.
3. **Selective Accuracy Under Tri-State Abstention:** When selective abstention is triggered on the conflicting study report (the Ginseng–Digoxin immunoassay artifact deadlock in Study Anderson 2003), clinical coverage remains high at **96.15% (25/26)** while selective accuracy increases from 92.31% to **96.00% (24/25 correct)** (Wilson 95% Confidence Interval: [80.5%, 99.3%], under abstention uncertainty threshold τ_u = 0.35 and conflict threshold τ_c = 0.40).
4. **Codebase & Prototype Verification:** The entire system is implemented in Python/PyTorch, achieves 100% test pass rate across 43 unit and integration tests (`pytest tests/ -v`), and operates as a standalone web application prototype.
5. **Contextual Evaluation Scope & Sample Size Bounding:** On the reported study-disjoint benchmark, the proposed model achieved an ROC-AUC of 0.9762, a PR-AUC of 0.9707, a Brier score of 0.0846, and an accuracy of 24/26 (92.31%), rising to 24/25 (96.00%) under selective abstention. The held-out test partition comprises 26 publications across 25 independent study clusters, representing a focused, auditable proof-of-concept laboratory benchmark. These results demonstrate the empirical validity of the citation-discounting and gating mechanisms under controlled study-disjoint splits, while broader generalization to wider clinical cohorts represents future translational evaluation.

---

### 9. What aspect(s) of the invention need(s) protection?

The proposed claims concern computer-implemented processing of biomedical evidence and associated clinical decision-support system operations. The scope and eligibility of the claims under applicable patent law, including the statutory exclusions relating to computer programs and algorithms (Section 3(k) of the Indian Patents Act, 1970) and methods of medical treatment (Section 3(i)), are framed in view of the complete technical specification and the following proposed claim categories:

1. **Computer-Implemented Method Claims:**
   - Methods for constructing citation-derivation directed acyclic graphs (DAGs) from scientific literature corpora and applying a parameterized evidence attenuation function (e.g., an exponential decay or constant discount factor $\alpha \in [0.05, 0.25]$, specifically $\alpha = 0.15$ in preferred embodiments) to secondary and derivative publications while isolating primary empirical investigations as independent evidence roots.
   - Methods executing a differentiable neural echo attenuation layer computing $C_{\text{attenuated}} = C \cdot \max(0.05, 1.0 - \sigma(\beta) \cdot \text{CEI})$ to eliminate circular citation echo-chamber inflation before evidence gating.
   - Methods for calculating a composite pharmacokinetic context compatibility score factoring biological species translation ($w_{\text{species}}$), botanical extract standardisation ($w_{\text{prep}}$), administration route ($w_{\text{route}}$), and an in vitro supra-physiological concentration ($>50\,\mu\text{M}$) penalty $w_{\text{conc}} = 0.25$ with learnable residual refinement parameters $(\Delta_s, \Delta_p, \Delta_r)$.
   - Methods for computing a continuous scalar reliability score $R \in [0, 1]$ as a function of citation graph density and evidence contradiction indices.
   - Methods for dynamically modulating multimodal representations by convexly fusing biological mechanism embeddings and contextual language embeddings via $\mathbf{z}_{\text{fused}} = (1 - R)\mathbf{z}_{\text{graph}} + R\mathbf{z}_{\text{contextual}}$.
   - Methods for executing a split conformal prediction tri-state decision rule guaranteeing finite-sample statistical coverage $P(Y \in \mathcal{C}(X)) \ge 1 - \epsilon$ at significance level $\epsilon \in (0, 0.5)$, classifying pairs as Interaction Present when $\mathcal{C}=\{1\}$, No Interaction when $\mathcal{C}=\{0\}$, and routing to clinical specialist review (Abstain state) when $|\mathcal{C}| \neq 1$ or contradiction exceeds threshold $\tau_c$.

2. **Computing System and Decision Support Architecture Claims:**
   - A computer-implemented clinical decision-support system configured to generate a structured interaction alert, a no-interaction-detected result under defined decision criteria, or an abstention state for review, comprising hardware processors, memory, and neural network modules executing the biological GAT encoder, the citation DAG processor, the neural reliability gating module, and the conformal tri-state decision engine.
   - In an implementation integrated with an authorised electronic health record (EHR) or hospital order management system, the resulting decision state may be communicated to a configured review or order-management workflow (e.g., communicating an automated prescription interlock advisory for detected interactions, communicating co-administration clearance for undetected interactions, or routing to an asynchronous pharmacist quarantine queue upon abstention). Such live clinical integration and its safety behaviour represent deployment embodiments requiring separate institutional validation.

3. **Non-Transitory Computer-Readable Medium Claims:**
   - Non-transitory storage media storing executable instructions that cause one or more processors to carry out the citation DAG noise attenuation, pharmacokinetic context compatibility scoring, reliability-conditioned multimodal fusion, conformal prediction set generation, and tri-state decision routing.

*(Statutory Scope Disclaimer: All claims are strictly directed to computer-implemented data processing, noise-filtering neural network architectures, and electronic clinical verification systems for assisting healthcare practitioners; the invention explicitly excludes methods for the direct therapeutic or surgical treatment of the human body).*


---

### 10. What is Technology readiness level of your invention? (Tick the appropriate TRL):

| TRL Level | Category | Description | Status for This Invention |
| :---: | :---: | :--- | :---: |
| **TRL 1** | Research | Basic principles observed | Passed |
| **TRL 2** | Research | Technology concept formulated | Passed |
| **TRL 3** | Research | Experimental proof of concept | Passed |
| **TRL 4** | **Development** | **Technology validated in a lab** | **[ ✔ ] TICK HERE (PRIMARY)** |
| **TRL 5** | **Development** | **Technology validated in a relevant environment** | **[ ✔ ] QUALIFIED TRANSITION** |
| **TRL 6** | Development | Technology demonstrated in a relevant environment | Future Roadmap |
| **TRL 7** | Deployment | System prototype demonstration in an operational environment | Future Roadmap |
| **TRL 8** | Deployment | System complete and qualified | Future Roadmap |
| **TRL 9** | Deployment | Actual system proven in operational environment | Future Roadmap |

#### Justification for TRL 4 / TRL 5:
- **TRL 4 (Validated in Lab):** The entire mathematical, algorithmic, and machine learning pipeline has been authored, tested, and validated in the laboratory using 141 peer-reviewed clinical publications across 18 botanicals and 40 prescription medications. It achieves 43/43 passing unit and integration tests and demonstrates 97.62% ROC-AUC and 96.00% selective accuracy.
- **Transitioning to TRL 5 (Validated in Relevant Environment):** An interactive, fully functional graphical user interface and simulated clinical decision support dashboard (`app/interactive_app.py`) has been built and executed, simulating real-world physician and clinical pharmacist review workflows.

---
**SIGNATURE OF INVENTOR(S):**  
Aditi Sharma: ___________________________  
**DATE:** 21 / 10 / 2026  
**DEPARTMENT / SCHOOL:** School of Computer Science and Engineering (SCOPE), Vellore Institute of Technology (VIT).
