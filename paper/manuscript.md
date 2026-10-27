# Reliability-Conditioned Prediction of Herb–Drug Interactions with Evidence Lineage Analysis, Citation Echo Detection, and Calibrated Abstention

**Authors**: Aditi Sharma, et al.  
**Affiliation**: Department of Computer Science & Engineering / Biomedical Data Mining Laboratory  
**Target Venue**: *Nature Communications* / *Bioinformatics* / *ACM Conference on Health, Inference, and Learning (CHIL)*  

---

## Abstract

Concurrent administration of botanical supplements and pharmaceutical agents is increasingly prevalent in clinical practice, yet predicting herb–drug interactions (HDIs) remains a critical open challenge. Computational models adapted from drug–drug interaction (DDI) literature typically treat published scientific reports as independent, identically distributed ground truth and rely solely on molecular topologies or static literature co-occurrences. However, botanical evidence in real-world biomedical corpora is heavily corrupted by **citation echo chambers** (where multiple publications reiterate a single historical index report), **experimental context heterogeneity** (in vitro cell assays vs. rodent models vs. human clinical trials), and **active scientific contradictions**. 

To address these vulnerabilities, we introduce a **Reliability-Conditioned Herb–Drug Interaction Prediction framework**. Our contributions are fourfold:
1. **Evidence Lineage & Echo Chamber Detection**: We construct citation-derivation directed acyclic graphs (DAGs) per herb–drug candidate pair, cluster derivative secondary reports around primary index studies while preserving independent clinical trials as distinct roots, and apply discounted independence scoring ($\alpha = 0.15$) to prevent circular citation inflation.
2. **Multi-Factor Context Compatibility**: We formulate a pharmacokinetic context compatibility metric scoring species translation (human = 1.00, in vivo mammalian = 0.50, in vitro = 0.20), preparation standardisation, and administration route, grounded in FDA and EMA botanical guidance.
3. **Reliability-Conditioned Multimodal Fusion**: We combine a 2-layer Biological Graph Attention Network (GAT) over human pharmacokinetic enzymes (CYP450s, ABCB1, OATP transporters) with frozen BiomedBERT language representations, dynamically gated by a learned reliability score $R \in [0, 1]$.
4. **Calibrated Tri-State Decision Engine with Abstention**: The model yields three clinical outcomes (*Interaction Present*, *No Interaction Observed*, or *Abstain*), withholding automated judgement when epistemic uncertainty or scientific contradiction exceeds rigorous safety thresholds.

Evaluated on an auditable benchmark of **141 curated clinical and pharmacokinetic publications** spanning 18 herbs and 40 prescription drugs across **123 independent study clusters** under strict **study-disjoint partitioning** (zero citation cluster leakage across splits), our framework achieves **0.8512 ROC-AUC [95% CI: 0.659, 0.987]**, **0.8794 PR-AUC**, **0.7826 F1-score**, and **80.77% accuracy (21/26 correct)**, providing competitive performance against external graph architectures under rigorous leakage-free evaluation. Furthermore, calibrated abstention selectively identifies uncertain and conflicting predictions, routing them safely to clinical pharmacist review.

---

## 1. Introduction

Herb–drug interactions (HDIs) represent a major, under-recognized public health risk. Over 35% of adult patients in North America and Europe, and more than 70% in Asia, concurrently consume botanical dietary supplements and prescription medications. Unlike synthetic small-molecule drugs that possess well-characterized single active pharmaceutical ingredients (APIs), botanicals (e.g., *Hypericum perforatum* [St. John's wort], *Ginkgo biloba*, *Panax ginseng*) are multi-component botanical mixtures containing dozens of biologically active phytochemicals that concurrently modulate phase I cytochrome P450 enzymes (CYP3A4, CYP2C9, CYP1A2, CYP2E1), phase II conjugation enzymes, and drug transporters (ABCB1/P-glycoprotein, OATP1A2, OATP1B1).

Despite the clinical severity of adverse HDIs—exemplified by St. John's wort precipitating acute transplant rejection by clearing cyclosporine and tacrolimus, or Ginkgo biloba precipitating fatal intracranial hemorrhages with warfarin—computational methods for HDI prediction remain severely lagging behind drug–drug interaction (DDI) algorithms. 

Existing machine learning approaches suffer from three systemic failure modes:
1. **The Citation Echo Chamber Fallacy**: Existing literature mining and DDI prediction systems treat every PubMed abstract or text mention as an independent observation. In reality, a single dramatic case report is frequently cited by dozens of subsequent reviews, secondary commentaries, and meta-analyses. Standard machine learning models mistake this circular citation volume for strong empirical consensus.
2. **Context Blindness**: An in vitro microsomal binding assay conducted with isolated phytochemical aglycones at supraphysiological concentrations (e.g., $100\,\mu\text{M}$) does not translate to oral in vivo bioavailability in humans, where intestinal absorption, hepatic first-pass metabolism, and protein binding attenuate systemic exposure. Standard algorithms average these findings without contextual weighting.
3. **Forced Binary Extrapolation on Conflicting Evidence**: Literature often contains direct contradictions (e.g., retrospective case reports attributing warfarin bleeding to cranberry juice vs. prospective randomized controlled trials demonstrating zero effect on INR). Existing binary classifiers force a hard classification, generating false positive alerts that lead to clinical alert fatigue.

To resolve these challenges, we develop the **Reliability-Conditioned Herb–Drug Interaction Prediction System**, establishing an auditable pipeline from raw literature verification to calibrated clinical decision-making.

---

## 2. Related Work & Novelty Differentiation

| System / Prior Art | Technical Approach | Evidentiary Rigor | Citation Lineage & Echo Discounting? | Abstention Policy? |
| :--- | :--- | :--- | :--- | :--- |
| **DeepDDI** (Ryu et al., *PNAS* 2018) | Deep MLP on molecular fingerprints | Single API only | None (assumes static labels) | None |
| **KGNN** (Wang et al., *WWW* 2019) | Knowledge Graph Convolution with relation attention | Graph topology only | None | None |
| **CompGCN** (Vashishth et al., *ICLR* 2020) | Relational composition ($\mathbf{e}_v \circ \mathbf{r}$) | Graph topology only | None | None |
| **CN115223731B** (Patent, 2026) | Dempster–Shafer evidence theory fusion with Dirichlet uncertainty | Multi-source sensor / feature fusion | **No** (hard rule combination; no citation graphs) | **No** (no clinical abstention) |
| **HTINet2** (PMC 2024) | Herb-Target KG + GCN + BPR link prediction | Binary threshold filtering ($p < 0.05$) | **No** (static filtering; drops lineage) | **No** |
| **RGEF** (IEEE 2024) | Modality-level trust gating | Sensor modalities (ECG/EEG) | **No** (applied to signals, not literature DAGs) | **No** |
| **This Work (Proposed)** | **Heterogeneous Biological GAT + BiomedBERT + Reliability Gating** | **Four-Tier Audited Source Registry (100% verified)** | **Yes (DAG connected components, $\alpha=0.15$ discount)** | **Yes (Epistemic MC Dropout + Conflict Abstention)** |

---

## 3. Methodological Architecture

```
                                  [ HERB-DRUG CANDIDATE PAIR ]
                                              │
                      ┌───────────────────────┴───────────────────────┐
                      ▼                                               ▼
         [ Pharmacological Text ]                       [ Pharmacokinetic Targets ]
                      │                                               │
             (Frozen BiomedBERT)                         (Heterogeneous Biological GAT)
                      │                                               │
           768D Contextual Embeddings                      Latent Mechanism Embeddings
                      │                                               │
             (SVD Train-Projected)                                    │
                      │                                               │
                      ▼                                               ▼
              [ z_contextual ]                                    [ z_graph ]
                      │                                               │
                      └───────────────────────┬───────────────────────┘
                                              │
                                 ┌────────────┴────────────┐
                                 │ Multi-Token Cross-Attn  │
                                 └────────────┬────────────┘
                                              │
                                              ▼
                             [ Dynamic Reliability Gating ]
                               z_fused = (1-R)*z_graph + R*z_contextual
                                              │
                                  ┌───────────┴───────────┐
                                  │   Lineage Metadata    │
                                  │   • Echo chamber DAG  │
                                  │   • Species weight    │
                                  │   • Route & Prep      │
                                  │   • Conflict score    │
                                  └───────────┬───────────┘
                                              │
                                              ▼
                                 [ Link Prediction Head ]
                                              │
                                              ▼
                           [ Validation Temperature Scaling ]
                                              │
                                              ▼
                                 [ Calibrated Tri-State ]
                                 ├── Interaction Present
                                 ├── No Interaction Observed
                                 └── ABSTAIN (Conflict / Uncertainty)
```

### 3.1 Formal Mathematical Formulation

#### Definition 1 (Evidence Lineage Graph & Edge Semantics)
For any herb–drug candidate pair $(h, d)$, let $\mathcal{E}_{h,d} = \{e_1, e_2, \dots, e_N\}$ denote the set of extracted evidence records indexed by verified DOIs and PMIDs. We construct a directed citation-derivation graph $\mathcal{G}_L = (\mathcal{V}_L, \mathcal{E}_L)$, where each node $v_i \in \mathcal{V}_L$ represents a primary study report $s_i$, and a directed edge $(v_i, v_j) \in \mathcal{E}_L$ exists if study $s_i$ explicitly cites or derives its interaction findings from study $s_j$. Independent clinical trials possessing distinct clinical trial registrations (e.g., distinct NCT IDs) are maintained as separate independent evidence roots even when co-cited by common review articles.

#### Definition 2 (Citation Cluster & Echo Attenuation)
Let $\mathcal{C}_1, \mathcal{C}_2, \dots, \mathcal{C}_K$ be the weakly connected components of $\mathcal{G}_L$. In each component $\mathcal{C}_k$, the study with the earliest verified publication timestamp is designated as the index study $s_{\text{primary}}^{(k)}$. All remaining studies $s_j \in \mathcal{C}_k \setminus \{s_{\text{primary}}^{(k)}\}$ are classified as secondary echo reports. The effective independent evidence cluster strength $S(\mathcal{C}_k)$ is computed as:
$$S(\mathcal{C}_k) = q(s_{\text{primary}}^{(k)}) + \sum_{s_j \in \mathcal{C}_k, j \neq \text{primary}} \alpha \cdot q(s_j)$$
where $q(s) \in [0, 1]$ is the verified extraction quality of study $s$, and $\alpha \in [0.05, 0.25]$ (nominal $\alpha = 0.15$) is the empirical echo attenuation discount factor.

#### Definition 3 (Pharmacokinetic Context Compatibility)
Each evidence record $e_i$ is parameterized by its experimental context tuple grounded in FDA and EMA guidelines:
$$\mathcal{K}(e_i) = w_{\text{species}}(e_i) \cdot w_{\text{prep}}(e_i) \cdot w_{\text{route}}(e_i)$$
where:
$$w_{\text{species}} = \begin{cases} 1.00 & \text{Human (clinical trial, PK study)} \\ 0.50 & \text{In vivo mammalian (rat, dog, mouse)} \\ 0.20 & \text{In vitro (cell line, microsomes, recombinant)} \end{cases}$$
$$w_{\text{prep}} = \begin{cases} 1.00 & \text{Standardized botanical extract (LI 160, EGb 761)} \\ 0.85 & \text{Crude whole-herb extract / decoction} \\ 0.70 & \text{Isolated pure aglycone / phytochemical} \end{cases}$$
$$w_{\text{route}} = \begin{cases} 1.00 & \text{Oral administration (canonical clinical route)} \\ 0.75 & \text{Intravenous / Intraperitoneal} \end{cases}$$

#### Definition 4 (Evidentiary Conflict Metric)
Let $S_{\text{pos}}$ and $S_{\text{neg}}$ denote the discounted cluster strengths supporting interaction presence and absence, respectively. The scientific contradiction index $C_{h,d} \in [0, 1]$ is formally defined as:
$$C_{h,d} = \frac{2 \cdot \min(S_{\text{pos}}, S_{\text{neg}})}{S_{\text{pos}} + S_{\text{neg}} + \epsilon}$$
where $C_{h,d} \to 1.0$ indicates perfect scientific deadlock between conflicting trials.

#### Definition 5 (Reliability Score Formulation)
The composite evidence lineage metadata vector $\mathbf{m}_{h,d} \in \mathbb{R}^{10}$ encompasses:
$$\mathbf{m}_{h,d} = \left[ S_{\text{pos}}, S_{\text{neg}}, N_{\text{primary}}, N_{\text{echo}}, \bar{\kappa}_{\text{context}}, C_{h,d}, \bar{q}, \Delta_{\text{time}}, N_{\text{raw,pos}}, N_{\text{raw,neg}} \right]$$
The scalar reliability gating coefficient $R_{h,d} \in [0, 1]$ is computed via a multi-layer perceptron:
$$R_{h,d} = \sigma\left(\mathbf{W}_2 \cdot \text{GELU}(\mathbf{W}_1 \mathbf{m}_{h,d} + \mathbf{b}_1) + b_2\right) \cdot \left(1 - \beta \cdot C_{h,d}\right)$$
where $\beta = 0.40$ ensures that active scientific conflicts attenuate the trust placed in literature representations.

#### Definition 6 (Reliability-Conditioned Multimodal Gating)
Given biological graph pair representation $\mathbf{z}_{\text{graph}}$ (learned via Heterogeneous Biological GAT over human CYP450s and transporters) and contextual evidence representation $\mathbf{z}_{\text{contextual}}$ (frozen BiomedBERT projected via training SVD), the fused interaction state $\mathbf{z}_{\text{fused}}$ is dynamically gated:
$$\mathbf{z}_{\text{fused}} = (1 - R_{h,d}) \cdot \mathbf{z}_{\text{graph}} + R_{h,d} \cdot \mathbf{z}_{\text{contextual}}$$

**Pharmacodynamic Rationale:** The biological graph $\mathbf{z}_{\text{graph}}$ captures *theoretical mechanistic potential* (in vitro enzyme affinities). The text branch $\mathbf{z}_{\text{contextual}}$ captures *empirical human trial observations*. In pharmacology, verified in vivo human outcomes supersede in vitro potential. As literature reliability $R \to 1$, in vivo human findings dominate. When literature is absent or circular ($R \to 0$), the model gracefully retreats to the mechanistic graph prior.

#### Definition 7 (Validation-Set Temperature Scaling)
Raw link prediction logits $z$ are calibrated using optimal temperature $T^* > 0$ learned exclusively on the held-out validation split $\mathcal{D}_{\text{val}}$ by minimizing negative log-likelihood:
$$T^* = \arg\min_T -\sum_{(x, y) \in \mathcal{D}_{\text{val}}} \left[ y \log \sigma\left(\frac{z(x)}{T}\right) + (1-y) \log\left(1 - \sigma\left(\frac{z(x)}{T}\right)\right) \right]$$
The calibrated probability is $\hat{p} = \sigma(z / T^*)$.

#### Definition 8 (Calibrated Tri-State Abstention Policy)
For each query $(h, d)$, let $u_{\text{epistemic}}$ be the predictive variance estimated over $M=10$ Monte Carlo Dropout forward passes ($\text{dropout}=0.15$). The decision rule $\hat{Y}(h, d)$ is defined as:
$$\hat{Y}(h, d) = \begin{cases}
\text{ABSTAIN} & \text{if } u_{\text{epistemic}} > \tau_u \;\lor\; \left( C_{h,d} > \tau_c \;\land\; |\hat{p} - 0.5| < \tau_{\text{margin}} \right) \\
\text{INTERACTION PRESENT} & \text{else if } \hat{p} \ge 0.50 \\
\text{NO INTERACTION} & \text{else if } \hat{p} < 0.50
\end{cases}$$
with clinical safety thresholds $\tau_u = 0.25$, $\tau_c = 0.50$, and $\tau_{\text{margin}} = 0.35$.

---

## 4. Empirical Evaluation & Benchmark Results

### 4.1 Benchmark Description
The benchmark dataset consists of **141 audited, real-world scientific publications** verified against the Auditable Source Registry with 100% PubMed PMID and DOI matching (zero synthetic placeholders), encompassing **123 independent study clusters**. The dataset spans **18 distinct botanicals** and **40 critical prescription medications** across cardiovascular, immunosuppressive, antineoplastic, psychiatric, and antiviral classes.

To guarantee zero data contamination, all evaluations enforce **Study-Disjoint Partitioning**: weakly connected citation components are assigned indivisibly to train (60%, 90 records, 73 clusters), validation (20%, 25 records, 25 clusters), and test (20%, 26 records, 25 clusters). Zero citation ancestors, descendants, or shared empirical cohorts appear across splits.

### 4.2 Comparative Evaluation Against External Baselines

| Model Architecture | Input Features | ROC-AUC (95% CI) | PR-AUC | F1-Score | Brier Score | Accuracy (Wilson 95% CI) |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **Reliability-Conditioned (Quality-Only)** | **Biological Graph + Lineage + Text** | **0.8512 [0.659, 0.987]** | **0.8794** | **0.7826** | **0.1554** | **80.77% (21/26) [62.7%, 91.1%]** |
| Standard Multimodal Fusion | Biological Graph + Text (Fixed $R=1$) | 0.8274 [0.650, 0.960] | 0.8504 | 0.6087 | 0.1795 | 65.38% (17/26) [46.2%, 80.6%] |
| Re-implemented DeepDDI (Ryu et al.) | Target Profile Fingerprints | 0.8452 [0.680, 0.970] | 0.8414 | 0.7500 | 0.1687 | 76.92% (20/26) [57.9%, 89.0%] |
| Re-implemented KGNN (Wang et al.) | Target Knowledge Graph | 0.7202 [0.490, 0.930] | 0.7805 | 0.6667 | 0.2194 | 69.23% (18/26) [50.0%, 83.5%] |
| Re-implemented CompGCN (Vashishth) | Compositional Relational Graph | 0.7262 [0.510, 0.920] | 0.7774 | 0.7273 | 0.1918 | 76.92% (20/26) [57.9%, 89.0%] |
| Evidence Text Only Ablation | Text Tokens / BiomedBERT Only | 0.8095 [0.610, 0.980] | 0.7647 | 0.7692 | 0.1934 | 76.92% (20/26) [57.9%, 89.0%] |
| Biological Graph Only Ablation | GAT Mechanism Graph Only | 0.7202 [0.490, 0.920] | 0.6858 | 0.6667 | 0.2108 | 69.23% (18/26) [50.0%, 83.5%] |

**Performance Analysis & Numerical Audit:**
- **Competitive Performance Under Leakage-Free Evaluation:** Evaluated on the strictly partitioned study-disjoint test partition, the proposed quality-only model achieves an ROC-AUC of **0.8512** [0.659, 0.987] and PR-AUC of **0.8794**, outperforming target knowledge graph baselines CompGCN (0.7262) and KGNN (0.7202), and remaining competitive with DeepDDI (0.8452).
- **ROC-AUC Calculation & Split Details:** The test partition comprises 26 publications across 25 independent study clusters, with 12 positive records and 14 negative/non-positive records ($12 \times 14 = 168$ evaluated rank pairs). The leakage-free model correctly ranks pairs without label-adjacent leakage.
- **Selective Abstention:** Rather than forcing classifications under extreme epistemic uncertainty or irreconcilable evidence conflict, the tri-state decision engine routes ambiguous cases to clinical pharmacist review queues.
- **Probability Calibration via Brier Score:** Validated post-hoc temperature scaling achieves a Brier score of **0.1554** and ECE of **0.1621**, providing calibrated probabilities that substantially improve over unconditioned baselines.

### 4.3 Systematic Ablation Studies & Anti-Circularity Analysis

| Ablation Variant | Description | ROC-AUC | F1-Score | Brier Loss | Accuracy |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **Proposed Quality-Only Framework** | Complete reliability gating + lineage | **0.8512** | **0.7826** | **0.1554** | **80.77% (21/26)** |
| Graph + Naive Evidence | Graph fused with unweighted text | 0.8512 | 0.7273 | 0.1652 | 76.92% (20/26) |
| Standard Multimodal Fusion | Fixed unconditioned fusion $R = 1.0$ | 0.8274 | 0.6087 | 0.1795 | 65.38% (17/26) |
| Evidence Text Only (w/o Graph Priors) | BiomedBERT text representation only | 0.8095 | 0.7692 | 0.1934 | 76.92% (20/26) |
| Biological Graph Only (w/o Text Evidence) | Biological GAT mechanism graph only | 0.7202 | 0.6667 | 0.2108 | 69.23% (18/26) |
| w/o Study Dependence (Ablation) | Lineage without cluster clustering | 0.7321 | 0.6667 | 0.1952 | 69.23% (18/26) |

*\*Critical Note on Label Circularity and Echo Attenuation:* If polarity features ($S_{\text{pos}}, S_{\text{neg}}$) are unattenuated, the model acts as a direct label regurgitator. Isolating lineage features to `quality_only` mode eliminates target leakage and guarantees that literature quality, context compatibility, and contradiction indices are evaluated solely for their capacity to weight evidence without label contamination. Echo attenuation ($\alpha = 0.15$) enforces epistemic safety, preventing circular citation inflation while enabling the system to retreat gracefully to biological graph priors $(1-R)\mathbf{z}_{\text{graph}}$ when literature is absent or noisy.

### 4.4 Baseline Parity and Negative Results
We report transparently that on the focused study-disjoint test partition ($N=26$), simpler non-gated architectures achieve comparable or slightly higher point estimates:
- **Naive Concat MLP**: Achieves **0.9048 ROC-AUC**, **0.9207 PR-AUC**, and **80.77% accuracy** (21/26 correct).
- **Logistic Regression (13D Tabular)**: Achieves **0.8274 ROC-AUC**, **0.8616 PR-AUC**, and **84.62% accuracy** (22/26 correct).
- **DeepDDI**: Achieves **0.8452 ROC-AUC** and **76.92% accuracy** (20/26 correct).

These results demonstrate that when evaluated purely on uncorrupted in-distribution candidate pairs with available text, direct concatenation or tabular modeling can achieve strong point estimates without explicit reliability gating. However, the essential value of the proposed reliability-conditioned framework lies not in raw test-set metric maximization on clean small cohorts, but in four safety-critical properties:
1. **Calibrated Posterior Probabilities**: Achieves a Brier score of **0.1554** and ECE of **0.1621**, compared to 0.1687–0.2194 for graph baselines.
2. **Principled Fallback Mechanism**: When literature is absent or uncorroborated, reliability collapses ($R \to 0$), gracefully defaulting to biological GAT mechanism priors without hallucination.
3. **Echo Chamber Attenuation**: Connected-component discounting ($\alpha = 0.15$) eliminates circular citation inflation.
4. **Calibrated Tri-State Abstention**: Epistemic uncertainty and contradiction thresholds safely route ambiguous and contested pairs to clinical pharmacist review queues rather than issuing forced guesses.

---

## 5. Limitations

While this work establishes an auditable foundation for reliability-conditioned HDI prediction, several important methodological and empirical limitations must be acknowledged:
1. **Small Benchmark Sample Size ($N=26$ test records)**: Although our benchmark of 141 publications across 123 independent study clusters is fully audited with verified PubMed PMIDs and DOIs, the held-out study-disjoint test split contains 26 records (12 positive, 14 non-positive). Consequently, bootstrap confidence intervals are wide (ROC-AUC 95% CI: [0.659, 0.987]), and differences between top-performing models do not reach statistical significance after Holm-Bonferroni correction.
2. **Zero-Literature External Validation Behavior**: When applied to external benchmarks without curated literature corpora (e.g., NaPDI clinical template pairs), the reliability module correctly detects complete literature missingness ($M = 1.0$) and sets $R = 0.0$. Under strict conformal calibration bands ($\epsilon = 0.10$), the resulting biological GAT prior predictions ($\hat{p} \in [0.63, 0.80]$) fall within the non-singleton prediction set region, triggering 100% abstention. While this represents safe protective clinical behavior, prospective deployment requires automated literature retrieval pipelines to populate evidence lineage graphs for novel pairs.
3. **Decision Sensitivity to Initialization Seeds**: On small sample sizes, candidate pairs with predicted probabilities near the decision boundary ($\hat{p} \approx 0.50$, such as Kava–Chlorzoxazone at $\hat{p} = 0.4419$) can flip classifications across different neural network initialization seeds, highlighting the necessity of deep ensembling for clinical deployment.
4. **Translational Scope**: The system provides advisory Clinical Decision Support (CDS) alerts and review queue routing; it has not undergone prospective multi-center clinical trial validation in active Electronic Health Record (EHR) environments.

---

## 6. Clinical Case Studies

### Case 1: *Hypericum perforatum* and Cyclosporine (Echo Chamber Identification)
In our citation graph, 4 separate publications reported catastrophic drops in cyclosporine blood levels following St. John's wort ingestion. Lineage analysis revealed that `Breidenbach_2000`, `Mai_2000`, and `Ernst_2000` all directly derived from the index clinical landmark report `Ruschitzka_1999` (Lancet). Our connected component algorithm grouped these into a single independent cluster ($S_{\text{pos}} = 1.0 + 3 \times 0.15 = 1.45$), correctly assigning a high reliability score ($R = 0.94$) grounded in genuine clinical replication rather than an artificial 4.0× count.

### Case 2: *Vaccinium macrocarpon* and Warfarin (Scientific Contradiction Resolution)
Cranberry juice co-administration with warfarin presents severe evidentiary conflict: retrospective case reports (BMJ 2003) reported fatal gastrointestinal bleeding, while a prospective randomized double-blind clinical trial (J Thromb Haemost 2011) observed no alteration in S-warfarin clearance or INR. Standard classifiers force a positive interaction prediction, triggering clinical alert fatigue. Our framework computed a conflict index $C_{h,d} = 0.82$ and epistemic uncertainty $u = 0.31$, triggering the **ABSTAIN** policy and generating an auditable report detailing the specific clinical trials in opposition.

---

## 7. Conclusion

We presented a reliability-conditioned prediction architecture specifically formulated for herb–drug interactions that replaces unconditioned literature trust with auditable evidence lineage graphs, context compatibility scoring, and calibrated selective abstention. By demonstrating that simpler concatenation models can achieve parity in raw test accuracy while lacking safety mechanisms, our findings emphasize that responsible clinical AI must prioritize uncertainty calibration, circularity attenuation, and transparent fallback over uncalibrated metric optimization.
