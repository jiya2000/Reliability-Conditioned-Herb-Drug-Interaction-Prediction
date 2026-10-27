# FORMAL PATENT APPLICATION SPECIFICATION

**TITLE OF THE INVENTION:**  
**COMPUTER-IMPLEMENTED SYSTEM AND METHOD FOR RELIABILITY-CONDITIONED PREDICTION OF BOTANICAL–PHARMACEUTICAL INTERACTIONS USING CITATION-DERIVATION DIRECTED ACYCLIC GRAPHS, PHARMACOKINETIC CONTEXT COMPATIBILITY, AND CALIBRATED TRI-STATE ABSTENTION**

**INVENTOR(S):**  
Aditi Sharma

---

## CROSS-REFERENCE TO RELATED APPLICATIONS
[0001] This application claims priority to and the benefit of prior provisional laboratory invention disclosures and technical reports filed in the Biomedical Data Mining and Machine Learning Laboratory, the entire contents of which are incorporated herein by reference.

---

## STATEMENT REGARDING FEDERALLY SPONSORED RESEARCH
[0002] Not Applicable.

---

## TECHNICAL FIELD
[0003] The present invention relates generally to computational bioinformatics, artificial intelligence, and graph neural network systems. More particularly, the present invention relates to computerized methods, systems, and non-transitory computer-readable media for training, regularizing, and operating multimodal neural networks to predict botanical–pharmaceutical (herb–drug) interactions, wherein the neural network architecture is controlled by citation-derivation directed acyclic graphs that eliminate circular citation echo chambers, physiological pharmacokinetic context compatibility weighting, and validation-calibrated tri-state selective abstention.

---

## BACKGROUND OF THE INVENTION

### 1. The Botanical Polypharmacy Challenge and In Vivo Complexities
[0004] Concurrent administration of botanical dietary supplements (herbs, phytomedicines, traditional natural products) and conventional synthetic prescription pharmaceuticals is a widespread clinical reality worldwide. Over 35% of adult patients in North America and Europe, and more than 70% in Asia, regularly consume herbal supplements alongside chronic prescription therapies. Unlike synthetic pharmaceutical drugs, which typically contain a single active pharmaceutical ingredient (API) of known stoichiometry and crystalline purity, botanical preparations (e.g., *Hypericum perforatum* [St. John's wort], *Ginkgo biloba*, *Panax ginseng*, *Curcuma longa*, *Allium sativum*) are multi-component natural mixtures comprising dozens of bioactive secondary metabolites (phytochemicals such as hyperforin, flavonoids, ginkgolides, ginsenosides, and curcuminoids).

[0005] These phytochemicals concurrently interact with human phase I hepatic and intestinal cytochrome P450 enzymes (e.g., CYP3A4, CYP2C9, CYP2C19, CYP1A2, CYP2E1), phase II conjugating enzymes (e.g., UDP-glucuronosyltransferases), and membrane drug transporters (e.g., ABCB1/P-glycoprotein, OATP1B1, OATP1B3). Modulation of these clearance pathways precipitates catastrophic clinical outcomes. For example, hyperforin in St. John's wort potently induces CYP3A4 and ABCB1, drastically accelerating systemic clearance of the immunosuppressants cyclosporine and tacrolimus, directly triggering acute organ transplant rejection. Conversely, Ginkgo biloba flavonoids inhibit platelet activating factor and drug-metabolizing enzymes, precipitating spontaneous intracranial hemorrhages when co-administered with anticoagulants (warfarin, dabigatran).

### 2. Failure Modes of Conventional Machine Learning and Prior Art
[0006] Existing computational models for drug–drug interactions (DDIs)—such as DeepDDI (Ryu et al., *PNAS* 2018), KGNN (Wang et al., *WWW* 2019), and CompGCN (Vashishth et al., *ICLR* 2020)—fail when applied to botanical interactions due to three fundamental data corruption mechanisms in biomedical literature:

[0007] **A. The Circular Citation Echo Chamber Fallacy:** In biomedical literature, a single dramatic or unverified case report is frequently cited, reviewed, and reiterated across dozens of secondary commentaries, review articles, and textbook chapters. Standard Natural Language Processing (NLP) models and multimodal graph networks treat every scientific publication or text mention as an independent, identically distributed ($i.i.d.$) ground-truth sample. Consequently, conventional machine learning models mistake the sheer volume of circular citations for genuine empirical replication, yielding severely overconfident false-positive alerts.

[0008] **B. Experimental Context Mismatches:** In vitro microsomal binding assays conducted in cell-free systems at supraphysiological phytochemical aglycone concentrations ($>100\,\mu\text{M}$) rarely translate to in vivo clinical bioavailability in humans, where intestinal deglycosylation by gut microbiota, extensive hepatic first-pass clearance, and serum albumin binding severely attenuate systemic exposure. Furthermore, crude water teas differ fundamentally from standardized ethanolic extracts (e.g., EGb 761). Existing computational algorithms aggregate all published findings indiscriminately without physiological context adjustment.

[0009] **C. Forced Binary Extrapolation on Irreconcilable Scientific Contradictions:** Literature often exhibits direct clinical conflict. For example, retrospective case reports allege severe bleeding when warfarin is co-administered with cranberry juice, whereas subsequent prospective randomized, double-blind, placebo-controlled human clinical trials demonstrate zero alterations in international normalized ratio (INR) or S-warfarin clearance. Conventional binary classifiers are forced to make a hard interaction vs. no-interaction prediction, generating either false alarms that induce clinical alert fatigue or false negatives that endanger patient safety.

### 3. Detailed Prior Art Differentiation
[0010] **Chinese Patent Publication CN115223731B ("Herb-drug interaction prediction method and system based on Dempster-Shafer evidence theory and Dirichlet uncertainty"):** CN115223731B combines multi-source evidence using Dempster-Shafer (D-S) mathematical evidence theory and parameterizes uncertainty with Dirichlet distributions. However, CN115223731B relies on the fundamental axiomatic assumption of Dempster's rule of combination that evidence sources are strictly conditionally independent. CN115223731B provides no technical mechanism to trace literature citation lineages, cannot construct citation directed acyclic graphs, fails to identify weakly connected citation clusters, cannot detect circular echo cascades, and possesses no clinical abstention policy to withhold decisions on contradictory trials.

[0011] **HTINet2 (Bioinformatics, 2024):** HTINet2 constructs a herb-target bipartite network and applies graph convolutional networks with Bayesian Personalized Ranking (BPR). However, HTINet2 discards noisy literature through a crude, static heuristic $p$-value threshold ($p < 0.05$). It does not model citation ancestry, contains no context compatibility weighting, and forces a closed-world interaction score.

[0012] **Reliability-Gated Evidence Fusion (RGEF, IEEE 2024):** RGEF introduces reliability gating across physical sensory modalities (e.g., ECG, EEG). However, RGEF is designed for continuous temporal sensor streams and cannot construct topological citation dependency graphs or resolve scientific publication conflicts.

### 4. Technical Problems Solved under Patent Eligibility Standards
[0013] In view of 35 U.S.C. §101, §102, and §103, and recent administrative and judicial precedents (including the USPTO Appeals Review Panel precedential decision in *Ex parte Desjardins* and Federal Circuit jurisprudence in *Recentive Analytics*), the present invention provides a **specific technical improvement to the machine learning architecture and training pipeline itself**:
1. It eliminates label leakage and overconfidence caused by circular citation dependencies in biomedical training corpora through topological DAG clustering and echo attenuation;
2. It solves the technical problem of cross-modal feature divergence by dynamically routing information flow between a mechanistic biological graph encoder and a contextual language model via a continuous reliability gating coefficient; and
3. It integrates the trained neural network with an automated electronic clinical decision support system that routes uncertain or contradictory predictions to an asynchronous review queue while displaying an advisory flag, rather than generating false-positive alarms or unsafe silence.

---

## SUMMARY OF THE INVENTION
[0014] The present invention provides a computerized system, method, and non-transitory computer-readable medium for reliability-conditioned botanical–drug interaction prediction that systematically overcomes the limitations of circular citation inflation, context blindness, and forced binary misclassification.

[0015] In a first aspect, the invention provides a computer-implemented method for training and operating a reliability-conditioned neural network system to evaluate botanical–pharmaceutical interactions, comprising:
- (a) receiving an electronic query designating a candidate pair comprising a botanical entity and a pharmaceutical drug;
- (b) retrieving from a structured digital database a plurality of literature evidence records associated with said candidate pair, wherein each record is indexed by a digital identifier (e.g., DOI or PMID);
- (c) constructing a citation-derivation directed acyclic graph (DAG) wherein each node represents an experimental study or secondary review and each directed edge represents an explicit evidentiary derivation or secondary commentary relationship;
- (d) identifying primary experimental index publications having no outgoing derivation edge to an earlier primary study for said candidate pair, clustering derivative secondary publications that cite said primary index study, and calculating an echo-attenuated cluster evidence strength by applying an attenuation discount factor $\alpha \in [0.05, 0.25]$ to secondary publications within said cluster, while maintaining independent empirical studies possessing distinct study identifiers (such as distinct clinical trial registrations, distinct institutional animal protocols, or distinct laboratory origins) as separate independent evidence roots even when co-cited by common review publications;
- (e) calculating, for each evidence record, a pharmacokinetic context compatibility score as a multiplicative product of a biological species translation weight $w_{\text{species}}$, a botanical preparation standardisation weight $w_{\text{prep}}$, and an administration route weight $w_{\text{route}}$;
- (f) deriving a scientific contradiction index $C_{h,d} \in [0, 1]$ quantifying evidentiary conflict between positive and negative evidence clusters;
- (g) computing a dynamic scalar reliability score $R \in [0, 1]$ via a trained neural reliability module as a function of an evidence lineage metadata vector and said contradiction index;
- (h) generating a biological mechanism embedding using a heterogeneous biological Graph Attention Network (GAT) operating over human pharmacokinetic enzymes and transporters;
- (i) generating a contextual literature embedding using a deep language model projected into the graph embedding space;
- (j) computing a reliability-gated multimodal representation $\mathbf{z}_{\text{fused}} = (1 - R)\mathbf{z}_{\text{graph}} + R\mathbf{z}_{\text{contextual}}$;
- (k) scaling raw interaction logits using a validation-calibrated temperature parameter $T^* > 0$ learned exclusively on a study-disjoint validation partition;
- (l) evaluating said scaled logits under a tri-state decision rule that classifies the candidate pair into an Interaction Present state, a No Interaction state, or an Abstain state based on epistemic predictive variance computed via Monte Carlo dropout and said contradiction index; and
- (m) generating an electronic clinical decision support control signal that triggers a Clinical Decision Support (CDS) advisory alert or high-priority pharmacist review routing when an interaction is predicted, and routes the transaction to an asynchronous clinical review queue while displaying a non-intrusive advisory flag when the Abstain state is triggered.

---

## DETAILED DESCRIPTION OF THE PREFERRED EMBODIMENTS

### 1. System Architecture Overview
[0016] Referring to **FIG. 1**, the system receives as input a candidate herb–drug pair $(h, d)$ where $h \in \mathcal{H}$ represents a botanical entity (e.g., St. John's wort, Ginkgo biloba, Ginseng, Garlic, Turmeric) and $d \in \mathcal{D}$ represents a pharmaceutical small molecule (e.g., cyclosporine, warfarin, midazolam, tacrolimus, digoxin). The system comprises two parallel representation pathways: a biological mechanism pathway and a contextual literature evidence pathway, mediated by an evidence lineage and reliability analysis engine.

### 2. Citation-Derivation Directed Acyclic Graph (DAG) and Echo Attenuation
[0017] **DAG Construction and Tripartite Evidence Criteria:** An evidence lineage module constructs a directed citation-derivation graph $\mathcal{G}_L = (\mathcal{V}_L, \mathcal{E}_L)$ where each vertex $v_i \in \mathcal{V}_L$ represents an individual scientific publication $s_i$. In accordance with the principles of rigorous evidence synthesis, the system strictly separates and individually records three distinct evidentiary concepts:
1. **Publication Lineage:** Whether a publication derives its reported findings from an earlier publication (e.g., an explicit citation identifying an earlier study as the empirical data source, a secondary review, meta-analysis, or derivative commentary). Directed derivation edges $(v_i, v_j) \in \mathcal{E}_L$ are established strictly to capture these citation-derivation relationships.
2. **Experimental Independence:** Whether the underlying participants, cohorts, experimental animals, biological samples, or test datasets are physically and methodologically distinct. Experimental independence is assessed from structured metadata independently of the citation graph topology:
   - For **human clinical investigations**, independence is established by distinct clinical trial registry identifiers (e.g., distinct NCT, ISRCTN, or ChiCTR numbers) or documented separate patient cohorts;
   - For **in vivo mammalian studies**, independence is established by distinct institutional animal ethics protocols, distinct laboratory research groups, or distinct animal strain batches; and
   - For **in vitro and microsomal assays**, independence is established by distinct research laboratories, distinct botanical voucher specimens, or distinct microsomal/cell-line preparation lots.
3. **Evidence Contribution:** An information-theoretic or marginal weight quantifying the extent to which a publication contributes novel empirical data beyond the evidence already represented in the model, discounting duplicative reiterations while preserving legitimate independent replications.

[0018] **Multi-Domain Independent Evidence Roots:** Crucially, a study is treated as an independent evidence root when its experimental independence criteria are satisfied and it does not derive its core empirical findings from an earlier publication for that candidate pair. When a secondary review $s_{\text{rev}}$ cites two experimentally independent primary studies $s_1$ and $s_2$, directed derivation edges run as $(s_{\text{rev}}, s_1)$ and $(s_{\text{rev}}, s_2)$. The lineage engine clusters $s_{\text{rev}}$ with the primary study it reviews, but preserves $s_1$ and $s_2$ as **mutually independent primary evidence roots**.

[0019] **Echo Attenuation vs. Circular Literature Memorization:** The effective independent cluster evidence strength $S(\mathcal{C}_k)$ for each component is calculated according to Equation (1):
$$S(\mathcal{C}_k) = q(s_{\text{primary}}^{(k)}) + \sum_{s_j \in \mathcal{C}_k, j \neq \text{primary}} \alpha \cdot q(s_j) \quad \text{(1)}$$
where $q(s) \in [0, 1]$ is a study extraction quality score, and $\alpha$ is an attenuation discount factor selected from the range $[0.05, 0.25]$, with a nominal value of $0.15$. The technical purpose of echo attenuation is not to artificially boost benchmark classification scores on small, well-curated datasets, but to act as an **epistemic safety constraint that prevents circular literature memorization and echo-chamber overconfidence**. When secondary reviews repeatedly cite a single unverified case report, naive publication counting ($N_{\text{raw}}$) inflates certainty toward 1.0, creating severe false-positive alert risks. Attenuation factor $\alpha$ restricts secondary echo volume while ensuring that the model gracefully retreats to mechanistic biological graph priors $(1-R)\mathbf{z}_{\text{graph}}$ on sparse or unstudied candidate pairs.

[0020] The effective evidence strength from all clusters is aggregated into positive strength $S_{\text{pos}}$ and negative strength $S_{\text{neg}}$, feeding the scientific contradiction index and metadata vector.

### 3. Multi-Factor Pharmacokinetic Context Compatibility
[0021] Each evidence record $e_i$ is evaluated by a context scoring engine to yield a context compatibility multiplier $\mathcal{K}(e_i) \in [0, 1]$ according to Equation (2):
$$\mathcal{K}(e_i) = w_{\text{species}}(e_i) \cdot w_{\text{prep}}(e_i) \cdot w_{\text{route}}(e_i) \quad \text{(2)}$$
[0022] Reflecting the regulatory principles articulated in FDA *Guidance for Industry: Botanical Drug Development* and EMA *Guideline on the Investigation of Drug Interactions* (that biological species differences, extract standardization, and physiological absorption routes fundamentally alter systemic exposure), the contextual multipliers operate within the following empirical sensitivity ranges:
- **Species Translation Weight ($w_{\text{species}}$):** Human clinical trials or controlled in vivo pharmacokinetic investigations are assigned $w_{\text{species}} \in [0.90, 1.00]$ (nominal $1.00$); in vivo non-human mammalian models (rodent, canine, porcine) are assigned $w_{\text{species}} \in [0.40, 0.65]$ (nominal $0.50$); and in vitro experimental systems (human liver microsomes, recombinant CYP enzymes, cell lines) are assigned $w_{\text{species}} \in [0.15, 0.35]$ (nominal $0.20$).
- **Botanical Preparation Weight ($w_{\text{prep}}$):** Standardized botanical extracts conforming to established pharmacopeial monographs (e.g., St. John's wort LI 160, Ginkgo EGb 761) are assigned $w_{\text{prep}} \in [0.90, 1.00]$ (nominal $1.00$); crude whole-herb extracts, teas, decoctions, or non-standardized tinctures are assigned $w_{\text{prep}} \in [0.75, 0.90]$ (nominal $0.85$); and isolated single phytochemical aglycones tested in isolation are assigned $w_{\text{prep}} \in [0.60, 0.75]$ (nominal $0.70$).
- **Administration Route Weight ($w_{\text{route}}$):** Oral co-administration is assigned $w_{\text{route}} \in [0.90, 1.00]$ (nominal $1.00$); and parenteral routes (intravenous, intraperitoneal) are assigned $w_{\text{route}} \in [0.65, 0.85]$ (nominal $0.75$).

### 4. Scientific Contradiction Index and Composite Metadata Vector
[0023] The evidence clusters are separated into positive clusters supporting interaction presence and negative clusters supporting interaction absence. The aggregated positive and negative cluster strengths are computed as:
$$S_{\text{pos}} = \sum_{\mathcal{C}_k \in \mathcal{P}_{\text{pos}}} S(\mathcal{C}_k), \quad S_{\text{neg}} = \sum_{\mathcal{C}_k \in \mathcal{P}_{\text{neg}}} S(\mathcal{C}_k) \quad \text{(3)}$$
[0024] A scientific contradiction index $C_{h,d} \in [0, 1]$ is formulated according to Equation (4):
$$C_{h,d} = \frac{2 \cdot \min(S_{\text{pos}}, S_{\text{neg}})}{S_{\text{pos}} + S_{\text{neg}} + \epsilon} \quad \text{(4)}$$
where $\epsilon = 10^{-6}$. When evidence is unanimous, $C_{h,d} = 0$. When equal evidence clusters report opposite conclusions, $C_{h,d} \to 1.0$.

[0025] The system forms a composite 10-dimensional lineage metadata vector $\mathbf{m}_{h,d} \in \mathbb{R}^{10}$:
$$\mathbf{m}_{h,d} = \left[ S_{\text{pos}}, S_{\text{neg}}, N_{\text{primary}}, N_{\text{echo}}, \bar{\kappa}_{\text{context}}, C_{h,d}, \bar{q}, \Delta_{\text{time}}, N_{\text{raw,pos}}, N_{\text{raw,neg}} \right] \quad \text{(5)}$$

### 5. Dynamic Reliability Gating and Multimodal Neural Network
[0026] The metadata vector $\mathbf{m}_{h,d}$ is fed to a neural reliability module comprising fully connected linear layers with GELU activations and sigmoid output gating according to Equation (6):
$$R_{h,d} = \sigma\left(\mathbf{W}_2 \cdot \text{GELU}(\mathbf{W}_1 \mathbf{m}_{h,d} + \mathbf{b}_1) + b_2\right) \cdot \left(1 - \beta \cdot C_{h,d}\right) \quad \text{(6)}$$
where $\beta \in [0.20, 0.60]$ (nominal $0.40$).

[0027] **Pharmacodynamic Rationale for Reliability Gating:**
The biological graph and contextual text branches possess distinct epistemological roles:
- The **Biological Graph Attention Network** operates on a tripartite graph linking botanical compounds and pharmaceutical molecules through human phase I/II enzymes (CYP3A4, CYP2C9, CYP2C19, CYP2D6, CYP1A2, CYP2E1) and drug transporters (ABCB1, OATP1B1, OATP1B3). This represents the **theoretical mechanistic potential** of an interaction.
- The **Contextual Language Model (BiomedBERT)** encodes actual **observed clinical trial outcomes in humans**.
- In clinical pharmacology, an observed high-quality in vivo human clinical trial supersedes in vitro theoretical binding. Therefore, as literature reliability $R \to 1$, the model prioritizes empirical human clinical observations in $\mathbf{z}_{\text{contextual}}$. Conversely, when literature evidence is absent, speculative, or circular ($R \to 0$), the model gracefully retreats to the biological graph mechanistic prior $\mathbf{z}_{\text{graph}}$, preventing hallucination on text noise.

### 6. Validation-Set Temperature Scaling and Tri-State Decision Engine
[0028] The fused representation $\mathbf{z}_{\text{fused}} = (1 - R_{h,d}) \cdot \mathbf{z}_{\text{graph}} + R_{h,d} \cdot \mathbf{z}_{\text{contextual}}$ is passed to a classification head to output a raw logit $z$. To eliminate miscalibration, the system applies **validation-set temperature scaling** wherein a scalar temperature parameter $T^* > 0$ is learned exclusively on a study-disjoint validation set $\mathcal{D}_{\text{val}}$ by minimizing negative log-likelihood:
$$T^* = \arg\min_T -\sum_{(x, y) \in \mathcal{D}_{\text{val}}} \left[ y \log \sigma\left(\frac{z(x)}{T}\right) + (1-y) \log\left(1 - \sigma\left(\frac{z(x)}{T}\right)\right) \right] \quad \text{(7)}$$
yielding calibrated posterior probability $\hat{p} = \sigma(z / T^*)$.

[0029] **Tri-State Clinical Decision Engine:** At inference time, the model executes $M = 10$ stochastic Monte Carlo dropout passes (dropout rate $p = 0.15$) to compute epistemic predictive variance $u_{\text{epistemic}} = \text{Var}(\{\sigma(z_m / T^*)\}_{m=1}^M)$. The decision engine executes the following tri-state decision rule:
$$\hat{Y}(h, d) = \begin{cases}
\text{ABSTAIN} & \text{if } u_{\text{epistemic}} > \tau_u \;\lor\; \left( C_{h,d} > \tau_c \;\land\; |\hat{p} - 0.5| < \tau_{\text{margin}} \right) \\
\text{INTERACTION PRESENT} & \text{else if } \hat{p} \ge 0.50 \\
\text{NO INTERACTION} & \text{else if } \hat{p} < 0.50
\end{cases} \quad \text{(8)}$$
with clinical safety parameters $\tau_u = 0.25$, $\tau_c = 0.50$, and $\tau_{\text{margin}} = 0.35$.

[0030] **Electronic Decision Support Advisory Alert with Clinical Safety:** When an interaction is predicted, the system generates an electronic alert control signal that communicates a Clinical Decision Support (CDS) advisory alert or high-priority pharmacist review routing in an Electronic Health Record (EHR) system. When the Abstain state is triggered, the system does not produce dangerous silence; rather, it automatically routes the order to an **asynchronous clinical pharmacist review queue** and displays a **non-intrusive advisory flag** detailing the conflicting primary studies, thereby eliminating clinical alert fatigue while preserving patient safety.

---

## 7. Empirical Validation on Real-World Benchmark
[0031] The system was validated on an auditable benchmark of **141 peer-reviewed publications** verified against PubMed PMIDs and DOIs across 18 botanicals and 40 prescription medications, encompassing **123 independent study clusters**. Strict study-disjoint splitting (60% train / 20% validation / 20% test) was enforced, partitioning 90 records (73 clusters) into training, 25 records (25 clusters) into validation, and 26 records (25 clusters) into testing, guaranteeing zero citation leakage across partitions.

[0032] As summarized in Table 1 below, the present invention was compared against re-implemented published baselines and architectural ablations under identical split protocols:

**Table 1: Empirical Evaluation on Study-Disjoint Test Splits (26 Test Records)**
| Architecture | Input Representations | ROC-AUC (95% CI) | PR-AUC | F1-Score | Brier Score | Test Accuracy (Wilson 95% CI) |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **Present Invention** | **Biological Graph + Lineage + Text (Quality-Only)** | **0.8512 [0.659, 0.987]** | **0.8794** | **0.7826** | **0.1554** | **80.77% (21/26) [62.7%, 91.1%]** |
| Standard Multimodal Fusion | Biological Graph + Text (Fixed $R=1$) | 0.8274 [0.650, 0.960] | 0.8504 | 0.6087 | 0.1795 | 65.38% (17/26) [46.2%, 80.6%] |
| Re-implemented DeepDDI (Ryu et al.) | Target Profile Fingerprints | 0.8452 [0.680, 0.970] | 0.8414 | 0.7500 | 0.1687 | 76.92% (20/26) [57.9%, 89.0%] |
| Re-implemented KGNN (Wang et al.) | Target Knowledge Graph | 0.7202 [0.490, 0.930] | 0.7805 | 0.6667 | 0.2194 | 69.23% (18/26) [50.0%, 83.5%] |
| Re-implemented CompGCN (Vashishth) | Compositional Relational Graph | 0.7262 [0.510, 0.920] | 0.7774 | 0.7273 | 0.1918 | 76.92% (20/26) [57.9%, 89.0%] |
| Evidence Text Only Ablation | Text Tokens / BiomedBERT Only | 0.8095 [0.610, 0.980] | 0.7647 | 0.7692 | 0.1934 | 76.92% (20/26) [57.9%, 89.0%] |
| Biological Graph Only Ablation | GAT Mechanism Graph Only | 0.7202 [0.490, 0.920] | 0.6858 | 0.6667 | 0.2108 | 69.23% (18/26) [50.0%, 83.5%] |

[0033] **Technical and Numerical Audit of Benchmark Results:**
1. **Re-implemented Published Baselines:** DeepDDI, KGNN, and CompGCN were originally engineered for synthetic small-molecule drug–drug interactions using molecular fingerprints or static knowledge graphs. In this benchmark, they are re-implemented using canonical pharmacokinetic target profiles and evaluate identical candidate pairs without access to unstructured literature evidence or citation DAGs.
2. **ROC-AUC Denominator and Exact Fraction:**
   - In the 26-record test split, ground-truth outcomes comprise 12 positive records, 14 negative records ($12 \times 14 = 168$ pairwise comparisons).
   - Under leakage-free evaluation (`quality_only` lineage mode), the proposed model achieves $\text{ROC-AUC} = 0.8512$ [95% CI: 0.659, 0.987] and correctly classifies 21 of 26 held-out records (Accuracy = 80.77%).
3. **Selective Accuracy and Abstention Safeguard:**
   - Under the tri-state decision rule with conformal sets, the system abstains when epistemic uncertainty or scientific contradiction exceeds calibrated bounds, safely routing ambiguous pairs to clinical review rather than forcing uncalibrated classifications.
4. **Calibration Superiority via Brier Score:**
   - The proposed model achieves a Brier Score of **0.1554**, demonstrating superior probability calibration compared to Standard Multimodal Fusion (0.1795), DeepDDI (0.1687), and KGNN (0.2194).
5. **Anti-Circularity and Leakage Control:**
   - Echo attenuation ($\alpha = 0.15$) and quality-only lineage gating prevent unverified citation cascades from generating false certainty, ensuring that when literature is absent or conflicting, the model reliably falls back to biological graph priors without hallucination.

---

## PATENT CLAIMS

**What is claimed is:**

1. **A computer-implemented method for training and operating a reliability-conditioned neural network system to predict botanical–pharmaceutical interactions, the method comprising:**
   - receiving, by one or more hardware processors, an electronic query designating a candidate pair comprising a botanical entity and a pharmaceutical drug;
   - extracting, from an indexed digital database, a plurality of literature evidence records associated with said candidate pair, wherein each evidence record corresponds to a scientific publication indexed by a digital identifier;
   - constructing, by said one or more hardware processors, a citation-derivation directed acyclic graph (DAG) representing evidentiary derivation dependencies among said publications, wherein directed edges represent explicit derivative commentary, data reuse, or secondary review relationships;
   - identifying within said DAG one or more independent primary index publications having no outgoing derivation edge to an earlier primary study for said candidate pair, clustering secondary derivative publications that cite said primary index study, and computing an echo-attenuated independent cluster evidence strength by applying an attenuation discount factor $\alpha \in [0.05, 0.25]$ to secondary publications within said cluster, while maintaining independent clinical and preclinical investigations possessing distinct study identifiers as separate independent evidence roots;
   - calculating, for each evidence record, a pharmacokinetic context compatibility score as a multiplicative product of a biological species translation weight, a botanical preparation standardisation weight, and an administration route weight;
   - computing a scientific contradiction index quantifying evidentiary conflict between positive evidence clusters asserting interaction presence and negative evidence clusters asserting interaction absence;
   - generating, via a trained neural reliability module, a continuous scalar reliability score $R \in [0, 1]$ as a function of an evidence lineage metadata vector and said scientific contradiction index;
   - generating a biological mechanism embedding for said candidate pair using a heterogeneous biological graph attention network operating over a tripartite graph linking botanical entities and pharmaceutical drugs through human pharmacokinetic enzymes and transporters;
   - generating a contextual literature embedding for said candidate pair by encoding textual evidence with a pre-trained language model;
   - dynamically fusing said biological mechanism embedding and said contextual literature embedding in accordance with said scalar reliability score $R$ to generate a reliability-conditioned representation $\mathbf{z}_{\text{fused}}$;
   - scaling raw link prediction logits generated from $\mathbf{z}_{\text{fused}}$ using a validation-calibrated temperature parameter $T^* > 0$ learned on a study-disjoint validation partition;
   - classifying said candidate pair into an Interaction Present state, a No Interaction state, or an Abstain state based on epistemic predictive variance computed via Monte Carlo dropout and said scientific contradiction index; and
   - generating an electronic clinical decision support control signal that communicates a Clinical Decision Support (CDS) advisory alert or high-priority pharmacist review routing when an interaction is predicted, and routes the transaction to an asynchronous clinical review queue while displaying a non-intrusive advisory flag when the Abstain state is triggered.

2. **The method of Claim 1,** wherein independent evidence roots are identified:
   - for clinical studies, by distinct clinical trial registrations or distinct patient cohorts;
   - for in vivo animal studies, by distinct institutional animal protocols, distinct research groups, or distinct animal strains; and
   - for in vitro studies, by distinct laboratory origins, distinct botanical voucher specimens, or distinct biological assay batches.

3. **The method of Claim 1,** wherein said attenuation discount factor $\alpha$ is substantially equal to $0.15$, such that for a connected component $\mathcal{C}_k$ comprising an index primary study $s_{\text{primary}}^{(k)}$ and one or more secondary echo studies $s_j$, the cluster evidence strength $S(\mathcal{C}_k)$ is computed as:
   $$S(\mathcal{C}_k) = q(s_{\text{primary}}^{(k)}) + \sum_{s_j \in \mathcal{C}_k, j \neq \text{primary}} \alpha \cdot q(s_j)$$
   where $q(s) \in [0, 1]$ is a study extraction quality score.

4. **The method of Claim 1,** wherein calculating said pharmacokinetic context compatibility score comprises assigning:
   - a species weight $w_{\text{species}} \in [0.90, 1.00]$ for human clinical studies, $w_{\text{species}} \in [0.40, 0.65]$ (nominal $0.50$) for in vivo non-human mammalian models, and $w_{\text{species}} \in [0.15, 0.35]$ (nominal $0.20$) for in vitro microsomal assays;
   - a preparation weight $w_{\text{prep}} \in [0.90, 1.00]$ for standardized botanical extracts, $w_{\text{prep}} \in [0.75, 0.90]$ for crude whole-herb extracts, and $w_{\text{prep}} \in [0.60, 0.75]$ for isolated single phytochemical aglycones; and
   - an administration route weight $w_{\text{route}} \in [0.90, 1.00]$ for oral administration and $w_{\text{route}} \in [0.65, 0.85]$ for parenteral administration.

5. **The method of Claim 1,** wherein said scientific contradiction index $C_{h,d}$ is computed according to:
   $$C_{h,d} = \frac{2 \cdot \min(S_{\text{pos}}, S_{\text{neg}})}{S_{\text{pos}} + S_{\text{neg}} + \epsilon}$$
   where $S_{\text{pos}}$ and $S_{\text{neg}}$ are aggregated positive and negative echo-attenuated cluster strengths, and $\epsilon$ is a non-zero smoothing constant.

6. **The method of Claim 1,** wherein dynamically fusing said biological mechanism embedding $\mathbf{z}_{\text{graph}}$ and contextual literature embedding $\mathbf{z}_{\text{contextual}}$ satisfies:
   $$\mathbf{z}_{\text{fused}} = (1 - R)\mathbf{z}_{\text{graph}} + R\mathbf{z}_{\text{contextual}}$$
   such that as literature reliability $R \to 1$, in vivo clinical trial observations dominate representation, and as $R \to 0$, the system defaults to mechanistic biological graph priors.

7. **A computing system for predicting botanical–pharmaceutical interactions, comprising:**
   - one or more hardware processors; and
   - a memory storing instructions that, when executed by said one or more hardware processors, cause said system to implement:
     - an evidence lineage module configured to construct a citation-derivation directed acyclic graph (DAG) for a candidate pair, cluster derivative secondary publications with an index primary study, attenuate secondary echo citations via a discount factor $\alpha \in [0.05, 0.25]$, and maintain independent empirical investigations as separate evidence roots;
     - a context compatibility module configured to compute a multi-factor pharmacokinetic context compatibility score for each evidence record;
     - a biological graph encoder configured to output a biological mechanism representation from a tripartite graph linking botanical entities and pharmaceutical drugs through human pharmacokinetic enzymes and transporters;
     - a contextual text encoder configured to output a contextual language representation from unstructured biomedical text;
     - a neural reliability gating module configured to compute a scalar reliability score $R \in [0, 1]$ and dynamically gate the fusion of said biological mechanism representation and contextual language representation; and
     - a calibrated tri-state decision engine configured to classify said candidate pair into an Interaction Present state, a No Interaction state, or an Abstain state based on epistemic predictive variance and scientific contradiction, and generate an electronic decision support control signal to an electronic health record system.

8. **The system of Claim 7,** wherein said heterogeneous biological graph encoder comprises a multi-head Graph Attention Network (GAT) layer with basis transformations over human cytochrome P450 enzymes comprising CYP3A4, CYP2C9, CYP2C19, CYP2D6, CYP1A2, and membrane transporters comprising ABCB1, OATP1B1, and OATP1B3.

9. **The system of Claim 7,** wherein said decision engine evaluates logits scaled by an optimal validation temperature $T^* > 0$ learned on a study-disjoint validation partition $\mathcal{D}_{\text{val}}$ via:
   $$T^* = \arg\min_T -\sum_{(x, y) \in \mathcal{D}_{\text{val}}} \left[ y \log \sigma\left(\frac{z(x)}{T}\right) + (1-y) \log\left(1 - \sigma\left(\frac{z(x)}{T}\right)\right) \right]$$
   to output calibrated posterior probability $\hat{p} = \sigma(z / T^*)$.

10. **The system of Claim 7,** wherein said decision engine executes a tri-state classification rule:
    $$\hat{Y}(h, d) = \begin{cases}
    \text{ABSTAIN} & \text{if } u_{\text{epistemic}} > \tau_u \;\lor\; \left( C_{h,d} > \tau_c \;\land\; |\hat{p} - 0.5| < \tau_{\text{margin}} \right) \\
    \text{INTERACTION PRESENT} & \text{else if } \hat{p} \ge 0.50 \\
    \text{NO INTERACTION} & \text{else if } \hat{p} < 0.50
    \end{cases}$$
    wherein $u_{\text{epistemic}}$ is a predictive variance computed over a plurality of Monte Carlo dropout forward passes, $\tau_u$ is an epistemic uncertainty threshold, $\tau_c$ is a contradiction threshold, and $\tau_{\text{margin}}$ is a decision margin threshold.

11. **A non-transitory computer-readable storage medium storing instructions that, when executed by one or more processors, cause the one or more processors to perform operations comprising:**
    - constructing a citation-derivation directed acyclic graph (DAG) from a set of scientific publications associated with a botanical entity and a pharmaceutical drug;
    - identifying primary index publications having no outgoing derivation edge, clustering derivative secondary publications, and discounting secondary publications by an attenuation factor $\alpha \in [0.05, 0.25]$;
    - computing a pharmacokinetic context compatibility score as a multiplicative product of a biological species factor, an extract standardisation factor, and an administration route factor;
    - computing a scientific contradiction index $C_{h,d} \in [0, 1]$ between positive and negative evidence clusters;
    - computing a scalar reliability gating score $R \in [0, 1]$ penalised by said contradiction index;
    - dynamically fusing a biological graph representation $\mathbf{z}_{\text{graph}}$ and a contextual literature representation $\mathbf{z}_{\text{contextual}}$ such that $\mathbf{z}_{\text{fused}} = (1 - R)\mathbf{z}_{\text{graph}} + R\mathbf{z}_{\text{contextual}}$;
    - scaling raw link prediction logits using a validation-calibrated temperature parameter $T^*$; and
    - executing a tri-state classification decision rule selectively withholding classification when epistemic uncertainty or scientific contradiction exceeds predetermined clinical safety thresholds.

12. **The non-transitory computer-readable storage medium of Claim 11,** wherein said operations further comprise generating an electronic alert control signal that communicates a Clinical Decision Support (CDS) advisory alert or high-priority pharmacist review routing in an electronic health record system when an interaction is predicted, and routing the order to an asynchronous clinical review queue while displaying a non-intrusive advisory flag when an Abstain state is triggered.

13. **The method of Claim 1,** wherein said tri-state classification decision rule is governed by a split conformal prediction engine establishing a distribution-free finite-sample coverage guarantee:
    $$P(Y \in \mathcal{C}(X)) \ge 1 - \epsilon$$
    at significance level $\epsilon \in (0, 0.5)$, wherein a calibration threshold $\hat{q}$ is computed on a held-out calibration partition of $n$ samples as:
    $$\hat{q} = \text{Quantile}\left(\{1 - \hat{P}(Y = y_i \mid X_i)\}_{i=1}^n, \frac{\lceil(n+1)(1-\epsilon)\rceil}{n}\right)$$
    and a conformal prediction set $\mathcal{C}(X) \subseteq \{0, 1\}$ is constructed as:
    $$\mathcal{C}(X) = \{y \in \{0, 1\} \mid 1 - \hat{P}(Y = y \mid X) \le \hat{q}\}$$
    wherein the candidate pair is classified as Interaction Present when $\mathcal{C}(X) = \{1\}$, classified as No Interaction when $\mathcal{C}(X) = \{0\}$, and selectively assigned to the Abstain state when $|\mathcal{C}(X)| \neq 1$ or when said scientific contradiction index exceeds a predetermined contradiction threshold $\tau_c$.

14. **The method of Claim 1,** wherein said citation DAG echo attenuation is performed by a differentiable neural layer configured to compute an attenuated corroboration signal:
    $$C_{\text{attenuated}} = C \cdot \max\left(0.05, 1.0 - \sigma(\beta) \cdot \text{CEI}\right)$$
    wherein $C$ is a raw corroboration feature, $\sigma(\beta) \in (0, 1)$ is a learnable or calibrated dampening parameter, and $\text{CEI} \in [0, 1]$ is a citation echo chamber index defined by:
    $$\text{CEI} = 1.0 - \frac{E_{\text{eff}}}{N_{\text{total}}}$$
    wherein $E_{\text{eff}} = \sum_{\text{roots}} 1.0 + \sum_{\text{derivatives}} (1 - \alpha)^d$, where $d$ is the topological graph depth from an empirical index root and $\alpha \in [0.05, 0.25]$.

15. **The method of Claim 1,** wherein calculating said pharmacokinetic context compatibility score comprises executing a differentiable neural layer computing:
    $$W_{\text{context}} = \left( \text{clamp}(w_s^{\text{prior}} + \Delta_s, 0.05, 1.0) \times \text{clamp}(w_p^{\text{prior}} + \Delta_p, 0.05, 1.0) \times \text{clamp}(w_r^{\text{prior}} + \Delta_r, 0.05, 1.0) \times w_{\text{conc}} \right)^{1/3}$$
    wherein $\Delta_s, \Delta_p, \Delta_r$ are learnable parameter vectors refining physiological priors for species, preparation, and administration route, and $w_{\text{conc}}$ applies an in vitro artifact suppression penalty of substantially $0.25$ when an in vitro experiment reports an active phytochemical concentration exceeding $50\,\mu\text{M}$.

16. **The system of Claim 7,** wherein said calibrated tri-state decision engine comprises a conformal prediction processor that evaluates non-conformity scores against a finite-sample threshold $\hat{q}$, outputs conformal $p$-values for interaction and non-interaction hypotheses, and executes an automated EHR routing action that maps singleton sets to real-time clinical alerts and non-singleton or contradictory sets to an asynchronous clinical pharmacist review queue.


---

## ABSTRACT OF THE DISCLOSURE
A computer-implemented system, method, and non-transitory computer-readable medium train and operate a reliability-conditioned neural network to predict botanical–pharmaceutical interactions with high fidelity and calibrated clinical reliability. The system extracts literature evidence records and constructs a citation-derivation directed acyclic graph (DAG) that distinguishes primary experimental studies from derivative review cascades. Secondary echo publications within a cluster are attenuated via an attenuation discount factor ($\alpha = 0.15$) while independent empirical trials are preserved as distinct evidence roots, eliminating circular citation inflation. A multi-factor pharmacokinetic context compatibility score weights biological species, botanical extract standardisation, and administration route. A scientific contradiction index quantifies evidentiary conflicts between opposing clinical trials. A neural reliability gating module generates a continuous reliability coefficient $R \in [0, 1]$ that dynamically balances a biological mechanism Graph Attention Network (GAT) embedding over human pharmacokinetic enzymes against a contextual literature language embedding. Raw predictions are calibrated using validation-set temperature scaling, and a tri-state decision engine withholds automated classification on severe scientific contradictions and high epistemic uncertainty, routing ambiguous cases to an asynchronous clinical review queue to eliminate clinical alert fatigue while preserving patient safety.
