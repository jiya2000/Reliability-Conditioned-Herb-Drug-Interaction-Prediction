# Experimental Results — HDI Prediction

**Generated**: 2026-09-08 18:18:18
**Data mode**: synthetic
**Seeds**: [42]
**Epochs**: 3
**Gating mode**: multiplicative

## Table 1: Ablation Study Results

| Variant | AUC-ROC | AUC-PR | Precision | Recall | F1 | MRR | Hits@10 | Params |
|---------|---------|--------|-----------|--------|-----|-----|---------|--------|
| (a) GNN-only | 0.7602±0.0000 | 0.3121±0.0000 | 0.0000±0.0000 | 0.0000±0.0000 | 0.0000±0.0000 | 0.0214±0.0000 | 0.0375±0.0000 | 1,018,369 |
| (b) Unconditioned | 0.7409±0.0000 | 0.2953±0.0000 | 0.0000±0.0000 | 0.0000±0.0000 | 0.0000±0.0000 | 0.0198±0.0000 | 0.0375±0.0000 | 1,199,617 |
| (c) **Full R-conditioned (Ours)** | 0.7395±0.0000 | 0.2873±0.0000 | 0.0000±0.0000 | 0.0000±0.0000 | 0.0000±0.0000 | 0.0190±0.0000 | 0.0375±0.0000 | 1,208,359 |

## Table 2: Critical Comparison — (b) vs (c)

The (b)-vs-(c) gap is the evidentiary core of the paper and patent.

| Metric | Unconditioned (b) | Full Model (c) | Δ (c−b) | Improvement |
|--------|-------------------|----------------|---------|-------------|
| auc_roc | 0.7409 | 0.7395 | ✗ -0.0014 | -0.2% |
| auc_pr | 0.2953 | 0.2873 | ✗ -0.0080 | -2.7% |
| f1 | 0.0000 | 0.0000 | ✗ +0.0000 | +0.0% |
| mrr | 0.0198 | 0.0190 | ✗ -0.0008 | -3.8% |
| hits@10 | 0.0375 | 0.0375 | ✗ +0.0000 | +0.0% |

## Table 3: Statistical Significance

- **AUC-ROC gap (c−b)**: -0.0014
- **Direction**: unconditioned > full
- **Per-seed differences**: [-0.001375000000000015]

## Table 4: Model Configuration

| Setting | Value |
|---------|-------|
| Data mode | synthetic |
| Epochs | 3 |
| Learning rate | 0.0001 |
| Batch size | 64 |
| Hidden dim | 128 |
| Attention heads | 4 |
| Gating mode | multiplicative |
| Seeds | [42] |
