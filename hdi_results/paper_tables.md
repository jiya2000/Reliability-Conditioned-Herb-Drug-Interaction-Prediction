# Experimental Results — HDI Prediction

**Generated**: 2026-10-09 20:39:04
**Data mode**: synthetic
**Seeds**: [42, 123, 456]
**Epochs**: 30
**Gating mode**: multiplicative

## Table 1: Ablation Study Results

| Variant | AUC-ROC | AUC-PR | Precision | Recall | F1 | MRR | Hits@10 | Params |
|---------|---------|--------|-----------|--------|-----|-----|---------|--------|
| (a) GNN-only | 0.7568±0.0072 | 0.3163±0.0020 | 0.3645±0.1423 | 0.0917±0.0328 | 0.1388±0.0383 | 0.6043±0.0217 | 1.0000±0.0000 | 1,018,369 |
| (b) Unconditioned | 0.7554±0.0014 | 0.3137±0.0016 | 0.3379±0.0639 | 0.0583±0.0118 | 0.0974±0.0127 | 0.6152±0.0097 | 1.0000±0.0000 | 1,199,617 |
| (c) **Full R-conditioned (Ours)** | 0.7538±0.0025 | 0.3071±0.0114 | 0.3587±0.0906 | 0.0583±0.0156 | 0.1003±0.0265 | 0.5772±0.0102 | 1.0000±0.0000 | 1,208,359 |

## Table 2: Critical Comparison — (b) vs (c)

The (b)-vs-(c) gap is the evidentiary core of the paper and patent.

| Metric | Unconditioned (b) | Full Model (c) | Δ (c−b) | Improvement |
|--------|-------------------|----------------|---------|-------------|
| auc_roc | 0.7554 | 0.7538 | ✗ -0.0017 | -0.2% |
| auc_pr | 0.3137 | 0.3071 | ✗ -0.0067 | -2.1% |
| f1 | 0.0974 | 0.1003 | ✓ +0.0029 | +3.0% |
| mrr | 0.6152 | 0.5772 | ✗ -0.0380 | -6.2% |
| hits@10 | 1.0000 | 1.0000 | ✗ +0.0000 | +0.0% |

## Table 3: Statistical Significance

- **AUC-ROC gap (c−b)**: -0.0017
- **Direction**: unconditioned > full
- **Per-seed differences**: [-0.0009062499999999973, 0.0018437500000000329, -0.005906250000000002]

## Table 4: Model Configuration

| Setting | Value |
|---------|-------|
| Data mode | synthetic |
| Epochs | 30 |
| Learning rate | 0.0001 |
| Batch size | 64 |
| Hidden dim | 128 |
| Attention heads | 4 |
| Gating mode | multiplicative |
| Seeds | [42, 123, 456] |
