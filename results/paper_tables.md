# Experimental Results — HDI Prediction

**Generated**: 2026-09-07 20:13:35
**Data mode**: synthetic
**Seeds**: [42, 123, 456]
**Epochs**: 30
**Gating mode**: multiplicative

## Table 1: Ablation Study Results

| Variant | AUC-ROC | AUC-PR | Precision | Recall | F1 | MRR | Hits@10 | Params |
|---------|---------|--------|-----------|--------|-----|-----|---------|--------|
| (a) GNN-only | 0.7982±0.0053 | 0.3691±0.0219 | 0.5404±0.3456 | 0.0292±0.0236 | 0.0526±0.0405 | 0.0298±0.0031 | 0.0292±0.0156 | 1,018,369 |
| (b) Unconditioned | 1.0000±0.0000 | 1.0000±0.0000 | 1.0000±0.0000 | 1.0000±0.0000 | 1.0000±0.0000 | 0.0621±0.0000 | 0.1250±0.0000 | 1,199,617 |
| (c) **Full R-conditioned (Ours)** | 1.0000±0.0000 | 1.0000±0.0000 | 1.0000±0.0000 | 1.0000±0.0000 | 1.0000±0.0000 | 0.0621±0.0000 | 0.1250±0.0000 | 1,208,359 |

## Table 2: Critical Comparison — (b) vs (c)

The (b)-vs-(c) gap is the evidentiary core of the paper and patent.

| Metric | Unconditioned (b) | Full Model (c) | Δ (c−b) | Improvement |
|--------|-------------------|----------------|---------|-------------|
| auc_roc | 1.0000 | 1.0000 | ✗ +0.0000 | +0.0% |
| auc_pr | 1.0000 | 1.0000 | ✗ -0.0000 | -0.0% |
| f1 | 1.0000 | 1.0000 | ✗ +0.0000 | +0.0% |
| mrr | 0.0621 | 0.0621 | ✗ +0.0000 | +0.0% |
| hits@10 | 0.1250 | 0.1250 | ✗ +0.0000 | +0.0% |

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
