"""Quick smoke test to verify the full pipeline works end-to-end."""
import sys
import os
os.environ["PYTHONIOENCODING"] = "utf-8"
sys.stdout.reconfigure(encoding='utf-8') if hasattr(sys.stdout, 'reconfigure') else None
import torch
sys.path.insert(0, ".")

from src.data.data_pipeline import DataPipeline
from src.models.hdi_model import HDIModel
from src.models.calibration_loss import CalibrationLoss, ExpectedCalibrationError
from src.models.reliability_scorer import ReliabilityScorer
from src.training.contrastive_loss import ReliabilityContrastiveRegularizer
from src.training.statistical_tests import bootstrap_confidence_interval, mcnemars_test
from src.models.temporal_drift import TemporalDriftDetector
from src.data.expanded_corpus import get_corpus_statistics

print("=" * 60)
print("SMOKE TEST: Full Pipeline Verification")
print("=" * 60)

# 1. Data pipeline
print("\n[1] Data Pipeline...")
pipeline = DataPipeline(mode="synthetic", batch_size=32, feature_dim=64)
data = pipeline.build()
print(f"    ✓ {data['graph_data']['node_features'].shape[0]} nodes, "
      f"{data['graph_data']['edge_index'].shape[1]} edges")

# 2. Model instantiation (GNN-only, no text encoder to avoid downloading BERT)
print("\n[2] Model Instantiation...")
model_a = HDIModel.gnn_only(gnn_input_dim=64, gnn_hidden_dim=128, gnn_output_dim=64, num_relations=6)
model_c = HDIModel.full_model(gnn_input_dim=64, gnn_hidden_dim=128, gnn_output_dim=64, num_relations=6)
params_a = sum(p.numel() for p in model_a.parameters())
params_c = sum(p.numel() for p in model_c.parameters())
print(f"    ✓ GNN-only: {params_a:,} params")
print(f"    ✓ Full model: {params_c:,} params")

# 3. Forward pass (GNN-only, no text needed)
print("\n[3] Forward Pass (GNN-only)...")
batch = next(iter(data["train_loader"]))
graph = data["graph_data"]
output = model_a(
    node_features=graph["node_features"],
    edge_index=graph["edge_index"],
    edge_type=graph["edge_type"],
    source_indices=batch["source_indices"],
    target_indices=batch["target_indices"],
    metadata=batch["metadata"],
    labels=batch["labels"],
)
print(f"    ✓ Loss: {output['loss'].item():.4f}")
print(f"    ✓ Predictions shape: {output['probabilities'].shape}")
print(f"    ✓ Fused embeddings shape: {output['fused_embeddings'].shape}")

# 4. Calibration loss
print("\n[4] ARC Calibration Loss...")
cal_loss = CalibrationLoss(lambda_calibration=0.1, lambda_diversity=0.01)
R_fake = torch.rand(32, 1)
preds_fake = output["probabilities"].detach()
labels_fake = batch["labels"]
arc_loss, breakdown = cal_loss(R_fake, preds_fake, labels_fake, batch["metadata"])
print(f"    ✓ ARC loss: {arc_loss.item():.4f}")
print(f"    ✓ Breakdown: cal={breakdown['calibration_loss']:.4f}, "
      f"div={breakdown['diversity_loss']:.4f}, ord={breakdown['ordering_loss']:.4f}")

# 5. Contrastive regularizer
print("\n[5] Contrastive Regularizer...")
contrastive = ReliabilityContrastiveRegularizer(margin=0.2)
c_loss = contrastive(output["fused_embeddings"].detach(), R_fake, labels_fake)
print(f"    ✓ Contrastive loss: {c_loss.item():.4f}")

# 6. Reliability scorer with uncertainty
print("\n[6] MC Dropout Uncertainty...")
scorer = ReliabilityScorer(mc_dropout_rate=0.15, mc_samples=10)
metadata = batch["metadata"][:8]
R_mean, R_uncertainty, mc_breakdown = scorer.forward_with_uncertainty(metadata, n_samples=20)
print(f"    ✓ R_mean: {R_mean.mean().item():.4f}")
print(f"    ✓ R_uncertainty: {R_uncertainty.mean().item():.4f}")
print(f"    ✓ MC samples: {mc_breakdown['mc_samples']}")

# 7. Temporal drift detector
print("\n[7] Temporal Drift Detector...")
detector = TemporalDriftDetector(cusum_threshold=3.0, min_observations=5)
import random
random.seed(42)
for i in range(20):
    r_val = random.gauss(0.7, 0.1)
    detector.update("DB00001", "HERB0001", "Warfarin", "Ashwagandha", r_val)
# Inject a sudden shift
for i in range(10):
    r_val = random.gauss(0.3, 0.1)
    alerts = detector.update("DB00001", "HERB0001", "Warfarin", "Ashwagandha", r_val)
summary = detector.get_interaction_summary("DB00001", "HERB0001")
print(f"    ✓ Observations: {summary['observations']}")
print(f"    ✓ R trend: {summary['R_trend']:.4f}")
print(f"    ✓ Total alerts: {len(detector.get_all_alerts())}")

# 8. Statistical tests
print("\n[8] Statistical Testing...")
import numpy as np
np.random.seed(42)
y_true = np.random.randint(0, 2, 100)
y_pred_a = np.random.rand(100)
y_pred_b = y_pred_a + 0.05 * np.random.rand(100)

def accuracy(y_t, y_p):
    return ((y_p > 0.5).astype(int) == y_t).mean()

est, ci_lo, ci_hi = bootstrap_confidence_interval(y_true, y_pred_b, accuracy, n_bootstrap=1000)
print(f"    ✓ Accuracy: {est:.4f} [{ci_lo:.4f}, {ci_hi:.4f}]")

chi2, p = mcnemars_test(y_true, (y_pred_a > 0.5).astype(int), (y_pred_b > 0.5).astype(int))
print(f"    ✓ McNemar's: χ²={chi2:.4f}, p={p:.4f}")

# 9. Corpus stats
print("\n[9] Code-Mixed Corpus...")
stats = get_corpus_statistics()
print(f"    ✓ Sentences: {stats['total_sentences']}")
print(f"    ✓ Entities: {stats['total_entities']}")
print(f"    ✓ Relations: {stats['total_relations']}")
print(f"    ✓ Entity types: {stats['entity_types']}")
print(f"    ✓ Scripts: {stats['scripts']}")
print(f"    ✓ Sources: {stats['sources']}")

# 10. ECE monitor
print("\n[10] Expected Calibration Error...")
ece_monitor = ExpectedCalibrationError(n_bins=5)
ece_val, bin_stats = ece_monitor(R_fake[:32], preds_fake[:32], labels_fake[:32])
print(f"    ✓ ECE: {ece_val:.4f}")
print(f"    ✓ Bins with data: {sum(1 for b in bin_stats if b['count'] > 0)}/5")

print("\n" + "=" * 60)
print("ALL SMOKE TESTS PASSED ✓")
print("=" * 60)
