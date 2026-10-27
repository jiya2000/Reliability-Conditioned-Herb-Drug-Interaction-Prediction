"""
HDI Prediction — Training & Evaluation Module
"""

from src.training.trainer import HDITrainer
from src.training.evaluator import HDIEvaluator
from src.training.ablation import AblationRunner
from src.training.calibration import (
    TemperatureScaler,
    compute_calibration_metrics,
    evaluate_abstention,
    CalibrationMetrics,
    AbstentionEvaluationResult,
)
from src.training.evaluation import (
    LeakageSafeSplitter,
    compute_comprehensive_metrics,
    ComprehensiveEvaluationReport,
    RepeatedNestedCVResult,
    format_ablation_markdown_table,
    format_repeated_cv_markdown_table,
    run_repeated_nested_cv,
    compare_models_significance,
)

__all__ = [
    "HDITrainer",
    "HDIEvaluator",
    "AblationRunner",
    "TemperatureScaler",
    "compute_calibration_metrics",
    "evaluate_abstention",
    "CalibrationMetrics",
    "AbstentionEvaluationResult",
    "LeakageSafeSplitter",
    "compute_comprehensive_metrics",
    "ComprehensiveEvaluationReport",
    "RepeatedNestedCVResult",
    "format_ablation_markdown_table",
    "format_repeated_cv_markdown_table",
    "run_repeated_nested_cv",
    "compare_models_significance",
]

