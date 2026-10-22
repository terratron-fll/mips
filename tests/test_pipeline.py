import os
from PIL import Image
import pytest
from core.evaluator import ModelEvaluator
from core.pipeline import PipelineEvaluator

def test_pipeline_decision_logic():
    pe = PipelineEvaluator()
    stage1_preds = [("not_plant", 0.95), ("plant", 0.88), ("plant", 0.92)]
    stage2_preds = [("invasive", 0.70), ("invasive", 0.85), ("non_invasive", 0.90)]

    final_labels = pe.combine_predictions(stage1_preds, stage2_preds)
    assert final_labels[0] == "not_plant"
    assert final_labels[1] == "invasive"
    assert final_labels[2] == "non_invasive"

def test_pipeline_metrics_calculation():
    pe = PipelineEvaluator()
    y_true = [0, 1, 0]
    final_preds = ["not_plant", "invasive", "non_invasive"]
    metrics = pe.compute_pipeline_metrics(y_true, final_preds)

    assert metrics["total"] == 3
    assert metrics["invasive_caught"] == 1
    assert metrics["false_alarms"] == 0
    assert metrics["missed_invasives"] == 0
    assert metrics["pipeline_accuracy"] == 1.0
    assert "precision" in metrics
    assert "recall" in metrics
    assert "f1" in metrics
    assert "accuracy" in metrics
