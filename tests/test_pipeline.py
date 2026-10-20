import pytest
from core.pipeline import PipelineEvaluator

def test_pipeline_decision_logic():
    # End-to-end logic test:
    # If Stage 1 says 'not_plant', Stage 2 is skipped (predicted as not_plant / harmless)
    # If Stage 1 says 'plant', Stage 2 makes the final invasive call
    pe = PipelineEvaluator()
    # Image 1: Not a plant -> robot ignores it (safe)
    # Image 2: Plant -> Invasive -> robot flags it as invasive
    # Image 3: Plant -> Non-invasive -> robot ignores it
    stage1_preds = [("not_plant", 0.95), ("plant", 0.88), ("plant", 0.92)]
    stage2_preds = [("invasive", 0.70), ("invasive", 0.85), ("non_invasive", 0.90)]

    final_labels = pe.combine_predictions(stage1_preds, stage2_preds)
    assert final_labels[0] == "not_plant"
    assert final_labels[1] == "invasive"
    assert final_labels[2] == "non_invasive"

def test_pipeline_metrics_calculation():
    pe = PipelineEvaluator()
    # Target: invasive detection in the wild
    # Ground truth: Image 1 is not_plant (0), Image 2 is invasive (1), Image 3 is non_invasive (0)
    y_true = [0, 1, 0]
    final_preds = ["not_plant", "invasive", "non_invasive"]
    metrics = pe.compute_pipeline_metrics(y_true, final_preds)

    assert metrics["total"] == 3
    assert metrics["invasive_caught"] == 1
    assert metrics["false_alarms"] == 0
    assert metrics["missed_invasives"] == 0
    assert metrics["pipeline_accuracy"] == 1.0
