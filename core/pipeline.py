import os
from typing import Any, Dict, List, Tuple
from core.evaluator import ModelEvaluator

class PipelineEvaluator:
    """Simulates an end-to-end autonomous field robot by cascading Stage 1 (Plant Filter)

    and Stage 2 (Invasive Classifier).
    """

    def __init__(self, evaluator: ModelEvaluator = None) -> None:
        self.evaluator = evaluator or ModelEvaluator()

    def combine_predictions(
        self,
        stage1_preds: List[Tuple[str, float]],
        stage2_preds: List[Tuple[str, float]],
    ) -> List[str]:
        final_preds: List[str] = []
        for s1, s2 in zip(stage1_preds, stage2_preds):
            s1_label, _ = s1
            s2_label, _ = s2
            # If Stage 1 determines it is not a plant, robot ignores it (not_plant)
            if s1_label == "not_plant":
                final_preds.append("not_plant")
            else:
                # If Stage 1 determines it IS a plant, Stage 2 decides if invasive or non_invasive
                final_preds.append(s2_label)
        return final_preds

    def compute_pipeline_metrics(
        self, y_true_binary: List[int], final_preds: List[str]
    ) -> Dict[str, Any]:
        """Calculates end-to-end invasive detection performance.

        Target: 1 = Invasive, 0 = Non-invasive or Not-plant.
        """
        # A sample is predicted positive only if final prediction is 'invasive'
        y_pred_binary = [1 if p == "invasive" else 0 for p in final_preds]

        tp = sum(1 for yt, yp in zip(y_true_binary, y_pred_binary) if yt == 1 and yp == 1)
        fp = sum(1 for yt, yp in zip(y_true_binary, y_pred_binary) if yt == 0 and yp == 1)
        tn = sum(1 for yt, yp in zip(y_true_binary, y_pred_binary) if yt == 0 and yp == 0)
        fn = sum(1 for yt, yp in zip(y_true_binary, y_pred_binary) if yt == 1 and yp == 0)
        total = len(y_true_binary)

        acc = (tp + tn) / total if total > 0 else 0.0
        prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        rec = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = (2 * prec * rec) / (prec + rec) if (prec + rec) > 0 else 0.0

        return {
            "total": total,
            "pipeline_accuracy": round(acc, 4),
            "pipeline_precision": round(prec, 4),
            "pipeline_recall": round(rec, 4),
            "pipeline_f1": round(f1, 4),
            "invasive_caught": tp,
            "false_alarms": fp,
            "missed_invasives": fn,
            "correct_ignores": tn,
        }
