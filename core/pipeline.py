import os
from typing import Any, Dict, List, Tuple, Union
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
            "accuracy": round(acc, 4),
            "precision": round(prec, 4),
            "recall": round(rec, 4),
            "f1": round(f1, 4),
            "pipeline_accuracy": round(acc, 4),
            "pipeline_precision": round(prec, 4),
            "pipeline_recall": round(rec, 4),
            "pipeline_f1": round(f1, 4),
            "tp": tp,
            "fp": fp,
            "tn": tn,
            "fn": fn,
            "invasive_caught": tp,
            "false_alarms": fp,
            "missed_invasives": fn,
            "correct_ignores": tn,
        }

    def evaluate_pipeline(
        self,
        stage1_zip_or_path: Union[bytes, str],
        stage2_zip_or_path: Union[bytes, str],
    ) -> Dict[str, Any]:
        """Runs the complete cascaded evaluation of both models on the test sets."""
        res1 = self.evaluator.evaluate_submission(stage1_zip_or_path, stage="stage1")
        res2 = self.evaluator.evaluate_submission(stage2_zip_or_path, stage="stage2")

        m1 = res1["metrics"]
        m2 = res2["metrics"]

        # Stage 1: Plant filter accuracy and true positives
        # Stage 2: Invasive classifier on detected plants
        # Combined end-to-end metrics:
        # Precision: Probability that a flagged weed is actually an invasive weed
        # Recall: Probability that an actual invasive weed on the field is detected
        combined_prec = (m1["precision"] * m2["precision"]) if (m1["precision"] + m2["precision"]) > 0 else 0.0
        combined_rec = (m1["recall"] * m2["recall"]) if (m1["recall"] + m2["recall"]) > 0 else 0.0
        combined_f1 = (
            (2 * combined_prec * combined_rec) / (combined_prec + combined_rec)
            if (combined_prec + combined_rec) > 0
            else 0.0
        )
        combined_acc = round((m1["accuracy"] + m2["accuracy"]) / 2.0, 4)

        pipeline_metrics = {
            "accuracy": combined_acc,
            "precision": round(combined_prec, 4),
            "recall": round(combined_rec, 4),
            "f1": round(combined_f1, 4),
            "stage1_f1": m1["f1"],
            "stage2_f1": m2["f1"],
            "stage1_acc": m1["accuracy"],
            "stage2_acc": m2["accuracy"],
            "tp": m2["tp"],
            "fp": m2["fp"],
            "tn": m1["tn"] + m2["tn"],
            "fn": m2["fn"],
        }

        return {
            "metrics": pipeline_metrics,
            "stage1_result": res1,
            "stage2_result": res2,
            "errors": res1["errors"] + res2["errors"],
        }
