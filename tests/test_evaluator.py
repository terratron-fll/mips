import io
import os
import zipfile
import numpy as np
from PIL import Image
import pytest
from core.evaluator import ModelEvaluator

def test_compute_binary_metrics_exact():
    evaluator = ModelEvaluator()
    # 5 samples: 3 actual positives (1), 2 actual negatives (0)
    # y_pred: TP=2, FN=1, TN=2, FP=0
    y_true = [1, 1, 1, 0, 0]
    y_pred = [1, 1, 0, 0, 0]
    metrics = evaluator.compute_binary_metrics(
        y_true, y_pred, class_names=["negative", "positive"]
    )

    assert metrics["tp"] == 2
    assert metrics["fn"] == 1
    assert metrics["tn"] == 2
    assert metrics["fp"] == 0
    assert metrics["accuracy"] == 0.80
    assert metrics["precision"] == 1.0
    assert pytest.approx(metrics["recall"], 0.01) == 0.667
    assert pytest.approx(metrics["f1"], 0.01) == 0.80

def test_compute_binary_metrics_division_by_zero_safe():
    evaluator = ModelEvaluator()
    # All predictions are negative: TP=0, FP=0
    y_true = [1, 1, 0, 0]
    y_pred = [0, 0, 0, 0]
    metrics = evaluator.compute_binary_metrics(
        y_true, y_pred, class_names=["negative", "positive"]
    )
    assert metrics["tp"] == 0
    assert metrics["fp"] == 0
    assert metrics["precision"] == 0.0
    assert metrics["recall"] == 0.0
    assert metrics["f1"] == 0.0

def test_sanitize_labels_handling():
    evaluator = ModelEvaluator()
    raw_labels = ["0 Plant\n", "1 Not Plant\r\n"]
    cleaned = evaluator.sanitize_labels(raw_labels)
    assert cleaned == ["plant", "not_plant"]

def test_unpack_nested_model_zip(tmp_path):
    evaluator = ModelEvaluator()
    zip_buf = io.BytesIO()
    with zipfile.ZipFile(zip_buf, "w") as z:
        # Put inside a nested directory
        z.writestr("my_model_folder/labels.txt", "0 plant\n1 not_plant\n")
        z.writestr("my_model_folder/model.tflite", b"dummy_tflite_bytes")

    extract_dir = str(tmp_path / "extracted")
    model_path, labels = evaluator.unpack_model_zip(zip_buf.getvalue(), extract_dir)
    assert os.path.exists(model_path)
    assert model_path.endswith("model.tflite")
    assert labels == ["plant", "not_plant"]

def test_preprocess_image(tmp_path):
    evaluator = ModelEvaluator()
    img_path = str(tmp_path / "sample.jpg")
    img = Image.new("RGB", (300, 300), color=(128, 128, 128))
    img.save(img_path)

    arr = evaluator.preprocess_image(img_path, target_size=(224, 224))
    assert arr.shape == (1, 224, 224, 3)
    assert -1.01 <= np.min(arr) <= 1.01
    assert -1.01 <= np.max(arr) <= 1.01
