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

def test_resolve_class_indices_exact_and_inverted_order():
    evaluator = ModelEvaluator()
    # Case 1: Standard order
    pos_idx, neg_idx = evaluator.resolve_class_indices(["plant", "not_plant"], stage="stage1")
    assert pos_idx == 0
    assert neg_idx == 1

    # Case 2: Inverted order (not_plant first) - must NOT misidentify not_plant as plant!
    pos_idx, neg_idx = evaluator.resolve_class_indices(["not_plant", "plant"], stage="stage1")
    assert pos_idx == 1
    assert neg_idx == 0

    # Case 3: Stage 2 inverted order (non_invasive first)
    pos_idx, neg_idx = evaluator.resolve_class_indices(["non_invasive", "invasive"], stage="stage2")
    assert pos_idx == 1
    assert neg_idx == 0

def test_unpack_ignores_macosx_appledouble(tmp_path):
    evaluator = ModelEvaluator()
    zip_buf = io.BytesIO()
    with zipfile.ZipFile(zip_buf, "w") as z:
        # Add AppleDouble corrupt metadata
        z.writestr("__MACOSX/._model.tflite", b"junk_metadata")
        z.writestr("__MACOSX/._labels.txt", b"junk_metadata")
        # Add real model and labels
        z.writestr("model.tflite", b"real_tflite_bytes")
        z.writestr("labels.txt", "0 plant\n1 not_plant\n")

    extract_dir = str(tmp_path / "extracted")
    model_path, labels = evaluator.unpack_model_zip(zip_buf.getvalue(), extract_dir)
    assert not os.path.basename(model_path).startswith("._")
    assert "__MACOSX" not in model_path
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

def test_fix_h5_depthwise_conv_strips_groups(tmp_path):
    import h5py
    import json

    h5_file = str(tmp_path / "test_model.h5")
    mock_config = {
        "class_name": "Sequential",
        "config": {
            "layers": [
                {
                    "class_name": "DepthwiseConv2D",
                    "config": {
                        "name": "expanded_conv_depthwise",
                        "groups": 1,
                        "depth_multiplier": 1,
                        "kernel_size": [3, 3],
                    },
                },
                {
                    "class_name": "Conv2D",
                    "config": {
                        "name": "conv2d_1",
                        "groups": 1,
                    },
                },
            ]
        },
    }

    with h5py.File(h5_file, "w") as f:
        f.attrs["model_config"] = json.dumps(mock_config)

    # Apply the patch
    ModelEvaluator.fix_h5_depthwise_conv(h5_file)

    # Verify that 'groups' was removed from DepthwiseConv2D but preserved elsewhere
    with h5py.File(h5_file, "r") as f:
        saved_config = json.loads(f.attrs["model_config"])
        layers = saved_config["config"]["layers"]
        assert "groups" not in layers[0]["config"]
        assert layers[0]["config"]["depth_multiplier"] == 1
        assert layers[1]["config"]["groups"] == 1
