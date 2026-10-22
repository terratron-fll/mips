import gc
import io
import os
import re
import tempfile
import zipfile
from typing import Any, Dict, List, Optional, Tuple, Union
import numpy as np
from PIL import Image

class ModelEvaluator:
    """Evaluates student Google Teachable Machine models on locked test datasets."""

    STAGE_TARGETS: Dict[str, Dict[str, int]] = {
        # class_name: binary_int (1 = positive/target class, 0 = negative/noise class)
        "intro": {"cat": 1, "not_cat": 0},
        "stage1": {"plant": 1, "not_plant": 0},
        "stage2": {"invasive": 1, "non_invasive": 0},
    }

    def __init__(self, test_dataset_dir: str = "data/test_dataset") -> None:
        self.test_dataset_dir = test_dataset_dir

    def sanitize_labels(self, raw_lines: List[str]) -> List[str]:
        cleaned: List[str] = []
        for line in raw_lines:
            text = line.strip()
            if not text:
                continue
            # Remove leading numbers/indexes like "0 ", "1.", "0 - "
            text = re.sub(r"^\d+[\s\.\-_:]*", "", text)
            # Normalize whitespace and separators to underscores and lowercase
            text = re.sub(r"[\s\-]+", "_", text).lower()
            cleaned.append(text)
        return cleaned

    def unpack_model_zip(
        self, zip_bytes_or_path: Union[bytes, str], extract_to: str
    ) -> Tuple[str, List[str]]:
        os.makedirs(extract_to, exist_ok=True)
        if isinstance(zip_bytes_or_path, bytes):
            with zipfile.ZipFile(io.BytesIO(zip_bytes_or_path), "r") as z:
                z.extractall(extract_to)
        else:
            with zipfile.ZipFile(zip_bytes_or_path, "r") as z:
                z.extractall(extract_to)

        # Recursively search for model file (.h5, .keras, or .tflite) and labels.txt
        # Ignore macOS AppleDouble metadata (__MACOSX and ._ files)
        model_path: Optional[str] = None
        labels_path: Optional[str] = None

        for root, _, files in os.walk(extract_to):
            if "__macosx" in root.lower():
                continue
            for f in files:
                if f.startswith("._"):
                    continue
                f_lower = f.lower()
                if f_lower.endswith((".h5", ".keras", ".tflite")):
                    if model_path is None or f_lower.endswith((".h5", ".keras")):
                        model_path = os.path.join(root, f)
                elif f_lower == "labels.txt":
                    labels_path = os.path.join(root, f)

        if not model_path:
            raise ValueError(
                "No valid model file (.h5, .keras, or .tflite) found in the zip archive."
            )
        if not labels_path:
            raise ValueError("No 'labels.txt' file found in the zip archive.")

        with open(labels_path, "r", encoding="utf-8", errors="ignore") as f:
            raw_lines = f.readlines()
        labels = self.sanitize_labels(raw_lines)
        if not labels:
            raise ValueError("'labels.txt' was empty or contained no valid class names.")

        return model_path, labels

    def resolve_class_indices(self, labels: List[str], stage: str) -> Tuple[int, int]:
        if stage not in self.STAGE_TARGETS:
            raise ValueError(f"Unknown stage: {stage}")

        stage_mapping = self.STAGE_TARGETS[stage]
        pos_class = [k for k, v in stage_mapping.items() if v == 1][0]
        neg_class = [k for k, v in stage_mapping.items() if v == 0][0]

        pos_idx: Optional[int] = None
        neg_idx: Optional[int] = None

        # 1. Exact match first
        for idx, label in enumerate(labels):
            if label == pos_class:
                pos_idx = idx
            elif label == neg_class:
                neg_idx = idx

        # 2. Check negative prefix first (not_, non_) to avoid substring collision
        if pos_idx is None or neg_idx is None:
            for idx, label in enumerate(labels):
                if label.startswith("not_") or label.startswith("non_") or "not" in label or "non" in label:
                    if neg_idx is None:
                        neg_idx = idx
                elif pos_class in label or any(p in label for p in ["yes", "pos", "target"]):
                    if pos_idx is None:
                        pos_idx = idx

        # 3. Two-class fallback if labels were customized
        if (pos_idx is None or neg_idx is None) and len(labels) == 2:
            neg_idx = 0 if any(n in labels[0] for n in ["not", "non", "no", "neg"]) else 1
            pos_idx = 1 if neg_idx == 0 else 0

        if pos_idx is None or neg_idx is None or pos_idx == neg_idx:
            raise ValueError(
                f"Could not map model labels {labels} to stage '{stage}' classes ('{pos_class}', '{neg_class}')."
            )

        return pos_idx, neg_idx

    def preprocess_image(
        self, image_path: str, target_size: Tuple[int, int] = (224, 224)
    ) -> np.ndarray:
        with Image.open(image_path) as img:
            img = img.convert("RGB")
            img = img.resize(target_size, Image.Resampling.LANCZOS)
            arr = np.asarray(img, dtype=np.float32)
            # Normalize to [-1, 1] as expected by standard Teachable Machine MobileNet
            arr = (arr / 127.5) - 1.0
            return np.expand_dims(arr, axis=0)

    def compute_binary_metrics(
        self,
        y_true: List[int],
        y_pred: List[int],
        class_names: Optional[List[str]] = None,
    ) -> Dict[str, Any]:
        tp = sum(1 for yt, yp in zip(y_true, y_pred) if yt == 1 and yp == 1)
        fp = sum(1 for yt, yp in zip(y_true, y_pred) if yt == 0 and yp == 1)
        tn = sum(1 for yt, yp in zip(y_true, y_pred) if yt == 0 and yp == 0)
        fn = sum(1 for yt, yp in zip(y_true, y_pred) if yt == 1 and yp == 0)
        total = len(y_true)

        accuracy = (tp + tn) / total if total > 0 else 0.0
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = (
            (2 * precision * recall) / (precision + recall)
            if (precision + recall) > 0
            else 0.0
        )

        return {
            "accuracy": round(float(accuracy), 4),
            "precision": round(float(precision), 4),
            "recall": round(float(recall), 4),
            "f1": round(float(f1), 4),
            "tp": int(tp),
            "fp": int(fp),
            "tn": int(tn),
            "fn": int(fn),
            "total_samples": int(total),
        }

    @staticmethod
    def fix_h5_depthwise_conv(model_path: str) -> None:
        """Removes incompatible 'groups' parameter from DepthwiseConv2D configs in

        Google Teachable Machine .h5 exports when loaded in modern Keras / TensorFlow.
        """
        try:
            import h5py
            import json

            with h5py.File(model_path, "r+") as f:
                config_key = None
                if "model_config" in f.attrs:
                    config_key = "model_config"
                elif "config" in f.attrs:
                    config_key = "config"

                if not config_key:
                    return

                raw_config = f.attrs[config_key]
                if isinstance(raw_config, bytes):
                    raw_config = raw_config.decode("utf-8")

                model_config = json.loads(raw_config)

                def clean_node(node: Any) -> None:
                    if isinstance(node, dict):
                        if node.get("class_name") == "DepthwiseConv2D":
                            if "config" in node and isinstance(node["config"], dict):
                                node["config"].pop("groups", None)
                        for val in node.values():
                            clean_node(val)
                    elif isinstance(node, list):
                        for item in node:
                            clean_node(item)

                clean_node(model_config)
                f.attrs[config_key] = json.dumps(model_config)
        except Exception:
            pass

    def _load_predictor(self, model_path: str):
        if model_path.endswith(".tflite"):
            try:
                import tensorflow as tf
                interpreter = tf.lite.Interpreter(model_path=model_path)
            except Exception:
                from tflite_runtime import interpreter as tflite
                interpreter = tflite.Interpreter(model_path=model_path)
            interpreter.allocate_tensors()
            input_details = interpreter.get_input_details()
            output_details = interpreter.get_output_details()

            def predict(arr: np.ndarray) -> np.ndarray:
                # Support quantized models if uint8 expected
                if input_details[0]["dtype"] == np.uint8:
                    arr_q = ((arr + 1.0) * 127.5).astype(np.uint8)
                    interpreter.set_tensor(input_details[0]["index"], arr_q)
                else:
                    interpreter.set_tensor(input_details[0]["index"], arr)
                interpreter.invoke()
                return interpreter.get_tensor(output_details[0]["index"])

            return predict
        else:
            import tensorflow as tf

            # 1. Patch H5 model config on disk to strip 'groups': 1 from DepthwiseConv2D
            self.fix_h5_depthwise_conv(model_path)

            # 2. Provide custom DepthwiseConv2D layer that ignores 'groups'
            custom_objects = {}
            try:
                class PatchedDepthwiseConv2D(tf.keras.layers.DepthwiseConv2D):
                    def __init__(self, *args, **kwargs):
                        kwargs.pop("groups", None)
                        super().__init__(*args, **kwargs)

                custom_objects["DepthwiseConv2D"] = PatchedDepthwiseConv2D
            except Exception:
                pass

            try:
                model = tf.keras.models.load_model(
                    model_path, compile=False, custom_objects=custom_objects
                )
            except Exception:
                model = tf.keras.models.load_model(model_path, compile=False)

            return lambda arr: model.predict(arr, verbose=0)

    def evaluate_submission(
        self, zip_bytes_or_path: Union[bytes, str], stage: str
    ) -> Dict[str, Any]:
        if stage not in self.STAGE_TARGETS:
            raise ValueError(f"Unknown stage: {stage}. Expected one of {list(self.STAGE_TARGETS.keys())}")

        stage_mapping = self.STAGE_TARGETS[stage]
        pos_class = [k for k, v in stage_mapping.items() if v == 1][0]
        neg_class = [k for k, v in stage_mapping.items() if v == 0][0]

        try:
            with tempfile.TemporaryDirectory() as temp_dir:
                model_path, labels = self.unpack_model_zip(zip_bytes_or_path, temp_dir)
                predict_fn = self._load_predictor(model_path)

                pos_idx, neg_idx = self.resolve_class_indices(labels, stage)

                # Gather test samples
                y_true: List[int] = []
                y_pred: List[int] = []
                errors: List[Dict[str, Any]] = []

                for cls_name, true_binary in [(neg_class, 0), (pos_class, 1)]:
                    cls_folder = os.path.join(self.test_dataset_dir, stage, cls_name)
                    if not os.path.exists(cls_folder):
                        continue

                    for fname in sorted(os.listdir(cls_folder)):
                        if not fname.lower().endswith((".jpg", ".jpeg", ".png", ".webp")):
                            continue
                        img_path = os.path.join(cls_folder, fname)
                        try:
                            arr = self.preprocess_image(img_path)
                            preds = predict_fn(arr)[0]
                            pred_label_idx = int(np.argmax(preds))
                            confidence = float(np.max(preds))
                            pred_binary = 1 if pred_label_idx == pos_idx else 0

                            y_true.append(true_binary)
                            y_pred.append(pred_binary)

                            if pred_binary != true_binary:
                                errors.append({
                                    "filename": fname,
                                    "image_path": img_path,
                                    "true_label": cls_name,
                                    "predicted_label": pos_class if pred_binary == 1 else neg_class,
                                    "confidence": round(confidence, 4),
                                    "error_type": "False Positive" if pred_binary == 1 else "False Negative",
                                })
                        except Exception:
                            continue

                if not y_true:
                    raise ValueError(
                        f"No test images found in '{self.test_dataset_dir}/{stage}'. Please run Dataset Admin to create test splits first."
                    )

                metrics = self.compute_binary_metrics(y_true, y_pred, class_names=[neg_class, pos_class])
                errors.sort(key=lambda x: x["confidence"], reverse=True)

                return {
                    "stage": stage,
                    "labels": labels,
                    "metrics": metrics,
                    "errors": errors[:10],
                }
        finally:
            # Prevent TensorFlow memory leak on repeated submissions
            try:
                import tensorflow as tf
                tf.keras.backend.clear_session()
            except Exception:
                pass
            gc.collect()
