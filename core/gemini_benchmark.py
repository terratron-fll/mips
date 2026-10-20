import json
import os
import re
from typing import Any, Dict, List, Optional
from PIL import Image

class GeminiBenchmarkRunner:
    """Evaluates benchmark reference scores using Gemini 2.5 Flash and botanical System Instructions."""

    SYSTEM_INSTRUCTIONS: Dict[str, str] = {
        "stage1": (
            "You are an expert botanical image classifier. Identify whether the given image contains "
            "a plant or natural plant foliage, or if it is a non-plant object (e.g. tools, shoes, ground, rocks, artificial items). "
            "Respond ONLY with a valid JSON object formatted as: "
            "{\"label\": \"plant\" | \"not_plant\", \"confidence\": float (0.0 to 1.0), \"reasoning\": string}"
        ),
        "stage2": (
            "You are an expert invasive species botanist. Identify whether the plant shown in the image "
            "is an invasive weed/plant species or a non-invasive (native/harmless) plant. "
            "Respond ONLY with a valid JSON object formatted as: "
            "{\"label\": \"invasive\" | \"non_invasive\", \"confidence\": float (0.0 to 1.0), \"reasoning\": string}"
        ),
    }

    def __init__(
        self,
        cache_file: str = "data/gemini_cache.json",
        api_key: Optional[str] = None,
        model_name: str = "gemini-2.5-flash",
    ) -> None:
        self.cache_file = cache_file
        self.api_key = api_key or os.environ.get("GEMINI_API_KEY")
        self.model_name = model_name

    def get_system_instruction(self, stage: str) -> str:
        return self.SYSTEM_INSTRUCTIONS.get(stage, "")

    def load_cache(self) -> Dict[str, Any]:
        if os.path.exists(self.cache_file):
            try:
                with open(self.cache_file, "r", encoding="utf-8") as f:
                    return json.load(f)
            except Exception:
                return {}
        return {}

    def save_cache(self, data: Dict[str, Any]) -> None:
        parent = os.path.dirname(self.cache_file)
        if parent:
            os.makedirs(parent, exist_ok=True)
        with open(self.cache_file, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)

    def evaluate_stage_benchmark(
        self, stage: str, test_images_dir: str = "data/test_dataset"
    ) -> Dict[str, Any]:
        if not self.api_key:
            raise ValueError(
                "GEMINI_API_KEY is not set. Please set the GEMINI_API_KEY environment variable or Streamlit secret."
            )

        from google import genai
        from google.genai import types

        client = genai.Client(api_key=self.api_key)
        sys_instruction = self.get_system_instruction(stage)
        pos_class = "plant" if stage == "stage1" else "invasive"
        neg_class = "not_plant" if stage == "stage1" else "non_invasive"

        stage_dir = os.path.join(test_images_dir, stage)
        if not os.path.exists(stage_dir):
            raise FileNotFoundError(f"Test directory not found: {stage_dir}")

        y_true: List[int] = []
        y_pred: List[int] = []
        errors: List[Dict[str, Any]] = []

        for cls_name, true_binary in [(neg_class, 0), (pos_class, 1)]:
            cls_folder = os.path.join(stage_dir, cls_name)
            if not os.path.exists(cls_folder):
                continue

            for fname in sorted(os.listdir(cls_folder)):
                if not fname.lower().endswith((".jpg", ".jpeg", ".png", ".webp")):
                    continue
                img_path = os.path.join(cls_folder, fname)
                try:
                    pil_img = Image.open(img_path)
                    response = client.models.generate_content(
                        model=self.model_name,
                        contents=[
                            types.Content(
                                role="user",
                                parts=[
                                    types.Part.from_text(text="Classify this image:"),
                                    pil_img,
                                ],
                            )
                        ],
                        config=types.GenerateContentConfig(
                            system_instruction=sys_instruction,
                            response_mime_type="application/json",
                            temperature=0.0,
                        ),
                    )

                    text_resp = response.text or "{}"
                    # Clean any markdown code blocks
                    text_resp = re.sub(r"^```json\s*", "", text_resp.strip())
                    text_resp = re.sub(r"\s*```$", "", text_resp.strip())
                    data = json.loads(text_resp)

                    pred_label = data.get("label", "").lower()
                    conf = float(data.get("confidence", 0.90))
                    pred_binary = 1 if pred_label == pos_class else 0

                    y_true.append(true_binary)
                    y_pred.append(pred_binary)

                    if pred_binary != true_binary:
                        errors.append({
                            "filename": fname,
                            "image_path": img_path,
                            "true_label": cls_name,
                            "predicted_label": pred_label,
                            "confidence": round(conf, 4),
                            "reasoning": data.get("reasoning", ""),
                            "error_type": "False Positive" if pred_binary == 1 else "False Negative",
                        })
                except Exception:
                    continue

        tp = sum(1 for yt, yp in zip(y_true, y_pred) if yt == 1 and yp == 1)
        fp = sum(1 for yt, yp in zip(y_true, y_pred) if yt == 0 and yp == 1)
        tn = sum(1 for yt, yp in zip(y_true, y_pred) if yt == 0 and yp == 0)
        fn = sum(1 for yt, yp in zip(y_true, y_pred) if yt == 1 and yp == 0)
        total = len(y_true)

        acc = (tp + tn) / total if total > 0 else 0.0
        prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        rec = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = (2 * prec * rec) / (prec + rec) if (prec + rec) > 0 else 0.0

        benchmark_result = {
            "stage": stage,
            "metrics": {
                "accuracy": round(acc, 4),
                "precision": round(prec, 4),
                "recall": round(rec, 4),
                "f1": round(f1, 4),
                "tp": tp,
                "fp": fp,
                "tn": tn,
                "fn": fn,
                "total_samples": total,
            },
            "errors": errors,
        }

        # Save to cache
        cache = self.load_cache()
        cache[stage] = benchmark_result
        self.save_cache(cache)

        return benchmark_result
