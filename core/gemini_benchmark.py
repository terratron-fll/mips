import base64
import io
import json
import os
import re
import urllib.request
from typing import Any, Dict, List, Optional, Tuple
from PIL import Image


class GeminiBenchmarkRunner:
    """Evaluates benchmark reference scores using Gemini Flash-Lite, Flash, and Decisions API."""

    BENCHMARK_PRESETS: Dict[str, Dict[str, str]] = {
        "flash_lite": {
            "label": "Gemini 3.5 Flash-Lite",
            "model": "gemini-3.5-flash-lite",
            "api_type": "generate_content",
            "version": "Zero-Shot VLM (SI)",
        },
        "flash": {
            "label": "Gemini 3.8 Flash",
            "model": "gemini-3.8-flash",
            "api_type": "generate_content",
            "version": "Zero-Shot VLM (SI)",
        },
        "decision": {
            "label": "Gemini Decision 1",
            "model": "gemini-decision-1",
            "api_type": "decisions",
            "version": "Decisions API (Zero Output Tokens)",
        },
    }

    SYSTEM_INSTRUCTIONS: Dict[str, str] = {
        "intro": (
            "You are an expert image classifier. Identify whether the given image contains "
            "a cat or if it is not a cat (e.g. dog, other animal, object, or background). "
            "Respond ONLY with a valid JSON object formatted as: "
            "{\"label\": \"cat\" | \"not_cat\", \"confidence\": float (0.0 to 1.0), \"reasoning\": string}"
        ),
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

    DECISION_CRITERIA: Dict[str, Dict[str, Any]] = {
        "intro": {
            "instructions": "Classify whether this image shows a cat or not a cat.",
            "criteria": {
                "cat": "Image contains a cat (domestic cat or feline).",
                "not_cat": "Image does not contain a cat (e.g., dog, other animal, object, or scenery).",
            },
        },
        "stage1": {
            "instructions": "Classify whether this image shows a real plant or a non-plant object.",
            "criteria": {
                "plant": "Image contains a plant, weed, tree, grass, or natural plant foliage.",
                "not_plant": "Image shows a non-plant object such as tools, shoes, soil, rocks, or artificial items.",
            },
        },
        "stage2": {
            "instructions": "Classify whether the plant in this image is an invasive species or a non-invasive plant.",
            "criteria": {
                "invasive": "An invasive weed or invasive plant species.",
                "non_invasive": "A native, harmless, or non-invasive plant.",
            },
        },
    }

    def __init__(
        self,
        cache_file: str = "data/gemini_cache.json",
        api_key: Optional[str] = None,
        model_name: str = "gemini-3.5-flash-lite",
    ) -> None:
        self.cache_file = cache_file
        self.api_key = api_key or os.environ.get("GEMINI_API_KEY")
        if not self.api_key:
            try:
                import streamlit as st
                self.api_key = st.secrets.get("GEMINI_API_KEY") or None
            except Exception:
                pass
        self.model_name = model_name

    STAGE_CLASSES: Dict[str, Tuple[str, str]] = {
        "intro": ("cat", "not_cat"),
        "stage1": ("plant", "not_plant"),
        "stage2": ("invasive", "non_invasive"),
    }

    def get_system_instruction(self, stage: str) -> str:
        return self.SYSTEM_INSTRUCTIONS.get(stage, "")

    def get_default_prompt(self, stage: str, api_type: str = "generate_content") -> str:
        """Return the formatted default prompt/instructions for a stage and API type."""
        if api_type == "decisions":
            spec = self.DECISION_CRITERIA.get(stage, {})
            criteria_lines = "\n".join(
                f"- {k}: {v}" for k, v in spec.get("criteria", {}).items()
            )
            return f"{spec.get('instructions', '')}\n\nCriteria:\n{criteria_lines}".strip()
        return self.get_system_instruction(stage)

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

    def get_stage_benchmarks(self, stage: str) -> List[Dict[str, Any]]:
        """Return cached benchmark entries for a stage (supports both legacy single-entry and multi-preset caches)."""
        cache = self.load_cache()
        stage_data = cache.get(stage)
        if not stage_data:
            return []
        if "metrics" in stage_data:
            return [stage_data]
        ordered_keys = [k for k in self.BENCHMARK_PRESETS if k in stage_data]
        extra_keys = [k for k in stage_data if k not in self.BENCHMARK_PRESETS]
        return [stage_data[k] for k in ordered_keys + extra_keys if isinstance(stage_data[k], dict)]

    def _classify_with_generate_content(
        self, client: Any, types: Any, model: str, pil_img: Image.Image, sys_instruction: str
    ) -> Tuple[str, float, str]:
        response = client.models.generate_content(
            model=model,
            contents=["Classify this image:", pil_img],
            config=types.GenerateContentConfig(
                system_instruction=sys_instruction,
                response_mime_type="application/json",
            ),
        )
        text_resp = response.text or "{}"
        json_match = re.search(r"\{.*\}", text_resp, re.DOTALL)
        json_str = json_match.group(0) if json_match else "{}"
        data = json.loads(json_str)
        return (
            str(data.get("label", "")).lower(),
            float(data.get("confidence", 0.90)),
            str(data.get("reasoning", "")),
        )

    def _classify_with_decisions(
        self,
        client: Any,
        model: str,
        pil_img: Image.Image,
        stage: str,
        instructions_override: Optional[str] = None,
    ) -> Tuple[str, float, str]:
        buf = io.BytesIO()
        pil_img.convert("RGB").save(buf, format="JPEG", quality=90)
        img_b64 = base64.b64encode(buf.getvalue()).decode("ascii")

        spec = self.DECISION_CRITERIA[stage]
        questions = {
            "classification": {
                "type": "choice",
                "instructions": instructions_override or spec["instructions"],
                "criteria": spec["criteria"],
            }
        }
        input_payload = [
            {"type": "text", "text": "Classify this image."},
            {
                "type": "image",
                "inline_data": {
                    "mime_type": "image/jpeg",
                    "data": img_b64,
                },
            },
        ]

        if hasattr(client, "decisions"):
            result = client.decisions.create(
                model=model,
                input=input_payload,
                questions=questions,
            )
            answer = result.answers["classification"]
            if isinstance(answer, dict):
                pred_label = str(answer.get("choice", answer.get("answer", ""))).lower()
                conf = float(answer.get("confidence", 0.90))
            else:
                pred_label = str(getattr(answer, "choice", getattr(answer, "answer", ""))).lower()
                conf = float(getattr(answer, "confidence", 0.90))
            return pred_label, conf, "Decisions API choice"

        # REST fallback for older google-genai versions (< 3.2)
        payload = json.dumps({
            "model": model,
            "input": input_payload,
            "questions": questions,
        }).encode("utf-8")
        req = urllib.request.Request(
            "https://generativelanguage.googleapis.com/v1beta/decisions",
            data=payload,
            headers={
                "x-goog-api-key": self.api_key or "",
                "Content-Type": "application/json",
            },
            method="POST",
        )
        with urllib.request.urlopen(req, timeout=30) as resp:
            body = json.loads(resp.read().decode("utf-8"))
        answer = body.get("answers", {}).get("classification", {})
        pred_label = str(answer.get("choice", answer.get("answer", ""))).lower()
        conf = float(answer.get("confidence", 0.90))
        return pred_label, conf, "Decisions API choice"

    def evaluate_stage_benchmark(
        self,
        stage: str,
        test_images_dir: str = "data/test_dataset",
        preset: str = "flash_lite",
        custom_prompt: Optional[str] = None,
    ) -> Dict[str, Any]:
        if not self.api_key:
            raise ValueError(
                "GEMINI_API_KEY is not set. Please set GEMINI_API_KEY in .streamlit/secrets.toml or your environment."
            )
        if stage not in self.STAGE_CLASSES:
            raise ValueError(f"Unknown stage: {stage}")
        if preset not in self.BENCHMARK_PRESETS:
            raise ValueError(f"Unknown Gemini benchmark preset: {preset}")

        preset_cfg = self.BENCHMARK_PRESETS[preset]
        model = preset_cfg["model"]
        api_type = preset_cfg["api_type"]

        from google import genai
        from google.genai import types

        client_kwargs: Dict[str, Any] = {"api_key": self.api_key}
        if api_type == "decisions":
            client_kwargs["http_options"] = {"api_version": "v1beta"}
        client = genai.Client(**client_kwargs)

        default_prompt = self.get_default_prompt(stage, api_type=api_type)
        prompt_used = (custom_prompt or "").strip() or default_prompt
        sys_instruction = (custom_prompt or "").strip() or self.get_system_instruction(stage)
        pos_class, neg_class = self.STAGE_CLASSES[stage]

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
                    with Image.open(img_path) as pil_img:
                        if api_type == "decisions":
                            pred_label, conf, reasoning = self._classify_with_decisions(
                                client,
                                model,
                                pil_img,
                                stage,
                                instructions_override=(custom_prompt or "").strip() or None,
                            )
                        else:
                            pred_label, conf, reasoning = self._classify_with_generate_content(
                                client, types, model, pil_img, sys_instruction
                            )

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
                            "reasoning": reasoning,
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
            "preset": preset,
            "model_name": preset_cfg["label"],
            "model_id": model,
            "version": preset_cfg["version"],
            "prompt": prompt_used,
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

        # Save to cache under cache[stage][preset]
        cache = self.load_cache()
        existing_stage = cache.get(stage)
        if not isinstance(existing_stage, dict) or "metrics" in existing_stage:
            existing_stage = {}
        existing_stage[preset] = benchmark_result
        cache[stage] = existing_stage
        self.save_cache(cache)

        return benchmark_result
