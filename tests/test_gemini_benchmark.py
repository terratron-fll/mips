import os
import json
from PIL import Image
import pytest
from core.gemini_benchmark import GeminiBenchmarkRunner

def test_cache_loading_and_saving(tmp_path):
    cache_path = str(tmp_path / "gemini_cache.json")
    runner = GeminiBenchmarkRunner(cache_file=cache_path)
    mock_results = {
        "stage1": {
            "metrics": {
                "accuracy": 0.92,
                "precision": 0.90,
                "recall": 0.95,
                "f1": 0.924,
                "tp": 19,
                "fp": 2,
                "tn": 18,
                "fn": 1,
            }
        }
    }
    runner.save_cache(mock_results)
    assert os.path.exists(cache_path)
    loaded = runner.load_cache()
    assert loaded["stage1"]["metrics"]["f1"] == 0.924

def test_system_instructions_present():
    runner = GeminiBenchmarkRunner()
    assert runner.model_name == "gemini-3.5-flash-lite"
    assert set(runner.BENCHMARK_PRESETS.keys()) == {"flash_lite", "flash", "decision"}
    assert runner.BENCHMARK_PRESETS["flash_lite"]["model"] == "gemini-3.5-flash-lite"
    assert runner.BENCHMARK_PRESETS["flash"]["model"] == "gemini-3.8-flash"
    assert runner.BENCHMARK_PRESETS["decision"]["model"] == "gemini-decision-1"
    si1 = runner.get_system_instruction("stage1")
    si2 = runner.get_system_instruction("stage2")
    assert "botanical" in si1.lower()
    assert "plant" in si1.lower()
    assert "invasive" in si2.lower()


def test_get_stage_benchmarks_multi_and_legacy(tmp_path):
    cache_path = str(tmp_path / "gemini_cache.json")
    runner = GeminiBenchmarkRunner(cache_file=cache_path)

    # Multi-preset cache
    runner.save_cache({
        "stage1": {
            "flash_lite": {"model_name": "Gemini 3.5 Flash-Lite", "metrics": {"f1": 0.9}},
            "flash": {"model_name": "Gemini 3.8 Flash", "metrics": {"f1": 0.94}},
            "decision": {"model_name": "Gemini Decision 1", "metrics": {"f1": 0.91}},
        },
        # Legacy single-entry cache
        "stage2": {"metrics": {"f1": 0.88}},
    })

    s1 = runner.get_stage_benchmarks("stage1")
    assert [e["model_name"] for e in s1] == [
        "Gemini 3.5 Flash-Lite",
        "Gemini 3.8 Flash",
        "Gemini Decision 1",
    ]
    s2 = runner.get_stage_benchmarks("stage2")
    assert len(s2) == 1
    assert s2[0]["metrics"]["f1"] == 0.88


def test_classify_with_decisions_sdk_mock():
    class FakeAnswer:
        choice = "invasive"
        confidence = 0.96

    class FakeResult:
        answers = {"classification": FakeAnswer()}

    class FakeDecisions:
        def create(self, model, input, questions):
            assert model == "gemini-decision-1"
            assert questions["classification"]["type"] == "choice"
            assert any(item.get("type") == "image" for item in input)
            return FakeResult()

    class FakeClient:
        decisions = FakeDecisions()

    runner = GeminiBenchmarkRunner(api_key="test-key")
    img = Image.new("RGB", (32, 32), color=(0, 128, 0))
    label, conf, reasoning = runner._classify_with_decisions(
        FakeClient(), "gemini-decision-1", img, "stage2"
    )
    assert label == "invasive"
    assert conf == 0.96
    assert "Decisions API" in reasoning
