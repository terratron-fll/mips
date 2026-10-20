import os
import json
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
    si1 = runner.get_system_instruction("stage1")
    si2 = runner.get_system_instruction("stage2")
    assert "botanical" in si1.lower()
    assert "plant" in si1.lower()
    assert "invasive" in si2.lower()
