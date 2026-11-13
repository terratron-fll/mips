import json
from core.dataset_config import DatasetConfig, StageConfig
from core.evaluator import ModelEvaluator
from core.gemini_benchmark import GeminiBenchmarkRunner
from core.pipeline import PipelineEvaluator
from core.r2_sync import FakeS3Client, R2SyncManager


def test_model_evaluator_resolves_custom_dataset_labels(tmp_path):
    ds_cfg = DatasetConfig(
        id="waste_sorter",
        name="Waste Sorter",
        stages={
            "stage1": StageConfig(
                stage_id="stage1",
                name="Recyclable vs Trash",
                pos_label="recyclable",
                neg_label="trash",
            )
        },
    )
    evaluator = ModelEvaluator(test_dataset_dir=str(tmp_path), dataset_config=ds_cfg)
    pos_idx, neg_idx = evaluator.resolve_class_indices(["trash", "recyclable"], "stage1")
    assert (pos_idx, neg_idx) == (1, 0)


def test_pipeline_evaluator_accepts_test_dataset_dir(tmp_path):
    pe = PipelineEvaluator(test_dataset_dir=str(tmp_path))
    assert pe.evaluator.test_dataset_dir == str(tmp_path)


def test_gemini_benchmark_runner_per_dataset_cache_and_r2_push(tmp_path):
    s3 = FakeS3Client()
    sync = R2SyncManager(s3_client=s3, bucket="mips", local_data_dir=str(tmp_path))

    ds_cfg = DatasetConfig(
        id="waste_sorter",
        name="Waste Sorter",
        split_version=2,
        stages={
            "stage1": StageConfig(
                stage_id="stage1",
                name="Recyclable vs Trash",
                pos_label="recyclable",
                neg_label="trash",
            )
        },
    )
    runner = GeminiBenchmarkRunner(
        cache_dir=str(tmp_path / "gemini_cache"),
        dataset_config=ds_cfg,
        r2_sync=sync,
    )
    assert "recyclable" in runner.get_default_prompt("stage1", api_type="generate_content")
    spec = runner.get_decision_spec("stage1")
    assert "recyclable" in spec["criteria"] and "trash" in spec["criteria"]

    runner.save_stage_cache("stage1", {"flash": {"stage": "stage1", "split_version": 2, "metrics": {"f1": 0.91}}})
    assert "gemini_cache/waste_sorter/stage1.json" in s3.objects

    # Should not collide with invasive_plants stage1
    benches = runner.get_stage_benchmarks("stage1")
    assert len(benches) == 1
    assert benches[0]["metrics"]["f1"] == 0.91
