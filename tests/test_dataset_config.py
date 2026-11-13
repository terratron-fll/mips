import json
import pytest
from core.dataset_config import StageConfig, DatasetConfig, DatasetRegistry
from core.r2_sync import FakeS3Client, R2SyncManager


def test_default_presets_seeded_and_loaded(tmp_path):
    ds_dir = str(tmp_path / "datasets")
    registry = DatasetRegistry(datasets_dir=ds_dir)
    datasets = registry.list_datasets()
    ids = {d.id for d in datasets}
    assert "invasive_plants" in ids
    assert "tutorial_cat" in ids

    invasive = registry.get_dataset("invasive_plants")
    assert invasive is not None
    assert invasive.has_pipeline is True
    assert invasive.stages["stage1"].classes == ["plant", "not_plant"]
    assert invasive.stages["stage1"].targets == {"plant": 1, "not_plant": 0}


def test_infer_dataset_id_for_legacy_stages():
    assert DatasetRegistry.infer_dataset_id("intro") == "tutorial_cat"
    assert DatasetRegistry.infer_dataset_id("stage1") == "invasive_plants"
    assert DatasetRegistry.infer_dataset_id("stage2") == "invasive_plants"
    assert DatasetRegistry.infer_dataset_id("pipeline") == "invasive_plants"


def test_slug_validation_and_duplicate_rejection(tmp_path):
    registry = DatasetRegistry(datasets_dir=str(tmp_path / "datasets"))
    assert DatasetRegistry.slugify("Waste Sorting!") == "waste_sorting"
    assert DatasetRegistry.validate_slug("waste_sorter-1") is True
    assert DatasetRegistry.validate_slug("../bad") is False
    assert DatasetRegistry.validate_slug("has space") is False

    custom = DatasetConfig(
        id="waste_sorter",
        name="Waste Sorting",
        icon="♻️",
        description="Recyclable vs trash",
        stages={
            "stage1": StageConfig(
                stage_id="stage1",
                name="Recyclable vs Trash",
                pos_label="recyclable",
                neg_label="trash",
            )
        },
    )
    registry.register_dataset(custom)

    with pytest.raises(ValueError, match="already exists"):
        registry.register_dataset(custom)


def test_register_dataset_persists_per_dataset_config_and_pushes_to_r2(tmp_path):
    s3 = FakeS3Client()
    sync = R2SyncManager(s3_client=s3, bucket="mips", local_data_dir=str(tmp_path))
    registry = DatasetRegistry(datasets_dir=str(tmp_path / "datasets"), r2_sync=sync)

    custom = DatasetConfig(
        id="pollinators",
        name="Pollinators",
        icon="🐝",
        description="Bee vs wasp",
        stages={
            "stage1": StageConfig(
                stage_id="stage1",
                name="Bee vs Wasp",
                pos_label="bee",
                neg_label="wasp",
            )
        },
    )
    registry.register_dataset(custom)

    cfg_path = tmp_path / "datasets" / "pollinators" / "config.json"
    assert cfg_path.exists()
    assert "datasets/pollinators/config.json" in s3.objects

    registry2 = DatasetRegistry(datasets_dir=str(tmp_path / "datasets"))
    loaded = registry2.get_dataset("pollinators")
    assert loaded is not None
    assert loaded.stages["stage1"].pos_label == "bee"
