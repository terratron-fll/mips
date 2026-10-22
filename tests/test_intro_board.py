import os
import zipfile
import pytest
from PIL import Image
from core.dataset_manager import DatasetManager
from core.evaluator import ModelEvaluator
from scripts.seed_intro_data import seed_cat_dataset

def test_dataset_manager_intro_stage():
    dm = DatasetManager(base_data_dir="data")
    assert "intro" in dm.STAGES
    assert dm.STAGES["intro"] == ["cat", "not_cat"]

def test_evaluator_intro_stage_labels():
    evaluator = ModelEvaluator()
    assert "intro" in evaluator.STAGE_TARGETS
    assert evaluator.STAGE_TARGETS["intro"] == {"cat": 1, "not_cat": 0}

    # Verify label resolution order
    pos_idx, neg_idx = evaluator.resolve_class_indices(["cat", "not_cat"], stage="intro")
    assert pos_idx == 0
    assert neg_idx == 1

    # Inverted order: not_cat first, cat second
    pos_idx, neg_idx = evaluator.resolve_class_indices(["not_cat", "cat"], stage="intro")
    assert pos_idx == 1
    assert neg_idx == 0

def test_seed_intro_data_and_split(tmp_path):
    intro_raw_dir = str(tmp_path / "raw" / "intro")
    res = seed_cat_dataset(target_dir=intro_raw_dir, count_per_class=5)
    assert res["cat_saved"] == 5
    assert res["not_cat_saved"] == 5

    cat_files = os.listdir(os.path.join(intro_raw_dir, "cat"))
    assert len(cat_files) == 5

    # Run DatasetManager splitting on this tmp_path
    dm = DatasetManager(base_data_dir=str(tmp_path))
    stats = dm.get_dataset_stats()
    assert stats["intro"]["cat"] == 5
    assert stats["intro"]["not_cat"] == 5

    split_res = dm.process_and_split_dataset(test_ratio=0.40)
    assert split_res["intro"]["train"] == 6
    assert split_res["intro"]["test"] == 4

    packs = dm.create_student_training_packs()
    assert "intro" in packs
    assert os.path.exists(packs["intro"])
    with zipfile.ZipFile(packs["intro"], "r") as z:
        names = z.namelist()
        assert any("cat/" in n for n in names)
        assert any("not_cat/" in n for n in names)
