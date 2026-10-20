import os
import zipfile
from PIL import Image
import pytest
from core.dataset_manager import DatasetManager

@pytest.fixture
def dummy_raw_dataset(tmp_path):
    base_dir = tmp_path / "data"
    for stage, classes in [
        ("stage1", ["plant", "not_plant"]),
        ("stage2", ["invasive", "non_invasive"]),
    ]:
        for cls in classes:
            folder = base_dir / "raw" / stage / cls
            folder.mkdir(parents=True, exist_ok=True)
            for i in range(10):
                img = Image.new("RGB", (800, 600), color=(i * 20, 100, 50))
                img.save(folder / f"img_{i}.jpg")
    return str(base_dir)

def test_dataset_stats_and_downsample(dummy_raw_dataset):
    dm = DatasetManager(base_data_dir=dummy_raw_dataset)
    stats = dm.get_dataset_stats()
    assert stats["stage1"]["plant"] == 10
    assert stats["stage1"]["not_plant"] == 10
    assert stats["stage2"]["invasive"] == 10
    assert stats["stage2"]["non_invasive"] == 10

def test_downsample_single_image(tmp_path):
    dm = DatasetManager(base_data_dir=str(tmp_path))
    src = str(tmp_path / "large.jpg")
    dest = str(tmp_path / "small.jpg")
    img = Image.new("RGB", (1200, 900), color=(255, 0, 0))
    img.save(src)

    success = dm.downsample_and_save_image(src, dest, max_size=448)
    assert success is True
    assert os.path.exists(dest)
    with Image.open(dest) as out_img:
        w, h = out_img.size
        assert max(w, h) <= 448

def test_split_and_training_pack_generation(dummy_raw_dataset):
    dm = DatasetManager(base_data_dir=dummy_raw_dataset)
    result = dm.process_and_split_dataset(test_ratio=0.30, seed=42)
    assert result["stage1"]["train"] == 14
    assert result["stage1"]["test"] == 6

    # Verify locked test directory has images
    test_plant_dir = os.path.join(dummy_raw_dataset, "test_dataset", "stage1", "plant")
    assert os.path.exists(test_plant_dir)
    assert len(os.listdir(test_plant_dir)) == 3

    # Verify training pack zip was created and contains only training samples
    packs = dm.create_student_training_packs()
    assert os.path.exists(packs["stage1"])
    with zipfile.ZipFile(packs["stage1"], "r") as z:
        names = z.namelist()
        assert any("plant/" in n for n in names)
        # Should contain exactly 14 training images
        assert len([n for n in names if not n.endswith("/")]) == 14
