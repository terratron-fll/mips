import io
import json
import os
from PIL import Image

from core.dataset_config import DatasetConfig, StageConfig
from core.dataset_manager import DatasetManager
from core.r2_sync import FakeS3Client, R2SyncManager
from core.storage import StorageManager


def _make_jpeg_bytes(color=(120, 200, 80), size=(1200, 900)) -> bytes:
    buf = io.BytesIO()
    Image.new("RGB", size, color=color).save(buf, format="JPEG")
    return buf.getvalue()


def test_storage_r2_push_pull_and_dataset_filtering(tmp_path):
    s3 = FakeS3Client()
    dir_a = tmp_path / "a"
    dir_b = tmp_path / "b"
    sync_a = R2SyncManager(s3_client=s3, bucket="mips", local_data_dir=str(dir_a))
    sync_b = R2SyncManager(s3_client=s3, bucket="mips", local_data_dir=str(dir_b))

    st_a = StorageManager(
        db_path=str(dir_a / "leaderboard_history.json"),
        models_dir=str(dir_a / "models"),
        r2_sync=sync_a,
    )
    st_b = StorageManager(
        db_path=str(dir_b / "leaderboard_history.json"),
        models_dir=str(dir_b / "models"),
        r2_sync=sync_b,
    )

    rec1 = st_a.save_run({"team_name": "Alpha", "stage": "stage1", "metrics": {"f1": 0.9}})
    assert rec1["dataset_id"] == "invasive_plants"
    assert rec1["split_version"] == 1
    st_a.save_model_artifact(rec1["id"], b"zip-bytes-1")

    st_a.save_run({"team_name": "Beta", "stage": "stage1", "dataset_id": "waste_sorter", "metrics": {"f1": 0.85}})

    # Sync B from R2
    sync_b.pull_metadata(db_path=str(dir_b / "leaderboard_history.json"))
    assert len(st_b.get_leaderboard("stage1", dataset_id="invasive_plants")) == 1
    assert len(st_b.get_leaderboard("stage1", dataset_id="waste_sorter")) == 1
    assert len(st_b.get_history(dataset_id="waste_sorter")) == 1

    # Lazy download model artifact on B
    dl = st_b.get_model_artifact_bytes(rec1["id"])
    assert dl is not None
    assert dl[1] == b"zip-bytes-1"


def test_dataset_manager_append_only_split_and_force_resplit(tmp_path):
    s3 = FakeS3Client()
    sync = R2SyncManager(s3_client=s3, bucket="mips", local_data_dir=str(tmp_path))
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
    mgr = DatasetManager(base_data_dir=str(tmp_path), dataset_config=ds_cfg, r2_sync=sync)

    # Upload 4 recyclable and 4 trash images (with same filename to test uuid8 prefixing)
    files = [("IMG_0001.jpg", _make_jpeg_bytes()) for _ in range(4)]
    saved_pos, _ = mgr.save_uploaded_images("stage1", "recyclable", files)
    saved_neg, _ = mgr.save_uploaded_images("stage1", "trash", files)
    assert saved_pos == 4
    assert saved_neg == 4

    # Verify raw photos were downsampled and pushed to R2 (no train_dataset pushed)
    raw_keys = [k for k in s3.objects if k.startswith("datasets/waste_sorter/raw/")]
    assert len(raw_keys) == 8

    mgr.process_and_split_dataset(test_ratio=0.3, seed=42, force_resplit=False)
    manifest_path = os.path.join(mgr.dataset_root, "split_manifest.json")
    manifest1 = json.loads(open(manifest_path).read())
    assert manifest1["split_version"] == 1
    first_assignments = dict(manifest1["assignments"])
    assert len(first_assignments) == 8
    assert not any(k.startswith("datasets/waste_sorter/train_dataset/") for k in s3.objects)
    assert "datasets/waste_sorter/training_packs/stage1_train.zip" in s3.objects

    # Add 2 more images and run append-only split: existing 8 assignments MUST NOT move
    mgr.save_uploaded_images("stage1", "recyclable", [("IMG_0001.jpg", _make_jpeg_bytes())])
    mgr.save_uploaded_images("stage1", "trash", [("IMG_0001.jpg", _make_jpeg_bytes())])
    mgr.process_and_split_dataset(test_ratio=0.3, seed=42, force_resplit=False)
    manifest2 = json.loads(open(manifest_path).read())
    assert manifest2["split_version"] == 1
    assert len(manifest2["assignments"]) == 10
    for key, split in first_assignments.items():
        assert manifest2["assignments"][key] == split

    # Force full re-split bumps split_version
    mgr.process_and_split_dataset(test_ratio=0.3, seed=42, force_resplit=True)
    manifest3 = json.loads(open(manifest_path).read())
    assert manifest3["split_version"] == 2
