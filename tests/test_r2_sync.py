import json
import os
from core.r2_sync import FakeS3Client, R2SyncManager


def test_from_secrets_offline_and_degraded(tmp_path):
    offline = R2SyncManager.from_secrets({}, local_data_dir=str(tmp_path))
    assert offline.is_configured() is False
    assert offline.last_sync_status == "offline"

    degraded = R2SyncManager.from_secrets(
        {"r2": {"endpoint_url": "https://example.r2.cloudflarestorage.com", "bucket": "mips"}},
        local_data_dir=str(tmp_path),
    )
    assert degraded.is_configured() is False
    assert degraded.last_sync_status == "degraded"


def test_push_and_pull_metadata_merges_and_skips_existing_runs(tmp_path):
    s3 = FakeS3Client(max_keys=2)
    dir_a = tmp_path / "a"
    dir_b = tmp_path / "b"
    dir_a.mkdir()
    dir_b.mkdir()

    sync_a = R2SyncManager(s3_client=s3, bucket="mips", local_data_dir=str(dir_a))
    sync_b = R2SyncManager(s3_client=s3, bucket="mips", local_data_dir=str(dir_b))

    # Local-only unsynced run in A
    db_a = dir_a / "leaderboard_history.json"
    db_a.write_text(
        json.dumps([{"id": "run-local", "timestamp": "2026-11-12T01:00:00Z", "stage": "stage1"}]),
        encoding="utf-8",
    )

    # Remote runs from A and B (3 runs > max_keys=2 to exercise pagination)
    sync_a.push_run({"id": "run-111", "timestamp": "2026-11-12T02:00:00Z", "stage": "stage1"})
    sync_b.push_run({"id": "run-222", "timestamp": "2026-11-12T03:00:00Z", "stage": "stage1"})
    sync_b.push_run({"id": "run-333", "timestamp": "2026-11-12T04:00:00Z", "stage": "stage1"})

    # Also put a dataset config and gemini cache file
    s3.put_object(Bucket="mips", Key="datasets/invasive_plants/config.json", Body=b'{"id": "invasive_plants"}')
    s3.put_object(Bucket="mips", Key="gemini_cache/invasive_plants/stage1.json", Body=b'{"flash": {}}')

    sync_a.pull_metadata(db_path=str(db_a))
    merged = json.loads(db_a.read_text(encoding="utf-8"))
    assert [r["id"] for r in merged] == ["run-local", "run-111", "run-222", "run-333"]
    assert (dir_a / "datasets" / "invasive_plants" / "config.json").exists()
    assert (dir_a / "gemini_cache" / "invasive_plants" / "stage1.json").exists()

    # Second pull skips already-local runs (no extra get_object for runs)
    calls_before = s3.get_count
    sync_a.pull_metadata(db_path=str(db_a))
    # Only config.json / gemini_cache (2 non-run metadata files), not the 3 runs
    assert s3.get_count - calls_before == 2


def test_pull_dataset_prefix_skips_matching_size_and_prunes_stale_test_files(tmp_path):
    s3 = FakeS3Client()
    sync = R2SyncManager(s3_client=s3, bucket="mips", local_data_dir=str(tmp_path))

    s3.put_object(Bucket="mips", Key="datasets/ds1/test_dataset/stage1/plant/a.jpg", Body=b"12345")
    stale = tmp_path / "datasets" / "ds1" / "test_dataset" / "stage1" / "plant" / "old.jpg"
    stale.parent.mkdir(parents=True, exist_ok=True)
    stale.write_bytes(b"stale")

    downloaded = sync.pull_dataset_prefix("ds1", "test_dataset")
    assert downloaded == 1
    assert not stale.exists()
    assert (tmp_path / "datasets" / "ds1" / "test_dataset" / "stage1" / "plant" / "a.jpg").read_bytes() == b"12345"

    # Second pull skips matching-size file
    calls_before = s3.get_count
    downloaded2 = sync.pull_dataset_prefix("ds1", "test_dataset")
    assert downloaded2 == 0
    assert s3.get_count == calls_before


def test_ensure_training_pack_and_model_artifact_local(tmp_path):
    s3 = FakeS3Client()
    sync = R2SyncManager(s3_client=s3, bucket="mips", local_data_dir=str(tmp_path))

    s3.put_object(Bucket="mips", Key="datasets/ds1/training_packs/stage1_train.zip", Body=b"packzip")
    s3.put_object(Bucket="mips", Key="models/run-99/stage1/submission.zip", Body=b"modelzip")

    assert sync.ensure_training_pack_local("ds1", "stage1") is True
    assert (tmp_path / "datasets" / "ds1" / "training_packs" / "stage1_train.zip").read_bytes() == b"packzip"
    assert sync.ensure_training_pack_local("ds1", "missing") is False

    models_dir = str(tmp_path / "models")
    assert sync.ensure_model_artifact_local("run-99", models_dir, subdir="stage1") is True
    assert (tmp_path / "models" / "run-99" / "stage1" / "submission.zip").read_bytes() == b"modelzip"
    assert sync.ensure_model_artifact_local("run-missing", models_dir) is False
