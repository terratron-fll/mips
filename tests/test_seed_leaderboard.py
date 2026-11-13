import json
from scripts.seed_leaderboard import seed_history


def test_seed_preserves_existing_runs_and_merges(tmp_path):
    db_file = tmp_path / "leaderboard_history.json"
    existing = [
        {
            "id": "run-user-1",
            "stage": "stage1",
            "team_name": "Trial Team",
            "model_version": "trial-1",
            "metrics": {"f1": 0.88, "accuracy": 0.90},
            "errors": [],
        }
    ]
    db_file.write_text(json.dumps(existing), encoding="utf-8")

    added = seed_history(db_path=str(db_file))
    assert added == 6

    data = json.loads(db_file.read_text(encoding="utf-8"))
    ids = [r["id"] for r in data]
    assert "run-user-1" in ids
    assert len(data) == 7


def test_seed_is_idempotent(tmp_path):
    db_file = tmp_path / "leaderboard_history.json"
    assert seed_history(db_path=str(db_file)) == 6
    assert seed_history(db_path=str(db_file)) == 0
    data = json.loads(db_file.read_text(encoding="utf-8"))
    assert len(data) == 6


def test_seed_force_overwrite_replaces_existing(tmp_path):
    db_file = tmp_path / "leaderboard_history.json"
    db_file.write_text(json.dumps([{"id": "old-run"}]), encoding="utf-8")

    added = seed_history(db_path=str(db_file), force_overwrite=True)
    assert added == 6
    data = json.loads(db_file.read_text(encoding="utf-8"))
    assert [r["id"] for r in data] != ["old-run"]
    assert len(data) == 6
