import os
import json
import pytest
from core.storage import StorageManager

def test_save_and_retrieve_run(tmp_path):
    db_file = str(tmp_path / "leaderboard.json")
    storage = StorageManager(db_path=db_file)
    run_record = {
        "id": "run-001",
        "stage": "stage1",
        "team_name": "Team Kudzu",
        "model_version": "v1.0",
        "notes": "Initial baseline",
        "metrics": {"accuracy": 0.85, "precision": 0.80, "recall": 0.90, "f1": 0.847, "tp": 18, "fp": 4, "tn": 16, "fn": 2},
        "errors": []
    }
    saved = storage.save_run(run_record)
    assert saved["id"] == "run-001"
    assert "timestamp" in saved

    leaderboard = storage.get_leaderboard(stage="stage1")
    assert len(leaderboard) == 1
    assert leaderboard[0]["team_name"] == "Team Kudzu"

def test_leaderboard_sorting_by_f1(tmp_path):
    db_file = str(tmp_path / "leaderboard.json")
    storage = StorageManager(db_path=db_file)
    storage.save_run({"stage": "stage2", "team_name": "Team Low", "metrics": {"accuracy": 0.70, "f1": 0.65}})
    storage.save_run({"stage": "stage2", "team_name": "Team High", "metrics": {"accuracy": 0.88, "f1": 0.89}})
    
    board = storage.get_leaderboard(stage="stage2")
    assert len(board) == 2
    assert board[0]["team_name"] == "Team High"
    assert board[1]["team_name"] == "Team Low"

def test_get_history_filtering(tmp_path):
    db_file = str(tmp_path / "leaderboard.json")
    storage = StorageManager(db_path=db_file)
    storage.save_run({"stage": "stage1", "team_name": "Team A", "metrics": {"f1": 0.8}})
    storage.save_run({"stage": "stage2", "team_name": "Team A", "metrics": {"f1": 0.75}})
    storage.save_run({"stage": "stage1", "team_name": "Team B", "metrics": {"f1": 0.9}})

    history_a = storage.get_history(team_name="Team A")
    assert len(history_a) == 2

    history_stage1 = storage.get_history(stage="stage1")
    assert len(history_stage1) == 2
