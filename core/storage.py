import os
import json
import uuid
from datetime import datetime
from typing import Any, Dict, List, Optional

class StorageManager:
    """Manages persistent JSON storage for model evaluation runs and leaderboards."""

    def __init__(self, db_path: str = "data/leaderboard_history.json") -> None:
        self.db_path = db_path
        self._ensure_db()

    def _ensure_db(self) -> None:
        parent_dir = os.path.dirname(self.db_path)
        if parent_dir and not os.path.exists(parent_dir):
            os.makedirs(parent_dir, exist_ok=True)
        if not os.path.exists(self.db_path):
            with open(self.db_path, "w", encoding="utf-8") as f:
                json.dump([], f)

    def _load_all(self) -> List[Dict[str, Any]]:
        try:
            with open(self.db_path, "r", encoding="utf-8") as f:
                data = json.load(f)
                return data if isinstance(data, list) else []
        except (json.JSONDecodeError, FileNotFoundError):
            return []

    def _save_all(self, runs: List[Dict[str, Any]]) -> None:
        with open(self.db_path, "w", encoding="utf-8") as f:
            json.dump(runs, f, indent=2)

    def save_run(self, run_data: Dict[str, Any]) -> Dict[str, Any]:
        record = dict(run_data)
        if "id" not in record or not record["id"]:
            record["id"] = f"run-{uuid.uuid4().hex[:8]}"
        if "timestamp" not in record or not record["timestamp"]:
            record["timestamp"] = datetime.utcnow().isoformat() + "Z"

        runs = self._load_all()
        runs.append(record)
        self._save_all(runs)
        return record

    def get_leaderboard(self, stage: str) -> List[Dict[str, Any]]:
        runs = self._load_all()
        stage_runs = [r for r in runs if r.get("stage") == stage]

        # Sort descending by F1, then by Accuracy
        def sort_key(item: Dict[str, Any]) -> tuple:
            metrics = item.get("metrics", {})
            f1 = metrics.get("f1", 0.0)
            acc = metrics.get("accuracy", 0.0)
            return (f1 if f1 is not None else 0.0, acc if acc is not None else 0.0)

        stage_runs.sort(key=sort_key, reverse=True)
        return stage_runs

    def get_history(
        self, team_name: Optional[str] = None, stage: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        runs = self._load_all()
        filtered = runs
        if team_name:
            filtered = [r for r in filtered if r.get("team_name") == team_name]
        if stage:
            filtered = [r for r in filtered if r.get("stage") == stage]
        return filtered
