import os
import json
import shutil
import uuid
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

class StorageManager:
    """Manages persistent JSON storage for model evaluation runs, leaderboards, and model artifacts."""

    def __init__(
        self,
        db_path: str = "data/leaderboard_history.json",
        models_dir: str = "data/models",
    ) -> None:
        self.db_path = db_path
        self.models_dir = models_dir
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

    def _artifact_dir(self, run_id: str, subdir: Optional[str] = None) -> str:
        if subdir:
            return os.path.join(self.models_dir, run_id, subdir)
        return os.path.join(self.models_dir, run_id)

    def save_model_artifact(
        self,
        run_id: str,
        zip_bytes: bytes,
        extracted_model_path: Optional[str] = None,
        labels: Optional[List[str]] = None,
        subdir: Optional[str] = None,
    ) -> Dict[str, str]:
        """Persist uploaded model zip and extracted model/labels under data/models/<run_id>/."""
        target_dir = self._artifact_dir(run_id, subdir=subdir)
        os.makedirs(target_dir, exist_ok=True)

        zip_path = os.path.join(target_dir, "submission.zip")
        with open(zip_path, "wb") as f:
            f.write(zip_bytes)

        artifacts: Dict[str, str] = {
            "model_dir": target_dir,
            "zip_path": zip_path,
        }

        if extracted_model_path and os.path.exists(extracted_model_path):
            dest_model = os.path.join(target_dir, os.path.basename(extracted_model_path))
            shutil.copy2(extracted_model_path, dest_model)
            artifacts["model_path"] = dest_model

        if labels is not None:
            labels_path = os.path.join(target_dir, "labels.txt")
            with open(labels_path, "w", encoding="utf-8") as f:
                for idx, lbl in enumerate(labels):
                    f.write(f"{idx} {lbl}\n")
            artifacts["labels_path"] = labels_path

        return artifacts

    def get_model_artifact_bytes(
        self, run_id: str, subdir: Optional[str] = None
    ) -> Optional[Tuple[str, bytes]]:
        """Return (download_filename, bytes) for a run's archived submission.zip, or None."""
        if not run_id:
            return None
        zip_path = os.path.join(self._artifact_dir(run_id, subdir=subdir), "submission.zip")
        if not os.path.exists(zip_path):
            return None
        with open(zip_path, "rb") as f:
            suffix = f"_{subdir}" if subdir else ""
            return (f"{run_id}{suffix}_model.zip", f.read())
