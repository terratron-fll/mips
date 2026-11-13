import argparse
import os
import json
import tempfile
from datetime import datetime, timedelta

def seed_history(
    db_path: str = "data/leaderboard_history.json",
    force_overwrite: bool = False,
) -> int:
    """Seed demo leaderboard runs safely, preserving any existing runs unless force_overwrite=True."""
    parent_dir = os.path.dirname(db_path)
    if parent_dir:
        os.makedirs(parent_dir, exist_ok=True)
    
    # Check what images exist in test_dataset/intro
    cat_test = []
    not_cat_test = []
    if os.path.exists("data/test_dataset/intro/cat"):
        cat_test = [os.path.join("data/test_dataset/intro/cat", f) for f in os.listdir("data/test_dataset/intro/cat") if f.endswith(".jpg")]
    if os.path.exists("data/test_dataset/intro/not_cat"):
        not_cat_test = [os.path.join("data/test_dataset/intro/not_cat", f) for f in os.listdir("data/test_dataset/intro/not_cat") if f.endswith(".jpg")]
        
    err_img1 = cat_test[0] if cat_test else ""
    err_img2 = not_cat_test[0] if not_cat_test else ""
    err_img3 = cat_test[1] if len(cat_test) > 1 else err_img1

    runs = [
        # Intro Board (Tutorial) - Team Terratron Hill Climbing
        {
            "id": "run-001",
            "stage": "intro",
            "team_name": "Team Terratron",
            "model_version": "v1.0 (Baseline)",
            "timestamp": (datetime.utcnow() - timedelta(days=4)).isoformat() + "Z",
            "notes": "Initial model trained with default 50 epochs & batch size 16",
            "metrics": {
                "accuracy": 0.833,
                "precision": 0.800,
                "recall": 0.889,
                "f1": 0.842,
                "tp": 8,
                "fp": 2,
                "tn": 7,
                "fn": 1
            },
            "errors": [
                {
                    "filename": os.path.basename(err_img1) if err_img1 else "cat_shadow.jpg",
                    "image_path": err_img1,
                    "true_label": "cat",
                    "predicted_label": "not_cat",
                    "confidence": 0.74,
                    "error_type": "False Negative"
                },
                {
                    "filename": os.path.basename(err_img2) if err_img2 else "dog_pointy_ears.jpg",
                    "image_path": err_img2,
                    "true_label": "not_cat",
                    "predicted_label": "cat",
                    "confidence": 0.81,
                    "error_type": "False Positive"
                }
            ]
        },
        {
            "id": "run-002",
            "stage": "intro",
            "team_name": "Team Terratron",
            "model_version": "v1.1 (Augmented)",
            "timestamp": (datetime.utcnow() - timedelta(days=2)).isoformat() + "Z",
            "notes": "Added 15 lighting variation photos and increased epochs to 60",
            "metrics": {
                "accuracy": 0.944,
                "precision": 0.900,
                "recall": 1.000,
                "f1": 0.947,
                "tp": 9,
                "fp": 1,
                "tn": 8,
                "fn": 0
            },
            "errors": [
                {
                    "filename": os.path.basename(err_img2) if err_img2 else "dog_fur_texture.jpg",
                    "image_path": err_img2,
                    "true_label": "not_cat",
                    "predicted_label": "cat",
                    "confidence": 0.62,
                    "error_type": "False Positive"
                }
            ]
        },
        {
            "id": "run-003",
            "stage": "intro",
            "team_name": "Team Terratron",
            "model_version": "v1.2 (Championship)",
            "timestamp": (datetime.utcnow() - timedelta(hours=6)).isoformat() + "Z",
            "notes": "Tuned learning rate to 0.001 with 70 epochs. Perfect recall!",
            "metrics": {
                "accuracy": 0.972,
                "precision": 0.947,
                "recall": 1.000,
                "f1": 0.973,
                "tp": 9,
                "fp": 1,
                "tn": 9,
                "fn": 0
            },
            "errors": [
                {
                    "filename": os.path.basename(err_img2) if err_img2 else "tricky_lighting.jpg",
                    "image_path": err_img2,
                    "true_label": "not_cat",
                    "predicted_label": "cat",
                    "confidence": 0.55,
                    "error_type": "False Positive"
                }
            ]
        },
        # Board 1: Plant vs Not-Plant
        {
            "id": "run-004",
            "stage": "stage1",
            "team_name": "Team Terratron",
            "model_version": "Stage 1 v1.1",
            "timestamp": (datetime.utcnow() - timedelta(days=1)).isoformat() + "Z",
            "notes": "Boat hull vs attached vegetation filter. Clean pass rate: 94%",
            "metrics": {
                "accuracy": 0.940,
                "precision": 0.925,
                "recall": 0.960,
                "f1": 0.942,
                "tp": 48,
                "fp": 4,
                "tn": 46,
                "fn": 2
            },
            "errors": []
        },
        # Board 2: Invasive vs Non-Invasive
        {
            "id": "run-005",
            "stage": "stage2",
            "team_name": "Team Terratron",
            "model_version": "Stage 2 v1.2",
            "timestamp": (datetime.utcnow() - timedelta(hours=12)).isoformat() + "Z",
            "notes": "Eurasian Milfoil vs native pondweed classifier with shadow augmentation",
            "metrics": {
                "accuracy": 0.935,
                "precision": 0.948,
                "recall": 0.920,
                "f1": 0.934,
                "tp": 46,
                "fp": 2,
                "tn": 47,
                "fn": 4
            },
            "errors": []
        },
        # Board 3: Field Robot Pipeline Simulation
        {
            "id": "run-006",
            "stage": "pipeline",
            "team_name": "Team Terratron (TerraBot)",
            "model_version": "Pipeline Run #3",
            "timestamp": (datetime.utcnow() - timedelta(hours=3)).isoformat() + "Z",
            "notes": "Cascaded Dock Scanner simulation at Lake Sebago boat ramp",
            "metrics": {
                "accuracy": 0.925,
                "precision": 0.935,
                "recall": 0.940,
                "f1": 0.937,
                "stage1_acc": 0.960,
                "stage2_acc": 0.935,
                "tp": 45,
                "fp": 3,
                "tn": 48,
                "fn": 3
            },
            "errors": []
        }
    ]
    
    existing_runs = []
    if not force_overwrite and os.path.exists(db_path):
        with open(db_path, "r", encoding="utf-8") as f:
            loaded = json.load(f)
            if not isinstance(loaded, list):
                raise ValueError(f"Expected a JSON list in '{db_path}', got {type(loaded).__name__}.")
            existing_runs = loaded

    existing_ids = {r.get("id") for r in existing_runs if isinstance(r, dict)}
    new_runs = [r for r in runs if r.get("id") not in existing_ids] if not force_overwrite else runs
    final_runs = new_runs if force_overwrite else existing_runs + new_runs

    # Atomic write so a crash mid-write never corrupts the existing leaderboard
    fd, tmp_path = tempfile.mkstemp(dir=parent_dir or ".", prefix=".leaderboard_", suffix=".tmp")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            json.dump(final_runs, f, indent=2)
        os.replace(tmp_path, db_path)
    except Exception:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)
        raise

    print(f"Added {len(new_runs)} seed runs into {db_path} (total: {len(final_runs)}).")
    return len(new_runs)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Seed demo leaderboard runs safely.")
    parser.add_argument("--db-path", default="data/leaderboard_history.json", help="Path to leaderboard JSON file.")
    parser.add_argument("--overwrite", action="store_true", help="Overwrite existing runs instead of merging.")
    args = parser.parse_args()
    seed_history(db_path=args.db_path, force_overwrite=args.overwrite)
