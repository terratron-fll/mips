# MIPS (Model for Invasive Plant Species Detection) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a multi-board evaluation platform, dataset manager, and experiment tracking leaderboard for First Lego League (FLL) students (ages 9–11) to evaluate Google Teachable Machine models on invasive plant detection.

**Architecture:** A Streamlit application backed by lightweight modular Python engines (`storage.py`, `dataset_manager.py`, `evaluator.py`, `gemini_benchmark.py`) with persistent local JSON tracking. Raw images are downsampled to $448 \times 448$ on intake to keep storage under 25 MB, partitioned into a 70% downloadable training zip and a 30% locked test set, and evaluated using standard $2 \times 2$ Confusion Matrices (TP, FP, TN, FN) and F1-scores.

**Tech Stack:** Python 3.10+, Streamlit, TensorFlow / Keras (or TFLite runtime), Pillow, NumPy, Pandas, Scikit-Learn, Plotly, `google-genai` SDK, Pytest.

**Spec:** `docs/superpowers/specs/2026-10-18-mips-design.md`

## Global Constraints

- Python version floor: Python 3.10+.
- Image downsampling maximum dimensions on intake: $448 \times 448$ at 85% JPEG quality.
- Model input standard dimensions: $224 \times 224$ normalized to $[-1, 1]$.
- Supported model export bundles: `.zip` containing `keras_model.h5` or `model.tflite` plus `labels.txt`.
- Binary stages:
  - Stage 1: `plant` vs. `not_plant`
  - Stage 2: `invasive` vs. `non_invasive`
- Metrics: Accuracy, Precision, Recall, F1-Score, and full $2 \times 2$ Confusion Matrix counts (TP, FP, TN, FN).
- Zero-leakage rule: Test split images in `data/test_dataset/` must never be packaged into downloadable training zips.

## Review Focus

1. **Missing or misspelled labels in `labels.txt`**: When a student enters classes as `Plant` vs `NotPlant` or has extra whitespace/casing differences, the loader should sanitize and map cleanly or raise a descriptive human-readable error instead of an internal KeyError.
2. **Empty or corrupted image in test set**: If an image is unreadable or 0 bytes, `evaluator.py` must skip or handle it gracefully without crashing mid-evaluation of the batch.
3. **Division by zero in metrics**: When a model predicts all samples as negative ($TP=0, FP=0$), Precision $\frac{0}{0}$ must return `0.0` safely rather than raising `ZeroDivisionError`.
4. **Unsupported archive structure**: If a student zips a parent folder rather than the files directly (nested directory inside zip), `evaluator.py` must locate `keras_model.h5` / `model.tflite` and `labels.txt` recursively.
5. **Missing Gemini API Key (P2)**: When `GEMINI_API_KEY` is unset or invalid, the benchmark should display an informative notice in the UI rather than failing the entire app.

---

### Task 1: Environment Dependencies & Persistent Storage Core

**Files:**
- Create: `requirements.txt`
- Create: `core/__init__.py`
- Create: `core/storage.py`
- Create: `tests/__init__.py`
- Create: `tests/test_storage.py`

**Interfaces:**
- Produces: `StorageManager` in `core/storage.py`:
  - `__init__(db_path: str = "data/leaderboard_history.json") -> None`
  - `save_run(run_data: dict) -> dict`
  - `get_leaderboard(stage: str) -> list[dict]` (sorted descending by `metrics.f1`, then `metrics.accuracy`)
  - `get_history(team_name: str | None = None, stage: str | None = None) -> list[dict]`

- [ ] **Step 1: Write requirements.txt**

Write `requirements.txt` containing:
```text
streamlit>=1.35.0
tensorflow>=2.15.0
pillow>=10.0.0
numpy>=1.24.0
pandas>=2.0.0
scikit-learn>=1.3.0
plotly>=5.20.0
google-genai>=0.1.1
pytest>=8.0.0
```

- [ ] **Step 2: Write failing tests in `tests/test_storage.py`**

```python
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
```

- [ ] **Step 3: Run pytest to verify tests fail**

Run: `pytest tests/test_storage.py -v`  
Expected: FAIL (ModuleNotFoundError: No module named 'core.storage')

- [ ] **Step 4: Implement `core/storage.py`**

Implement `StorageManager` handling file initialization, reading/writing JSON records, appending timestamps, and sorting leaderboards by F1-score descending.

- [ ] **Step 5: Run pytest to verify tests pass**

Run: `pytest tests/test_storage.py -v`  
Expected: PASS

- [ ] **Step 6: Commit**

```bash
git add requirements.txt core/__init__.py core/storage.py tests/__init__.py tests/test_storage.py
git commit -m "feat: add dependencies and storage manager core"
```

---

### Task 2: Dataset Manager & Splitter Core

**Files:**
- Create: `core/dataset_manager.py`
- Create: `tests/test_dataset_manager.py`

**Interfaces:**
- Consumes: Raw images from `data/raw/<stage>/<class>/`
- Produces: `DatasetManager` in `core/dataset_manager.py`:
  - `__init__(base_data_dir: str = "data") -> None`
  - `get_dataset_stats() -> dict[str, dict[str, int]]`
  - `downsample_and_save_image(src_path: str, dest_path: str, max_size: int = 448) -> bool`
  - `process_and_split_dataset(test_ratio: float = 0.30, seed: int = 42) -> dict`
  - `create_student_training_packs() -> dict[str, str]` (returns paths to generated zip files)

- [ ] **Step 1: Write failing tests in `tests/test_dataset_manager.py`**

```python
import os
import zipfile
from PIL import Image
import pytest
from core.dataset_manager import DatasetManager

@pytest.fixture
def dummy_raw_dataset(tmp_path):
    base_dir = tmp_path / "data"
    for stage, classes in [("stage1", ["plant", "not_plant"]), ("stage2", ["invasive", "non_invasive"])]:
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

def test_split_and_training_pack_generation(dummy_raw_dataset):
    dm = DatasetManager(base_data_dir=dummy_raw_dataset)
    result = dm.process_and_split_dataset(test_ratio=0.30, seed=42)
    assert result["stage1"]["train"] == 14
    assert result["stage1"]["test"] == 6

    # Verify locked test directory has images
    test_plant_dir = os.path.join(dummy_raw_dataset, "test_dataset", "stage1", "plant")
    assert len(os.listdir(test_plant_dir)) == 3

    # Verify training pack zip was created and contains only training samples
    packs = dm.create_student_training_packs()
    assert os.path.exists(packs["stage1"])
    with zipfile.ZipFile(packs["stage1"], "r") as z:
        names = z.namelist()
        assert any("plant/" in n for n in names)
        assert len([n for n in names if not n.endswith("/")]) == 14
```

- [ ] **Step 2: Run pytest to verify tests fail**

Run: `pytest tests/test_dataset_manager.py -v`  
Expected: FAIL (ModuleNotFoundError: No module named 'core.dataset_manager')

- [ ] **Step 3: Implement `core/dataset_manager.py`**

Implement image validation, $448 \times 448$ resizing (using Pillow's `Image.Resampling.LANCZOS`), stratified train/test split without data leakage, and zip packaging for students.

- [ ] **Step 4: Run pytest to verify tests pass**

Run: `pytest tests/test_dataset_manager.py -v`  
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add core/dataset_manager.py tests/test_dataset_manager.py
git commit -m "feat: add dataset manager, downsampler, and train-test splitter"
```

---

### Task 3: Model Evaluator & Real ML Metrics Engine

**Files:**
- Create: `core/evaluator.py`
- Create: `tests/test_evaluator.py`

**Interfaces:**
- Consumes: Test images in `data/test_dataset/<stage>/<class>/` and student `.zip` upload
- Produces: `ModelEvaluator` in `core/evaluator.py`:
  - `__init__(test_dataset_dir: str = "data/test_dataset") -> None`
  - `unpack_model_zip(zip_bytes_or_path: bytes | str, extract_to: str) -> tuple[str, list[str]]` (returns model file path and normalized labels list)
  - `preprocess_image(image_path: str, target_size: tuple[int, int] = (224, 224)) -> np.ndarray` (returns normalized array $[-1, 1]$)
  - `compute_binary_metrics(y_true: list[int], y_pred: list[int], class_names: list[str]) -> dict`
  - `evaluate_submission(zip_bytes_or_path: bytes | str, stage: str) -> dict`

- [ ] **Step 1: Write failing tests in `tests/test_evaluator.py`**

```python
import pytest
from core.evaluator import ModelEvaluator

def test_compute_binary_metrics_exact():
    evaluator = ModelEvaluator()
    # 5 samples: 3 actual positives (1), 2 actual negatives (0)
    # y_pred: TP=2, FN=1, TN=2, FP=0
    y_true = [1, 1, 1, 0, 0]
    y_pred = [1, 1, 0, 0, 0]
    metrics = evaluator.compute_binary_metrics(y_true, y_pred, class_names=["negative", "positive"])
    
    assert metrics["tp"] == 2
    assert metrics["fn"] == 1
    assert metrics["tn"] == 2
    assert metrics["fp"] == 0
    assert metrics["accuracy"] == 0.80
    assert metrics["precision"] == 1.0
    assert pytest.approx(metrics["recall"], 0.01) == 0.667
    assert pytest.approx(metrics["f1"], 0.01) == 0.80

def test_compute_binary_metrics_division_by_zero_safe():
    evaluator = ModelEvaluator()
    # All predictions are negative: TP=0, FP=0
    y_true = [1, 1, 0, 0]
    y_pred = [0, 0, 0, 0]
    metrics = evaluator.compute_binary_metrics(y_true, y_pred, class_names=["negative", "positive"])
    assert metrics["tp"] == 0
    assert metrics["fp"] == 0
    assert metrics["precision"] == 0.0
    assert metrics["recall"] == 0.0
    assert metrics["f1"] == 0.0

def test_normalize_labels_handling():
    evaluator = ModelEvaluator()
    raw_labels = ["0 Plant\n", "1 Not Plant\r\n"]
    cleaned = evaluator.sanitize_labels(raw_labels)
    assert cleaned == ["plant", "not_plant"]
```

- [ ] **Step 2: Run pytest to verify tests fail**

Run: `pytest tests/test_evaluator.py -v`  
Expected: FAIL (ModuleNotFoundError: No module named 'core.evaluator')

- [ ] **Step 3: Implement `core/evaluator.py`**

Implement zip extraction (supporting flat or nested zips), `labels.txt` normalization (handling numbering prefixes like `0 plant`), $224 \times 224$ preprocessing, TensorFlow / Keras / TFLite inference runner, metric computation, and error inspection extraction.

- [ ] **Step 4: Run pytest to verify tests pass**

Run: `pytest tests/test_evaluator.py -v`  
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add core/evaluator.py tests/test_evaluator.py
git commit -m "feat: add model evaluator and ML metrics calculation engine"
```

---

### Task 4: Streamlit Frontend - Core Boards, Submission, & Visual Metrics

**Files:**
- Create: `app.py`
- Modify: `README.md`

**Interfaces:**
- Consumes: `core/storage.py`, `core/evaluator.py`
- Renders:
  - Header & Stage selection
  - **Board 1 (Plant vs. Not-Plant)** Leaderboard table & Plotly hill-climb timeline
  - **Board 2 (Invasive vs. Non-Invasive)** Leaderboard table & Plotly hill-climb timeline
  - **Submit Model** tab with file uploader for `.zip`, team name, version, notes, and run trigger
  - Interactive Diagnostics: $2 \times 2$ Confusion Matrix plot and Error Gallery ("Bloopers")

- [ ] **Step 1: Write `app.py` with multi-tab layout and submission pipeline**

Implement:
- Tab navigation: `["🌿 Board 1: Plant vs Not-Plant", "⚠️ Board 2: Invasive vs Non-Invasive", "🏆 Board 3: Field Robot Pipeline", "📤 Submit Model", "🛠️ Coach Dataset Admin"]`.
- The "Submit Model" tab with drag-and-drop file uploader, stage selector, team metadata inputs, and live evaluation trigger.
- Rendering of Leaderboard tables with Rank, Team, Model Version, F1-Score, Accuracy, Precision, Recall, Date.
- Confusion Matrix display using `plotly.figure_factory.create_annotated_heatmap` showing TP, FP, TN, FN.
- Error inspection section rendering the top false positive / false negative images with prediction confidence.

- [ ] **Step 2: Run Streamlit headless test or sanity check**

Run: `python -c "import app; print('App syntax valid')"`  
Expected: Output `App syntax valid`

- [ ] **Step 3: Update `README.md`**

Document how to run the app (`streamlit run app.py`), how students export from Google Teachable Machine, and how to submit to the leaderboard.

- [ ] **Step 4: Commit**

```bash
git add app.py README.md
git commit -m "feat: implement Streamlit UI for leaderboards, submission, and ML diagnostics"
```

---

### Task 5: Streamlit Frontend - Coach Dataset Admin & End-to-End Field Robot Pipeline

**Files:**
- Modify: `app.py`
- Create: `core/pipeline.py`
- Create: `tests/test_pipeline.py`

**Interfaces:**
- Consumes: `core/dataset_manager.py`, `core/evaluator.py`
- Produces: `PipelineEvaluator` in `core/pipeline.py`:
  - `evaluate_pipeline(stage1_zip: bytes | str, stage2_zip: bytes | str) -> dict`
- Renders:
  - **Board 3 (Field Robot Simulation)**: Evaluates Stage 1 filter $\rightarrow$ Stage 2 classifier on an end-to-end test set (calculating overall robot precision & recall).
  - **Coach Dataset Admin**: PIN-protected tab to view class balance, click "Downsample & Generate 70/30 Splits", and download student training pack zips.

- [ ] **Step 1: Write failing tests in `tests/test_pipeline.py`**

```python
import pytest
from core.pipeline import PipelineEvaluator

def test_pipeline_decision_logic():
    # End-to-end logic test:
    # If Stage 1 says 'not_plant', Stage 2 is skipped (predicted as non_invasive / safe)
    # If Stage 1 says 'plant', Stage 2 makes the final invasive call
    pe = PipelineEvaluator()
    # Image 1: Not a plant -> robot ignores it (safe)
    # Image 2: Plant -> Invasive -> robot flags it as invasive
    # Image 3: Plant -> Non-invasive -> robot ignores it
    stage1_preds = [("not_plant", 0.95), ("plant", 0.88), ("plant", 0.92)]
    stage2_preds = [("invasive", 0.70), ("invasive", 0.85), ("non_invasive", 0.90)]
    
    final_labels = pe.combine_predictions(stage1_preds, stage2_preds)
    assert final_labels[0] == "not_plant"
    assert final_labels[1] == "invasive"
    assert final_labels[2] == "non_invasive"
```

- [ ] **Step 2: Run pytest to verify tests fail**

Run: `pytest tests/test_pipeline.py -v`  
Expected: FAIL (ModuleNotFoundError: No module named 'core.pipeline')

- [ ] **Step 3: Implement `core/pipeline.py` and update `app.py`**

Implement `PipelineEvaluator` and wire both Board 3 and the Coach Dataset Admin tab into `app.py`.

- [ ] **Step 4: Run pytest to verify tests pass**

Run: `pytest tests/test_pipeline.py -v`  
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add core/pipeline.py tests/test_pipeline.py app.py
git commit -m "feat: add end-to-end pipeline evaluation and coach dataset admin panel"
```

---

### Task 6 (P2): Gemini 2.5 Flash VLM Benchmark Client (Coach Built)

**Files:**
- Create: `core/gemini_benchmark.py`
- Create: `tests/test_gemini_benchmark.py`
- Modify: `app.py`

**Interfaces:**
- Consumes: `GEMINI_API_KEY` from environment or Streamlit secrets
- Produces: `GeminiBenchmarkRunner` in `core/gemini_benchmark.py`:
  - `__init__(cache_file: str = "data/gemini_cache.json") -> None`
  - `evaluate_stage_benchmark(stage: str, test_images_dir: str) -> dict`
- Integrates into `app.py` as a pinned reference entry on Board 1 and Board 2:
  `🤖 [BENCHMARK] Gemini 2.5 Flash (System Instruction)`

- [ ] **Step 1: Write failing tests in `tests/test_gemini_benchmark.py`**

```python
import os
import json
import pytest
from core.gemini_benchmark import GeminiBenchmarkRunner

def test_cache_loading_and_saving(tmp_path):
    cache_path = str(tmp_path / "gemini_cache.json")
    runner = GeminiBenchmarkRunner(cache_file=cache_path)
    mock_results = {
        "stage1": {
            "metrics": {"accuracy": 0.92, "precision": 0.90, "recall": 0.95, "f1": 0.924, "tp": 19, "fp": 2, "tn": 18, "fn": 1}
        }
    }
    runner.save_cache(mock_results)
    assert os.path.exists(cache_path)
    loaded = runner.load_cache()
    assert loaded["stage1"]["metrics"]["f1"] == 0.924
```

- [ ] **Step 2: Run pytest to verify tests fail**

Run: `pytest tests/test_gemini_benchmark.py -v`  
Expected: FAIL (ModuleNotFoundError: No module named 'core.gemini_benchmark')

- [ ] **Step 3: Implement `core/gemini_benchmark.py` and link to `app.py`**

Implement `GeminiBenchmarkRunner` with structured System Instructions for Stage 1 and Stage 2, caching responses to avoid redundant API queries, and displaying the pinned benchmark row on the leaderboards.

- [ ] **Step 4: Run pytest to verify tests pass**

Run: `pytest tests/test_gemini_benchmark.py -v`  
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add core/gemini_benchmark.py tests/test_gemini_benchmark.py app.py
git commit -m "feat: add Gemini 2.5 Flash VLM benchmark with System Instructions"
```

---

### Task 7: Full Test Suite Run & Remote Git Push

**Files:**
- Modify: `tests/`
- Test: All tests across the test suite

- [ ] **Step 1: Run the full test suite**

Run: `pytest tests/ -v`  
Expected: All tests PASS.

- [ ] **Step 2: Push all commits to GitHub remote**

Run: `git push origin main`  
Expected: Remote `https://github.com/terratron-fll/mips` updated with all tasks.
