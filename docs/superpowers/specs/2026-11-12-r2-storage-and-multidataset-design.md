# Cloudflare R2 Shared Storage, Multi-Dataset Organization & Secrets Security

*   **Author:** Chetan Ahuja
*   **Status:** Approved
*   **Date:** 2026-11-12

--------------------------------------------------------------------------------

## Overview

**MIPS (Model for Invasive Plant Species Detection)** currently stores all images, train/test splits, training packs (student-downloadable `.zip`s of training images), uploaded student models, and leaderboard runs on local disk under `data/`. Because `data/` is excluded from Git and Streamlit Community Cloud containers use ephemeral storage, runs and datasets created locally and in the cloud diverge and cloud state is lost on restart.

This design backs `StorageManager`, `DatasetManager`, and `GeminiBenchmarkRunner` with a single-bucket **Cloudflare R2** (S3-compatible) object store, introduces **multi-dataset organization** with isolated splits and per-`(dataset, stage)` leaderboards, adds **incremental append-only train/test splits** with versioned force re-splits, and gates live Gemini API calls and destructive re-splits behind a **coach password** while keeping all credentials in `.streamlit/secrets.toml` and Streamlit Cloud Secrets.

## The Problem

1. **Ephemeral cloud state & local/cloud divergence:** Runs submitted on Streamlit Community Cloud disappear on container restart, and local runs/images are invisible to the cloud app.
2. **Concurrent write collisions:** A single shared `leaderboard_history.json`, `datasets.json`, or `gemini_cache.json` risks last-writer-wins data loss when multiple students submit models or create datasets at the same time.
3. **Single-dataset hardcoding:** `DatasetManager`, `ModelEvaluator`, `PipelineEvaluator`, `GeminiBenchmarkRunner`, and `app.py` hardcode `"intro"`, `"stage1"`, `"stage2"`, and `"pipeline"` for two fixed tasks.
4. **Test-set drift on new photo uploads:** `DatasetManager.process_and_split_dataset()` currently deletes and rebuilds `train_dataset/` and `test_dataset/` from scratch, invalidating prior leaderboard comparisons whenever students add new photos.
5. **Unprotected Gemini quota:** Anyone visiting the Streamlit Cloud URL can trigger live Gemini API calls if `GEMINI_API_KEY` is configured in Cloud Secrets.

## Goals

*   Synchronize runs (`runs/<run_id>.json`), model artifacts (`models/<run_id>/...`), dataset configs (`datasets/<dataset_id>/config.json`), images, training packs, and Gemini benchmark cache between local disk and a bucket-scoped Cloudflare R2 store.
*   Avoid cold-boot sync storms and concurrent-write races using a process-wide `@st.cache_resource` + `threading.Lock` coordinator, lightweight metadata sync on boot, lazy dataset/model pulls, and per-entity files.
*   Support multiple configurable datasets with isolated storage, custom binary stages/classes, and per-`(dataset, stage)` leaderboards (Precision, Recall, F1, Accuracy, Confusion Matrix, and Bloopers—a gallery of misclassified test images).
*   Preserve locked test sets across new photo uploads via incremental append-only splitting (`split_manifest.json` + `split_version`), with an explicit coach-gated versioned force re-split option.
*   Preserve raw photo downsampling on upload inside the MIPS web UI and push downsampled images to R2.
*   Protect live Gemini API calls and destructive re-splits behind a coach password (`GEMINI_ADMIN_PASSWORD` checked with `hmac.compare_digest`), while keeping cached Gemini results publicly visible.
*   Keep all API keys and R2 credentials in `.streamlit/secrets.toml` / Streamlit Cloud Secrets, linking to official docs for obtaining them.
*   Maintain backward compatibility with offline local execution (preserving legacy `data/` paths and `STAGE_TARGETS` fallback) and keep all 31 existing unit tests passing via test-driven development (TDD).

## Non-Goals

*   Cross-dataset aggregate or blended team leaderboards (leaderboards remain per `(dataset, stage)`).
*   Password-gating dataset creation or append-only train/test splitting (kept open in the UI).
*   External Google Drive folder browsing for student photo drops (all student photo uploads go through MIPS).
*   Per-student user accounts or OAuth login flows.

## Critical User Journeys (CUJs)

### CUJ 1: Student Uploads Photos, Downloads Training Pack & Submits Model (Cloud or Local)
*   **Given**: MIPS is running on Streamlit Cloud (or locally) connected to the shared Cloudflare R2 bucket.
*   **When**: A student selects a dataset, uploads new photos in the Dataset Manager (downsampled on intake), triggers an incremental split (which assigns only new photos 70/30 without altering existing test images), downloads the training pack, trains in Teachable Machine, and submits their `.zip` model.
*   **Then**: MIPS evaluates the model against the locked test set, writes `runs/<run_id>.json` (stamped with `dataset_id` and `split_version`) and `models/<run_id>/submission.zip` locally and to R2, and updates the per-`(dataset, stage)` leaderboard visible to both local and cloud sessions.

### CUJ 2: Classroom Cold Boot & Dataset Switching
*   **Given**: 20 students open the Streamlit Cloud URL simultaneously after a container restart.
*   **When**: Sessions initialize and students switch between datasets, download training packs, or download saved model `.zip`s.
*   **Then**: The server performs one shared cold-boot metadata sync (datasets, split manifests, runs, and cached Gemini benchmarks). Dataset images and training packs are fetched lazily only for the active dataset and action, skipping unchanged files, and saved model `.zip`s are downloaded only when clicked.

### CUJ 3: Coach Unlocks and Runs Live Gemini Benchmarks
*   **Given**: `GEMINI_API_KEY` and `GEMINI_ADMIN_PASSWORD` are stored in `.streamlit/secrets.toml` or Streamlit Cloud Secrets.
*   **When**: The coach enters `GEMINI_ADMIN_PASSWORD` in the **Coach Reference** section (unlocking Coach Mode for that browser session only) and runs a Gemini benchmark for a dataset stage.
*   **Then**: MIPS runs the benchmark, saves the score under `(dataset_id, stage, split_version)` locally and to R2, and allows the coach to click **Lock Coach Mode** afterward. Students can view cached Gemini reference bars (benchmark comparison lines on the leaderboard) without entering a password.

--------------------------------------------------------------------------------

## Technical Plan (Architecture)

### 1. Component Architecture & Data Flow

```text
+-------------------------------------------------------------------+
|                        Streamlit UI (app.py)                      |
|  - Dataset Switcher + Create Dataset (slug-validated)             |
|  - Dynamic Stage/Pipeline Boards (filtered by dataset_id, stage)  |
|  - Coach Gemini Gate (GEMINI_ADMIN_PASSWORD via hmac)             |
|  - Sync Status Badge + Manual Sync Button                         |
+-------------------+-------------------------+---------------------+
                    |                         |
                    v                         v
+-------------------------------+   +-------------------------------+
|  Sync Coordinator             |   | Domain Modules (Local Disk)   |
|  (@st.cache_resource + Lock)  |   | - DatasetRegistry             |
|  - Boot: metadata only        |   | - DatasetManager (append-only |
|  - Lazy: targeted prefix pull |   |   splits + split_manifest)    |
|  - Size check + atomic write  |   | - StorageManager (runs/models)|
+---------------+---------------+   | - ModelEvaluator / Pipeline   |
                |                   | - GeminiBenchmarkRunner       |
                v                   +---------------+---------------+
+-------------------------------+                   |
|   core/r2_sync.py             |<------------------+
|   - R2SyncManager             |  (push on write / lazy pull on read)
|   - boto3 S3 client           |
|   - FakeS3Client (TDD tests)  |
+---------------+---------------+
                |
                v
+-------------------------------------------------------------------+
|              Cloudflare R2 Bucket (Single Scoped Bucket)          |
|  runs/<run_id>.json                                               |
|  models/<run_id>/[stage1|stage2/]submission.zip, labels.txt       |
|  datasets/<dataset_id>/config.json                                |
|  datasets/<dataset_id>/split_manifest.json                        |
|  datasets/<dataset_id>/raw/<stage>/<class>/<img.jpg>              |
|  datasets/<dataset_id>/test_dataset/<stage>/<class>/<img.jpg>     |
|  datasets/<dataset_id>/training_packs/<stage>_train.zip           |
|  gemini_cache/<dataset_id>/<stage>.json                           |
+-------------------------------------------------------------------+
```

### 2. R2 Sync Layer (`core/r2_sync.py`)

*   **S3 Client Integration:**
    *   `R2SyncManager` accepts an optional `s3_client` (duck-typed for `FakeS3Client` in TDD unit tests). In production, `from_secrets(secrets, local_data_dir="data")` builds a `boto3.client("s3", endpoint_url=..., aws_access_key_id=..., aws_secret_access_key=..., region_name="auto")` when all `[r2]` keys are present. If `[r2]` is absent, it returns a no-op manager (`is_configured() == False`, status `offline`); if `[r2]` is partially populated (missing required keys), it logs a warning, returns an unconfigured manager, and surfaces `degraded` status in the sidebar badge.
    *   Uses paginated `list_objects_v2` (`ContinuationToken`), `put_object`, `get_object`, and batch `delete_objects`.
*   **Sync Flows:**
    *   **Metadata Sync (`pull_metadata()`):** Guarded by a process-wide `threading.Lock` inside `@st.cache_resource` in `app.py`, running once on cold boot (`last_sync_at is None`) and whenever a user clicks `🔄 Sync Now`. Skips already-local immutable `runs/<run_id>.json` files. Pulls `datasets/*/config.json`, `datasets/*/split_manifest.json`, `runs/*.json`, and `gemini_cache/*/*.json`, merging `runs/*.json` into local `data/leaderboard_history.json` deduplicated by `id`.
    *   **Targeted Lazy Pulls:** `pull_dataset_prefix(dataset_id, prefix)` downloads only the needed prefix (`test_dataset/` on model evaluation/benchmarking, `raw/` when Dataset Manager opens), skipping files where `os.path.exists(path) and os.path.getsize(path) == remote_size` and pruning local files under `test_dataset/` that are absent from R2 (so `force_resplit=True` never leaves stale test images). `training_packs/*.zip` and `models/<run_id>/...` (`subdir` is `None` for single-stage models, or `"stage1"` / `"stage2"` for 2-stage pipeline models) are downloaded on demand when a user clicks download.
    *   **Atomic Writes & Observability:** All downloaded JSON/binary files write to a temp file in the target directory and swap via `os.replace`. Tracks `last_sync_at`, `last_sync_status` (`ok | degraded | offline`), and sanitized `last_sync_error` (never logging secrets). Surfaces an explicit UI warning banner if a remote `push` fails after 1 inline retry.

### 3. Multi-Dataset Schema & Registry (`core/dataset_config.py`)

*   **Dataclasses:** `StageConfig(stage_id, name, pos_label, neg_label, description)` (deriving `classes = [pos_label, neg_label]` and `targets = {pos_label: 1, neg_label: 0}` as properties) and `DatasetConfig(id, name, icon, description, stages, has_pipeline=False, split_version=1)`.
*   **Create New Dataset Form (`app.py`):** Collects dataset name, emoji icon, short description, and 1–4 binary stages (each with stage name, positive class label, and negative class label). Auto-slugifies names and rejects duplicate `dataset_id`s with a validation error. Dataset deletion is omitted from the open UI.
*   **Built-in Presets:** `invasive_plants` (`stage1`, `stage2`, `has_pipeline=True`) and `tutorial_cat` (`intro`). Generic `PipelineConfig` for arbitrary custom 2-stage cascades is deferred until a second pipeline dataset is needed.
*   **Isolation & Slug Validation:** Custom dataset IDs, stage IDs, and class names are validated against `^[a-z0-9_-]+$` to prevent path traversal or R2 key collisions. Each dataset is persisted separately at `data/datasets/<dataset_id>/config.json` (and `datasets/<dataset_id>/config.json` in R2).
*   **Legacy Mapping:** `DatasetRegistry.infer_dataset_id(stage)` maps legacy built-in stages (`intro -> tutorial_cat`, `stage1`/`stage2`/`pipeline -> invasive_plants`) only when a historical run lacks `dataset_id`.

### 4. Incremental Append-Only Splits & Raw Downsampling (`core/dataset_manager.py`)

*   **Path Resolution:** When `dataset_config` is `None`, `DatasetManager` preserves exact legacy paths (`data/raw`, `data/train_dataset`, `data/test_dataset`, `data/training_packs`) for backward compatibility. When `dataset_config` is provided, it isolates data under `data/datasets/<dataset_id>/`.
*   **Raw Downsampling & Safe Filenames on Intake:** `save_uploaded_images()` already sanitizes `os.path.basename`, converts valid images via Pillow to `.jpg`, and downsamples (`max_size=896, quality=90`); it is extended to prefix filenames unconditionally with `<uuid8>_` (preventing mobile camera collisions like `IMG_0001.jpg`) and push saved raw images to R2.
*   **Append-Only Splitting (`split_manifest.json`):** `process_and_split_dataset(test_ratio=0.3, seed=42, force_resplit=False)` reads `split_manifest.json` (`{split_version: int, assignments: { "<stage>/<class>/<filename>": "train" | "test" }}`).
    *   Before splitting, if R2 is configured, it pulls `raw/` so all prior images are present locally, and reconstructs any missing local `train_dataset/` and `test_dataset/` files from `raw/` using `split_manifest.json` assignments.
    *   When `force_resplit=False` (default): existing assigned images stay in their current `train_dataset/` or `test_dataset/` folders (ignoring missing `raw/` entries unless `Force Full Re-Split` is used); only newly added `raw/` files are split 70/30 and appended to the manifest and folders, and `training_packs/*.zip` are rebuilt. Only `raw/`, `test_dataset/`, `training_packs/`, and `split_manifest.json` are synced to R2 (`train_dataset/` is not duplicated to R2).
    *   When `force_resplit=True` (coach-gated via `GEMINI_ADMIN_PASSWORD` in Dataset Manager): increments `split_version`, clears and rebuilds local `train_dataset/` and `test_dataset/`, deletes stale R2 `test_dataset/` objects, and pushes the updated manifest, `test_dataset/`, and `training_packs/`.

### 5. Evaluators, Storage & Gemini Cache (`core/storage.py`, `core/evaluator.py`, `core/pipeline.py`, `core/gemini_benchmark.py`)

*   **`StorageManager` (`core/storage.py`):**
    *   `save_run()` stamps `dataset_id` and `split_version` onto the run record, appends to local `leaderboard_history.json`, and pushes `runs/<run_id>.json` to R2.
    *   `get_leaderboard(stage, dataset_id=None)` and `get_history(team_name=None, stage=None, dataset_id=None)` filter by `(dataset_id, stage)` (using `infer_dataset_id` for legacy runs). In the UI, leaderboards default to showing runs for the active `split_version` (with a toggle to inspect prior versions).
    *   `app.py` calls `storage.save_model_artifact()` after evaluation so `models/<run_id>/[stage1|stage2/]submission.zip` and `labels.txt` are always pushed to R2.
*   **`ModelEvaluator` & `PipelineEvaluator` (`core/evaluator.py`, `core/pipeline.py`):**
    *   `ModelEvaluator` accepts `DatasetConfig` / `StageConfig` so positive/negative class indices are resolved from config instead of the hardcoded `STAGE_TARGETS` dictionary (which maps built-in class names to binary `0`/`1` targets) and heuristic fallbacks (retaining `STAGE_TARGETS` when `dataset_config is None`).
    *   `PipelineEvaluator` stays scoped to `invasive_plants`, accepting `test_dataset_dir: str` so it reads from `data/datasets/invasive_plants/test_dataset` when active.
*   **`GeminiBenchmarkRunner` (`core/gemini_benchmark.py`):**
    *   Keys cached results by `(dataset_id, stage)` in `data/gemini_cache/<dataset_id>/<stage>.json` (recording `split_version` on each entry, with read fallback for legacy `data/gemini_cache.json`). Generates default VLM (Vision-Language Model) and Gemini Decisions API prompts from `StageConfig` class labels when a custom dataset has no preset prompt.

--------------------------------------------------------------------------------

## Non-Functional Requirements

*   **Security & Secrets Isolation:**
    *   All credentials (`GEMINI_API_KEY`, `GEMINI_ADMIN_PASSWORD`, `[r2].endpoint_url`, `[r2].bucket`, `[r2].access_key_id`, `[r2].secret_access_key`) live in `.streamlit/secrets.toml` (gitignored) or Streamlit Cloud Secrets.
    *   `.streamlit/secrets.toml.example` and `README.md` link directly to official Cloudflare R2 API token docs, Google AI Studio, and Streamlit Secrets Management docs.
    *   Live Gemini runs and `Force Full Re-Split` in `app.py` require a non-empty `GEMINI_ADMIN_PASSWORD` verified via `hmac.compare_digest` (failing closed if `GEMINI_ADMIN_PASSWORD` is unset or empty), with a "Lock Coach Mode" button and sanitized error messages that never echo secrets.
*   **Performance & Reliability:**
    *   Cold boot performs one coordinated metadata sync; `raw/` and `test_dataset/` prefixes are lazy-loaded with local `size`-matching checks (`os.path.getsize(path) == remote_size`) and `test_dataset/` pruning, while `training_packs/*.zip` and model `.zip`s are downloaded on demand.
    *   R2 transient failures log structured diagnostics (`mips.*`) and fall back to local cache without crashing the app, surfacing a UI warning if a remote push fails.
*   **Backward Compatibility:**
    *   All 31 existing unit tests must continue to pass without modification; local offline execution works with zero secrets configured.

--------------------------------------------------------------------------------

## Alternatives Considered

### Alternative A: Personal Google Drive via OAuth Refresh Token
*   **Pros:** Zero hosting cost (15 GB free), and kids could drop photos directly into a shared Drive folder outside MIPS.
*   **Cons:** Requires full `https://www.googleapis.com/auth/drive` scope, exposing the coach's personal Drive files to any app compromise; personal Google Cloud OAuth apps in "Testing" mode expire refresh tokens every 7 days; Drive v3 `files.create` allows duplicate filenames and paginates at 100 items.
*   **Reason for Rejection:** The coach did not want personal Google Drive exposed, and in-app photo upload via MIPS is acceptable for students.

### Alternative B: Dedicated Club Google Account + Google Drive
*   **Pros:** Isolates the coach's personal Drive while still allowing external Drive folder drops.
*   **Cons:** Still requires creating and maintaining a separate Google account, configuring Google Cloud OAuth consent screens ("In Production"), and handling Drive v3 duplicate-file/pagination quirks.
*   **Reason for Rejection:** Unnecessary once in-app photo upload through MIPS was accepted, making bucket-scoped object storage simpler.

### Alternative C: Private Hugging Face Dataset Repo (`huggingface_hub`)
*   **Pros:** Free without a credit card on file, single-repo fine-grained token, and built-in Git version history.
*   **Cons:** Backed by Git commits, so concurrent student submissions (`runs/<run_id>.json`) and photo uploads (~1–2s per commit) contend on commit locks and require retry loops.
*   **Reason for Rejection:** Cloudflare R2 avoids Git commit lock contention and handles concurrent student writes directly.

### Alternative D: Single Shared `leaderboard_history.json` on Remote Storage
*   **Pros:** Mirrors the current local `StorageManager` file format with one object read/write.
*   **Cons:** Concurrent submissions from two students at the same time create a read-modify-write race that silently drops one student's run.
*   **Reason for Rejection:** Per-run `runs/<run_id>.json` files merged locally eliminate concurrent submission overwrites.

### Alternative E: GitHub Snapshots or Manual Export/Import `.zip` Bundles
*   **Pros:** Zero external infrastructure or credentials beyond GitHub.
*   **Cons:** Streamlit Cloud cannot push runs back to GitHub automatically, and manual export/import bundles require coach intervention before container sleep wipes cloud runs.
*   **Reason for Rejection:** Shared external storage keeps local and Streamlit Cloud runs synchronized automatically.

### Alternative F: Cross-Dataset Aggregate Team Leaderboard
*   **Pros:** Gives a single summary view across all datasets.
*   **Cons:** Blending F1/Accuracy across tasks with different class balances, difficulties, and test-set sizes is misleading, especially when teams complete different subsets of datasets.
*   **Reason for Rejection:** Averaging metrics across unrelated datasets with different class balances and test-set sizes produces misleading comparisons.

--------------------------------------------------------------------------------

## Detailed Implementation Plan

All implementation follows test-driven development (TDD): each module's failing unit tests are written and verified before implementing the module, and all 31 existing tests in `./tests/` must continue to pass.

### Target Files to Create / Modify

#### 1. `./requirements.txt` (Modify)
*   **What & Why:** Add `boto3>=1.34.0` so `R2SyncManager` can communicate with Cloudflare R2's S3-compatible API.

#### 2. `./.gitignore` (Modify)
*   **What & Why:** Add `data/datasets/` and `data/gemini_cache/` so local multi-dataset caches, split manifests, and Gemini cache files are never committed to Git.

#### 3. `./.streamlit/secrets.toml.example` (Create)
*   **What & Why:** Provide a non-secret template documenting `GEMINI_API_KEY`, `GEMINI_ADMIN_PASSWORD`, and `[r2]` (`endpoint_url`, `bucket`, `access_key_id`, `secret_access_key`) with inline links to official Cloudflare R2, Google AI Studio, and Streamlit Secrets documentation.

#### 4. `./README.md` (Modify)
*   **What & Why:** Update the deployment/configuration section with official documentation links for obtaining Cloudflare R2 API tokens, `GEMINI_API_KEY`, and configuring Streamlit Community Cloud Secrets (`.streamlit/secrets.toml`).

#### 5. `./core/r2_sync.py` (Create)
*   **What & Why:** Implement `FakeS3Client` (supporting `list_objects_v2` with `MaxKeys`/`IsTruncated`/`NextContinuationToken`, `put_object`, `get_object` returning `{"Body": BytesIO(...)}`, and `delete_objects`) and `R2SyncManager` accepting an optional `s3_client`:
    *   `from_secrets(secrets, local_data_dir)` returning a safe no-op manager when `[r2]` is absent (status `offline`), or `degraded` when `[r2]` is partially populated.
    *   `pull_metadata()` (cold-boot sync for `datasets/*/config.json`, `datasets/*/split_manifest.json`, `runs/*.json`, `gemini_cache/*/*.json`, skipping `runs/<run_id>.json` whose `id` is already in `leaderboard_history.json`, preserving local-only runs, and sorting merged runs by `timestamp`).
    *   `pull_dataset_prefix(dataset_id: str, prefix: str) -> int` (targeted lazy sync of `"raw"` or `"test_dataset"` skipping existing local files with matching `size`, and pruning local `test_dataset/` files absent from R2).
    *   `ensure_training_pack_local(dataset_id: str, stage: str) -> bool` and `ensure_model_artifact_local(run_id: str, local_models_dir: str, subdir: Optional[str] = None) -> bool` for on-demand `.zip` downloads (`subdir` is `None` for single-stage models, `"stage1"`/`"stage2"` for pipeline models).
    *   `push_run(record)`, `push_model_artifact(run_id, local_dir, subdir=None)`, `push_file(remote_key, local_path)` (for configs, manifests, and Gemini cache), and `delete_prefix(prefix)` (for `force_resplit=True` cleanup).
    *   Atomic local writes (`os.replace`), `threading.Lock`, structured `mips.r2_sync` logging, and `last_sync_at` / `last_sync_status` / sanitized `last_sync_error` tracking.

#### 6. `./core/dataset_config.py` (Create)
*   **What & Why:** Define `StageConfig(stage_id, name, pos_label, neg_label, description)` (with derived `classes` and `targets` properties), `DatasetConfig`, and `DatasetRegistry(datasets_dir: str = "data/datasets", r2_sync: Optional[R2SyncManager] = None)`:
    *   Expose `list_datasets()`, `get_dataset(dataset_id)`, `register_dataset(dataset)`, `validate_slug(value)` (`^[a-z0-9_-]+$`), and `infer_dataset_id(stage)` for legacy runs lacking `dataset_id`.
    *   Auto-seed `invasive_plants` (`has_pipeline=True`) and `tutorial_cat` `config.json` files when `datasets_dir` is empty; persist each dataset as `data/datasets/<dataset_id>/config.json` and sync via `R2SyncManager`.

#### 7. `./core/dataset_manager.py` (Modify)
*   **What & Why:**
    *   Accept optional `dataset_config: Optional[DatasetConfig] = None` and `r2_sync: Optional[R2SyncManager] = None` while keeping legacy `data/` paths when `dataset_config is None`.
    *   Update `save_uploaded_images()` to prefix filenames unconditionally with `<uuid8>_` and push saved raw images to R2.
    *   Persist `split_manifest.json` (`split_version`, `assignments`) and update `process_and_split_dataset(test_ratio=0.3, seed=42, force_resplit=False)` to reconstruct missing local `train_dataset/`/`test_dataset/` from `raw/` + `split_manifest.json`, perform append-only splitting by default (`n_test = max(1, int(round(n_new * test_ratio))) if n_new >= 2 else 0` for new files), invoke `create_student_training_packs()` internally, and on `force_resplit=True` increment `split_version` in both `split_manifest.json` and `config.json` + clean stale `test_dataset/`.
    *   Push uploaded raw images, `test_dataset/`, `training_packs/`, and `split_manifest.json` to R2 (without duplicating `train_dataset/`).

#### 8. `./core/storage.py` (Modify)
*   **What & Why:**
    *   Accept optional `r2_sync: Optional[R2SyncManager] = None`.
    *   `save_run()` stamps `dataset_id` (defaulting via `DatasetRegistry.infer_dataset_id(stage)`) and `split_version` (defaulting to `1`) onto the run record and pushes `runs/<run_id>.json` to R2.
    *   `get_leaderboard(stage, dataset_id=None)` and `get_history(team_name=None, stage=None, dataset_id=None)` filter by `dataset_id` only when `dataset_id is not None` (using `DatasetRegistry.infer_dataset_id` fallback for legacy runs).
    *   `save_model_artifact()` pushes `models/<run_id>/[stage1|stage2/]...` to R2; `get_model_artifact_bytes()` lazy-downloads missing model `.zip`s via `ensure_model_artifact_local()`.
    *   Add `verify_admin_password(candidate: str, expected: Optional[str]) -> bool` (fail-closed if `expected` is empty/None, verified with `hmac.compare_digest`) so `tests/test_gemini_gate.py` can test it without importing Streamlit.

#### 9. `./core/evaluator.py` (Modify)
*   **What & Why:** Accept optional `dataset_config: Optional[DatasetConfig] = None` in `ModelEvaluator` so `resolve_class_indices()` and `evaluate_submission()` use `StageConfig.pos_label` / `neg_label` (with the existing `not_`/`non_` and 2-class fallback) instead of hardcoded `STAGE_TARGETS` (retaining `STAGE_TARGETS` when `dataset_config is None`).

#### 10. `./core/pipeline.py` (Modify)
*   **What & Why:** Keep `PipelineEvaluator` scoped to `invasive_plants` with `__init__(self, evaluator: Optional[ModelEvaluator] = None, test_dataset_dir: Optional[str] = None)` (preferring `evaluator` if both are supplied) so it reads from `data/datasets/invasive_plants/test_dataset` when active.

#### 11. `./core/gemini_benchmark.py` (Modify)
*   **What & Why:** Accept optional `dataset_config: Optional[DatasetConfig] = None`, `r2_sync: Optional[R2SyncManager] = None`, and `cache_dir: str = "data/gemini_cache"`. Store `{ "<preset>": { ..., "split_version": int } }` in `data/gemini_cache/<dataset_id>/<stage>.json` (restricting legacy `data/gemini_cache.json` fallback to `invasive_plants` and `tutorial_cat`), add `get_decision_spec(stage)` and default prompt generation from `StageConfig` for custom datasets, and push updated cache files to R2.

#### 12. `./app.py` (Modify)
*   **What & Why:**
    *   Initialize a process-wide `@st.cache_resource` `R2SyncManager` (`threading.Lock`) calling `pull_metadata()` once on cold boot, `pull_dataset_prefix(active_dataset_id, ...)` lazily for `raw/` and `test_dataset/`, and `ensure_training_pack_local(...)` on pack download.
    *   Add sidebar controls: Dataset Switcher, `➕ Create New Dataset` form (slug-validated), and R2 Sync status badge + `🔄 Sync Now` button.
    *   Render board tabs, `render_submit_tab`, and `render_dataset_tab` dynamically from `active_dataset.stages` instead of `DatasetManager.STAGES` / `STAGE_LABELS` (showing Pipeline tab only when `active_dataset.has_pipeline` is `True`), displaying `split_version` badges on leaderboard runs.
    *   Update `seed_cat_dataset()` to write into `os.path.join(dataset_mgr.raw_dir, "intro")` and push seeded files to R2 (shown only when `tutorial_cat` is active).
    *   Pass `dataset_config`, `r2_sync`, and `test_images_dir=dataset_mgr.test_dir` to `GeminiBenchmarkRunner` and use `get_decision_spec(stage)` instead of direct `DECISION_CRITERIA` indexing.
    *   Buffer uploaded model bytes (`s1_bytes`, `s2_bytes`) and pass `save_dir=None` to `evaluate_submission()` / `evaluate_pipeline()`, then call `storage.save_model_artifact()` directly so all submitted models are pushed to R2.
    *   Gate live Gemini benchmark execution and `Force Full Re-Split` behind `GEMINI_ADMIN_PASSWORD` using `verify_admin_password` from `core/storage.py` (per-session `st.session_state` unlock, "Lock Coach Mode" button, and sanitized error messages).

#### 13. Test Suite Additions (`./tests/`) (Create)
*   **`./tests/test_r2_sync.py`:** Paginated `FakeS3Client`, `from_secrets` unconfigured/partial (`degraded`) fallback, concurrent `push_run` + `pull_metadata` merge, lazy `ensure_model_artifact_local`, `size`-match skip, and atomic writes.
*   **`./tests/test_dataset_config.py`:** Built-in presets, `infer_dataset_id`, slug validation, and per-dataset `config.json` persistence + R2 push.
*   **`./tests/test_storage_r2_and_datasets.py`:** `StorageManager` + `DatasetManager` with `FakeS3Client`, `get_leaderboard`/`get_history` filtering by `dataset_id`, append-only `split_manifest.json` preservation vs. `force_resplit=True` version bump, and raw upload downsampling.
*   **`./tests/test_multidataset_eval.py`:** Dynamic `ModelEvaluator`, `PipelineEvaluator`, and `(dataset_id, stage)`-scoped `GeminiBenchmarkRunner` caching.
*   **`./tests/test_gemini_gate.py`:** Constant-time `GEMINI_ADMIN_PASSWORD` verification helper (accepts exact match, rejects empty/None/mismatch).
