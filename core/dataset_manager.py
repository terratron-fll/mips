import io
import json
import os
import random
import re
import shutil
import uuid
import zipfile
from typing import Dict, List, Optional, Tuple
from PIL import Image

from core.dataset_config import DatasetConfig
from core.r2_sync import R2SyncManager

class DatasetManager:
    """Manages raw intake, downsampling, train/test splitting, and zip packaging."""

    STAGES: Dict[str, List[str]] = {
        "intro": ["cat", "not_cat"],
        "stage1": ["plant", "not_plant"],
        "stage2": ["invasive", "non_invasive"],
    }

    def __init__(
        self,
        base_data_dir: str = "data",
        dataset_config: Optional[DatasetConfig] = None,
        r2_sync: Optional[R2SyncManager] = None,
    ) -> None:
        self.base_data_dir = base_data_dir
        self.dataset_config = dataset_config
        self.r2_sync = r2_sync
        if dataset_config is not None:
            self.dataset_root = os.path.join(base_data_dir, "datasets", dataset_config.id)
            self.stages: Dict[str, List[str]] = {
                sid: scfg.classes for sid, scfg in dataset_config.stages.items()
            }
        else:
            self.dataset_root = base_data_dir
            self.stages = dict(self.STAGES)

        self.raw_dir = os.path.join(self.dataset_root, "raw")
        self.test_dir = os.path.join(self.dataset_root, "test_dataset")
        self.train_dir = os.path.join(self.dataset_root, "train_dataset")
        self.packs_dir = os.path.join(self.dataset_root, "training_packs")
        self.manifest_path = os.path.join(self.dataset_root, "split_manifest.json")
        self._ensure_dirs()

    def _ensure_dirs(self) -> None:
        for stage, classes in self.stages.items():
            for cls in classes:
                os.makedirs(os.path.join(self.raw_dir, stage, cls), exist_ok=True)
                os.makedirs(os.path.join(self.test_dir, stage, cls), exist_ok=True)
                os.makedirs(os.path.join(self.train_dir, stage, cls), exist_ok=True)
        os.makedirs(self.packs_dir, exist_ok=True)

    def get_dataset_stats(self) -> Dict[str, Dict[str, int]]:
        stats: Dict[str, Dict[str, int]] = {}
        for stage, classes in self.stages.items():
            stats[stage] = {}
            for cls in classes:
                folder = os.path.join(self.raw_dir, stage, cls)
                if os.path.exists(folder):
                    files = [
                        f
                        for f in os.listdir(folder)
                        if f.lower().endswith((".jpg", ".jpeg", ".png", ".webp"))
                    ]
                    stats[stage][cls] = len(files)
                else:
                    stats[stage][cls] = 0
        return stats

    def downsample_and_save_image(
        self, src_path: str, dest_path: str, max_size: int = 448, quality: int = 85
    ) -> bool:
        try:
            with Image.open(src_path) as img:
                img = img.convert("RGB")
                img.thumbnail((max_size, max_size), Image.Resampling.LANCZOS)
                os.makedirs(os.path.dirname(dest_path), exist_ok=True)
                img.save(dest_path, "JPEG", quality=quality, optimize=True)
            return True
        except Exception:
            return False

    def _load_manifest(self) -> Dict[str, object]:
        if os.path.exists(self.manifest_path):
            try:
                with open(self.manifest_path, "r", encoding="utf-8") as f:
                    data = json.load(f)
                    if isinstance(data, dict):
                        return {
                            "split_version": int(data.get("split_version", 1)),
                            "assignments": dict(data.get("assignments", {})),
                        }
            except (json.JSONDecodeError, ValueError, TypeError):
                pass
        return {"split_version": 1, "assignments": {}}

    def _save_manifest(self, manifest: Dict[str, object]) -> None:
        os.makedirs(os.path.dirname(self.manifest_path), exist_ok=True)
        with open(self.manifest_path, "w", encoding="utf-8") as f:
            json.dump(manifest, f, indent=2)

    def get_split_version(self) -> int:
        return int(self._load_manifest()["split_version"])

    def process_and_split_dataset(
        self, test_ratio: float = 0.30, seed: int = 42, force_resplit: bool = False
    ) -> Dict[str, Dict[str, int]]:
        random.seed(seed)
        if self.dataset_config is not None and self.r2_sync and self.r2_sync.is_configured():
            self.r2_sync.pull_dataset_prefix(self.dataset_config.id, "raw")

        manifest = self._load_manifest()
        assignments: Dict[str, str] = dict(manifest["assignments"])  # type: ignore[arg-type]
        split_version = int(manifest["split_version"])  # type: ignore[arg-type]

        if force_resplit:
            split_version += 1
            assignments = {}

        summary: Dict[str, Dict[str, int]] = {}
        new_test_paths: List[Tuple[str, str]] = []

        for stage, classes in self.stages.items():
            summary[stage] = {"train": 0, "test": 0}

            for cls in classes:
                t_dir = os.path.join(self.test_dir, stage, cls)
                tr_dir = os.path.join(self.train_dir, stage, cls)
                if force_resplit:
                    if os.path.exists(t_dir):
                        shutil.rmtree(t_dir)
                    if os.path.exists(tr_dir):
                        shutil.rmtree(tr_dir)
                os.makedirs(t_dir, exist_ok=True)
                os.makedirs(tr_dir, exist_ok=True)

                raw_folder = os.path.join(self.raw_dir, stage, cls)
                if not os.path.exists(raw_folder):
                    continue

                images = sorted(
                    f
                    for f in os.listdir(raw_folder)
                    if f.lower().endswith((".jpg", ".jpeg", ".png", ".webp"))
                )

                # Separate already-assigned images from newly added images
                unassigned: List[str] = []
                for fname in images:
                    key = f"{stage}/{cls}/{fname}"
                    existing_split = assignments.get(key)
                    base_name = os.path.splitext(fname)[0] + ".jpg"
                    src = os.path.join(raw_folder, fname)
                    if existing_split == "test":
                        dest = os.path.join(t_dir, base_name)
                        if os.path.exists(dest) or self.downsample_and_save_image(src, dest):
                            summary[stage]["test"] += 1
                    elif existing_split == "train":
                        dest = os.path.join(tr_dir, base_name)
                        if os.path.exists(dest) or self.downsample_and_save_image(src, dest):
                            summary[stage]["train"] += 1
                    else:
                        unassigned.append(fname)

                if unassigned:
                    random.shuffle(unassigned)
                    n_new = len(unassigned)
                    n_test = max(1, int(round(n_new * test_ratio))) if n_new >= 2 else 0
                    test_images = unassigned[:n_test]
                    train_images = unassigned[n_test:]

                    for fname in test_images:
                        key = f"{stage}/{cls}/{fname}"
                        base_name = os.path.splitext(fname)[0] + ".jpg"
                        src = os.path.join(raw_folder, fname)
                        dest = os.path.join(t_dir, base_name)
                        if self.downsample_and_save_image(src, dest):
                            assignments[key] = "test"
                            summary[stage]["test"] += 1
                            new_test_paths.append((f"{stage}/{cls}/{base_name}", dest))

                    for fname in train_images:
                        key = f"{stage}/{cls}/{fname}"
                        base_name = os.path.splitext(fname)[0] + ".jpg"
                        src = os.path.join(raw_folder, fname)
                        dest = os.path.join(tr_dir, base_name)
                        if self.downsample_and_save_image(src, dest):
                            assignments[key] = "train"
                            summary[stage]["train"] += 1

        self._save_manifest({"split_version": split_version, "assignments": assignments})
        self.create_student_training_packs()

        if self.dataset_config is not None:
            self.dataset_config.split_version = split_version
            cfg_path = os.path.join(self.dataset_root, "config.json")
            if os.path.exists(cfg_path):
                with open(cfg_path, "w", encoding="utf-8") as f:
                    json.dump(self.dataset_config.to_dict(), f, indent=2)

            if self.r2_sync and self.r2_sync.is_configured():
                ds_id = self.dataset_config.id
                if force_resplit:
                    self.r2_sync.delete_prefix(f"datasets/{ds_id}/test_dataset/")
                    for stage, classes in self.stages.items():
                        for cls in classes:
                            t_dir = os.path.join(self.test_dir, stage, cls)
                            if os.path.isdir(t_dir):
                                for fname in os.listdir(t_dir):
                                    fpath = os.path.join(t_dir, fname)
                                    if os.path.isfile(fpath):
                                        self.r2_sync.push_file(
                                            f"datasets/{ds_id}/test_dataset/{stage}/{cls}/{fname}",
                                            fpath,
                                        )
                else:
                    for rel_key, dest in new_test_paths:
                        self.r2_sync.push_file(f"datasets/{ds_id}/test_dataset/{rel_key}", dest)

                self.r2_sync.push_file(f"datasets/{ds_id}/split_manifest.json", self.manifest_path)
                if os.path.exists(cfg_path):
                    self.r2_sync.push_file(f"datasets/{ds_id}/config.json", cfg_path)

        return summary

    def save_uploaded_images(
        self,
        stage: str,
        cls: str,
        files: List[Tuple[str, bytes]],
        max_size: int = 896,
        quality: int = 90,
    ) -> Tuple[int, List[str]]:
        """Validate, sanitize, and save uploaded image bytes into raw/<stage>/<cls>."""
        if stage not in self.stages:
            raise ValueError(f"Unknown stage: {stage}")
        if cls not in self.stages[stage]:
            raise ValueError(f"Invalid class '{cls}' for stage '{stage}'")

        dest_dir = os.path.join(self.raw_dir, stage, cls)
        os.makedirs(dest_dir, exist_ok=True)

        saved_count = 0
        errors: List[str] = []

        for orig_name, data in files:
            safe_base = os.path.basename(orig_name or "upload.jpg")
            stem = re.sub(r"[^A-Za-z0-9_-]", "_", os.path.splitext(safe_base)[0]).strip("_") or "image"
            fname = f"{uuid.uuid4().hex[:8]}_{stem}.jpg"
            dest_path = os.path.join(dest_dir, fname)

            try:
                with Image.open(io.BytesIO(data)) as img:
                    img = img.convert("RGB")
                    img.thumbnail((max_size, max_size), Image.Resampling.LANCZOS)
                    img.save(dest_path, "JPEG", quality=quality, optimize=True)
                saved_count += 1
                if self.dataset_config is not None and self.r2_sync and self.r2_sync.is_configured():
                    self.r2_sync.push_file(
                        f"datasets/{self.dataset_config.id}/raw/{stage}/{cls}/{fname}",
                        dest_path,
                    )
            except Exception as exc:
                errors.append(f"{safe_base}: {exc}")

        return saved_count, errors

    def create_student_training_packs(self) -> Dict[str, str]:
        pack_paths: Dict[str, str] = {}
        for stage, classes in self.stages.items():
            zip_filename = os.path.join(self.packs_dir, f"{stage}_train.zip")
            with zipfile.ZipFile(zip_filename, "w", zipfile.ZIP_DEFLATED) as z:
                for cls in classes:
                    folder = os.path.join(self.train_dir, stage, cls)
                    if os.path.exists(folder):
                        for fname in sorted(os.listdir(folder)):
                            fpath = os.path.join(folder, fname)
                            if os.path.isfile(fpath):
                                arcname = f"{cls}/{fname}"
                                z.write(fpath, arcname)
            pack_paths[stage] = zip_filename
            if self.dataset_config is not None and self.r2_sync and self.r2_sync.is_configured():
                self.r2_sync.push_file(
                    f"datasets/{self.dataset_config.id}/training_packs/{stage}_train.zip",
                    zip_filename,
                )
        return pack_paths
