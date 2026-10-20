import os
import random
import shutil
import zipfile
from typing import Dict, List, Tuple
from PIL import Image

class DatasetManager:
    """Manages raw intake, downsampling, train/test splitting, and zip packaging."""

    STAGES: Dict[str, List[str]] = {
        "stage1": ["plant", "not_plant"],
        "stage2": ["invasive", "non_invasive"],
    }

    def __init__(self, base_data_dir: str = "data") -> None:
        self.base_data_dir = base_data_dir
        self.raw_dir = os.path.join(base_data_dir, "raw")
        self.test_dir = os.path.join(base_data_dir, "test_dataset")
        self.train_dir = os.path.join(base_data_dir, "train_dataset")
        self.packs_dir = os.path.join(base_data_dir, "training_packs")
        self._ensure_dirs()

    def _ensure_dirs(self) -> None:
        for stage, classes in self.STAGES.items():
            for cls in classes:
                os.makedirs(os.path.join(self.raw_dir, stage, cls), exist_ok=True)
                os.makedirs(os.path.join(self.test_dir, stage, cls), exist_ok=True)
                os.makedirs(os.path.join(self.train_dir, stage, cls), exist_ok=True)
        os.makedirs(self.packs_dir, exist_ok=True)

    def get_dataset_stats(self) -> Dict[str, Dict[str, int]]:
        stats: Dict[str, Dict[str, int]] = {}
        for stage, classes in self.STAGES.items():
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

    def process_and_split_dataset(
        self, test_ratio: float = 0.30, seed: int = 42
    ) -> Dict[str, Dict[str, int]]:
        random.seed(seed)
        summary: Dict[str, Dict[str, int]] = {}

        for stage, classes in self.STAGES.items():
            summary[stage] = {"train": 0, "test": 0}

            # Clear existing test and train folders for this stage
            for cls in classes:
                t_dir = os.path.join(self.test_dir, stage, cls)
                tr_dir = os.path.join(self.train_dir, stage, cls)
                if os.path.exists(t_dir):
                    shutil.rmtree(t_dir)
                if os.path.exists(tr_dir):
                    shutil.rmtree(tr_dir)
                os.makedirs(t_dir, exist_ok=True)
                os.makedirs(tr_dir, exist_ok=True)

                raw_folder = os.path.join(self.raw_dir, stage, cls)
                if not os.path.exists(raw_folder):
                    continue

                images = [
                    f
                    for f in os.listdir(raw_folder)
                    if f.lower().endswith((".jpg", ".jpeg", ".png", ".webp"))
                ]
                random.shuffle(images)

                # Determine test split count (at least 1 if >=2 images exist)
                n_total = len(images)
                n_test = max(1, int(round(n_total * test_ratio))) if n_total >= 2 else 0
                test_images = images[:n_test]
                train_images = images[n_test:]

                for fname in test_images:
                    src = os.path.join(raw_folder, fname)
                    dest = os.path.join(t_dir, fname)
                    if self.downsample_and_save_image(src, dest):
                        summary[stage]["test"] += 1

                for fname in train_images:
                    src = os.path.join(raw_folder, fname)
                    dest = os.path.join(tr_dir, fname)
                    if self.downsample_and_save_image(src, dest):
                        summary[stage]["train"] += 1

        return summary

    def create_student_training_packs(self) -> Dict[str, str]:
        pack_paths: Dict[str, str] = {}
        for stage, classes in self.STAGES.items():
            zip_filename = os.path.join(self.packs_dir, f"{stage}_train.zip")
            with zipfile.ZipFile(zip_filename, "w", zipfile.ZIP_DEFLATED) as z:
                for cls in classes:
                    folder = os.path.join(self.train_dir, stage, cls)
                    if os.path.exists(folder):
                        for fname in os.listdir(folder):
                            fpath = os.path.join(folder, fname)
                            if os.path.isfile(fpath):
                                arcname = f"{cls}/{fname}"
                                z.write(fpath, arcname)
            pack_paths[stage] = zip_filename
        return pack_paths
