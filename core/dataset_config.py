import json
import os
import re
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional

from core.r2_sync import R2SyncManager

SLUG_PATTERN = re.compile(r"^[a-z0-9_-]+$")


@dataclass
class StageConfig:
    """Configuration for a single binary classification stage within a dataset."""

    stage_id: str
    name: str
    pos_label: str
    neg_label: str
    description: str = ""

    @property
    def classes(self) -> List[str]:
        return [self.pos_label, self.neg_label]

    @property
    def targets(self) -> Dict[str, int]:
        return {self.pos_label: 1, self.neg_label: 0}

    def to_dict(self) -> Dict[str, Any]:
        return {
            "stage_id": self.stage_id,
            "name": self.name,
            "pos_label": self.pos_label,
            "neg_label": self.neg_label,
            "description": self.description,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "StageConfig":
        return cls(
            stage_id=data["stage_id"],
            name=data["name"],
            pos_label=data["pos_label"],
            neg_label=data["neg_label"],
            description=data.get("description", ""),
        )


@dataclass
class DatasetConfig:
    """Configuration for an isolated MIPS dataset and its classification stages."""

    id: str
    name: str
    icon: str = "📦"
    description: str = ""
    stages: Dict[str, StageConfig] = field(default_factory=dict)
    has_pipeline: bool = False
    split_version: int = 1

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "name": self.name,
            "icon": self.icon,
            "description": self.description,
            "stages": {k: v.to_dict() for k, v in self.stages.items()},
            "has_pipeline": self.has_pipeline,
            "split_version": self.split_version,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "DatasetConfig":
        raw_stages = data.get("stages", {})
        stages = {k: StageConfig.from_dict(v) for k, v in raw_stages.items()}
        return cls(
            id=data["id"],
            name=data["name"],
            icon=data.get("icon", "📦"),
            description=data.get("description", ""),
            stages=stages,
            has_pipeline=bool(data.get("has_pipeline", False)),
            split_version=int(data.get("split_version", 1)),
        )


def _built_in_presets() -> List[DatasetConfig]:
    return [
        DatasetConfig(
            id="invasive_plants",
            name="Invasive Plants",
            icon="🌿",
            description="FLL two-stage plant filter and invasive species classifier.",
            has_pipeline=True,
            stages={
                "stage1": StageConfig(
                    stage_id="stage1",
                    name="Board 1: Plant vs Not-Plant",
                    pos_label="plant",
                    neg_label="not_plant",
                ),
                "stage2": StageConfig(
                    stage_id="stage2",
                    name="Board 2: Invasive vs Non-Invasive",
                    pos_label="invasive",
                    neg_label="non_invasive",
                ),
            },
        ),
        DatasetConfig(
            id="tutorial_cat",
            name="Tutorial: Cat vs Not-Cat",
            icon="🐱",
            description="Onboarding Hello World dataset.",
            has_pipeline=False,
            stages={
                "intro": StageConfig(
                    stage_id="intro",
                    name="Tutorial: Cat vs Not-Cat",
                    pos_label="cat",
                    neg_label="not_cat",
                ),
            },
        ),
    ]


class DatasetRegistry:
    """Manages per-dataset config.json files under data/datasets/<dataset_id>/."""

    LEGACY_STAGE_TO_DATASET: Dict[str, str] = {
        "intro": "tutorial_cat",
        "stage1": "invasive_plants",
        "stage2": "invasive_plants",
        "pipeline": "invasive_plants",
    }

    def __init__(
        self,
        datasets_dir: str = "data/datasets",
        r2_sync: Optional[R2SyncManager] = None,
    ) -> None:
        self.datasets_dir = datasets_dir
        self.r2_sync = r2_sync
        self._ensure_presets()

    @staticmethod
    def validate_slug(value: str) -> bool:
        return bool(value and SLUG_PATTERN.match(value))

    @staticmethod
    def slugify(text: str) -> str:
        slug = re.sub(r"[^a-z0-9_-]+", "_", text.strip().lower())
        slug = re.sub(r"_+", "_", slug).strip("_-")
        return slug

    @classmethod
    def infer_dataset_id(cls, stage: str) -> str:
        return cls.LEGACY_STAGE_TO_DATASET.get(stage, "invasive_plants")

    def _config_path(self, dataset_id: str) -> str:
        return os.path.join(self.datasets_dir, dataset_id, "config.json")

    def _ensure_presets(self) -> None:
        os.makedirs(self.datasets_dir, exist_ok=True)
        existing_configs = [
            name
            for name in os.listdir(self.datasets_dir)
            if os.path.exists(self._config_path(name))
        ]
        if not existing_configs:
            for preset in _built_in_presets():
                self._write_config(preset, push_r2=False)

    def _write_config(self, dataset: DatasetConfig, push_r2: bool = True) -> None:
        path = self._config_path(dataset.id)
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(dataset.to_dict(), f, indent=2)
        if push_r2 and self.r2_sync and self.r2_sync.is_configured():
            self.r2_sync.push_file(f"datasets/{dataset.id}/config.json", path)

    def list_datasets(self) -> List[DatasetConfig]:
        if not os.path.isdir(self.datasets_dir):
            return []
        configs: List[DatasetConfig] = []
        for entry in sorted(os.listdir(self.datasets_dir)):
            cfg_path = self._config_path(entry)
            if os.path.isfile(cfg_path):
                try:
                    with open(cfg_path, "r", encoding="utf-8") as f:
                        configs.append(DatasetConfig.from_dict(json.load(f)))
                except (json.JSONDecodeError, KeyError, TypeError):
                    continue
        # Keep built-in presets first in stable order
        order = {"invasive_plants": 0, "tutorial_cat": 1}
        configs.sort(key=lambda d: (order.get(d.id, 2), d.name.lower()))
        return configs

    def get_dataset(self, dataset_id: str) -> Optional[DatasetConfig]:
        cfg_path = self._config_path(dataset_id)
        if not os.path.isfile(cfg_path):
            return None
        with open(cfg_path, "r", encoding="utf-8") as f:
            return DatasetConfig.from_dict(json.load(f))

    def register_dataset(self, dataset: DatasetConfig, overwrite: bool = False) -> None:
        if not self.validate_slug(dataset.id):
            raise ValueError(f"Invalid dataset id '{dataset.id}'. Use lowercase letters, digits, _ or -.")
        if not dataset.stages:
            raise ValueError("Dataset must define at least one stage.")
        for stage_id, stage in dataset.stages.items():
            if not self.validate_slug(stage_id):
                raise ValueError(f"Invalid stage id '{stage_id}'.")
            if not self.validate_slug(stage.pos_label) or not self.validate_slug(stage.neg_label):
                raise ValueError(f"Invalid class label in stage '{stage_id}'.")
            if stage.pos_label == stage.neg_label:
                raise ValueError(f"Positive and negative labels must differ in stage '{stage_id}'.")
        if not overwrite and os.path.exists(self._config_path(dataset.id)):
            raise ValueError(f"Dataset '{dataset.id}' already exists.")
        self._write_config(dataset, push_r2=True)

    def update_split_version(self, dataset_id: str, split_version: int) -> None:
        ds = self.get_dataset(dataset_id)
        if ds is None:
            return
        ds.split_version = split_version
        self._write_config(ds, push_r2=True)
