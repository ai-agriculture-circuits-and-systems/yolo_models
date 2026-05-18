"""Model registry: map every checkpoint in ``models/`` to a training backend."""

from __future__ import annotations

import re
from dataclasses import dataclass
from enum import Enum
from pathlib import Path

from darknet_support import DARKNET_CFG_FILES, darknet_cfg_for

YOLO_ROOT = Path(__file__).resolve().parent
REPO_ROOT = YOLO_ROOT.parent
MODELS_DIR = REPO_ROOT / "models"

DEFAULT_MODEL = "yolov5n"

# Default mini-VOC smoke-test set (one checkpoint per backend; detection only).
REGRESSION_MODEL_IDS: tuple[str, ...] = (
    "yolov3-tiny",
    "yolov5n",
    "yolov6n",
    "yolov7-tiny",
    "yolov8n",
    "yolov9-s",
    "yolov10n",
    "yolo11n",
)


class TrainBackend(str, Enum):
    """Training implementation used for a checkpoint."""

    YOLOV3_DET = "yolov3_det"
    YOLOV5_DET = "yolov5_det"
    YOLOV5_SEG = "yolov5_seg"
    YOLOV5_CLS = "yolov5_cls"
    YOLOV6 = "yolov6"
    ULTRALYTICS = "ultralytics"
    DARKNET = "darknet"
    INFERENCE_ONLY = "inference_only"


@dataclass(frozen=True)
class ModelSpec:
    """Resolved trainable (or inference-only) model specification."""

    model_id: str
    weights_file: str
    backend: TrainBackend
    work_dir: Path
    description: str
    yolov6_conf: str | None = None

    @property
    def trainable(self) -> bool:
        return self.backend is not TrainBackend.INFERENCE_ONLY


def _stem_from_name(filename: str) -> str:
    return Path(filename).stem


def _yolov6_conf_for(model_id: str) -> str | None:
    match = re.match(r"yolov6([nsml])", model_id)
    if not match:
        return None
    size = match.group(1)
    weights = MODELS_DIR / f"yolov6{size}.pt"
    finetune = YOLO_ROOT / "YOLOv6" / "configs" / f"yolov6{size}_finetune.py"
    if weights.is_file() and finetune.is_file():
        return f"configs/yolov6{size}_finetune.py"
    conf = YOLO_ROOT / "YOLOv6" / "configs" / f"yolov6{size}.py"
    if conf.is_file():
        return f"configs/yolov6{size}.py"
    return "configs/yolov6n.py"


def infer_backend(model_id: str, *, suffix: str) -> TrainBackend:
    """Infer training backend from checkpoint stem and file type."""
    if suffix == ".weights":
        if model_id in DARKNET_CFG_FILES and darknet_cfg_for(model_id) is not None:
            return TrainBackend.DARKNET
        return TrainBackend.INFERENCE_ONLY
    if model_id.startswith("yolov3"):
        if "-seg" in model_id:
            return TrainBackend.YOLOV5_SEG
        if "-cls" in model_id:
            return TrainBackend.YOLOV5_CLS
        return TrainBackend.YOLOV3_DET
    if model_id.startswith("yolov5"):
        if "-seg" in model_id:
            return TrainBackend.YOLOV5_SEG
        if "-cls" in model_id:
            return TrainBackend.YOLOV5_CLS
        return TrainBackend.YOLOV5_DET
    if model_id.startswith("yolov6"):
        return TrainBackend.YOLOV6
    if re.match(r"yolo(v7|v8|v9|v10|11)", model_id):
        return TrainBackend.ULTRALYTICS
    if model_id.startswith("yolov1") or model_id.startswith("yolov2") or model_id.startswith("yolov4"):
        if model_id in DARKNET_CFG_FILES and darknet_cfg_for(model_id) is not None:
            return TrainBackend.DARKNET
        return TrainBackend.INFERENCE_ONLY
    return TrainBackend.INFERENCE_ONLY


def _work_dir_for(backend: TrainBackend) -> Path:
    mapping = {
        TrainBackend.YOLOV3_DET: YOLO_ROOT / "yolov3",
        TrainBackend.YOLOV5_DET: YOLO_ROOT / "yolov5",
        TrainBackend.YOLOV5_SEG: YOLO_ROOT / "yolov5",
        TrainBackend.YOLOV5_CLS: YOLO_ROOT / "yolov5",
        TrainBackend.YOLOV6: YOLO_ROOT / "YOLOv6",
        TrainBackend.ULTRALYTICS: YOLO_ROOT / "ultralytics",
        TrainBackend.DARKNET: YOLO_ROOT / "darknet",
        TrainBackend.INFERENCE_ONLY: YOLO_ROOT,
    }
    return mapping[backend]


def _description_for(spec_id: str, backend: TrainBackend) -> str:
    if backend is TrainBackend.INFERENCE_ONLY:
        return f"{spec_id} (Darknet weights — no cfg/weights for training in this repo)"
    if backend is TrainBackend.DARKNET:
        return f"{spec_id} (detection, backend=darknet, AlexeyAB detector train)"
    task = "detection"
    if "-seg" in spec_id:
        task = "segmentation"
    elif "-pose" in spec_id:
        task = "pose"
    elif "-cls" in spec_id:
        task = "classification"
    return f"{spec_id} ({task}, backend={backend.value})"


def spec_from_weights_path(path: Path) -> ModelSpec:
    """Build a :class:`ModelSpec` from a checkpoint path under ``models/``."""
    weights_file = path.name
    model_id = _stem_from_name(weights_file)
    backend = infer_backend(model_id, suffix=path.suffix.lower())
    yolov6_conf = _yolov6_conf_for(model_id) if backend is TrainBackend.YOLOV6 else None
    return ModelSpec(
        model_id=model_id,
        weights_file=weights_file,
        backend=backend,
        work_dir=_work_dir_for(backend),
        description=_description_for(model_id, backend),
        yolov6_conf=yolov6_conf,
    )


def discover_models(models_dir: Path | None = None) -> dict[str, ModelSpec]:
    """Discover all checkpoints in ``models/`` and index by model id."""
    root = models_dir or MODELS_DIR
    specs: dict[str, ModelSpec] = {}
    if not root.is_dir():
        return specs
    for path in sorted(root.iterdir()):
        if not path.is_file():
            continue
        if path.suffix.lower() not in {".pt", ".weights"}:
            continue
        spec = spec_from_weights_path(path)
        specs[spec.model_id] = spec
    return specs


def normalize_model(model: str, models_dir: Path | None = None) -> str:
    """Normalize and validate a model id."""
    model_id = model.strip().lower().replace(" ", "")
    if model_id.endswith(".pt") or model_id.endswith(".weights"):
        model_id = _stem_from_name(model_id)
    catalog = discover_models(models_dir)
    if model_id in catalog:
        return model_id
    # Allow legacy family aliases.
    aliases = {
        "yolov3": "yolov3-tiny",
        "yolov5": "yolov5n",
        "yolov6": "yolov6n",
        "ultralytics": "yolov8n",
        "yolov8": "yolov8n",
        "yolov9": "yolov9-s",
        "yolov10": "yolov10n",
        "yolo11": "yolo11n",
    }
    if model_id in aliases and aliases[model_id] in catalog:
        return aliases[model_id]
    if model_id in aliases:
        return aliases[model_id]
    known = ", ".join(sorted(catalog))
    raise ValueError(f"Unknown model '{model}'. Known models: {known}")


def resolve_model(model: str, models_dir: Path | None = None) -> ModelSpec:
    """Resolve model id to a :class:`ModelSpec` (from disk catalog)."""
    model_id = normalize_model(model, models_dir)
    catalog = discover_models(models_dir)
    if model_id not in catalog:
        weights = f"{model_id}.pt"
        path = (models_dir or MODELS_DIR) / weights
        if path.is_file():
            catalog[model_id] = spec_from_weights_path(path)
        else:
            spec = spec_from_weights_path(Path(weights))
            catalog[model_id] = spec
    return catalog[model_id]


def list_models(*, trainable_only: bool = False, models_dir: Path | None = None) -> list[tuple[str, str]]:
    """Return ``(model_id, description)`` pairs."""
    catalog = discover_models(models_dir)
    items = sorted(catalog.items(), key=lambda item: item[0])
    if trainable_only:
        items = [(mid, spec.description) for mid, spec in items if spec.trainable]
    else:
        items = [(mid, spec.description) for mid, spec in items]
    return items


def list_regression_models(models_dir: Path | None = None) -> list[str]:
    """Model ids used for default regression (must exist under ``models/``)."""
    catalog = discover_models(models_dir)
    return [mid for mid in REGRESSION_MODEL_IDS if mid in catalog and catalog[mid].trainable]


DEFAULT_DATA_YAML = REPO_ROOT / ".regression_configs" / "voc-mini.yaml"
DEFAULT_DATA_YAML_V6 = REPO_ROOT / ".regression_configs" / "voc-mini-yolov6.yaml"
DEFAULT_DATA_YAML_SEG = REPO_ROOT / ".regression_configs" / "voc-mini-seg.yaml"
DEFAULT_DATA_YAML_POSE = REPO_ROOT / ".regression_configs" / "voc-mini-pose.yaml"
DEFAULT_DATA_YAML_CLS = REPO_ROOT / ".regression_configs" / "voc-mini-cls.yaml"


def regression_data_yaml(spec: ModelSpec) -> Path:
    """Return the mini-VOC dataset YAML appropriate for a model task."""
    if spec.backend is TrainBackend.YOLOV6:
        return DEFAULT_DATA_YAML_V6
    if "-cls" in spec.model_id:
        return DEFAULT_DATA_YAML_CLS
    if "-seg" in spec.model_id:
        return DEFAULT_DATA_YAML_SEG
    if "-pose" in spec.model_id:
        return DEFAULT_DATA_YAML_POSE
    return DEFAULT_DATA_YAML


def list_all_regression_model_ids(models_dir: Path | None = None) -> list[str]:
    """All trainable checkpoint ids under ``models/`` (for full regression)."""
    catalog = discover_models(models_dir)
    return sorted(mid for mid, spec in catalog.items() if spec.trainable)
