"""Training backends for all YOLO checkpoints registered in ``registry``."""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

from darknet_support import darknet_cfg_for, ensure_darknet_dataset, find_darknet_binary, run_darknet_train
from registry import (
    DEFAULT_DATA_YAML,
    DEFAULT_DATA_YAML_V6,
    MODELS_DIR,
    REPO_ROOT,
    ModelSpec,
    TrainBackend,
    resolve_model,
)

DEFAULT_OUTPUT_ROOT = REPO_ROOT / "outputs" / "train"


def resolve_weights(weights: str) -> str:
    """Prefer repo ``models/`` checkpoints when present."""
    candidate = MODELS_DIR / weights
    if candidate.is_file():
        return str(candidate.resolve())
    path = Path(weights)
    if path.is_file():
        return str(path.resolve())
    return weights


def vendored_ultralytics_complete(work_dir: Path) -> bool:
    """Return True if vendored ultralytics package includes ``models/``."""
    return (work_dir / "ultralytics" / "models").is_dir()


def preflight(spec: ModelSpec) -> None:
    """Verify upstream code and weights exist before training."""
    if not spec.trainable:
        raise RuntimeError(
            f"{spec.model_id} cannot be trained in this repo "
            "(missing Darknet cfg/weights or unsupported checkpoint). "
            "Run: ./scripts/fetch_darknet_cfg.sh and ./scripts/download_yolo_models.sh"
        )
    weights_path = MODELS_DIR / spec.weights_file
    if spec.backend is not TrainBackend.YOLOV6 and not weights_path.is_file():
        raise FileNotFoundError(
            f"Weights not found: {weights_path}. Run ./scripts/download_yolo_models.sh"
        )

    if spec.backend is TrainBackend.YOLOV3_DET:
        if not (spec.work_dir / "models").is_dir():
            raise FileNotFoundError(
                f"Missing {spec.work_dir / 'models'}. "
                "Run: ./scripts/download_yolo_models.sh --sync-sources"
            )
    if spec.backend in {TrainBackend.YOLOV5_DET, TrainBackend.YOLOV5_SEG, TrainBackend.YOLOV5_CLS}:
        if not (spec.work_dir / "models").is_dir():
            raise FileNotFoundError(
                f"Missing {spec.work_dir / 'models'}. "
                "Run: ./scripts/download_yolo_models.sh --sync-sources"
            )
    if spec.backend is TrainBackend.YOLOV6:
        if not (spec.work_dir / "tools" / "train.py").is_file():
            raise FileNotFoundError(f"Missing YOLOv6 train.py under {spec.work_dir}")
    if spec.backend is TrainBackend.ULTRALYTICS:
        import importlib.util

        if importlib.util.find_spec("ultralytics") is None and not vendored_ultralytics_complete(
            spec.work_dir
        ):
            raise FileNotFoundError(
                "Ultralytics not installed. Run: pip install -r requirements.txt "
                "or ./scripts/download_yolo_models.sh --install-deps --sync-sources"
            )
    if spec.backend is TrainBackend.DARKNET:
        find_darknet_binary()
        cfg = darknet_cfg_for(spec.model_id)
        if cfg is None:
            raise FileNotFoundError(
                f"Missing Darknet cfg for {spec.model_id}. Run: ./scripts/fetch_darknet_cfg.sh"
            )
        yolo_marker = REPO_ROOT / "datasets" / "voc-mini-yolo" / ".ready"
        if not yolo_marker.is_file():
            raise FileNotFoundError(
                f"Mini VOC YOLO dataset not ready ({yolo_marker}). "
                "Run: ./scripts/prepare_mini_voc_yolo.py"
            )


def _run(cmd: list[str], *, cwd: Path, env: dict[str, str] | None = None) -> int:
    run_env = os.environ.copy()
    if env:
        run_env.update(env)
    print("[INFO]", " ".join(cmd), file=sys.stderr)
    completed = subprocess.run(cmd, cwd=cwd, env=run_env, check=False)
    return int(completed.returncode)


def _yolov5_fork_train(
    spec: ModelSpec,
    *,
    train_script: Path,
    data_yaml: Path,
    weights: str,
    epochs: int,
    device: str,
    batch_size: int,
    project: Path,
    name: str,
) -> int:
    cmd = [
        sys.executable,
        str(train_script),
        "--data",
        str(data_yaml),
        "--weights",
        weights,
        "--epochs",
        str(epochs),
        "--batch-size",
        str(batch_size),
        "--device",
        device,
        "--project",
        str(project),
        "--name",
        name,
        "--exist-ok",
        "--workers",
        "2",
        "--noplots",
    ]
    if epochs == 1:
        cmd.append("--noval")
    return _run(cmd, cwd=spec.work_dir)


def train_yolov6(
    spec: ModelSpec,
    *,
    data_yaml: Path,
    epochs: int,
    device: str,
    batch_size: int,
    output_dir: Path,
    name: str,
) -> int:
    """Train YOLOv6 via ``tools/train.py``."""
    conf = spec.yolov6_conf or "configs/yolov6n.py"
    train_py = spec.work_dir / "tools" / "train.py"
    cmd = [
        sys.executable,
        str(train_py),
        "--data-path",
        str(data_yaml),
        "--conf-file",
        conf,
        "--epochs",
        str(epochs),
        "--batch-size",
        str(batch_size),
        "--device",
        device,
        "--workers",
        "2",
        "--eval-interval",
        str(max(epochs, 1)),
        "--output-dir",
        str(output_dir),
        "--name",
        name,
    ]
    return _run(cmd, cwd=spec.work_dir)


def train_ultralytics(
    spec: ModelSpec,
    *,
    data_yaml: Path,
    weights: str,
    epochs: int,
    device: str,
    batch_size: int,
    project: Path,
    name: str,
) -> int:
    """Train YOLOv7–YOLO11 and task variants via the Ultralytics API."""
    code = f"""
import importlib.util
if importlib.util.find_spec("ultralytics") is None:
    raise SystemExit("ultralytics is not installed. Run: pip install -r requirements.txt")
from ultralytics import YOLO

model = YOLO({weights!r})
model.train(
    data={str(data_yaml)!r},
    epochs={epochs},
    device={device!r},
    batch={batch_size},
    project={str(project)!r},
    name={name!r},
    exist_ok=True,
    plots=False,
    verbose=True,
)
"""
    env: dict[str, str] | None = None
    if vendored_ultralytics_complete(spec.work_dir):
        env = {
            "PYTHONPATH": os.pathsep.join(
                [str(spec.work_dir), os.environ.get("PYTHONPATH", "")]
            ).rstrip(os.pathsep)
        }
    return _run([sys.executable, "-c", code], cwd=REPO_ROOT, env=env)


def run_train(
    model_id: str,
    *,
    data_yaml: Path | None = None,
    epochs: int = 1,
    device: str = "cpu",
    batch_size: int = 4,
    weights: str | None = None,
    output_root: Path | None = None,
) -> int:
    """Dispatch training for any registered checkpoint."""
    spec = resolve_model(model_id)
    preflight(spec)

    out_root = output_root or DEFAULT_OUTPUT_ROOT
    project = out_root / spec.model_id
    name = "exp"
    use_weights = resolve_weights(weights or spec.weights_file)

    if spec.backend is TrainBackend.YOLOV3_DET:
        data = data_yaml or DEFAULT_DATA_YAML
        return _yolov5_fork_train(
            spec,
            train_script=spec.work_dir / "train.py",
            data_yaml=data,
            weights=use_weights,
            epochs=epochs,
            device=device,
            batch_size=batch_size,
            project=project,
            name=name,
        )

    if spec.backend is TrainBackend.YOLOV5_DET:
        data = data_yaml or DEFAULT_DATA_YAML
        return _yolov5_fork_train(
            spec,
            train_script=spec.work_dir / "train.py",
            data_yaml=data,
            weights=use_weights,
            epochs=epochs,
            device=device,
            batch_size=batch_size,
            project=project,
            name=name,
        )

    if spec.backend is TrainBackend.YOLOV5_SEG:
        data = data_yaml or DEFAULT_DATA_YAML
        return _yolov5_fork_train(
            spec,
            train_script=spec.work_dir / "segment" / "train.py",
            data_yaml=data,
            weights=use_weights,
            epochs=epochs,
            device=device,
            batch_size=batch_size,
            project=project,
            name=name,
        )

    if spec.backend is TrainBackend.YOLOV5_CLS:
        raise RuntimeError(
            f"{spec.model_id} is a classification model; mini-VOC detection YAML is not suitable. "
            "Provide a classification dataset YAML via --data."
        )

    if spec.backend is TrainBackend.YOLOV6:
        data = data_yaml or DEFAULT_DATA_YAML_V6
        return train_yolov6(
            spec,
            data_yaml=data,
            epochs=epochs,
            device=device,
            batch_size=batch_size,
            output_dir=project,
            name=name,
        )

    if spec.backend is TrainBackend.ULTRALYTICS:
        data = data_yaml or DEFAULT_DATA_YAML
        if "-cls" in spec.model_id:
            raise RuntimeError(
                f"{spec.model_id} requires an ImageNet-style classification dataset YAML (--data)."
            )
        return train_ultralytics(
            spec,
            data_yaml=data,
            weights=use_weights,
            epochs=epochs,
            device=device,
            batch_size=batch_size,
            project=project,
            name=name,
        )

    if spec.backend is TrainBackend.DARKNET:
        cfg = darknet_cfg_for(spec.model_id)
        if cfg is None:
            raise FileNotFoundError(f"No Darknet cfg for {spec.model_id}")
        obj_data = ensure_darknet_dataset()
        gpus = "0" if device not in {"cpu", "-1"} else "cpu"
        return run_darknet_train(
            model_id=spec.model_id,
            weights=Path(use_weights),
            cfg_path=cfg,
            obj_data=obj_data,
            epochs=epochs,
            gpus=gpus,
            project_dir=project,
        )

    raise ValueError(f"No train handler for backend {spec.backend}")
