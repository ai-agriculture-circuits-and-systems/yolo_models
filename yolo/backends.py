"""Training backends for all YOLO checkpoints registered in ``registry``."""

from __future__ import annotations

import fcntl
import os
import re
import subprocess
import sys
from dataclasses import replace
from pathlib import Path

import yaml

from darknet_support import darknet_cfg_for, ensure_darknet_dataset, find_darknet_binary, run_darknet_train
from registry import (
    MODELS_DIR,
    REPO_ROOT,
    YOLO_ROOT,
    ModelSpec,
    TrainBackend,
    regression_data_yaml,
    resolve_model,
)

DEFAULT_OUTPUT_ROOT = REPO_ROOT / "outputs" / "train"
DARKNET_LOCK_FILE = REPO_ROOT / "outputs" / ".darknet.train.lock"


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


def _legacy_ultralytics_repo(model_id: str) -> Path | None:
    """Return vendored WongKinYiu tree for YOLOv7/v9 weight unpickling."""
    if model_id.startswith("yolov7"):
        return YOLO_ROOT / "yolov7"
    if model_id.startswith("yolov9"):
        return YOLO_ROOT / "yolov9"
    return None


def _ultralytics_env(spec: ModelSpec) -> dict[str, str] | None:
    """Build PYTHONPATH for Ultralytics training.

    YOLOv7/v9 need WongKinYiu ``models/`` on PYTHONPATH. YOLOv8+ uses pip ultralytics when
    installed; vendored ``ultralytics/`` is only a fallback (never mixed with v7/v9).
    """
    import importlib.util

    legacy = _legacy_ultralytics_repo(spec.model_id)
    if legacy is not None and (legacy / "models" / "yolo.py").is_file():
        prefix = str(legacy)
        existing = os.environ.get("PYTHONPATH", "")
        return {"PYTHONPATH": os.pathsep.join([prefix, existing]).rstrip(os.pathsep)}

    if importlib.util.find_spec("ultralytics") is not None:
        return None

    if vendored_ultralytics_complete(spec.work_dir):
        return {
            "PYTHONPATH": os.pathsep.join(
                [str(spec.work_dir), os.environ.get("PYTHONPATH", "")]
            ).rstrip(os.pathsep)
        }
    return None


def _subprocess_env(extra: dict[str, str] | None = None) -> dict[str, str]:
    """Environment for forked training (PyTorch 2.6+ needs legacy weight unpickling)."""
    env = os.environ.copy()
    env.setdefault("TORCH_FORCE_WEIGHTS_ONLY_LOAD", "0")
    env.setdefault("MPLBACKEND", "Agg")
    if extra:
        env.update(extra)
    return env


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

    data_yaml = regression_data_yaml(spec)
    if not data_yaml.is_file():
        raise FileNotFoundError(
            f"Dataset YAML missing: {data_yaml}. Run: ./scripts/prepare_mini_voc_yolo.py"
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

        legacy = _legacy_ultralytics_repo(spec.model_id)
        if legacy is not None and not (legacy / "models" / "yolo.py").is_file():
            raise FileNotFoundError(
                f"Missing {legacy / 'models'}. "
                "Run: ./scripts/download_yolo_models.sh --sync-sources"
            )
        if legacy is None and importlib.util.find_spec("ultralytics") is None and not (
            vendored_ultralytics_complete(spec.work_dir)
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
    run_env = _subprocess_env(env)
    print("[INFO]", " ".join(cmd), file=sys.stderr)
    completed = subprocess.run(cmd, cwd=cwd, env=run_env, check=False)
    return int(completed.returncode)


def _yaml_dataset_root(data_yaml: Path) -> str:
    """Read ``path`` from a dataset YAML (used by YOLOv5 classify)."""
    cfg = yaml.safe_load(data_yaml.read_text(encoding="utf-8"))
    if isinstance(cfg, dict) and cfg.get("path"):
        return str(cfg["path"])
    return str(data_yaml.parent)


def _regression_batch_size(spec: ModelSpec, batch_size: int) -> int:
    """Cap batch size for very large YOLOv7 P6 checkpoints (avoid OOM on 32GB GPUs)."""
    if re.match(r"yolov7-(d6|e6|e6e|w6)$", spec.model_id):
        return min(max(1, batch_size), 1)
    return max(1, batch_size)


def _legacy_yolov7_train_script(work_dir: Path, model_id: str) -> Path:
    """P6/E6 YOLOv7 checkpoints need ``train_aux.py`` (``ComputeLossAuxOTA``)."""
    if re.match(r"yolov7-(d6|e6|e6e|w6)$", model_id):
        aux = work_dir / "train_aux.py"
        if aux.is_file():
            return aux
    return work_dir / "train.py"


def _legacy_fork_extra_args(spec: ModelSpec) -> list[str]:
    """Extra CLI flags for WongKinYiu YOLOv7/v9 ``train.py`` forks."""
    if spec.model_id.startswith("yolov9"):
        cfg = spec.work_dir / "models" / "detect" / f"{spec.model_id}.yaml"
        if not cfg.is_file():
            cfg = spec.work_dir / "models" / "detect" / "yolov9-c.yaml"
        hyp = spec.work_dir / "data" / "hyps" / "hyp.scratch-high.yaml"
        args = ["--cfg", str(cfg)]
        if hyp.is_file():
            args.extend(["--hyp", str(hyp)])
        return args
    if spec.model_id.startswith("yolov7"):
        cfg = spec.work_dir / "cfg" / "training" / f"{spec.model_id}.yaml"
        if not cfg.is_file():
            cfg = spec.work_dir / "cfg" / "training" / "yolov7.yaml"
        if cfg.is_file():
            return ["--cfg", str(cfg)]
    return []


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
        str(max(1, batch_size)),
        "--device",
        device,
        "--project",
        str(project),
        "--name",
        name,
        "--exist-ok",
        "--workers",
        "0",
    ]
    legacy = _legacy_ultralytics_repo(spec.model_id)
    cmd.extend(_legacy_fork_extra_args(spec))
    if legacy is not None and epochs == 1:
        if spec.model_id.startswith("yolov7"):
            cmd.append("--notest")
        elif spec.model_id.startswith("yolov9"):
            cmd.append("--noval")
    elif spec.backend is not TrainBackend.YOLOV3_DET:
        cmd.append("--noplots")
        if epochs == 1:
            cmd.append("--noval")
    if spec.backend is TrainBackend.YOLOV5_SEG and epochs <= 1:
        hyp = spec.work_dir / "data/hyps/hyp.regression.yaml"
        if hyp.is_file():
            cmd.extend(["--hyp", str(hyp)])
    return _run(cmd, cwd=spec.work_dir)


def _yolov5_cls_train(
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
        "--model",
        weights,
        "--data",
        _yaml_dataset_root(data_yaml),
        "--epochs",
        str(epochs),
        "--batch-size",
        str(max(1, batch_size)),
        "--device",
        device,
        "--project",
        str(project),
        "--name",
        name,
        "--exist-ok",
        "--workers",
        "0",
    ]
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
    yolov6_env = {"PYTHONPATH": str(spec.work_dir)}
    return _run(cmd, cwd=spec.work_dir, env=yolov6_env)


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
    # Tiny mini-VOC (2 images): disable mosaic/mixup so seg/pose batches keep labels.
    smoke = epochs <= 1
    task_batch = max(1, batch_size)
    if smoke and ("-seg" in spec.model_id or "-pose" in spec.model_id):
        # Avoid mixed empty-label batches on mini-VOC (Ultralytics seg/pose loss).
        task_batch = 1
    if "-cls" in spec.model_id:
        data_arg = _yaml_dataset_root(data_yaml)
    else:
        data_arg = str(data_yaml)
    code = f"""
import importlib.util
if importlib.util.find_spec("ultralytics") is None:
    raise SystemExit("ultralytics is not installed. Run: pip install -r requirements.txt")
from ultralytics import YOLO

model = YOLO({weights!r})
kwargs = dict(
    data={data_arg!r},
    epochs={epochs},
    device={device!r},
    batch={task_batch},
    project={str(project)!r},
    name={name!r},
    exist_ok=True,
    plots=False,
    verbose=True,
    workers=0,
)
if {smoke!r}:
    kwargs.update(dict(
        mosaic=0.0, mixup=0.0, copy_paste=0.0, degrees=0.0, translate=0.0, scale=0.0,
        fliplr=0.0, flipud=0.0, hsv_h=0.0, hsv_s=0.0, hsv_v=0.0, close_mosaic=0,
        val=False,
    ))
    if "-seg" in {spec.model_id!r}:
        kwargs["overlap_mask"] = False
model.train(**kwargs)
"""
    return _run([sys.executable, "-c", code], cwd=REPO_ROOT, env=_ultralytics_env(spec))


def _run_darknet_locked(**kwargs) -> int:
    """Serialize Darknet training (not safe for concurrent detector processes)."""
    DARKNET_LOCK_FILE.parent.mkdir(parents=True, exist_ok=True)
    with open(DARKNET_LOCK_FILE, "w", encoding="utf-8") as lock_file:
        fcntl.flock(lock_file.fileno(), fcntl.LOCK_EX)
        try:
            return run_darknet_train(**kwargs)
        finally:
            fcntl.flock(lock_file.fileno(), fcntl.LOCK_UN)


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
    data = data_yaml or regression_data_yaml(spec)
    batch_size = _regression_batch_size(spec, batch_size)

    if spec.backend is TrainBackend.YOLOV3_DET:
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
        return _yolov5_cls_train(
            spec,
            train_script=spec.work_dir / "classify" / "train.py",
            data_yaml=data,
            weights=use_weights,
            epochs=epochs,
            device=device,
            batch_size=batch_size,
            project=project,
            name=name,
        )

    if spec.backend is TrainBackend.YOLOV6:
        models_dir = spec.work_dir / "yolov6" / "models"
        if not models_dir.is_dir():
            raise FileNotFoundError(
                f"Missing {models_dir}. Run: ./scripts/download_yolo_models.sh --sync-sources"
            )
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
        legacy = _legacy_ultralytics_repo(spec.model_id)
        if legacy is not None:
            for cache in REPO_ROOT.glob("datasets/**/*.cache"):
                cache.unlink(missing_ok=True)
        if legacy is not None and (legacy / "train.py").is_file():
            fork_spec = replace(spec, work_dir=legacy)
            train_py = (
                _legacy_yolov7_train_script(legacy, spec.model_id)
                if spec.model_id.startswith("yolov7")
                else legacy / "train.py"
            )
            return _yolov5_fork_train(
                fork_spec,
                train_script=train_py,
                data_yaml=data,
                weights=use_weights,
                epochs=epochs,
                device=device,
                batch_size=batch_size,
                project=project,
                name=name,
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
        darknet_gpus = "cpu"
        if device not in {"", "cpu"} and device != "auto":
            darknet_gpus = device.replace("cuda:", "").split(",")[0]
        return _run_darknet_locked(
            model_id=spec.model_id,
            weights=Path(use_weights),
            cfg_path=cfg,
            obj_data=obj_data,
            epochs=epochs,
            gpus=darknet_gpus,
            project_dir=project,
        )

    raise ValueError(f"No train handler for backend {spec.backend}")
