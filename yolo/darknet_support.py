"""Darknet (AlexeyAB) training helpers for legacy ``.weights`` checkpoints."""

from __future__ import annotations

import os
import re
import shutil
import subprocess
from pathlib import Path

YOLO_ROOT = Path(__file__).resolve().parent
REPO_ROOT = YOLO_ROOT.parent
DARKNET_CFG_DIR = YOLO_ROOT / "darknet" / "cfg"
DARKNET_DATA_DIR = REPO_ROOT / "datasets" / "voc-mini-darknet"
DARKNET_BACKUP_ROOT = REPO_ROOT / "outputs" / "darknet-train"
DEFAULT_OBJ_DATA = DARKNET_DATA_DIR / "voc-mini.data"

# model_id -> cfg filename under yolo/darknet/cfg/ (from AlexeyAB / pjreddie).
DARKNET_CFG_FILES: dict[str, str] = {
    "yolov1": "yolov1.cfg",
    "yolov2": "yolov2-voc.cfg",
    "yolov2-tiny": "yolov2-tiny.cfg",
    "yolov4": "yolov4.cfg",
    "yolov4-tiny": "yolov4-tiny.cfg",
    "yolov4-csp": "yolov4-csp.cfg",
    "yolov4x-mish": "yolov4x-mish.cfg",
}

VOC_NAMES = [
    "aeroplane",
    "bicycle",
    "bird",
    "boat",
    "bottle",
    "bus",
    "car",
    "cat",
    "chair",
    "cow",
    "diningtable",
    "dog",
    "horse",
    "motorbike",
    "person",
    "pottedplant",
    "sheep",
    "sofa",
    "train",
    "tvmonitor",
]


def darknet_cfg_for(model_id: str) -> Path | None:
    """Return path to network cfg for a Darknet weights stem, if supported."""
    name = DARKNET_CFG_FILES.get(model_id)
    if not name:
        return None
    path = DARKNET_CFG_DIR / name
    return path if path.is_file() else None


def find_darknet_binary() -> Path:
    """Locate the ``darknet`` executable."""
    env_bin = os.environ.get("DARKNET_BIN")
    if env_bin:
        path = Path(env_bin)
        if path.is_file() and os.access(path, os.X_OK):
            return path.resolve()
        raise FileNotFoundError(f"DARKNET_BIN is not executable: {path}")

    candidates = [
        REPO_ROOT / "tools" / "darknet" / "darknet",
        REPO_ROOT / "tools" / "darknet-src" / "darknet",
        Path("/usr/local/bin/darknet"),
    ]
    for path in candidates:
        if path.is_file() and os.access(path, os.X_OK):
            return path.resolve()

    found = shutil.which("darknet")
    if found:
        return Path(found).resolve()

    raise FileNotFoundError(
        "darknet binary not found. Build it with:\n"
        "  ./scripts/install_darknet.sh\n"
        "Or set DARKNET_BIN=/path/to/darknet"
    )


def ensure_darknet_dataset(
    yolo_dataset_root: Path | None = None,
    voc_mini_root: Path | None = None,
) -> Path:
    """Build Darknet ``obj.data`` + train/val lists pointing at YOLO-format mini VOC."""
    yolo_root = yolo_dataset_root or (REPO_ROOT / "datasets" / "voc-mini-yolo")
    marker = DARKNET_DATA_DIR / ".ready"
    if marker.is_file():
        # Regenerate lists when YOLO mini VOC changes (paths use JPEGImages alias).
        train_list = DARKNET_DATA_DIR / "train.txt"
        if train_list.is_file() and "/JPEGImages/" in train_list.read_text(encoding="utf-8"):
            return DEFAULT_OBJ_DATA
        marker.unlink(missing_ok=True)

    def _ensure_jpegimages_alias(split: str) -> Path:
        """Symlink ``JPEGImages/<split>`` -> ``images/<split>`` for Darknet label resolution."""
        src = yolo_root / "images" / split
        link = yolo_root / "JPEGImages" / split
        link.parent.mkdir(parents=True, exist_ok=True)
        if not link.exists():
            link.symlink_to(src.resolve(), target_is_directory=True)
        return link

    train_images = _ensure_jpegimages_alias("train2007")
    val_images = _ensure_jpegimages_alias("test2007")
    if not train_images.is_dir():
        raise FileNotFoundError(
            f"YOLO mini VOC images missing: {train_images}. "
            "Run: ./scripts/prepare_mini_voc_yolo.py"
        )

    DARKNET_DATA_DIR.mkdir(parents=True, exist_ok=True)
    names_path = DARKNET_DATA_DIR / "voc.names"
    names_path.write_text("\n".join(VOC_NAMES) + "\n", encoding="utf-8")

    def _darknet_train_path(image_path: Path) -> str:
        # AlexeyAB maps /JPEGImages/ -> /labels/ (see darknet src/utils.c).
        return str(image_path.resolve()).replace("/images/", "/JPEGImages/")

    def _write_list(split_dir: Path, out_file: Path) -> None:
        lines = sorted(_darknet_train_path(p) for p in split_dir.glob("*.jpg"))
        out_file.write_text("\n".join(lines) + "\n", encoding="utf-8")

    _write_list(train_images, DARKNET_DATA_DIR / "train.txt")
    _write_list(val_images, DARKNET_DATA_DIR / "val.txt")

    backup_dir = DARKNET_BACKUP_ROOT / "backup"
    backup_dir.mkdir(parents=True, exist_ok=True)

    data_content = f"""classes = 20
train = {DARKNET_DATA_DIR / 'train.txt'}
valid = {DARKNET_DATA_DIR / 'val.txt'}
names = {names_path}
backup = {backup_dir}
"""
    DEFAULT_OBJ_DATA.write_text(data_content, encoding="utf-8")
    marker.write_text("ok\n", encoding="utf-8")
    return DEFAULT_OBJ_DATA


def _patch_cfg_line(text: str, key: str, value: str) -> str:
    pattern = rf"^{re.escape(key)}\s*=.*$"
    replacement = f"{key}={value}"
    if re.search(pattern, text, flags=re.MULTILINE):
        return re.sub(pattern, replacement, text, count=1, flags=re.MULTILINE)
    return f"{replacement}\n{text}"


def patch_cfg_for_smoke_test(cfg_src: Path, cfg_dst: Path, *, max_batches: int) -> None:
    """Write a short-run cfg for regression (low batches, small batch size, no burn-in)."""
    text = cfg_src.read_text(encoding="utf-8")
    text = _patch_cfg_line(text, "max_batches", str(max_batches))
    text = _patch_cfg_line(text, "burn_in", "0")
    text = _patch_cfg_line(text, "batch", "4")
    text = _patch_cfg_line(text, "subdivisions", "1")
    cfg_dst.parent.mkdir(parents=True, exist_ok=True)
    cfg_dst.write_text(text, encoding="utf-8")


def run_darknet_train(
    *,
    model_id: str,
    weights: Path,
    cfg_path: Path,
    obj_data: Path,
    epochs: int,
    gpus: str,
    project_dir: Path,
) -> int:
    """Invoke ``darknet detector train`` for one legacy checkpoint."""
    del obj_data  # run uses per-project run.data derived from DEFAULT_OBJ_DATA
    darknet = find_darknet_binary()
    ensure_darknet_dataset()

    project_dir.mkdir(parents=True, exist_ok=True)
    run_cfg = project_dir / f"{model_id}-run.cfg"
    # Short smoke run on CPU (full training needs much higher max_batches).
    patch_cfg_for_smoke_test(
        cfg_path, run_cfg, max_batches=max(10, min(200, 10 * max(epochs, 1)))
    )

    backup_dir = project_dir / "backup"
    backup_dir.mkdir(parents=True, exist_ok=True)
    run_data = project_dir / "run.data"
    run_data.write_text(
        DEFAULT_OBJ_DATA.read_text(encoding="utf-8").replace(
            f"backup = {DARKNET_BACKUP_ROOT / 'backup'}",
            f"backup = {backup_dir}",
        ),
        encoding="utf-8",
    )

    cmd = [
        str(darknet),
        "detector",
        "train",
        str(run_data.resolve()),
        str(run_cfg.resolve()),
        str(weights.resolve()),
        "-dont_show",
    ]
    if gpus and gpus != "cpu":
        cmd.extend(["-gpus", gpus.replace("cuda", "").strip() or "0"])

    env = os.environ.copy()
    env.setdefault("OPENCV_IO_ENABLE_OPENEXR", "1")
    print("[INFO]", " ".join(cmd))
    completed = subprocess.run(cmd, cwd=REPO_ROOT, env=env, check=False)
    return int(completed.returncode)
