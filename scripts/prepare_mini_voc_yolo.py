#!/usr/bin/env python3
"""Convert mini PASCAL VOC (VOCdevkit layout) to YOLO train/val image+label dirs."""

from __future__ import annotations

import argparse
import os
import shutil
import sys
import xml.etree.ElementTree as ET
from pathlib import Path

VOC_NAMES: list[str] = [
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

READY_MARKER = ".ready"
PERSON_CLASS_ID = VOC_NAMES.index("person")
# COCO 17 keypoints (synthetic layout from person bbox for mini-VOC pose smoke tests).
POSE_KPT_SHAPE = (17, 3)
# Only duplicate images when the set is smaller than this (regression-sized VOC is larger).
MIN_IMAGES_BEFORE_DUPLICATE = 32
# Normalized (x, y) within person bbox for COCO-17 order (visibility filled as 2).
POSE_BBOX_SKELETON: list[tuple[float, float]] = [
    (0.50, 0.08),  # nose
    (0.45, 0.05),
    (0.55, 0.05),  # eyes
    (0.40, 0.10),
    (0.60, 0.10),  # ears
    (0.35, 0.22),
    (0.65, 0.22),  # shoulders
    (0.30, 0.40),
    (0.70, 0.40),  # elbows
    (0.28, 0.58),
    (0.72, 0.58),  # wrists
    (0.38, 0.55),
    (0.62, 0.55),  # hips
    (0.40, 0.75),
    (0.60, 0.75),  # knees
    (0.42, 0.95),
    (0.58, 0.95),  # ankles
]


def _repo_root() -> Path:
    return Path(__file__).resolve().parent.parent


def _convert_box(size: tuple[int, int], box: list[float]) -> tuple[float, float, float, float]:
    width, height = size
    dw, dh = 1.0 / width, 1.0 / height
    x = (box[0] + box[1]) / 2.0 - 1
    y = (box[2] + box[3]) / 2.0 - 1
    w = box[1] - box[0]
    h = box[3] - box[2]
    return x * dw, y * dh, w * dw, h * dh


def _write_label(annotation_xml: Path, label_path: Path) -> None:
    tree = ET.parse(annotation_xml)
    root = tree.getroot()
    size = root.find("size")
    if size is None:
        raise ValueError(f"Missing <size> in {annotation_xml}")
    width = int(size.find("width").text)  # type: ignore[union-attr]
    height = int(size.find("height").text)  # type: ignore[union-attr]
    lines: list[str] = []
    for obj in root.iter("object"):
        name_el = obj.find("name")
        diff_el = obj.find("difficult")
        if name_el is None or diff_el is None:
            continue
        cls = name_el.text
        if cls not in VOC_NAMES or int(diff_el.text) == 1:
            continue
        xmlbox = obj.find("bndbox")
        if xmlbox is None:
            continue
        bb = _convert_box(
            (width, height),
            [float(xmlbox.find(tag).text) for tag in ("xmin", "xmax", "ymin", "ymax")],  # type: ignore[union-attr]
        )
        cls_id = VOC_NAMES.index(cls)
        lines.append(" ".join(str(v) for v in (cls_id, *bb)))
    label_path.parent.mkdir(parents=True, exist_ok=True)
    label_path.write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")


def _copy_split(
    voc_root: Path,
    year: str,
    split_name: str,
    images_out: Path,
    labels_out: Path,
) -> int:
    """Copy one VOC split into YOLO image/label folders."""
    split_file = voc_root / f"VOC{year}/ImageSets/Main/{split_name}.txt"
    if not split_file.is_file():
        raise FileNotFoundError(f"Missing split file: {split_file}")
    count = 0
    for image_id in split_file.read_text(encoding="utf-8").split():
        image_id = image_id.strip()
        if not image_id:
            continue
        jpg = voc_root / f"VOC{year}/JPEGImages/{image_id}.jpg"
        xml = voc_root / f"VOC{year}/Annotations/{image_id}.xml"
        if not jpg.is_file() or not xml.is_file():
            continue
        images_out.mkdir(parents=True, exist_ok=True)
        labels_out.mkdir(parents=True, exist_ok=True)
        shutil.copy2(jpg, images_out / f"{image_id}.jpg")
        _write_label(xml, labels_out / f"{image_id}.txt")
        count += 1
    return count


def _link_or_copy(src: Path, dest: Path) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.exists() or dest.is_symlink():
        dest.unlink()
    try:
        os.symlink(src.resolve(), dest, target_is_directory=True)
    except OSError:
        if src.is_dir():
            shutil.copytree(src, dest, dirs_exist_ok=True)
        else:
            shutil.copy2(src, dest)


def prepare_mini_voc_yolo(
    voc_mini_root: Path,
    output_root: Path,
    year: str = "2007",
    force: bool = False,
) -> Path:
    """Build YOLO-format mini VOC under ``output_root``.

    Args:
        voc_mini_root: Directory containing ``VOCdevkit/`` (e.g. ``datasets/voc-mini``).
        output_root: Output root (e.g. ``datasets/voc-mini-yolo``).
        year: VOC year subdirectory name.
        force: Rebuild even if the ready marker exists.

    Returns:
        Path to the ready marker file.
    """
    voc_devkit = voc_mini_root / "VOCdevkit"
    marker = output_root / READY_MARKER
    if marker.is_file() and not force:
        return marker

    if output_root.exists() and force:
        shutil.rmtree(output_root)
    output_root.mkdir(parents=True, exist_ok=True)

    train_n = _copy_split(
        voc_devkit,
        year,
        "trainval",
        output_root / "images" / "train2007",
        output_root / "labels" / "train2007",
    )
    val_n = _copy_split(
        voc_devkit,
        year,
        "test",
        output_root / "images" / "test2007",
        output_root / "labels" / "test2007",
    )
    if train_n == 0 or val_n == 0:
        raise RuntimeError(
            f"No images converted (train={train_n}, val={val_n}). "
            f"Run scripts/create_mini_voc.sh first."
        )

    v6_root = output_root / "voc_07_12"
    _link_or_copy(output_root / "images" / "train2007", v6_root / "images" / "train")
    _link_or_copy(output_root / "images" / "test2007", v6_root / "images" / "val")
    _link_or_copy(output_root / "labels" / "train2007", v6_root / "labels" / "train")
    _link_or_copy(output_root / "labels" / "test2007", v6_root / "labels" / "val")

    if train_n + val_n < MIN_IMAGES_BEFORE_DUPLICATE:
        _expand_split_for_regression(output_root, "train2007", copies=3)
        _expand_split_for_regression(output_root, "test2007", copies=3)

    marker.write_text(
        f"train2007={train_n}\nval2007={val_n}\n",
        encoding="utf-8",
    )
    return marker


def _expand_split_for_regression(yolo_root: Path, split: str, *, copies: int = 3) -> None:
    """Duplicate images/labels so tiny mini-VOC works with seg/pose trainers (smoke only)."""
    img_dir = yolo_root / "images" / split
    lbl_dir = yolo_root / "labels" / split
    for img in sorted(img_dir.glob("*.jpg")):
        lbl = lbl_dir / f"{img.stem}.txt"
        if not lbl.is_file():
            continue
        label_text = lbl.read_text(encoding="utf-8")
        for idx in range(copies):
            dup_stem = f"{img.stem}_r{idx}"
            dup_img = img_dir / f"{dup_stem}.jpg"
            dup_lbl = lbl_dir / f"{dup_stem}.txt"
            if dup_img.exists():
                continue
            try:
                os.symlink(img.resolve(), dup_img)
            except OSError:
                shutil.copy2(img, dup_img)
            dup_lbl.write_text(label_text, encoding="utf-8")


def write_voc_mini_yaml(output_path: Path, yolo_dataset_root: Path) -> Path:
    """Write a dataset YAML for YOLOv3/v5/Ultralytics trainers."""
    dataset_path = yolo_dataset_root.absolute()
    names_block = "\n".join(f"  {idx}: {name}" for idx, name in enumerate(VOC_NAMES))
    content = f"""# Auto-generated for mini VOC regression tests. Do not edit by hand.
path: {dataset_path}
train:
  - images/train2007
val:
  - images/test2007
test:
  - images/test2007
nc: {len(VOC_NAMES)}

names:
{names_block}
"""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(content, encoding="utf-8")
    return output_path


def write_voc_mini_yolov3_yaml(output_path: Path, yolo_dataset_root: Path) -> Path:
    """Write dataset YAML for legacy YOLOv3 (absolute train/val image dirs)."""
    dataset_path = yolo_dataset_root.absolute()
    train_dir = dataset_path / "images" / "train2007"
    val_dir = dataset_path / "images" / "test2007"
    names_block = "\n".join(f"  {idx}: {name}" for idx, name in enumerate(VOC_NAMES))
    content = f"""# Auto-generated for mini VOC regression tests (YOLOv3 fork).
train: {train_dir}
val: {val_dir}
test: {val_dir}
nc: {len(VOC_NAMES)}

names:
{names_block}
"""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(content, encoding="utf-8")
    return output_path


def _read_det_labels(label_path: Path) -> list[tuple[int, float, float, float, float]]:
    if not label_path.is_file() or label_path.stat().st_size == 0:
        return []
    rows: list[tuple[int, float, float, float, float]] = []
    for line in label_path.read_text(encoding="utf-8").splitlines():
        parts = line.strip().split()
        if len(parts) < 5:
            continue
        cls_id = int(parts[0])
        rows.append((cls_id, float(parts[1]), float(parts[2]), float(parts[3]), float(parts[4])))
    return rows


def _min_box_dim(value: float, *, minimum: float = 0.02) -> float:
    """Clamp normalized width/height so seg polygons are non-degenerate."""
    return max(minimum, min(1.0, value))


def _bbox_to_seg_line(cls_id: int, xc: float, yc: float, w: float, h: float) -> str:
    """Convert a normalized bbox to a 4-point polygon (YOLO-seg line)."""
    w = _min_box_dim(w)
    h = _min_box_dim(h)
    x1 = max(0.0, min(1.0, xc - w / 2))
    y1 = max(0.0, min(1.0, yc - h / 2))
    x2 = max(0.0, min(1.0, xc + w / 2))
    y2 = max(0.0, min(1.0, yc + h / 2))
    return f"{cls_id} {x1} {y1} {x2} {y1} {x2} {y2} {x1} {y2}"


def _bbox_to_pose_line(cls_id: int, xc: float, yc: float, w: float, h: float) -> str:
    """Synthetic COCO-17 keypoints laid out inside the person bbox (smoke-test only)."""
    w = _min_box_dim(w)
    h = _min_box_dim(h)
    x1 = xc - w / 2
    y1 = yc - h / 2
    kpts: list[str] = []
    for rx, ry in POSE_BBOX_SKELETON:
        kx = max(0.0, min(1.0, x1 + rx * w))
        ky = max(0.0, min(1.0, y1 + ry * h))
        kpts.extend([f"{kx:.6f}", f"{ky:.6f}", "2"])
    return f"{cls_id} {xc:.6f} {yc:.6f} {w:.6f} {h:.6f} " + " ".join(kpts)


def _seg_lines_from_voc_mask(
    voc_devkit: Path,
    year: str,
    image_id: str,
    det_rows: list[tuple[int, float, float, float, float]],
) -> list[str] | None:
    """Build YOLO-seg label lines from VOC ``SegmentationObject`` mask (instance index = object order)."""
    seg_png = voc_devkit / f"VOC{year}/SegmentationObject/{image_id}.png"
    if not seg_png.is_file() or not det_rows:
        return None

    try:
        import cv2  # type: ignore[import-untyped]
        import numpy as np  # type: ignore[import-untyped]
    except ImportError:
        return None

    mask = cv2.imread(str(seg_png), cv2.IMREAD_UNCHANGED)
    if mask is None:
        return None
    if mask.ndim == 3:
        mask = mask[:, :, 0]

    lines: list[str] = []
    for inst_idx, (cls_id, _xc, _yc, _w, _h) in enumerate(det_rows, start=1):
        binary = (mask == inst_idx).astype("uint8")
        if binary.sum() < 4:
            continue
        contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            continue
        contour = max(contours, key=cv2.contourArea)
        if contour.shape[0] < 3:
            continue
        epsilon = 0.005 * cv2.arcLength(contour, True)
        approx = cv2.approxPolyDP(contour, epsilon, True)
        if len(approx) < 3:
            continue
        height, width = mask.shape[:2]
        coords: list[str] = []
        for point in approx.reshape(-1, 2):
            xn = max(0.0, min(1.0, float(point[0]) / width))
            yn = max(0.0, min(1.0, float(point[1]) / height))
            coords.extend([f"{xn:.6f}", f"{yn:.6f}"])
        if len(coords) >= 6:
            lines.append(f"{cls_id} " + " ".join(coords))
    return lines if lines else None


def _build_task_dataset(
    yolo_root: Path,
    task_name: str,
    *,
    label_writer,
    voc_mini_root: Path | None = None,
    year: str = "2007",
    force: bool = False,
) -> Path:
    """Create ``voc-mini-yolo-<task>`` with shared images and task-specific ``labels/``."""
    out_root = yolo_root.parent / f"voc-mini-yolo-{task_name}"
    marker = out_root / ".ready"
    if marker.is_file() and not force:
        return out_root

    if out_root.exists() and force:
        shutil.rmtree(out_root)

    voc_devkit = (voc_mini_root / "VOCdevkit") if voc_mini_root else None
    image_count = 0

    for split in ("train2007", "test2007"):
        img_src = yolo_root / "images" / split
        lbl_src = yolo_root / "labels" / split
        img_dst = out_root / "images" / split
        lbl_dst = out_root / "labels" / split
        img_dst.mkdir(parents=True, exist_ok=True)
        lbl_dst.mkdir(parents=True, exist_ok=True)
        for img in sorted(img_src.glob("*.jpg")):
            rows = _read_det_labels(lbl_src / f"{img.stem}.txt")
            lines = label_writer(img.stem, rows, voc_devkit, year)
            if not lines:
                continue
            image_count += 1
            (lbl_dst / f"{img.stem}.txt").write_text(
                "\n".join(lines) + "\n",
                encoding="utf-8",
            )
            try:
                os.symlink(img.resolve(), img_dst / img.name)
            except OSError:
                shutil.copy2(img, img_dst / img.name)

    if image_count < MIN_IMAGES_BEFORE_DUPLICATE:
        for split in ("train2007", "test2007"):
            _expand_split_for_regression(out_root, split, copies=3)

    marker.parent.mkdir(parents=True, exist_ok=True)
    marker.write_text("ok\n", encoding="utf-8")
    return out_root


def prepare_mini_voc_seg(
    yolo_root: Path,
    voc_mini_root: Path,
    *,
    year: str = "2007",
    force: bool = False,
) -> Path:
    """Build YOLO-seg dataset (VOC masks when available, else bbox quads)."""

    def _writer(
        image_id: str,
        rows: list[tuple[int, float, float, float, float]],
        voc_devkit: Path | None,
        year_name: str,
    ) -> list[str]:
        if voc_devkit is not None:
            from_mask = _seg_lines_from_voc_mask(voc_devkit, year_name, image_id, rows)
            if from_mask:
                return from_mask
        return [_bbox_to_seg_line(c, xc, yc, w, h) for c, xc, yc, w, h in rows]

    return _build_task_dataset(
        yolo_root,
        "seg",
        label_writer=_writer,
        voc_mini_root=voc_mini_root,
        year=year,
        force=force,
    )


def prepare_mini_voc_pose(yolo_root: Path, force: bool = False) -> Path:
    """Build YOLO-pose dataset (synthetic person keypoints inside bbox)."""

    def _writer(
        _image_id: str,
        rows: list[tuple[int, float, float, float, float]],
        _voc_devkit: Path | None,
        _year_name: str,
    ) -> list[str]:
        return [
            _bbox_to_pose_line(c, xc, yc, w, h)
            for c, xc, yc, w, h in rows
            if c == PERSON_CLASS_ID
        ]

    return _build_task_dataset(yolo_root, "pose", label_writer=_writer, force=force)


def prepare_mini_voc_cls(yolo_root: Path, force: bool = False) -> Path:
    """Build ImageFolder-style classification dataset from detection labels."""
    cls_root = yolo_root.parent / "voc-mini-cls"
    marker = cls_root / ".ready"
    if marker.is_file() and not force:
        return cls_root

    if cls_root.exists() and force:
        shutil.rmtree(cls_root)

    split_map = {"train2007": "train", "test2007": "val"}
    for yolo_split, cls_split in split_map.items():
        img_dir = yolo_root / "images" / yolo_split
        lbl_dir = yolo_root / "labels" / yolo_split
        for img in sorted(img_dir.glob("*.jpg")):
            rows = _read_det_labels(lbl_dir / f"{img.stem}.txt")
            if not rows:
                cls_name = "background"
            else:
                cls_name = VOC_NAMES[rows[0][0]]
            dest_dir = cls_root / cls_split / cls_name
            dest_dir.mkdir(parents=True, exist_ok=True)
            dest = dest_dir / img.name
            if dest.exists() or dest.is_symlink():
                dest.unlink()
            try:
                os.symlink(img.resolve(), dest)
            except OSError:
                shutil.copy2(img, dest)

    marker.parent.mkdir(parents=True, exist_ok=True)
    marker.write_text("ok\n", encoding="utf-8")
    return cls_root


def write_voc_mini_seg_yaml(output_path: Path, yolo_dataset_root: Path) -> Path:
    """Write dataset YAML for segmentation trainers (det images + seg labels)."""
    dataset_path = yolo_dataset_root.absolute()
    names_block = "\n".join(f"  {idx}: {name}" for idx, name in enumerate(VOC_NAMES))
    content = f"""# Auto-generated for mini VOC regression tests (VOC masks or bbox polygons).
path: {dataset_path}
train:
  - images/train2007
val:
  - images/test2007
test:
  - images/test2007
nc: {len(VOC_NAMES)}

names:
{names_block}
"""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(content, encoding="utf-8")
    return output_path


def write_voc_mini_pose_yaml(output_path: Path, yolo_dataset_root: Path) -> Path:
    """Write dataset YAML for pose trainers."""
    dataset_path = yolo_dataset_root.absolute()
    names_block = "\n".join(f"  {idx}: {name}" for idx, name in enumerate(VOC_NAMES))
    kpt_shape = list(POSE_KPT_SHAPE)
    content = f"""# Auto-generated for mini VOC regression tests (synthetic person keypoints).
path: {dataset_path}
train:
  - images/train2007
val:
  - images/test2007
test:
  - images/test2007

kpt_shape: {kpt_shape}
flip_idx: [0, 2, 1, 4, 3, 6, 5, 8, 7, 10, 9, 12, 11, 14, 13, 16, 15]

names:
{names_block}
"""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(content, encoding="utf-8")
    return output_path


def write_voc_mini_cls_yaml(output_path: Path, cls_dataset_root: Path) -> Path:
    """Write dataset YAML for classification trainers."""
    dataset_path = cls_dataset_root.absolute()
    names_block = "\n".join(f"  {idx}: {name}" for idx, name in enumerate(VOC_NAMES))
    content = f"""# Auto-generated for mini VOC regression tests (ImageFolder layout).
path: {dataset_path}
train: train
val: val
test: val

names:
{names_block}
"""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(content, encoding="utf-8")
    return output_path


def prepare_all_regression_datasets(
    voc_mini_root: Path,
    yolo_output: Path,
    config_dir: Path,
    *,
    year: str = "2007",
    force: bool = False,
) -> None:
    """Prepare detection, seg, pose, and cls mini-VOC layouts + YAML configs."""
    prepare_mini_voc_yolo(voc_mini_root, yolo_output, year=year, force=force)
    seg_root = prepare_mini_voc_seg(yolo_output, voc_mini_root, year=year, force=force)
    pose_root = prepare_mini_voc_pose(yolo_output, force=force)
    cls_root = prepare_mini_voc_cls(yolo_output, force=force)
    write_voc_mini_yaml(config_dir / "voc-mini.yaml", yolo_output)
    write_voc_mini_yolov3_yaml(config_dir / "voc-mini-yolov3.yaml", yolo_output)
    write_voc_mini_seg_yaml(config_dir / "voc-mini-seg.yaml", seg_root)
    write_voc_mini_pose_yaml(config_dir / "voc-mini-pose.yaml", pose_root)
    write_voc_mini_cls_yaml(config_dir / "voc-mini-cls.yaml", cls_root)
    write_yolov6_voc_mini_yaml(config_dir / "voc-mini-yolov6.yaml", yolo_output)


def write_yolov6_voc_mini_yaml(
    output_path: Path,
    yolo_dataset_root: Path,
    yolov6_root: Path | None = None,
) -> Path:
    """Write a dataset YAML for YOLOv6 (paths relative to YOLOv6 repo root)."""
    base = (yolov6_root or (_repo_root() / "yolo" / "YOLOv6")).absolute()
    dataset_path = yolo_dataset_root.absolute()
    rel = Path(os.path.relpath(str(dataset_path), str(base)))
    names_repr = repr(VOC_NAMES)
    content = f"""# Auto-generated for mini VOC regression tests. Do not edit by hand.
train: {rel.as_posix()}/images/train2007
val: {rel.as_posix()}/images/test2007
test: {rel.as_posix()}/images/test2007

is_coco: False
nc: 20
names: {names_repr}
"""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(content, encoding="utf-8")
    return output_path


def main(argv: list[str] | None = None) -> int:
    """CLI entry point."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--voc-mini",
        type=Path,
        default=_repo_root() / "datasets" / "voc-mini",
        help="Root containing VOCdevkit/ (default: datasets/voc-mini).",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=_repo_root() / "datasets" / "voc-mini-yolo",
        help="YOLO-format output root (default: datasets/voc-mini-yolo).",
    )
    parser.add_argument(
        "--config-dir",
        type=Path,
        default=_repo_root() / ".regression_configs",
        help="Directory for generated dataset YAML files.",
    )
    parser.add_argument("--year", default="2007", help="VOC year under VOCdevkit.")
    parser.add_argument("--force", action="store_true", help="Rebuild even if ready.")
    args = parser.parse_args(argv)

    prepare_all_regression_datasets(
        args.voc_mini,
        args.output,
        args.config_dir,
        year=args.year,
        force=args.force,
    )
    marker = args.output / READY_MARKER
    print(f"[INFO] YOLO mini VOC ready: {marker}")
    print(f"[INFO] Configs in: {args.config_dir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
