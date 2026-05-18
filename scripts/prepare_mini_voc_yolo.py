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

    marker.write_text(
        f"train2007={train_n}\nval2007={val_n}\n",
        encoding="utf-8",
    )
    return marker


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

names:
{names_block}
"""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(content, encoding="utf-8")
    return output_path


def write_yolov6_voc_mini_yaml(
    output_path: Path,
    yolo_dataset_root: Path,
    yolov6_root: Path | None = None,
) -> Path:
    """Write a dataset YAML for YOLOv6 (paths relative to YOLOv6 repo root)."""
    base = (yolov6_root or (_repo_root() / "yolo" / "YOLOv6")).absolute()
    dataset_path = yolo_dataset_root.absolute()
    rel = os.path.relpath(str(dataset_path), str(base))
    voc12 = Path(rel) / "voc_07_12"
    names_repr = repr(VOC_NAMES)
    content = f"""# Auto-generated for mini VOC regression tests. Do not edit by hand.
train: {voc12.as_posix()}/images/train
val: {voc12.as_posix()}/images/val
test: {voc12.as_posix()}/images/val

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

    marker = prepare_mini_voc_yolo(args.voc_mini, args.output, year=args.year, force=args.force)
    write_voc_mini_yaml(args.config_dir / "voc-mini.yaml", args.output)
    write_yolov6_voc_mini_yaml(args.config_dir / "voc-mini-yolov6.yaml", args.output)
    print(f"[INFO] YOLO mini VOC ready: {marker}")
    print(f"[INFO] Configs: {args.config_dir / 'voc-mini.yaml'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
