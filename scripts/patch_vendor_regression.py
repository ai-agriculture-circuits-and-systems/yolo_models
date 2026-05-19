#!/usr/bin/env python3
"""Re-apply mini-VOC regression fixes after vendor tree sync (rsync --delete)."""

from __future__ import annotations

import sys
from pathlib import Path


def _replace_once(path: Path, old: str, new: str) -> bool:
    text = path.read_text(encoding="utf-8")
    if new in text:
        return False
    if old not in text:
        return False
    path.write_text(text.replace(old, new, 1), encoding="utf-8")
    return True


def patch_yolov3(repo: Path) -> int:
    """Patches under ``yolo/yolov3`` for shared caches and NumPy 2 / PyTorch 2.x."""
    root = repo / "yolo" / "yolov3"
    if not root.is_dir():
        return 0
    n = 0
    datasets = root / "utils" / "datasets.py"
    if datasets.is_file():
        if _replace_once(
            datasets,
            "        labels, shapes = zip(*cache.values())",
            "        labels, shapes = zip(*((v[0], v[1]) for v in cache.values()))",
        ):
            n += 1
        if _replace_once(
            datasets,
            "        cache.pop('hash')  # remove hash\n        labels, shapes = zip(*((v[0], v[1]) for v in cache.values()))",
            "        cache.pop('hash')  # remove hash\n        cache.pop('version', None)\n        cache.pop('msgs', None)\n        labels, shapes = zip(*((v[0], v[1]) for v in cache.values()))",
        ):
            n += 1
        if _replace_once(datasets, "                x[im_file] = [l, shape]", "                x[im_file] = [l, shape, []]"):
            n += 1
        if _replace_once(datasets, ".astype(np.int)", ".astype(int)"):
            n += 1
    loss_py = root / "utils" / "loss.py"
    if loss_py.is_file():
        if _replace_once(
            loss_py,
            "        indices.append((b, a, gj.clamp_(0, gain[3] - 1), gi.clamp_(0, gain[2] - 1)))",
            "        indices.append((b, a, gj.clamp(0, gain[3].long() - 1), gi.clamp(0, gain[2].long() - 1)))",
        ):
            n += 1
    yolo_py = root / "models" / "yolo.py"
    if yolo_py.is_file():
        old = """                b = mi.bias.view(m.na, -1)  # conv.bias(255) to (3,85)
                b[:, 4] += math.log(8 / (640 / s) ** 2)  # obj (8 objects per 640 image)
                b[:, 5:] += math.log(0.6 / (m.nc - 0.99)) if cf is None else torch.log(cf / cf.sum())  # cls
                mi.bias = torch.nn.Parameter(b.view(-1), requires_grad=True)"""
        new = """                b = mi.bias.view(m.na, -1).clone()  # avoid in-place on leaf Parameter view
                b[:, 4] += math.log(8 / (640 / s) ** 2)  # obj (8 objects per 640 image)
                b[:, 5:] += math.log(0.6 / (m.nc - 0.99)) if cf is None else torch.log(cf / cf.sum())  # cls
                mi.bias = torch.nn.Parameter(b.view(-1), requires_grad=True)"""
        if _replace_once(yolo_py, old, new):
            n += 1
    return n


def main() -> int:
    repo = Path(__file__).resolve().parent.parent
    total = patch_yolov3(repo)
    print(f"[INFO] regression vendor patches applied: {total} edit(s)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
