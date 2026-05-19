#!/usr/bin/env python3
"""Patch vendored YOLO forks for PyTorch 2.6+ ``weights_only`` default in ``torch.load``."""

from __future__ import annotations

import re
import sys
from pathlib import Path


def _cleanup_bad_patches(text: str) -> str:
    """Undo incorrect regex patches from older versions of this script."""
    text = text.replace(", weights_only=False, weights_only=False", ", weights_only=False")
    text = re.sub(
        r"attempt_download\(([^,)]+),\s*weights_only=False\)",
        r"attempt_download(\1)",
        text,
    )
    text = text.replace(
        "map_location=torch.device('cpu', weights_only=False)",
        "map_location=torch.device('cpu')",
    )
    return text


def _patch_torch_load_calls(text: str) -> tuple[str, int]:
    """Add ``weights_only=False`` to each ``torch.load(...)`` that lacks it."""
    needle = "torch.load("
    out: list[str] = []
    i = 0
    count = 0
    while True:
        start = text.find(needle, i)
        if start == -1:
            out.append(text[i:])
            break
        out.append(text[i:start])
        j = start + len(needle)
        depth = 1
        while j < len(text) and depth:
            ch = text[j]
            if ch == "(":
                depth += 1
            elif ch == ")":
                depth -= 1
            j += 1
        call = text[start:j]
        if "weights_only" not in call:
            call = call[:-1] + ", weights_only=False)"
            count += 1
        out.append(call)
        i = j
    return "".join(out), count


def _patch_file(path: Path) -> bool:
    text = path.read_text(encoding="utf-8")
    if "torch.load" not in text:
        return False
    cleaned = _cleanup_bad_patches(text)
    new_text, count = _patch_torch_load_calls(cleaned)
    if new_text == text:
        return False
    path.write_text(new_text, encoding="utf-8")
    return count > 0 or cleaned != text


def main() -> int:
    repo = Path(__file__).resolve().parent.parent
    roots = [
        repo / "yolo" / "yolov3",
        repo / "yolo" / "yolov5",
        repo / "yolo" / "yolov7",
        repo / "yolo" / "yolov9",
        repo / "yolo" / "YOLOv6",
    ]
    patched = 0
    for root in roots:
        if not root.is_dir():
            continue
        for path in root.rglob("*.py"):
            if _patch_file(path):
                patched += 1
                print(f"[INFO] patched {path.relative_to(repo)}")
    print(f"[INFO] patched {patched} file(s)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
