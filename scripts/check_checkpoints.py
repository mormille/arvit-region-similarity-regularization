#!/usr/bin/env python3
"""Report whether historical checkpoint files are loadable binaries or LFS pointers."""
from __future__ import annotations

from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
CHECKPOINT_DIR = ROOT / "pretrained_models"
LFS_HEADER = b"version https://git-lfs.github.com/spec/v1"


def main() -> None:
    paths = sorted(CHECKPOINT_DIR.glob("*.pth"))
    if not paths:
        print("No historical checkpoint paths were found.")
        return

    loadable = 0
    pointers = 0
    for path in paths:
        prefix = path.read_bytes()[:128]
        if prefix.startswith(LFS_HEADER) or path.stat().st_size < 1024:
            pointers += 1
            print(f"POINTER/UNAVAILABLE: {path.relative_to(ROOT)} ({path.stat().st_size} bytes)")
            continue

        try:
            torch.load(path, map_location="cpu")
        except Exception as exc:
            print(f"BINARY PRESENT, LOAD FAILED: {path.relative_to(ROOT)}: {type(exc).__name__}: {exc}")
        else:
            loadable += 1
            print(f"LOADABLE: {path.relative_to(ROOT)}")

    print(f"Summary: {loadable} loadable; {pointers} pointer/unavailable; {len(paths)} total")


if __name__ == "__main__":
    main()
