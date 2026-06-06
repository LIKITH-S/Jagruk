#!/usr/bin/env python3
"""
promote_model.py — Copy a candidate model from staging to the live artifacts dir.

Usage:
    python scripts/promote_model.py --src /tmp/model.joblib --name severity_model.joblib

Populated fully in Phase 3.
"""
import argparse
import shutil
from pathlib import Path

ARTIFACTS_DIR = Path(__file__).resolve().parent.parent / "artifacts"


def promote(src: Path, name: str) -> None:
    dest = ARTIFACTS_DIR / name
    ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dest)
    print(f"[promote] {src} → {dest}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--src", required=True, type=Path)
    parser.add_argument("--name", required=True)
    args = parser.parse_args()
    promote(args.src, args.name)
