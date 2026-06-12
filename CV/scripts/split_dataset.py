"""
Dataset Splitting Utility for Jagruk CV.

Performs stratified train/val/test splitting (70/15/15) of the preprocessed
building crop dataset, producing physically separated directories with
per-split labels.csv files and metadata.

Usage (standalone):
    python scripts/split_dataset.py --data-dir data/processed/dataset

Usage (imported):
    from scripts.split_dataset import split_dataset
    split_dataset("data/processed/dataset", seed=42)
"""

import os
import sys
import json
import shutil
import logging
import argparse
from pathlib import Path
from datetime import datetime, timezone

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

SPLIT_RATIOS = {"train": 0.70, "val": 0.15, "test": 0.15}
SPLITS = ["train", "val", "test"]


def split_dataset(data_dir: str, seed: int = 42, force: bool = False) -> dict:
    """
    Splits the flat preprocessed dataset into train/val/test directories.

    Args:
        data_dir: Root of the preprocessed dataset (contains images/ and labels.csv).
        seed: Random seed for reproducibility.
        force: If True, overwrites existing split directories.

    Returns:
        Dictionary with split statistics and metadata.
    """
    data_path = Path(data_dir)
    flat_csv = data_path / "labels.csv"
    flat_images = data_path / "images"

    if not flat_csv.exists():
        logger.error(f"labels.csv not found at {flat_csv}. Run preprocessing first.")
        sys.exit(1)

    if not flat_images.exists():
        logger.error(f"images/ directory not found at {flat_images}. Run preprocessing first.")
        sys.exit(1)

    # Load the flat labels
    df = pd.read_csv(flat_csv)
    logger.info(f"Loaded {len(df)} records from {flat_csv}")

    if len(df) == 0:
        logger.error("labels.csv is empty. Nothing to split.")
        sys.exit(1)

    # Determine label column
    label_col = "damage_label" if "damage_label" in df.columns else "damage_level"
    if label_col not in df.columns:
        logger.error(f"Neither 'damage_label' nor 'damage_level' column found in labels.csv.")
        sys.exit(1)

    # Check for existing splits
    split_dirs_exist = any((data_path / s).exists() for s in SPLITS)
    if split_dirs_exist and not force:
        logger.warning("Split directories already exist. Use --force to overwrite.")
        logger.info("Skipping split. Use --force flag to rebuild splits.")
        return {}

    # Clean existing split directories
    for split_name in SPLITS:
        split_dir = data_path / split_name
        if split_dir.exists():
            shutil.rmtree(split_dir)
        (split_dir / "images").mkdir(parents=True, exist_ok=True)

    # -------------------------------------------------------------------
    # Stratified two-phase split: 70% train, then 50/50 of remaining 30%
    # -------------------------------------------------------------------
    labels = df[label_col].values
    indices = np.arange(len(df))

    # Phase 1: 70% train / 30% temp
    try:
        train_idx, temp_idx = train_test_split(
            indices,
            test_size=0.30,
            stratify=labels,
            random_state=seed
        )
    except ValueError:
        # Fallback if stratification fails (too few samples in a class)
        logger.warning("Stratified split failed, falling back to random split.")
        train_idx, temp_idx = train_test_split(
            indices,
            test_size=0.30,
            random_state=seed
        )

    # Phase 2: 50/50 split of the 30% → 15% val / 15% test
    temp_labels = labels[temp_idx]
    try:
        val_idx, test_idx = train_test_split(
            temp_idx,
            test_size=0.50,
            stratify=temp_labels,
            random_state=seed
        )
    except ValueError:
        logger.warning("Stratified val/test split failed, falling back to random split.")
        val_idx, test_idx = train_test_split(
            temp_idx,
            test_size=0.50,
            random_state=seed
        )

    split_indices = {"train": train_idx, "val": val_idx, "test": test_idx}

    # -------------------------------------------------------------------
    # Copy files and generate per-split labels.csv
    # -------------------------------------------------------------------
    metadata = {
        "seed": seed,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "total_samples": len(df),
        "splits": {},
        "label_column": label_col,
        "duplicate_leak_check": "PASS"
    }

    all_filenames = {}  # split_name -> set of filenames for leak check

    for split_name, idx_array in split_indices.items():
        split_df = df.iloc[idx_array].copy().reset_index(drop=True)
        split_dir = data_path / split_name
        split_img_dir = split_dir / "images"

        # Copy images
        copied = 0
        missing = 0
        for _, row in split_df.iterrows():
            img_name = row.get("image_path", "")
            src = flat_images / img_name
            dst = split_img_dir / img_name
            if src.exists():
                shutil.copy2(str(src), str(dst))
                copied += 1
            else:
                logger.warning(f"Missing image during split copy: {src}")
                missing += 1

        # Save per-split labels.csv
        split_csv = split_dir / "labels.csv"
        split_df.to_csv(split_csv, index=False)

        # Compute class distribution
        class_dist = split_df[label_col].value_counts().to_dict()

        metadata["splits"][split_name] = {
            "count": len(split_df),
            "copied": copied,
            "missing": missing,
            "class_distribution": class_dist
        }

        all_filenames[split_name] = set(split_df["image_path"].tolist())

        logger.info(f"  [{split_name.upper():>5}] {len(split_df):>6} samples | Copied: {copied} | Missing: {missing}")

    # -------------------------------------------------------------------
    # Duplicate leak check
    # -------------------------------------------------------------------
    for i, s1 in enumerate(SPLITS):
        for s2 in SPLITS[i + 1:]:
            overlap = all_filenames[s1] & all_filenames[s2]
            if overlap:
                metadata["duplicate_leak_check"] = "FAIL"
                logger.error(f"DATA LEAK DETECTED: {len(overlap)} files overlap between {s1} and {s2}!")
                for f in list(overlap)[:5]:
                    logger.error(f"  Leaked file: {f}")

    # Save metadata
    meta_path = data_path / "split_metadata.json"
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)
    logger.info(f"Split metadata saved to {meta_path}")

    # -------------------------------------------------------------------
    # Print summary table
    # -------------------------------------------------------------------
    _print_split_summary(metadata, label_col)

    return metadata


def _print_split_summary(metadata: dict, label_col: str):
    """Prints a formatted summary of the dataset split."""
    logger.info("")
    logger.info("=" * 60)
    logger.info("          DATASET SPLIT SUMMARY")
    logger.info("=" * 60)
    logger.info(f"  Total Samples : {metadata['total_samples']}")
    logger.info(f"  Random Seed   : {metadata['seed']}")
    logger.info(f"  Leak Check    : {metadata['duplicate_leak_check']}")
    logger.info("-" * 60)

    # Collect all class names across all splits
    all_classes = set()
    for split_info in metadata["splits"].values():
        all_classes.update(split_info["class_distribution"].keys())
    all_classes = sorted(all_classes)

    # Header
    header = f"  {'Class':<20}"
    for split_name in SPLITS:
        header += f"  {split_name:>8}"
    logger.info(header)
    logger.info("  " + "-" * (20 + len(SPLITS) * 10))

    # Rows
    for cls in all_classes:
        row = f"  {cls:<20}"
        for split_name in SPLITS:
            count = metadata["splits"][split_name]["class_distribution"].get(cls, 0)
            row += f"  {count:>8}"
        logger.info(row)

    # Totals
    logger.info("  " + "-" * (20 + len(SPLITS) * 10))
    total_row = f"  {'TOTAL':<20}"
    for split_name in SPLITS:
        total_row += f"  {metadata['splits'][split_name]['count']:>8}"
    logger.info(total_row)
    logger.info("=" * 60)


def main():
    parser = argparse.ArgumentParser(description="Split preprocessed xBD dataset into train/val/test.")
    parser.add_argument("--data-dir", default="data/processed/dataset",
                        help="Root directory of the preprocessed dataset.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility.")
    parser.add_argument("--force", action="store_true", help="Overwrite existing split directories.")
    args = parser.parse_args()

    split_dataset(args.data_dir, seed=args.seed, force=args.force)


if __name__ == "__main__":
    main()
