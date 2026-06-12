"""
Dataset Inspection Utility for Jagruk CV.

Provides visual and statistical inspection of the split dataset:
- Class distribution (table + bar chart)
- Split integrity verification (no file overlap)
- Random sample preview export
- Metadata statistics

Usage:
    python scripts/inspect_dataset.py --data-dir data/processed/dataset
    python scripts/inspect_dataset.py --data-dir data/processed/dataset --preview-count 5
"""

import os
import sys
import json
import random
import shutil
import argparse
import logging
from pathlib import Path

import pandas as pd
import numpy as np

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

SPLITS = ["train", "val", "test"]
CLASS_NAMES = ['no_damage', 'minor_damage', 'major_damage', 'destroyed']


def inspect_dataset(data_dir: str, preview_count: int = 5, output_dir: str = "reports"):
    """
    Inspects and validates the split dataset.

    Args:
        data_dir: Root directory of the split dataset.
        preview_count: Number of random crops per class to export for preview.
        output_dir: Directory to save inspection artifacts.
    """
    data_path = Path(data_dir)

    # -------------------------------------------------------------------
    # 1. Load all split labels
    # -------------------------------------------------------------------
    split_dfs = {}
    for split_name in SPLITS:
        csv_path = data_path / split_name / "labels.csv"
        if csv_path.exists():
            split_dfs[split_name] = pd.read_csv(csv_path)
            logger.info(f"Loaded {split_name}: {len(split_dfs[split_name])} samples")
        else:
            logger.warning(f"{split_name}/labels.csv not found — skipping.")

    if not split_dfs:
        logger.error("No split labels found. Run split_dataset.py first.")
        return

    # Detect label column
    sample_df = next(iter(split_dfs.values()))
    label_col = "damage_label" if "damage_label" in sample_df.columns else "damage_level"

    # -------------------------------------------------------------------
    # 2. Class Distribution Table
    # -------------------------------------------------------------------
    logger.info("")
    logger.info("=" * 70)
    logger.info("              CLASS DISTRIBUTION BY SPLIT")
    logger.info("=" * 70)

    all_classes = set()
    for df in split_dfs.values():
        all_classes.update(df[label_col].unique())
    all_classes = sorted(all_classes)

    header = f"  {'Class':<20}"
    for split_name in SPLITS:
        if split_name in split_dfs:
            header += f"  {split_name:>8}"
    header += f"  {'TOTAL':>8}"
    logger.info(header)
    logger.info("  " + "-" * (20 + (len(split_dfs) + 1) * 10))

    for cls in all_classes:
        row = f"  {cls:<20}"
        total_cls = 0
        for split_name in SPLITS:
            if split_name in split_dfs:
                count = int((split_dfs[split_name][label_col] == cls).sum())
                total_cls += count
                row += f"  {count:>8}"
        row += f"  {total_cls:>8}"
        logger.info(row)

    # Totals row
    logger.info("  " + "-" * (20 + (len(split_dfs) + 1) * 10))
    total_row = f"  {'TOTAL':<20}"
    grand_total = 0
    for split_name in SPLITS:
        if split_name in split_dfs:
            count = len(split_dfs[split_name])
            grand_total += count
            total_row += f"  {count:>8}"
    total_row += f"  {grand_total:>8}"
    logger.info(total_row)
    logger.info("=" * 70)

    # -------------------------------------------------------------------
    # 3. Split Integrity Verification
    # -------------------------------------------------------------------
    logger.info("")
    logger.info("=" * 70)
    logger.info("              SPLIT INTEGRITY CHECK")
    logger.info("=" * 70)

    filenames = {}
    for split_name, df in split_dfs.items():
        filenames[split_name] = set(df["image_path"].tolist()) if "image_path" in df.columns else set()

    integrity_pass = True
    checked_pairs = []
    for i, s1 in enumerate(SPLITS):
        for s2 in SPLITS[i + 1:]:
            if s1 in filenames and s2 in filenames:
                overlap = filenames[s1] & filenames[s2]
                status = "PASS" if len(overlap) == 0 else "FAIL"
                if len(overlap) > 0:
                    integrity_pass = False
                checked_pairs.append((s1, s2, len(overlap), status))
                logger.info(f"  {s1} ∩ {s2}: {len(overlap)} overlapping files — {status}")

    if integrity_pass:
        logger.info("  ✅ No data leakage detected across splits.")
    else:
        logger.error("  ❌ DATA LEAKAGE DETECTED! Re-run split_dataset.py.")
    logger.info("=" * 70)

    # -------------------------------------------------------------------
    # 4. Metadata Statistics
    # -------------------------------------------------------------------
    meta_path = data_path / "split_metadata.json"
    if meta_path.exists():
        with open(meta_path, "r") as f:
            meta = json.load(f)
        logger.info("")
        logger.info("  Split Metadata:")
        logger.info(f"    Seed      : {meta.get('seed', 'N/A')}")
        logger.info(f"    Timestamp : {meta.get('timestamp', 'N/A')}")
        logger.info(f"    Leak Check: {meta.get('duplicate_leak_check', 'N/A')}")

    # -------------------------------------------------------------------
    # 5. Crop dimension statistics
    # -------------------------------------------------------------------
    if "crop_width" in sample_df.columns and "crop_height" in sample_df.columns:
        logger.info("")
        logger.info("  Crop Dimension Statistics (from metadata):")
        for split_name, df in split_dfs.items():
            min_w = df["crop_width"].min()
            max_w = df["crop_width"].max()
            min_h = df["crop_height"].min()
            max_h = df["crop_height"].max()
            logger.info(f"    [{split_name:>5}] Width: [{min_w}, {max_w}] | Height: [{min_h}, {max_h}]")

    # -------------------------------------------------------------------
    # 6. Random Sample Preview Export
    # -------------------------------------------------------------------
    if preview_count > 0:
        preview_dir = Path(output_dir) / "previews"
        if preview_dir.exists():
            shutil.rmtree(preview_dir)
        preview_dir.mkdir(parents=True, exist_ok=True)

        logger.info(f"\n  Exporting {preview_count} random preview crops per class per split...")

        for split_name, df in split_dfs.items():
            split_preview = preview_dir / split_name
            split_preview.mkdir(parents=True, exist_ok=True)
            img_dir = data_path / split_name / "images"

            for cls in all_classes:
                cls_df = df[df[label_col] == cls]
                n_sample = min(preview_count, len(cls_df))
                if n_sample == 0:
                    continue

                sample_rows = cls_df.sample(n=n_sample, random_state=42)
                cls_dir = split_preview / cls
                cls_dir.mkdir(parents=True, exist_ok=True)

                for _, row in sample_rows.iterrows():
                    img_name = row.get("image_path", "")
                    src = img_dir / img_name
                    if src.exists():
                        shutil.copy2(str(src), str(cls_dir / img_name))

        logger.info(f"  Previews saved to {preview_dir}")

    # -------------------------------------------------------------------
    # 7. Generate class distribution bar chart
    # -------------------------------------------------------------------
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt

        os.makedirs(output_dir, exist_ok=True)

        fig, ax = plt.subplots(figsize=(12, 6))
        x = np.arange(len(all_classes))
        width = 0.25
        colors = ['#FF6B6B', '#4ECDC4', '#45B7D1']

        for i, split_name in enumerate(SPLITS):
            if split_name in split_dfs:
                counts = [int((split_dfs[split_name][label_col] == cls).sum()) for cls in all_classes]
                ax.bar(x + i * width, counts, width, label=split_name.capitalize(), color=colors[i])

        ax.set_xlabel("Damage Class", fontsize=12)
        ax.set_ylabel("Count", fontsize=12)
        ax.set_title("Class Distribution by Split", fontsize=14, fontweight='bold')
        ax.set_xticks(x + width)
        ax.set_xticklabels(all_classes, rotation=20, ha='right')
        ax.legend()
        ax.grid(True, alpha=0.3, axis='y')
        fig.tight_layout()

        chart_path = os.path.join(output_dir, "class_distribution.png")
        fig.savefig(chart_path, dpi=150)
        plt.close(fig)
        logger.info(f"  Class distribution chart saved to {chart_path}")
    except ImportError:
        logger.warning("  matplotlib not available — skipping chart generation.")


def main():
    parser = argparse.ArgumentParser(description="Inspect and validate Jagruk CV dataset splits.")
    parser.add_argument("--data-dir", default="data/processed/dataset",
                        help="Root directory of the split dataset.")
    parser.add_argument("--preview-count", type=int, default=5,
                        help="Number of random crops per class to export for preview.")
    parser.add_argument("--output-dir", default="reports",
                        help="Directory to save inspection artifacts.")
    args = parser.parse_args()

    inspect_dataset(args.data_dir, preview_count=args.preview_count, output_dir=args.output_dir)


if __name__ == "__main__":
    main()
