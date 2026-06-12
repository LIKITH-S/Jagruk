import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import glob
import json
import argparse
import logging
import shutil
import random
from pathlib import Path
import cv2
import pandas as pd
import numpy as np
import rasterio
from shapely.wkt import loads as load_wkt
from scripts.split_dataset import split_dataset

# Configure Logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def load_geotiff(img_path: str) -> np.ndarray:
    """Loads a GeoTIFF image correctly using rasterio, scaling it to 8-bit RGB BGR."""
    try:
        with rasterio.open(img_path) as src:
            # Read bands (C, H, W)
            data = src.read()
            # Convert to (H, W, C)
            img = data.transpose(1, 2, 0)
            
            # Extract basic properties for debug logging
            h, w, c = img.shape
            dtype = img.dtype
            img_min = img.min()
            img_max = img.max()
            
            logger.info(f"Loaded TIFF: {Path(img_path).name} - Shape: ({h}, {w}, {c}), Dtype: {dtype}, Min: {img_min}, Max: {img_max}, Bands: {src.count}")
            
            # Ensure we only use first 3 bands for RGB
            if c > 3:
                img = img[:, :, :3]
            elif c == 1:
                img = np.repeat(img, 3, axis=2)
                
            # Convert to float32 for normalization
            img_float = img.astype(np.float32)
            
            # Percentile clipping (1% - 99%) to eliminate extreme outliers and preserve visibility
            low = np.percentile(img_float, 1)
            high = np.percentile(img_float, 99)
            
            if high > low:
                img_norm = np.clip(img_float, low, high)
                img_norm = (img_norm - low) / (high - low) * 255.0
            else:
                # Fallback to direct scale
                if img_max > img_min:
                    img_norm = (img_float - img_min) / (img_max - img_min) * 255.0
                else:
                    img_norm = img_float
                    
            # Convert to standard uint8
            img_uint8 = np.clip(img_norm, 0, 255).astype(np.uint8)
            
            # Convert from RGB to BGR for OpenCV saving/operations
            img_bgr = cv2.cvtColor(img_uint8, cv2.COLOR_RGB2BGR)
            return img_bgr
            
    except Exception as e:
        logger.error(f"Error loading GeoTIFF {img_path}: {e}")
        raise ValueError(f"Corrupted or invalid TIFF file: {e}")

def process_file(json_path: str, out_dir: str, existing_paths: set) -> list:
    """Process a single xBD JSON label file to extract full BGR image from GeoTIFF with aggregated label."""
    json_path_obj = Path(json_path)
    
    # Identify split (e.g. tier1, tier3, test, hold)
    source_split = json_path_obj.parent.parent.name
    
    # Locate post-disaster GeoTIFF
    img_dir = json_path_obj.parent.parent / "images"
    base_name = json_path_obj.name
    img_name = base_name.replace('.json', '.tif')
    img_path = img_dir / img_name
    
    if not img_path.exists():
        raise FileNotFoundError(f"GeoTIFF image not found at {img_path} for JSON: {json_path}")
    
    # Load and preprocess GeoTIFF image correctly
    img = load_geotiff(str(img_path))
    h, w, c = img.shape

    with open(json_path, 'r') as f:
        data = json.load(f)
    
    results = []
    features = data.get("features", {}).get("xy", [])
    
    # Ensure images dir exists
    img_out_dir = os.path.join(out_dir, "images")
    os.makedirs(img_out_dir, exist_ok=True)
    
    # Compute aggregate max-severity label for the entire image
    severity_rank = {
        "destroyed": 3,
        "major_damage": 2,
        "minor_damage": 1,
        "no_damage": 0,
        "un_classified": 0
    }
    
    max_severity_val = -1
    max_severity_label = "no_damage"
    
    for feature in features:
        properties = feature.get("properties", {})
        subtype_raw = properties.get("subtype", "no_damage")
        subtype = subtype_raw.replace("-", "_")
        val = severity_rank.get(subtype, 0)
        if val > max_severity_val:
            max_severity_val = val
            max_severity_label = subtype
            
    if max_severity_val == -1:
        # Default to no_damage if no features are present
        max_severity_label = "no_damage"
        
    # Safe resize of the full-size image to 224x224
    resized_img = cv2.resize(img, (224, 224))
    
    # Create a structured, traceable filename
    source_tif_clean = img_path.stem
    image_filename = f"{source_split}_{source_tif_clean}.png"
    
    # If already processed and appending, skip writing/adding
    if image_filename not in existing_paths:
        out_path = os.path.join(img_out_dir, image_filename)
        cv2.imwrite(out_path, resized_img)
        
        results.append({
            "image_path": image_filename,
            "damage_label": max_severity_label,
            "source_split": source_split,
            "source_tif": img_name,
            "building_index": -1, # Indicates full-image classification
            "crop_width": w,
            "crop_height": h
        })
        
    return results

def validate_dataset(out_dir: str):
    """Performs visual validation, structural checks, and generates preview crops."""
    logger.info("Starting visual and structural dataset validation...")
    csv_path = os.path.join(out_dir, "labels.csv")
    img_dir = os.path.join(out_dir, "images")
    preview_dir = os.path.join(out_dir, "previews")
    
    # Re-create previews directory cleanly
    if os.path.exists(preview_dir):
        shutil.rmtree(preview_dir)
    os.makedirs(preview_dir, exist_ok=True)
    
    if not os.path.exists(csv_path):
        logger.error(f"labels.csv not found at {csv_path}. Cannot validate dataset.")
        return
        
    df = pd.read_csv(csv_path)
    if len(df) == 0:
        logger.warning("Dataset contains 0 rows in labels.csv.")
        return
        
    total_crops = len(df)
    dark_count = 0
    empty_count = 0
    corrupted_count = 0
    valid_count = 0
    
    # Save a random subset of previews for visual auditing (up to 20 crops)
    sample_size = min(total_crops, 20)
    sample_indices = set(random.sample(range(total_crops), sample_size))
    
    logger.info(f"Auditing crops. Exporting {sample_size} random verification previews to: {preview_dir}")
    
    for idx, row in df.iterrows():
        img_name = row['image_path']
        img_path = os.path.join(img_dir, img_name)
        
        if not os.path.exists(img_path):
            logger.warning(f"Image crop missing: {img_path}")
            corrupted_count += 1
            continue
            
        try:
            img = cv2.imread(img_path)
            if img is None:
                logger.warning(f"Corrupted crop (unreadable): {img_name}")
                corrupted_count += 1
                continue
                
            h, w, c = img.shape
            mean_val = img.mean()
            std_val = img.std()
            
            # Validation metrics
            is_empty = (std_val == 0) or (h == 0) or (w == 0)
            is_dark = mean_val < 15.0
            
            if is_empty:
                logger.warning(f"[EMPTY CROP WARNING] Flat texture/null dimensions in crop: {img_name}")
                empty_count += 1
            elif is_dark:
                logger.warning(f"[DARK CROP WARNING] Extremely low brightness (mean={mean_val:.2f}): {img_name}")
                dark_count += 1
            else:
                valid_count += 1
                
            # If selected, write to preview folder
            if idx in sample_indices:
                preview_path = os.path.join(preview_dir, img_name)
                cv2.imwrite(preview_path, img)
                
        except Exception as e:
            logger.error(f"Error checking crop {img_name}: {e}")
            corrupted_count += 1
            
    logger.info("=" * 40)
    logger.info("     DATASET VALIDATION REPORT")
    logger.info("=" * 40)
    logger.info(f"Total Crops Analyzed : {total_crops}")
    logger.info(f"Valid/Visual Crops   : {valid_count} ({valid_count/total_crops*100:.1f}%)")
    logger.info(f"Extremely Dark Crops : {dark_count}")
    logger.info(f"Empty/Flat Crops     : {empty_count}")
    logger.info(f"Corrupted or Missing : {corrupted_count}")
    logger.info("=" * 40)
    
    if dark_count > 0:
        logger.warning(f"[ACTION REQUIRED] {dark_count} dark crops detected. Verify raw imagery bounds.")
    if empty_count > 0:
        logger.warning(f"[ACTION REQUIRED] {empty_count} empty crops detected. Clean annotations or bounds.")

def main():
    parser = argparse.ArgumentParser(description="Extract building crops from xBD dataset GeoTIFFs.")
    parser.add_argument("--raw-dir", default="data/raw/xbd/geotiffs", help="Root Path to raw dataset.")
    parser.add_argument("--out-dir", default="data/processed/dataset", help="Output extraction directory.")
    parser.add_argument("--limit", type=int, default=None, help="Limit number of JSON files to process for testing.")
    parser.add_argument("--wipe", type=str, default=None, choices=["yes", "no"], help="Wipe old processed dataset cleanly.")
    parser.add_argument("--skip-split", action="store_true", help="Skip automatic train/val/test splitting after extraction.")
    args = parser.parse_args()
    
    # 1. Determine Wipe/Append Behavior
    out_path = Path(args.out_dir)
    images_dir = out_path / "images"
    csv_path = out_path / "labels.csv"
    
    wipe_dataset = False
    
    # Check if there's pre-existing data
    if images_dir.exists() and any(images_dir.iterdir()):
        if args.wipe is not None:
            wipe_dataset = (args.wipe == "yes")
        else:
            # Interactive prompt
            user_input = input("Wipe old processed dataset? [yes/no]: ").strip().lower()
            while user_input not in ['yes', 'no', 'y', 'n']:
                user_input = input("Please enter yes or no: ").strip().lower()
            wipe_dataset = user_input in ['yes', 'y']
            
    if wipe_dataset:
        logger.info(f"Wiping existing dataset directory cleanly: {args.out_dir}")
        if out_path.exists():
            shutil.rmtree(out_path)
        os.makedirs(args.out_dir, exist_ok=True)
        os.makedirs(images_dir, exist_ok=True)
        # Also clean any existing split directories
        for split_name in ["train", "val", "test"]:
            split_dir = out_path / split_name
            if split_dir.exists():
                shutil.rmtree(split_dir)
        existing_paths = set()
        existing_df = None
    else:
        # Append mode
        os.makedirs(args.out_dir, exist_ok=True)
        os.makedirs(images_dir, exist_ok=True)
        existing_paths = set()
        existing_df = None
        if csv_path.exists():
            try:
                existing_df = pd.read_csv(csv_path)
                if 'image_path' in existing_df.columns:
                    existing_paths = set(existing_df['image_path'].tolist())
                elif 'crop_id' in existing_df.columns:
                    existing_paths = set(existing_df['crop_id'].apply(lambda x: f"{x}.png").tolist())
                logger.info(f"Appending Mode: Loaded {len(existing_df)} existing crop references.")
            except Exception as e:
                logger.warning(f"Could not load existing labels.csv, starting fresh: {e}")
                
    # 2. Gather post-disaster JSON files
    raw_path = Path(args.raw_dir)
    json_files = list(raw_path.rglob("*_post_disaster.json"))
    
    if not json_files:
        logger.info(f"No post-disaster JSON files found in {args.raw_dir} (recursive). Nothing to process.")
        return
        
    logger.info(f"Found {len(json_files)} label files. Starting extraction...")
    
    if args.limit:
        logger.info(f"Limiting processing to {args.limit} files.")
        json_files = json_files[:args.limit]
        
    all_rows = []
    skipped_count = 0
    corrupted_count = 0
    
    # 3. Extract Crops
    for j_path in json_files:
        try:
            rows = process_file(str(j_path), args.out_dir, existing_paths)
            all_rows.extend(rows)
            logger.info(f"Successfully processed {j_path.name}")
        except ValueError as ve:
            logger.error(f"Corrupted image in {j_path.name}: {ve}")
            corrupted_count += 1
        except Exception as e:
            logger.error(f"Skipping {j_path.name} due to error: {e}")
            skipped_count += 1
            
    # 4. Save metadata to labels.csv
    if all_rows:
        new_df = pd.DataFrame(all_rows)
        if existing_df is not None:
            # Concatenate existing and new records
            combined_df = pd.concat([existing_df, new_df], ignore_index=True)
            # Drop duplicates based on image_path just in case
            combined_df.drop_duplicates(subset=['image_path'], keep='last', inplace=True)
        else:
            combined_df = new_df
            
        combined_df.to_csv(csv_path, index=False)
        logger.info(f"Saved dataset metadata successfully with {len(combined_df)} records to {csv_path}")
    elif existing_df is not None:
        logger.info(f"No new crops extracted. Maintained {len(existing_df)} existing records in {csv_path}")
    else:
        logger.warning("No crops extracted and no existing dataset found.")
        
    # 5. Visual dataset validation
    validate_dataset(args.out_dir)
    
    # Print class distribution from labels.csv
    if csv_path.exists():
        try:
            final_df = pd.read_csv(csv_path)
            dist_col = 'damage_label' if 'damage_label' in final_df.columns else 'damage_level'
            if dist_col in final_df.columns:
                logger.info("=" * 40)
                logger.info("     CLASS DISTRIBUTION")
                logger.info("=" * 40)
                dist = final_df[dist_col].value_counts()
                for cat, val in dist.items():
                    logger.info(f" {cat:<20}: {val}")
                logger.info("=" * 40)
        except Exception as e:
            logger.error(f"Failed to calculate final distribution: {e}")
            
    logger.info(f"Extraction execution finalized. Added {len(all_rows)} new crops. Skipped {skipped_count} images. Corrupted {corrupted_count} files.")
    
    # 6. Automatic dataset splitting
    if not args.skip_split and csv_path.exists():
        logger.info("")
        logger.info("=" * 40)
        logger.info("     AUTOMATIC DATASET SPLITTING")
        logger.info("=" * 40)
        try:
            split_dataset(args.out_dir, seed=42, force=True)
            logger.info("Dataset splitting completed successfully.")
        except Exception as e:
            logger.error(f"Dataset splitting failed: {e}")
            logger.info("You can run splitting manually: python scripts/split_dataset.py --data-dir " + args.out_dir)
    elif args.skip_split:
        logger.info("Skipping automatic dataset splitting (--skip-split flag set).")
        logger.info("Run manually: python scripts/split_dataset.py --data-dir " + args.out_dir)

if __name__ == "__main__":
    main()
