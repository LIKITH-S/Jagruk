"""
Interactive API Prediction Test Script for Jagruk CV.

Fetches real image crops of different damage levels from the test split,
sends them as base64 requests to the FastAPI application, and compares the
model's predicted outputs (score, label, explainability features) with the
actual ground truth labels.

Usage:
    python scripts/test_api_prediction.py
"""

import os
import sys
import base64
import logging
from tabulate import tabulate

# Add project root and src/ to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src')))

from fastapi.testclient import TestClient
from cv_service.api.main import app

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

import pandas as pd

def main():
    test_csv = "data/processed/dataset/test/labels.csv"
    data_dir = "data/processed/dataset/test/images"
    
    if not os.path.exists(test_csv):
        logger.error(f"Test labels file not found at {test_csv}. Please run split_dataset.py first.")
        sys.exit(1)
        
    df = pd.read_csv(test_csv)
    if len(df) == 0:
        logger.error("Test labels.csv is empty.")
        sys.exit(1)
        
    logger.info("Initializing API Client and triggering application lifespan events...")
    
    table_data = []
    
    # Using 'with' triggers FastAPI's ASGI lifespan startup and shutdown context manager automatically
    with TestClient(app) as client:
        logger.info("Lifespan events completed successfully. App initialized.")
        logger.info("-" * 80)
        
        # Process up to 5 images from test set
        for idx, row in df.head(5).iterrows():
            img_name = row["image_path"]
            actual_label = row["damage_label"]
            img_path = os.path.join(data_dir, img_name)
            
            if not os.path.exists(img_path):
                logger.error(f"Image not found at {img_path}. Skipping.")
                continue
                
            logger.info(f"Processing test image: {img_name}")
            logger.info(f"  └─ Ground Truth Label: {actual_label}")
            
            # Read image and encode to base64
            with open(img_path, "rb") as f:
                img_bytes = f.read()
            b64_str = base64.b64encode(img_bytes).decode('utf-8')
            
            # Send API request
            payload = {
                "report_id": f"test-{actual_label}-{idx}",
                "image_base64": b64_str
            }
            
            response = client.post("/cv/analyze", json=payload)
            
            if response.status_code != 200:
                logger.error(f"  └─ API failed with status {response.status_code}: {response.text}")
                continue
                
            res = response.json()
            pred_label = res["damage_label"]
            damage_score = res["damage_score"]
            confidence = res["confidence"]
            fake_score = res["fake_image_score"]
            
            explain = res["explain"]
            debris_ratio = explain["debris_ratio"]
            broken_edges = explain["broken_edges"]
            collapse_features = explain["collapse_features"]
            model_prob = explain["model_probability"]
            
            logger.info(f"  └─ Predicted Label   : {pred_label} (Match: {actual_label == pred_label})")
            logger.info(f"  └─ Damage Score      : {damage_score:.4f}")
            logger.info(f"  └─ Confidence        : {confidence:.4f}")
            logger.info(f"  └─ Explain Details   : Debris: {debris_ratio:.4f} | Edges: {broken_edges:.4f} | Collapse: {collapse_features:.4f}")
            logger.info("-" * 80)
            
            match_status = "MATCH" if actual_label == pred_label else "MISMATCH"
            
            table_data.append([
                img_name[:30] + "...",
                actual_label,
                pred_label,
                match_status,
                f"{damage_score:.4f}",
                f"{confidence:.4f}",
                f"{fake_score:.4f}",
                f"{model_prob:.4f}",
                f"{debris_ratio:.4f}",
                f"{broken_edges:.4f}",
                f"{collapse_features:.4f}"
            ])
            
    if table_data:
        # Print the beautiful summary comparison table
        print("\n" + "=" * 120)
        print("                      JAGRUK COMPUTER VISION API END-TO-END PREDICTION REPORT")
        print("=" * 120)
        headers = [
            "Image (Truncated)", "Actual Label", "Predicted Label", "Status",
            "Score", "Conf", "Fake", "Model P", "Debris", "Edges", "Collapse"
        ]
        print(tabulate(table_data, headers=headers, tablefmt="grid"))
        print("=" * 120 + "\n")
    else:
        logger.warning("No test data was evaluated.")

if __name__ == "__main__":
    main()
