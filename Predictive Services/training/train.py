"""
train.py – Entry-point for the full model training pipeline.

Usage:
    python training/train.py [--model rf|xgb] [--days 1825]
"""
import sys
import os
import logging
import argparse

# Ensure project root is on the path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pandas as pd

from data_generation.synthetic_dataset_generator import SyntheticDataGenerator
from feature_engineering.feature_assembler import FeatureAssembler
from training.trainer import RandomForestTrainer, XGBoostTrainer
from training.evaluator import Evaluator
from training.artifact_manager import ArtifactManager

# ------------------------------------------------------------------
# Logging
# ------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("train")

# ------------------------------------------------------------------
# Config
# ------------------------------------------------------------------
TARGET_COL = "hazard_label"
TEST_SIZE   = 0.20
ARTIFACTS_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "artifacts")


# ------------------------------------------------------------------
# Helpers
# ------------------------------------------------------------------

def build_dataset(days=1825):
    """Generates synthetic data and applies full feature engineering."""
    logger.info("Generating synthetic dataset (%d days) …", days)
    gen = SyntheticDataGenerator(start_date="2020-01-01", days=days, seed=42)
    raw_df = gen.generate_dataset()

    logger.info("Running feature engineering pipeline …")
    assembler = FeatureAssembler(is_training=True)
    df = assembler.assemble(raw_df)

    return df, assembler


def split_dataset(df):
    """Sequential (time-series safe) train/test split."""
    split_idx = int(len(df) * (1 - TEST_SIZE))
    train_df = df.iloc[:split_idx]
    test_df  = df.iloc[split_idx:]
    logger.info("Train: %d rows | Test: %d rows", len(train_df), len(test_df))
    return train_df, test_df


def get_xy(df):
    feature_cols = [c for c in df.columns if c != TARGET_COL]
    X = df[feature_cols].values
    y = df[TARGET_COL].values
    return X, y, feature_cols


def choose_trainer(model_type: str):
    if model_type == "xgb":
        # Compute positive class weight for imbalanced data
        return XGBoostTrainer(n_estimators=200, max_depth=6, learning_rate=0.05, scale_pos_weight=49)
    return RandomForestTrainer(n_estimators=200, max_depth=12, class_weight="balanced")


# ------------------------------------------------------------------
# Main
# ------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Train environmental hazard classifier")
    parser.add_argument("--model", choices=["rf", "xgb"], default="rf",
                        help="Model to train: 'rf' (RandomForest) or 'xgb' (XGBoost)")
    parser.add_argument("--days", type=int, default=1825,
                        help="Number of synthetic days to generate (default: 1825 ≈ 5 years)")
    args = parser.parse_args()

    logger.info("=== PART 5B Training Pipeline | model=%s | days=%d ===", args.model, args.days)

    # 1. Data
    df, assembler = build_dataset(days=args.days)
    train_df, test_df = split_dataset(df)
    X_train, y_train, feature_cols = get_xy(train_df)
    X_test,  y_test,  _            = get_xy(test_df)

    # 2. Train
    trainer = choose_trainer(args.model)
    trainer.fit(X_train, y_train)

    # 3. Persist model
    artifact_mgr = ArtifactManager(artifacts_dir=ARTIFACTS_DIR)
    artifact_mgr.save_model(trainer.model, filename="model.pkl")

    # 4. Metadata
    meta = {
        "model_type":    trainer.MODEL_NAME,
        "model_params":  trainer.get_params(),
        "feature_names": feature_cols,
        "train_rows":    int(len(X_train)),
        "test_rows":     int(len(X_test)),
        "target":        TARGET_COL,
        "class_names":   ["Normal", "Hazard"],
    }
    artifact_mgr.save_meta(meta, filename="model_meta.json")

    # 5. Evaluate
    evaluator = Evaluator(artifact_manager=artifact_mgr)
    metrics = evaluator.evaluate(trainer, X_test, y_test, feature_names=feature_cols)

    logger.info("=== Pipeline complete. Metrics: %s ===", metrics)
    return metrics


if __name__ == "__main__":
    main()
