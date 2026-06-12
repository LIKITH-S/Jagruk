"""
Test Set Evaluation Script for Jagruk CV.

Loads the best model checkpoint and runs final evaluation on the held-out
test split. Generates accuracy, precision, recall, F1-macro, per-class F1,
confusion matrix plot, and saves all metrics to reports/.

Usage:
    python scripts/evaluate.py --data-dir data/processed/dataset
    python scripts/evaluate.py --data-dir data/processed/dataset --model-path models/best_model.pth
"""

import os
import sys
import json
import argparse
import logging
from datetime import datetime, timezone

import torch
import numpy as np
from torch.utils.data import DataLoader
from torchvision import transforms
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report
)

# Add project root and src/ to path so scripts/ and cv_service are importable
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src')))

from cv_service.models.dataset import XBDDataset
from cv_service.models.architecture import DamageClassifier
from scripts.visualize import save_confusion_matrix

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

CLASS_NAMES = ['no_damage', 'minor_damage', 'major_damage', 'destroyed']


def evaluate_model(args):
    """Runs evaluation on the test split and generates reports."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Evaluating on device: {device}")

    # Resolve paths
    test_csv = os.path.join(args.data_dir, "test", "labels.csv")
    test_img_dir = os.path.join(args.data_dir, "test", "images")

    if not os.path.exists(test_csv):
        logger.error(f"Test labels.csv not found at {test_csv}. Run split_dataset.py first.")
        sys.exit(1)

    if not os.path.exists(args.model_path):
        logger.error(f"Model checkpoint not found at {args.model_path}.")
        sys.exit(1)

    # Transform (no augmentation — deterministic eval)
    eval_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    # Load test dataset
    test_dataset = XBDDataset(test_csv, test_img_dir, transform=eval_transform)
    if len(test_dataset) == 0:
        logger.error("Test dataset is empty.")
        sys.exit(1)

    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)
    logger.info(f"Test dataset loaded: {len(test_dataset)} samples")

    # Load model
    model = DamageClassifier(num_classes=4, pretrained=False)
    state_dict = torch.load(args.model_path, map_location=device)
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()
    logger.info(f"Model loaded from {args.model_path}")

    # Run inference
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for images, labels in test_loader:
            images = images.to(device)
            outputs = model(images)
            _, preds = torch.max(outputs, 1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)

    # Compute metrics
    accuracy = accuracy_score(all_labels, all_preds)
    precision = precision_score(all_labels, all_preds, average='macro', zero_division=0)
    recall = recall_score(all_labels, all_preds, average='macro', zero_division=0)
    f1_macro = f1_score(all_labels, all_preds, average='macro', zero_division=0)
    per_class_f1 = f1_score(all_labels, all_preds, average=None, zero_division=0)

    # Print classification report
    logger.info("\n" + "=" * 60)
    logger.info("          TEST SET EVALUATION RESULTS")
    logger.info("=" * 60)
    logger.info(f"  Accuracy       : {accuracy:.4f}")
    logger.info(f"  Precision (M)  : {precision:.4f}")
    logger.info(f"  Recall (M)     : {recall:.4f}")
    logger.info(f"  F1-Macro       : {f1_macro:.4f}")
    logger.info("-" * 60)
    for i, cls in enumerate(CLASS_NAMES):
        if i < len(per_class_f1):
            logger.info(f"  F1 [{cls:<14}] : {per_class_f1[i]:.4f}")
    logger.info("=" * 60)

    # Full sklearn report
    report = classification_report(
        all_labels, all_preds,
        labels=[0, 1, 2, 3],
        target_names=CLASS_NAMES,
        zero_division=0
    )
    logger.info("\nDetailed Classification Report:\n" + report)

    # Save metrics JSON
    os.makedirs(args.output_dir, exist_ok=True)
    metrics = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "model_path": args.model_path,
        "test_samples": len(test_dataset),
        "accuracy": float(accuracy),
        "precision_macro": float(precision),
        "recall_macro": float(recall),
        "f1_macro": float(f1_macro),
        "per_class_f1": {CLASS_NAMES[i]: float(per_class_f1[i]) for i in range(len(per_class_f1))},
        "device": str(device)
    }

    metrics_path = os.path.join(args.output_dir, "test_metrics.json")
    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)
    logger.info(f"Metrics saved to {metrics_path}")

    # Generate confusion matrix
    save_confusion_matrix(
        all_labels.tolist(),
        all_preds.tolist(),
        CLASS_NAMES,
        output_dir=args.output_dir
    )

    return metrics


def main():
    parser = argparse.ArgumentParser(description="Evaluate Jagruk CV model on test split.")
    parser.add_argument("--data-dir", default="data/processed/dataset",
                        help="Root of the split dataset.")
    parser.add_argument("--model-path", default="models/best_model.pth",
                        help="Path to the best model checkpoint.")
    parser.add_argument("--batch-size", type=int, default=32,
                        help="Batch size for evaluation.")
    parser.add_argument("--output-dir", default="reports",
                        help="Directory to save evaluation reports.")
    args = parser.parse_args()

    evaluate_model(args)


if __name__ == "__main__":
    main()
