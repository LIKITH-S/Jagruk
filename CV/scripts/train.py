"""
Jagruk CV Training Script.

Trains a ResNet50-based damage classifier on the pre-split xBD dataset.
Uses train/ for training, val/ for validation monitoring, and never touches test/.

Usage:
    # Full training
    python scripts/train.py --data-dir data/processed/dataset --epochs 10

    # Quick dry-run with subset
    python scripts/train.py --data-dir data/processed/dataset --epochs 2 --subset-size 100

    # GPU training
    python scripts/train.py --data-dir data/processed/dataset --epochs 20 --batch-size 64
"""

import os
import sys
import json
import argparse
import logging
from datetime import datetime, timezone

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Subset
from torchvision import transforms
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score, accuracy_score
import numpy as np

# Add project root and src/ to path so scripts/ and cv_service are importable
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src')))

from cv_service.models.dataset import XBDDataset
from cv_service.models.architecture import DamageClassifier
from scripts.visualize import save_training_plots

# Configure Logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def train_model(args):
    """Main training loop for damage classification."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Training using device: {device}")

    # 1. Define Augmentations (Moderate & Controlled)
    train_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.ColorJitter(brightness=0.2, contrast=0.2),
        transforms.GaussianBlur(kernel_size=(3, 3), sigma=(0.1, 2.0)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    val_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    # 2. Resolve split paths
    train_csv = os.path.join(args.data_dir, "train", "labels.csv")
    train_img_dir = os.path.join(args.data_dir, "train", "images")
    val_csv = os.path.join(args.data_dir, "val", "labels.csv")
    val_img_dir = os.path.join(args.data_dir, "val", "images")

    if not os.path.exists(train_csv):
        logger.error(f"Train labels.csv not found at {train_csv}. Run preprocessing + splitting first.")
        return

    if not os.path.exists(val_csv):
        logger.error(f"Val labels.csv not found at {val_csv}. Run preprocessing + splitting first.")
        return

    # 3. Load Datasets from pre-split directories
    train_dataset = XBDDataset(train_csv, train_img_dir, transform=train_transform)
    val_dataset = XBDDataset(val_csv, val_img_dir, transform=val_transform)

    if len(train_dataset) == 0:
        logger.error("Training dataset is empty. Ensure preprocessing and splitting were run correctly.")
        return

    if len(val_dataset) == 0:
        logger.error("Validation dataset is empty. Ensure preprocessing and splitting were run correctly.")
        return

    logger.info(f"Train set: {len(train_dataset)} samples | Val set: {len(val_dataset)} samples")

    # 4. Optional subsetting (for quick dry-run training)
    train_indices = np.arange(len(train_dataset))
    if args.subset_size > 0 and args.subset_size < len(train_dataset):
        logger.info(f"Subsetting training data to {args.subset_size} samples.")
        try:
            train_indices, _ = train_test_split(
                train_indices,
                train_size=args.subset_size,
                stratify=train_dataset.df[train_dataset.label_col],
                random_state=42
            )
        except Exception:
            train_indices, _ = train_test_split(
                train_indices,
                train_size=args.subset_size,
                random_state=42
            )
        train_subset = Subset(train_dataset, train_indices)
    else:
        train_subset = train_dataset

    train_loader = DataLoader(train_subset, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False)

    # 5. Model, Loss, Optimizer
    model = DamageClassifier(num_classes=4, pretrained=True).to(device)

    # Class weighting in CrossEntropyLoss (computed from training set)
    weights = train_dataset.get_class_weights().to(device)
    criterion = nn.CrossEntropyLoss(weight=weights)

    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.1, patience=2)

    # 6. Training Loop with history tracking
    best_f1 = 0.0
    os.makedirs("models", exist_ok=True)

    history = {
        "train_loss": [],
        "val_loss": [],
        "f1": [],
        "accuracy": []
    }

    effective_train_size = len(train_subset)

    for epoch in range(args.epochs):
        model.train()
        train_loss = 0.0

        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            train_loss += loss.item() * images.size(0)

        # Validation
        model.eval()
        val_preds = []
        val_labels = []
        val_loss = 0.0

        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                loss = criterion(outputs, labels)
                val_loss += loss.item() * images.size(0)

                _, preds = torch.max(outputs, 1)
                val_preds.extend(preds.cpu().numpy())
                val_labels.extend(labels.cpu().numpy())

        # Metrics
        avg_train_loss = train_loss / effective_train_size
        avg_val_loss = val_loss / len(val_dataset)
        acc = accuracy_score(val_labels, val_preds)
        f1 = f1_score(val_labels, val_preds, average='macro', zero_division=0)

        # Record history
        history["train_loss"].append(avg_train_loss)
        history["val_loss"].append(avg_val_loss)
        history["f1"].append(f1)
        history["accuracy"].append(acc)

        logger.info(f"Epoch {epoch+1}/{args.epochs} - "
                    f"Train Loss: {avg_train_loss:.4f}, Val Loss: {avg_val_loss:.4f}, "
                    f"Acc: {acc:.4f}, F1-Macro: {f1:.4f}")

        scheduler.step(f1)

        # Checkpointing — save BEST model only (by val F1-macro)
        if f1 > best_f1:
            best_f1 = f1
            torch.save(model.state_dict(), "models/best_model.pth")
            logger.info(f"Model saved to models/best_model.pth (Best F1: {best_f1:.4f})")

    # 7. Post-training: save plots and metrics
    reports_dir = "reports"
    os.makedirs(reports_dir, exist_ok=True)

    # Save training curves
    try:
        save_training_plots(history, output_dir=reports_dir)
    except Exception as e:
        logger.warning(f"Failed to save training plots: {e}")

    # Save training metrics JSON
    metrics = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "device": str(device),
        "epochs_completed": args.epochs,
        "best_val_f1": float(best_f1),
        "final_train_loss": float(history["train_loss"][-1]) if history["train_loss"] else None,
        "final_val_loss": float(history["val_loss"][-1]) if history["val_loss"] else None,
        "final_accuracy": float(history["accuracy"][-1]) if history["accuracy"] else None,
        "train_samples": effective_train_size,
        "val_samples": len(val_dataset),
        "history": history
    }

    metrics_path = os.path.join(reports_dir, "training_metrics.json")
    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)
    logger.info(f"Training metrics saved to {metrics_path}")

    logger.info(f"\nTraining complete. Best F1-Macro: {best_f1:.4f}")
    logger.info(f"Run evaluation on test set: python scripts/evaluate.py --data-dir {args.data_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Jagruk CV Training Script")
    parser.add_argument("--data-dir", default="data/processed/dataset",
                        help="Root of the split dataset (contains train/, val/, test/).")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--subset-size", type=int, default=0,
                        help="Subset training data for quick dry-run. 0 = use all data.")

    args = parser.parse_args()
    train_model(args)
