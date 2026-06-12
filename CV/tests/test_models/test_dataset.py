import os
import pandas as pd
import numpy as np
import torch
import torchvision.transforms as transforms
import pytest
from PIL import Image
import tempfile
from cv_service.models.dataset import XBDDataset

def create_mock_dataset(img_dir, csv_path):
    """Utility to provide mock images and CSV for testing."""
    os.makedirs(img_dir, exist_ok=True)
    
    # Create 3 images
    data = []
    for i in range(3):
        img_id = f"test_{i}"
        label = "no-damage" if i < 2 else "destroyed"
        
        # Save placeholder image
        img = Image.new('RGB', (100, 100), color=(i*50, 0, 0))
        img.save(os.path.join(img_dir, f"{img_id}.png"))
        
        data.append({
            "crop_id": img_id,
            "original_image": "test.png",
            "damage_level": label,
            "polygon_wkt": "POLYGON ((0 0, 0 1, 1 1, 1 0, 0 0))"
        })
        
    df = pd.DataFrame(data)
    df.to_csv(csv_path, index=False)

def test_xbd_dataset_loading():
    """Verify loading from mock CSV and images."""
    with tempfile.TemporaryDirectory() as tmp_dir:
        csv_path = os.path.join(tmp_dir, "labels.csv")
        img_dir = os.path.join(tmp_dir, "images")
        
        create_mock_dataset(img_dir, csv_path)
        
        transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor()
        ])
        
        dataset = XBDDataset(csv_path, img_dir, transform=transform)
        
        # Assert length is exactly 3
        assert len(dataset) == 3
        
        image, label = dataset[0]
        # Image should be resized to 224x224
        assert image.size() == (3, 224, 224)
        # First image has label "no-damage" (mapped to 0)
        assert label == 0

def test_xbd_dataset_class_weights():
    """Verify class weights are calculated correctly."""
    with tempfile.TemporaryDirectory() as tmp_dir:
        csv_path = os.path.join(tmp_dir, "labels.csv")
        img_dir = os.path.join(tmp_dir, "images")
        
        create_mock_dataset(img_dir, csv_path)
        dataset = XBDDataset(csv_path, img_dir)
        
        weights = dataset.get_class_weights()
        
        # We have 2 "no-damage" (0) and 1 "destroyed" (3)
        # Total = 3, Classes = 4
        # w0 = 3 / (4 * 2) = 0.375
        # w3 = 3 / (4 * 1) = 0.75
        # w1, w2 = 0.0
        
        assert weights[0] == 0.375
        assert weights[1] == 0.0
        assert weights[2] == 0.0
        assert weights[3] == 0.75
