import os
import pytest
import tempfile
import torch
import pandas as pd
from PIL import Image
from scripts.train import train_model


def create_mock_split_environment(tmp_dir):
    """Creates a mock split-based dataset directory with train/ and val/ subdirs."""
    classes = ["no_damage", "minor_damage", "major_damage", "destroyed"]

    for split_name in ["train", "val"]:
        csv_path = os.path.join(tmp_dir, split_name, "labels.csv")
        img_dir = os.path.join(tmp_dir, split_name, "images")
        os.makedirs(img_dir, exist_ok=True)

        data = []
        # Create enough mock images per class for stratification to work
        n_per_class = 3 if split_name == "train" else 2
        idx = 0
        for cls in classes:
            for j in range(n_per_class):
                img_id = f"{split_name}_mock_{idx:03d}_{cls}.png"
                img = Image.new('RGB', (224, 224), color=(idx * 10 % 256, j * 30 % 256, 0))
                img.save(os.path.join(img_dir, img_id))
                data.append({
                    "image_path": img_id,
                    "damage_label": cls,
                    "source_split": "mock",
                    "source_tif": "mock.tif",
                    "building_index": idx,
                    "crop_width": 100,
                    "crop_height": 100
                })
                idx += 1

        df = pd.DataFrame(data)
        df.to_csv(csv_path, index=False)

    return tmp_dir


class Args:
    """Mock args for train_model."""
    def __init__(self, data_dir, epochs=1, batch_size=2, lr=1e-4, subset_size=0):
        self.data_dir = data_dir
        self.epochs = epochs
        self.batch_size = batch_size
        self.lr = lr
        self.subset_size = subset_size


def test_train_model_execution():
    """Verify that train_model runs for 1 epoch without crash on CPU."""
    with tempfile.TemporaryDirectory() as tmp_dir:
        data_dir = create_mock_split_environment(tmp_dir)
        args = Args(data_dir)

        # Run 1 epoch training loop
        # We don't need a GPU for this check.
        try:
            train_model(args)
        except Exception as e:
            pytest.fail(f"train_model crashed: {e}")

        # Verify checkpoint was created
        assert os.path.exists("models/best_model.pth")
