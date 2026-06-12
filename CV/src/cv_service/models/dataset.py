import os
import pandas as pd
import torch
from torch.utils.data import Dataset
from PIL import Image

class XBDDataset(Dataset):
    """
    Custom Dataset for loading building crops from xBD dataset.
    Expects labels.csv and an images directory.
    """
    LABEL_MAP = {
        'no-damage': 0,
        'no_damage': 0,
        'minor-damage': 1,
        'minor_damage': 1,
        'major-damage': 2,
        'major_damage': 2,
        'destroyed': 3
    }
    
    def __init__(self, csv_path, img_dir, transform=None):
        self.df = pd.read_csv(csv_path)
        self.img_dir = img_dir
        self.transform = transform
        
        # Dynamically support both traceable metadata and old columns
        if 'image_path' in self.df.columns:
            self.image_col = 'image_path'
            self.has_extension = True
        else:
            self.image_col = 'crop_id'
            self.has_extension = False
            
        if 'damage_label' in self.df.columns:
            self.label_col = 'damage_label'
        else:
            self.label_col = 'damage_level'
        
        # Filter out rows where image files don't exist
        self.df = self.df[self.df[self.image_col].apply(
            lambda x: os.path.exists(os.path.join(self.img_dir, x if self.has_extension else f"{x}.png"))
        )].reset_index(drop=True)
        
    def __len__(self):
        return len(self.df)
        
    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
            
        img_file = self.df.iloc[idx][self.image_col]
        label_name = self.df.iloc[idx][self.label_col]
        label = self.LABEL_MAP.get(label_name, 0)
        
        img_name = img_file if self.has_extension else f"{img_file}.png"
        img_path = os.path.join(self.img_dir, img_name)
        image = Image.open(img_path).convert('RGB')
        
        if self.transform:
            image = self.transform(image)
            
        return image, label

    def get_class_weights(self):
        """
        Computes weights for each class based on frequencies in the dataset.
        Formula: w_i = Total / (Num_Classes * Count_i)
        """
        counts = self.df[self.label_col].map(self.LABEL_MAP).value_counts().sort_index()
        total = len(self.df)
        num_classes = 4 # Fixed number of damage classes
        
        weights = []
        for i in range(num_classes):
            count = counts.get(i, 0)
            if count > 0:
                weight = total / (num_classes * count)
            else:
                weight = 0.0
            weights.append(weight)
            
        return torch.tensor(weights, dtype=torch.float)

