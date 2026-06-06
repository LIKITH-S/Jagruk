import os
import pandas as pd

class DatasetExporter:
    """
    Handles exporting and splitting of the dataset.
    """
    def __init__(self, output_dir='data'):
        self.output_dir = output_dir
        os.makedirs(self.output_dir, exist_ok=True)
        
    def export_csv(self, df, filename='synthetic_dataset.csv'):
        """Exports the full dataset to CSV."""
        filepath = os.path.join(self.output_dir, filename)
        df.to_csv(filepath, index=False)
        print(f"Dataset exported to {filepath}")
        return filepath
        
    def export_train_test_split(self, df, test_size=0.2, prefix='dataset'):
        """Splits data sequentially (time-series) and exports."""
        # For time-series, we split sequentially rather than randomly
        split_idx = int(len(df) * (1 - test_size))
        
        train_df = df.iloc[:split_idx]
        test_df = df.iloc[split_idx:]
        
        train_path = os.path.join(self.output_dir, f"{prefix}_train.csv")
        test_path = os.path.join(self.output_dir, f"{prefix}_test.csv")
        
        train_df.to_csv(train_path, index=False)
        test_df.to_csv(test_path, index=False)
        
        print(f"Train split ({len(train_df)} rows) exported to {train_path}")
        print(f"Test split ({len(test_df)} rows) exported to {test_path}")
        
        return train_path, test_path
