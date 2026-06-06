import pandas as pd

class FeatureScaler:
    """Scales numerical features."""
    
    def __init__(self, method='standard'):
        """
        Args:
            method (str): 'standard' for z-score scaling, 'minmax' for [0,1] scaling.
        """
        self.method = method
        self.params = {}

    def fit_transform(self, df, columns):
        """Fits scaling parameters and transforms the data."""
        df_scaled = df.copy()
        
        for col in columns:
            if col not in df.columns:
                continue
                
            if self.method == 'standard':
                mean = df[col].mean()
                std = df[col].std()
                self.params[col] = {'mean': mean, 'std': std}
                # Add small epsilon to prevent division by zero
                df_scaled[col] = (df[col] - mean) / (std + 1e-9)
            elif self.method == 'minmax':
                cmin = df[col].min()
                cmax = df[col].max()
                self.params[col] = {'min': cmin, 'max': cmax}
                df_scaled[col] = (df[col] - cmin) / (cmax - cmin + 1e-9)
        return df_scaled
        
    def transform(self, df, columns):
        """Transforms data using previously fitted parameters."""
        df_scaled = df.copy()
        
        for col in columns:
            if col not in df.columns or col not in self.params:
                continue
                
            if self.method == 'standard':
                mean = self.params[col]['mean']
                std = self.params[col]['std']
                df_scaled[col] = (df[col] - mean) / (std + 1e-9)
            elif self.method == 'minmax':
                cmin = self.params[col]['min']
                cmax = self.params[col]['max']
                df_scaled[col] = (df[col] - cmin) / (cmax - cmin + 1e-9)
        return df_scaled
