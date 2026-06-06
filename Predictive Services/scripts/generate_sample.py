import os
import sys

# Ensure we can import from data_generation if run from project root or scripts directory
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from data_generation.synthetic_dataset_generator import SyntheticDataGenerator
from data_generation.dataset_exporter import DatasetExporter

def main():
    print("Initializing synthetic dataset generator...")
    # Generate 5 years of daily synthetic data
    generator = SyntheticDataGenerator(start_date='2020-01-01', days=1825, seed=42)
    
    print("Generating environmental features and hazard labels...")
    df = generator.generate_dataset()
    
    print(f"\nDataset generated successfully with {len(df)} records.")
    print("\nFeatures overview:")
    print(df.head())
    
    print("\nHazard distribution:")
    print(df['hazard_label'].value_counts(normalize=True).apply(lambda x: f"{x:.2%}"))
    
    # Export results
    output_directory = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'data')
    exporter = DatasetExporter(output_dir=output_directory)
    
    print("\nExporting datasets...")
    # Export full dataset
    exporter.export_csv(df, filename='synthetic_environmental_data.csv')
    
    # Export train/test splits (80/20 for time series)
    exporter.export_train_test_split(df, test_size=0.2, prefix='env_disaster')
    
if __name__ == '__main__':
    main()
