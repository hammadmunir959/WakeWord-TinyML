"""
Preprocess Script for WakeWord
Run the full preprocessing pipeline on downloaded data.
"""

import os
import sys
import argparse
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import yaml
from src.data.preprocessor import DataPreprocessor


def load_config(config_path: str) -> dict:
    """Load configuration from YAML file."""
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


def main():
    """Entry point."""
    parser = argparse.ArgumentParser(description='Preprocess audio data')
    parser.add_argument('--config', type=str, default='configs/config.yaml',
                        help='Path to config file')
    parser.add_argument('--val_split', type=float, default=0.1,
                        help='Validation split ratio')
    parser.add_argument('--test_split', type=float, default=0.1,
                        help='Test split ratio')
    parser.add_argument('--no_augment', action='store_true',
                        help='Disable data augmentation')
    parser.add_argument('--num_augmentations', type=int, default=2,
                        help='Number of augmented versions per sample')
    
    args = parser.parse_args()
    
    # Change to project root
    os.chdir(project_root)
    
    print("=" * 50)
    print("WakeWord Data Preprocessing")
    print("=" * 50)
    
    # Load config
    config = load_config(args.config)
    
    # Create preprocessor
    preprocessor = DataPreprocessor(config)
    
    # Get paths
    raw_dir = Path(config['paths']['data_raw'])
    processed_dir = Path(config['paths']['data_processed'])
    
    print(f"\nRaw data: {raw_dir}")
    print(f"Output: {processed_dir}")
    print(f"Classes: {len(config['dataset']['classes'])}")
    print(f"Augmentation: {'disabled' if args.no_augment else 'enabled'}")
    
    # Run preprocessing
    preprocessor.preprocess_dataset(
        raw_dir=raw_dir,
        output_dir=processed_dir,
        val_split=args.val_split,
        test_split=args.test_split,
        augment_train=not args.no_augment,
        num_augmentations=args.num_augmentations
    )
    
    print("\nDone!")


if __name__ == "__main__":
    main()
