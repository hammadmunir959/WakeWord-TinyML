"""
Data Downloader for Google Speech Commands Dataset
Downloads and extracts the dataset for keyword spotting training.
"""

import os
import sys
import tarfile
import urllib.request
from pathlib import Path
from tqdm import tqdm

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

import yaml


class DownloadProgressBar(tqdm):
    """Progress bar for downloads."""
    
    def update_to(self, b=1, bsize=1, tsize=None):
        if tsize is not None:
            self.total = tsize
        self.update(b * bsize - self.n)


def load_config(config_path: str = "configs/config.yaml") -> dict:
    """Load configuration from YAML file."""
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


def download_file(url: str, output_path: str) -> None:
    """Download a file with progress bar."""
    print(f"Downloading from: {url}")
    print(f"Saving to: {output_path}")
    
    with DownloadProgressBar(unit='B', unit_scale=True, miniters=1, desc="Downloading") as t:
        urllib.request.urlretrieve(url, filename=output_path, reporthook=t.update_to)
    
    print("Download complete!")


def extract_archive(archive_path: str, extract_path: str) -> None:
    """Extract tar.gz archive."""
    print(f"Extracting to: {extract_path}")
    
    with tarfile.open(archive_path, 'r:gz') as tar:
        members = tar.getmembers()
        for member in tqdm(members, desc="Extracting"):
            tar.extract(member, extract_path)
    
    print("Extraction complete!")


def verify_dataset(data_path: str, expected_classes: list) -> bool:
    """Verify the dataset was downloaded correctly."""
    data_dir = Path(data_path)
    
    if not data_dir.exists():
        print(f"ERROR: Data directory not found: {data_path}")
        return False
    
    # Check for expected class directories
    found_classes = []
    for class_name in expected_classes:
        if class_name.startswith('_'):
            continue  # Skip special classes like _silence_, _unknown_
        class_dir = data_dir / class_name
        if class_dir.exists() and class_dir.is_dir():
            num_files = len(list(class_dir.glob('*.wav')))
            found_classes.append((class_name, num_files))
    
    print("\nDataset verification:")
    print("-" * 40)
    for class_name, num_files in found_classes:
        print(f"  {class_name}: {num_files} samples")
    print("-" * 40)
    print(f"Total classes found: {len(found_classes)}")
    
    return len(found_classes) > 0


def download_speech_commands(config: dict) -> None:
    """Main function to download Speech Commands dataset."""
    # Get paths from config
    url = config['dataset']['url']
    raw_path = Path(config['paths']['data_raw'])
    
    # Create directories
    raw_path.mkdir(parents=True, exist_ok=True)
    
    # Define file paths
    archive_name = "speech_commands_v0.02.tar.gz"
    archive_path = raw_path / archive_name
    
    # Download if not exists
    if not archive_path.exists():
        download_file(url, str(archive_path))
    else:
        print(f"Archive already exists: {archive_path}")
    
    # Check if already extracted
    test_dir = raw_path / "yes"  # Check for one class directory
    if test_dir.exists():
        print("Dataset already extracted.")
    else:
        extract_archive(str(archive_path), str(raw_path))
    
    # Verify dataset
    expected_classes = config['dataset']['classes']
    if verify_dataset(str(raw_path), expected_classes):
        print("\nDataset ready for preprocessing!")
    else:
        print("\nWARNING: Dataset verification failed!")


def main():
    """Entry point."""
    # Get project root
    project_root = Path(__file__).parent.parent
    os.chdir(project_root)
    
    print("=" * 50)
    print("Google Speech Commands Dataset Downloader")
    print("=" * 50)
    
    # Load config
    config = load_config()
    
    # Download dataset
    download_speech_commands(config)
    
    print("\nDone!")


if __name__ == "__main__":
    main()
