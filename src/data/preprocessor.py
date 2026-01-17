"""
Data Preprocessor for WakeWord
Converts raw audio files to MFCC features and creates train/val/test splits.
"""

import os
import sys
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import numpy as np
import yaml
from tqdm import tqdm
import json

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.utils.audio import (
    load_audio,
    pad_or_trim,
    normalize_audio,
    extract_mfcc,
    augment_audio
)


class DataPreprocessor:
    """Preprocesses audio data for keyword spotting."""
    
    def __init__(self, config: dict):
        """
        Initialize preprocessor.
        
        Args:
            config: Configuration dictionary
        """
        self.config = config
        self.audio_config = config['audio']
        self.aug_config = config.get('augmentation', {})
        
        # Class mapping
        self.classes = config['dataset']['classes']
        self.class_to_idx = {c: i for i, c in enumerate(self.classes)}
        self.idx_to_class = {i: c for i, c in enumerate(self.classes)}
        
    def process_audio_file(
        self,
        audio_path: Path,
        augment: bool = False
    ) -> np.ndarray:
        """
        Process single audio file to MFCC features.
        
        Args:
            audio_path: Path to audio file
            augment: Whether to apply augmentation
            
        Returns:
            MFCC features array
        """
        sr = self.audio_config['sample_rate']
        duration = self.audio_config['duration']
        target_length = int(sr * duration)
        
        # Load audio
        audio, _ = load_audio(audio_path, sr=sr, duration=None)
        
        # Pad or trim
        audio = pad_or_trim(audio, target_length)
        
        # Normalize
        audio = normalize_audio(audio)
        
        # Apply augmentation if training
        if augment and self.aug_config.get('enabled', False):
            audio = augment_audio(
                audio,
                sr=sr,
                time_shift_ms=self.aug_config.get('time_shift_ms', 100),
                speed_range=tuple(self.aug_config.get('speed_range', [0.9, 1.1])),
                noise_snr_db=tuple(self.aug_config.get('noise_snr_db', [5, 20])),
                volume_db=self.aug_config.get('volume_db', 3),
                pitch_semitones=self.aug_config.get('pitch_shift_semitones', 2),
                target_length=target_length
            )
        
        # Extract MFCC
        mfcc = extract_mfcc(
            audio,
            sr=sr,
            n_mfcc=self.audio_config['n_mfcc'],
            n_fft=self.audio_config['n_fft'],
            hop_length=self.audio_config['hop_length'],
            n_mels=self.audio_config['n_mels'],
            fmin=self.audio_config['fmin'],
            fmax=self.audio_config['fmax']
        )
        
        return mfcc
    
    def get_file_list(self, data_dir: Path) -> Dict[str, List[Path]]:
        """
        Get list of audio files organized by class.
        
        Args:
            data_dir: Path to raw data directory
            
        Returns:
            Dictionary mapping class names to file lists
        """
        file_dict = {}
        
        for class_name in self.classes:
            if class_name.startswith('_'):
                continue  # Handle special classes separately
            
            class_dir = data_dir / class_name
            if class_dir.exists():
                files = list(class_dir.glob('*.wav'))
                file_dict[class_name] = files
        
        return file_dict
    
    def create_splits(
        self,
        file_dict: Dict[str, List[Path]],
        val_split: float = 0.1,
        test_split: float = 0.1,
        seed: int = 42
    ) -> Tuple[List, List, List]:
        """
        Create train/val/test splits.
        
        Args:
            file_dict: Dictionary of class to file lists
            val_split: Validation split ratio
            test_split: Test split ratio
            seed: Random seed
            
        Returns:
            Tuple of (train_files, val_files, test_files)
        """
        np.random.seed(seed)
        
        train_files = []
        val_files = []
        test_files = []
        
        for class_name, files in file_dict.items():
            # Shuffle files
            files = np.array(files)
            np.random.shuffle(files)
            
            # Calculate split indices
            n_files = len(files)
            n_test = int(n_files * test_split)
            n_val = int(n_files * val_split)
            
            # Split
            test_files.extend([(f, class_name) for f in files[:n_test]])
            val_files.extend([(f, class_name) for f in files[n_test:n_test + n_val]])
            train_files.extend([(f, class_name) for f in files[n_test + n_val:]])
        
        return train_files, val_files, test_files
    
    def process_and_save(
        self,
        files: List[Tuple[Path, str]],
        output_dir: Path,
        split_name: str,
        augment: bool = False,
        num_augmentations: int = 1
    ) -> None:
        """
        Process files and save to disk.
        
        Args:
            files: List of (file_path, class_name) tuples
            output_dir: Output directory
            split_name: Name of split (train/val/test)
            augment: Whether to apply augmentation
            num_augmentations: Number of augmented versions per sample
        """
        split_dir = output_dir / split_name
        split_dir.mkdir(parents=True, exist_ok=True)
        
        features = []
        labels = []
        
        desc = f"Processing {split_name}"
        for file_path, class_name in tqdm(files, desc=desc):
            label = self.class_to_idx[class_name]
            
            # Process original
            try:
                mfcc = self.process_audio_file(file_path, augment=False)
                features.append(mfcc)
                labels.append(label)
                
                # Create augmented versions for training
                if augment and split_name == 'train':
                    for _ in range(num_augmentations):
                        mfcc_aug = self.process_audio_file(file_path, augment=True)
                        features.append(mfcc_aug)
                        labels.append(label)
            except Exception as e:
                print(f"Error processing {file_path}: {e}")
                continue
        
        # Convert to numpy arrays
        features = np.array(features)
        labels = np.array(labels)
        
        # Save
        np.save(split_dir / 'features.npy', features)
        np.save(split_dir / 'labels.npy', labels)
        
        print(f"Saved {split_name}: {features.shape[0]} samples, shape: {features.shape}")
    
    def preprocess_dataset(
        self,
        raw_dir: Path,
        output_dir: Path,
        val_split: float = 0.1,
        test_split: float = 0.1,
        augment_train: bool = True,
        num_augmentations: int = 2
    ) -> None:
        """
        Full preprocessing pipeline.
        
        Args:
            raw_dir: Path to raw data
            output_dir: Path to save processed data
            val_split: Validation split ratio
            test_split: Test split ratio
            augment_train: Whether to augment training data
            num_augmentations: Number of augmented versions
        """
        print("Getting file list...")
        file_dict = self.get_file_list(raw_dir)
        
        total_files = sum(len(f) for f in file_dict.values())
        print(f"Found {total_files} files across {len(file_dict)} classes")
        
        print("\nCreating splits...")
        train_files, val_files, test_files = self.create_splits(
            file_dict, val_split, test_split
        )
        print(f"Train: {len(train_files)}, Val: {len(val_files)}, Test: {len(test_files)}")
        
        print("\nProcessing training data...")
        self.process_and_save(
            train_files, output_dir, 'train',
            augment=augment_train, num_augmentations=num_augmentations
        )
        
        print("\nProcessing validation data...")
        self.process_and_save(val_files, output_dir, 'val', augment=False)
        
        print("\nProcessing test data...")
        self.process_and_save(test_files, output_dir, 'test', augment=False)
        
        # Save metadata
        metadata = {
            'classes': self.classes,
            'class_to_idx': self.class_to_idx,
            'audio_config': self.audio_config,
            'num_train': len(train_files),
            'num_val': len(val_files),
            'num_test': len(test_files)
        }
        
        with open(output_dir / 'metadata.json', 'w') as f:
            json.dump(metadata, f, indent=2)
        
        print("\nPreprocessing complete!")
