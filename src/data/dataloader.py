"""
Data Loader for WakeWord
Loads preprocessed features for training.
"""

import numpy as np
from pathlib import Path
from typing import Tuple, Dict, Optional
import json
import tensorflow as tf


class WakeWordDataset:
    """Dataset class for loading preprocessed keyword spotting data."""
    
    def __init__(self, data_dir: str):
        """
        Initialize dataset.
        
        Args:
            data_dir: Path to processed data directory
        """
        self.data_dir = Path(data_dir)
        
        # Load metadata
        metadata_path = self.data_dir / 'metadata.json'
        if metadata_path.exists():
            with open(metadata_path, 'r') as f:
                self.metadata = json.load(f)
        else:
            self.metadata = {}
        
        self.classes = self.metadata.get('classes', [])
        self.class_to_idx = self.metadata.get('class_to_idx', {})
    
    def load_split(self, split: str) -> Tuple[np.ndarray, np.ndarray]:
        """
        Load a data split.
        
        Args:
            split: Split name ('train', 'val', 'test')
            
        Returns:
            Tuple of (features, labels)
        """
        split_dir = self.data_dir / split
        
        features = np.load(split_dir / 'features.npy')
        labels = np.load(split_dir / 'labels.npy')
        
        return features, labels
    
    def get_tf_dataset(
        self,
        split: str,
        batch_size: int = 64,
        shuffle: bool = True,
        augment: bool = False
    ) -> tf.data.Dataset:
        """
        Get TensorFlow Dataset for a split.
        
        Args:
            split: Split name
            batch_size: Batch size
            shuffle: Whether to shuffle
            augment: Whether to apply augmentation
            
        Returns:
            tf.data.Dataset
        """
        features, labels = self.load_split(split)
        
        # Add channel dimension if needed
        if len(features.shape) == 3:
            features = features[..., np.newaxis]
        
        # Create dataset
        dataset = tf.data.Dataset.from_tensor_slices((features, labels))
        
        if shuffle:
            dataset = dataset.shuffle(buffer_size=len(features))
        
        # Normalize features
        dataset = dataset.map(
            lambda x, y: (self._normalize(x), y),
            num_parallel_calls=tf.data.AUTOTUNE
        )
        
        if augment:
            dataset = dataset.map(
                lambda x, y: (self._augment(x), y),
                num_parallel_calls=tf.data.AUTOTUNE
            )
        
        dataset = dataset.batch(batch_size)
        dataset = dataset.prefetch(tf.data.AUTOTUNE)
        
        return dataset
    
    @staticmethod
    def _normalize(features: tf.Tensor) -> tf.Tensor:
        """Normalize features to zero mean and unit variance."""
        mean = tf.reduce_mean(features)
        std = tf.math.reduce_std(features) + 1e-8
        return (features - mean) / std
    
    @staticmethod
    def _augment(features: tf.Tensor) -> tf.Tensor:
        """Apply simple augmentation (time masking)."""
        # SpecAugment-style time masking
        time_steps = tf.shape(features)[0]
        mask_length = tf.random.uniform([], 0, time_steps // 10, dtype=tf.int32)
        mask_start = tf.random.uniform([], 0, time_steps - mask_length, dtype=tf.int32)
        
        # Create mask
        mask = tf.concat([
            tf.ones([mask_start, tf.shape(features)[1], tf.shape(features)[2]]),
            tf.zeros([mask_length, tf.shape(features)[1], tf.shape(features)[2]]),
            tf.ones([time_steps - mask_start - mask_length, tf.shape(features)[1], tf.shape(features)[2]])
        ], axis=0)
        
        return features * mask
    
    def get_num_classes(self) -> int:
        """Get number of classes."""
        return len(self.classes)
    
    def get_input_shape(self) -> Tuple[int, ...]:
        """Get input shape from first sample."""
        features, _ = self.load_split('train')
        shape = features[0].shape
        if len(shape) == 2:
            return (*shape, 1)
        return shape
