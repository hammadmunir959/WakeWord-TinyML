"""
Training Script for WakeWord
Trains CNN or DS-CNN model on Speech Commands dataset.
"""

import os
import sys
import argparse
from pathlib import Path
from datetime import datetime
import json

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import yaml
import numpy as np
import tensorflow as tf
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import (
    ModelCheckpoint,
    EarlyStopping,
    ReduceLROnPlateau,
    TensorBoard,
    CSVLogger
)

from src.model.cnn import create_cnn_model
from src.model.ds_cnn import create_ds_cnn_model, create_ds_cnn_small
from src.data.dataloader import WakeWordDataset


def load_config(config_path: str) -> dict:
    """Load configuration from YAML file."""
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


def create_model(config: dict, input_shape: tuple, num_classes: int) -> tf.keras.Model:
    """Create model based on configuration."""
    model_type = config['model']['type']
    dropout = config['model']['dropout']
    
    if model_type == 'cnn':
        model = create_cnn_model(input_shape, num_classes, dropout)
    elif model_type == 'ds_cnn':
        model = create_ds_cnn_model(input_shape, num_classes, dropout)
    elif model_type == 'ds_cnn_small':
        model = create_ds_cnn_small(input_shape, num_classes, dropout)
    else:
        raise ValueError(f"Unknown model type: {model_type}")
    
    return model


def get_callbacks(config: dict, checkpoint_dir: Path, run_name: str) -> list:
    """Create training callbacks."""
    callbacks = []
    
    # Model checkpoint
    checkpoint_path = checkpoint_dir / f"{run_name}_best.keras"
    callbacks.append(ModelCheckpoint(
        str(checkpoint_path),
        monitor='val_accuracy',
        save_best_only=True,
        mode='max',
        verbose=1
    ))
    
    # Early stopping
    callbacks.append(EarlyStopping(
        monitor='val_accuracy',
        patience=config['training']['early_stopping_patience'],
        restore_best_weights=True,
        verbose=1
    ))
    
    # Learning rate scheduler
    if config['training']['lr_scheduler'] == 'plateau':
        callbacks.append(ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=5,
            min_lr=1e-6,
            verbose=1
        ))
    
    # TensorBoard
    tb_dir = checkpoint_dir / 'logs' / run_name
    callbacks.append(TensorBoard(log_dir=str(tb_dir)))
    
    # CSV Logger
    csv_path = checkpoint_dir / f"{run_name}_history.csv"
    callbacks.append(CSVLogger(str(csv_path)))
    
    return callbacks


def train(args):
    """Main training function."""
    # Load config
    config = load_config(args.config)
    
    # Set random seeds
    np.random.seed(42)
    tf.random.set_seed(42)
    
    # Load dataset
    print("\nLoading dataset...")
    data_dir = Path(config['paths']['data_processed'])
    dataset = WakeWordDataset(str(data_dir))
    
    input_shape = dataset.get_input_shape()
    num_classes = dataset.get_num_classes()
    
    print(f"Input shape: {input_shape}")
    print(f"Number of classes: {num_classes}")
    
    # Get datasets
    batch_size = args.batch_size or config['training']['batch_size']
    train_ds = dataset.get_tf_dataset('train', batch_size, shuffle=True, augment=True)
    val_ds = dataset.get_tf_dataset('val', batch_size, shuffle=False, augment=False)
    
    # Create model
    print("\nCreating model...")
    model = create_model(config, input_shape, num_classes)
    model.summary()
    
    # Compile model
    lr = config['training']['learning_rate']
    model.compile(
        optimizer=Adam(learning_rate=lr),
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )
    
    # Setup callbacks
    checkpoint_dir = Path(config['paths']['checkpoints'])
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    
    run_name = f"{config['model']['type']}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    callbacks = get_callbacks(config, checkpoint_dir, run_name)
    
    # Get cosine scheduler if configured
    epochs = args.epochs or config['training']['epochs']
    if config['training']['lr_scheduler'] == 'cosine':
        # Calculate total steps
        train_features, _ = dataset.load_split('train')
        steps_per_epoch = len(train_features) // batch_size
        total_steps = steps_per_epoch * epochs
        
        lr_schedule = tf.keras.optimizers.schedules.CosineDecay(
            initial_learning_rate=lr,
            decay_steps=total_steps,
            alpha=1e-2
        )
        model.compile(
            optimizer=Adam(learning_rate=lr_schedule),
            loss='sparse_categorical_crossentropy',
            metrics=['accuracy']
        )
    
    # Train
    print(f"\nStarting training for {epochs} epochs...")
    history = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=epochs,
        callbacks=callbacks,
        verbose=1
    )
    
    # Save final model
    final_path = Path(config['paths']['exported']) / f"{run_name}_final.keras"
    final_path.parent.mkdir(parents=True, exist_ok=True)
    model.save(str(final_path))
    print(f"\nFinal model saved to: {final_path}")
    
    # Evaluate on test set
    print("\nEvaluating on test set...")
    test_ds = dataset.get_tf_dataset('test', batch_size, shuffle=False, augment=False)
    test_loss, test_acc = model.evaluate(test_ds)
    print(f"Test Loss: {test_loss:.4f}")
    print(f"Test Accuracy: {test_acc:.4f}")
    
    # Save training summary
    summary = {
        'run_name': run_name,
        'model_type': config['model']['type'],
        'epochs_trained': len(history.history['loss']),
        'best_val_accuracy': float(max(history.history['val_accuracy'])),
        'test_accuracy': float(test_acc),
        'test_loss': float(test_loss),
        'input_shape': list(input_shape),
        'num_classes': num_classes,
        'total_params': int(model.count_params())
    }
    
    summary_path = checkpoint_dir / f"{run_name}_summary.json"
    with open(summary_path, 'w') as f:
        json.dump(summary, f, indent=2)
    
    print(f"\nTraining complete! Summary saved to: {summary_path}")
    
    return model, history


def main():
    """Entry point."""
    parser = argparse.ArgumentParser(description='Train WakeWord model')
    parser.add_argument('--config', type=str, default='configs/config.yaml',
                        help='Path to config file')
    parser.add_argument('--epochs', type=int, default=None,
                        help='Override epochs from config')
    parser.add_argument('--batch_size', type=int, default=None,
                        help='Override batch size from config')
    parser.add_argument('--debug', action='store_true',
                        help='Run in debug mode with fewer samples')
    
    args = parser.parse_args()
    
    # Change to project root
    os.chdir(project_root)
    
    print("=" * 50)
    print("WakeWord Model Training")
    print("=" * 50)
    
    train(args)


if __name__ == "__main__":
    main()
