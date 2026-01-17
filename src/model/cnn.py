"""
CNN Model for Keyword Spotting
Standard Convolutional Neural Network architecture.
"""

import tensorflow as tf
from tensorflow.keras import layers, Model
from typing import Tuple


def create_cnn_model(
    input_shape: Tuple[int, int, int] = (98, 40, 1),
    num_classes: int = 12,
    dropout_rate: float = 0.5
) -> Model:
    """
    Create a standard CNN model for keyword spotting.
    
    Architecture:
        Conv2D(64) -> BN -> MaxPool
        Conv2D(64) -> BN -> MaxPool
        Conv2D(128) -> BN -> MaxPool
        Conv2D(128) -> BN
        GlobalAveragePooling
        Dense(128) -> Dropout
        Dense(num_classes)
    
    Args:
        input_shape: Input tensor shape (time_steps, n_mfcc, channels)
        num_classes: Number of output classes
        dropout_rate: Dropout rate
        
    Returns:
        Keras Model
    """
    inputs = layers.Input(shape=input_shape, name='input')
    
    # Block 1
    x = layers.Conv2D(64, (3, 3), padding='same', name='conv1')(inputs)
    x = layers.BatchNormalization(name='bn1')(x)
    x = layers.ReLU(name='relu1')(x)
    x = layers.MaxPooling2D((2, 2), name='pool1')(x)
    
    # Block 2
    x = layers.Conv2D(64, (3, 3), padding='same', name='conv2')(x)
    x = layers.BatchNormalization(name='bn2')(x)
    x = layers.ReLU(name='relu2')(x)
    x = layers.MaxPooling2D((2, 2), name='pool2')(x)
    
    # Block 3
    x = layers.Conv2D(128, (3, 3), padding='same', name='conv3')(x)
    x = layers.BatchNormalization(name='bn3')(x)
    x = layers.ReLU(name='relu3')(x)
    x = layers.MaxPooling2D((2, 2), name='pool3')(x)
    
    # Block 4
    x = layers.Conv2D(128, (3, 3), padding='same', name='conv4')(x)
    x = layers.BatchNormalization(name='bn4')(x)
    x = layers.ReLU(name='relu4')(x)
    
    # Global pooling and classifier
    x = layers.GlobalAveragePooling2D(name='gap')(x)
    x = layers.Dense(128, name='fc1')(x)
    x = layers.ReLU(name='relu_fc')(x)
    x = layers.Dropout(dropout_rate, name='dropout')(x)
    outputs = layers.Dense(num_classes, activation='softmax', name='output')(x)
    
    model = Model(inputs=inputs, outputs=outputs, name='cnn_keyword_spotting')
    
    return model


def get_cnn_model_summary(model: Model) -> str:
    """Get a string summary of the model."""
    string_list = []
    model.summary(print_fn=lambda x: string_list.append(x))
    return '\n'.join(string_list)


if __name__ == "__main__":
    # Test model creation
    model = create_cnn_model()
    model.summary()
    
    # Print parameter count
    total_params = model.count_params()
    print(f"\nTotal parameters: {total_params:,}")
