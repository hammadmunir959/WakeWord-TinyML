"""
DS-CNN Model for Keyword Spotting
Depthwise Separable CNN - optimized for edge deployment.
"""

import tensorflow as tf
from tensorflow.keras import layers, Model
from typing import Tuple


class DepthwiseSeparableConv(layers.Layer):
    """Depthwise Separable Convolution block."""
    
    def __init__(
        self,
        filters: int,
        kernel_size: Tuple[int, int] = (3, 3),
        strides: Tuple[int, int] = (1, 1),
        padding: str = 'same',
        name: str = None
    ):
        super().__init__(name=name)
        
        self.depthwise = layers.DepthwiseConv2D(
            kernel_size=kernel_size,
            strides=strides,
            padding=padding,
            use_bias=False,
            name=f'{name}_dw' if name else None
        )
        self.bn1 = layers.BatchNormalization(name=f'{name}_bn1' if name else None)
        self.relu1 = layers.ReLU(name=f'{name}_relu1' if name else None)
        
        self.pointwise = layers.Conv2D(
            filters,
            kernel_size=(1, 1),
            strides=(1, 1),
            padding='same',
            use_bias=False,
            name=f'{name}_pw' if name else None
        )
        self.bn2 = layers.BatchNormalization(name=f'{name}_bn2' if name else None)
        self.relu2 = layers.ReLU(name=f'{name}_relu2' if name else None)
    
    def call(self, inputs, training=None):
        x = self.depthwise(inputs)
        x = self.bn1(x, training=training)
        x = self.relu1(x)
        
        x = self.pointwise(x)
        x = self.bn2(x, training=training)
        x = self.relu2(x)
        
        return x


def create_ds_cnn_model(
    input_shape: Tuple[int, int, int] = (98, 40, 1),
    num_classes: int = 12,
    dropout_rate: float = 0.3
) -> Model:
    """
    Create a Depthwise Separable CNN model for keyword spotting.
    
    This architecture is optimized for edge deployment with fewer
    parameters while maintaining good accuracy.
    
    Architecture:
        Conv2D(64) -> BN -> ReLU
        DS-Conv(64) -> MaxPool
        DS-Conv(64) -> MaxPool
        DS-Conv(128) -> MaxPool
        DS-Conv(128)
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
    
    # Initial Conv layer
    x = layers.Conv2D(64, (3, 3), padding='same', use_bias=False, name='conv_initial')(inputs)
    x = layers.BatchNormalization(name='bn_initial')(x)
    x = layers.ReLU(name='relu_initial')(x)
    
    # DS-Conv Block 1
    x = DepthwiseSeparableConv(64, name='ds_conv1')(x)
    x = layers.MaxPooling2D((2, 2), name='pool1')(x)
    
    # DS-Conv Block 2
    x = DepthwiseSeparableConv(64, name='ds_conv2')(x)
    x = layers.MaxPooling2D((2, 2), name='pool2')(x)
    
    # DS-Conv Block 3
    x = DepthwiseSeparableConv(128, name='ds_conv3')(x)
    x = layers.MaxPooling2D((2, 2), name='pool3')(x)
    
    # DS-Conv Block 4
    x = DepthwiseSeparableConv(128, name='ds_conv4')(x)
    
    # Global pooling and classifier
    x = layers.GlobalAveragePooling2D(name='gap')(x)
    x = layers.Dense(128, name='fc1')(x)
    x = layers.ReLU(name='relu_fc')(x)
    x = layers.Dropout(dropout_rate, name='dropout')(x)
    outputs = layers.Dense(num_classes, activation='softmax', name='output')(x)
    
    model = Model(inputs=inputs, outputs=outputs, name='ds_cnn_keyword_spotting')
    
    return model


def create_ds_cnn_small(
    input_shape: Tuple[int, int, int] = (98, 40, 1),
    num_classes: int = 12,
    dropout_rate: float = 0.2
) -> Model:
    """
    Create a smaller DS-CNN model for ultra-low latency.
    
    Args:
        input_shape: Input tensor shape
        num_classes: Number of output classes
        dropout_rate: Dropout rate
        
    Returns:
        Keras Model
    """
    inputs = layers.Input(shape=input_shape, name='input')
    
    # Initial Conv layer
    x = layers.Conv2D(32, (3, 3), padding='same', use_bias=False, name='conv_initial')(inputs)
    x = layers.BatchNormalization(name='bn_initial')(x)
    x = layers.ReLU(name='relu_initial')(x)
    x = layers.MaxPooling2D((2, 2), name='pool_initial')(x)
    
    # DS-Conv Block 1
    x = DepthwiseSeparableConv(32, name='ds_conv1')(x)
    x = layers.MaxPooling2D((2, 2), name='pool1')(x)
    
    # DS-Conv Block 2
    x = DepthwiseSeparableConv(64, name='ds_conv2')(x)
    x = layers.MaxPooling2D((2, 2), name='pool2')(x)
    
    # DS-Conv Block 3
    x = DepthwiseSeparableConv(64, name='ds_conv3')(x)
    
    # Global pooling and classifier
    x = layers.GlobalAveragePooling2D(name='gap')(x)
    x = layers.Dense(64, name='fc1')(x)
    x = layers.ReLU(name='relu_fc')(x)
    x = layers.Dropout(dropout_rate, name='dropout')(x)
    outputs = layers.Dense(num_classes, activation='softmax', name='output')(x)
    
    model = Model(inputs=inputs, outputs=outputs, name='ds_cnn_small_keyword_spotting')
    
    return model


if __name__ == "__main__":
    # Test model creation
    print("DS-CNN Model:")
    print("=" * 50)
    model = create_ds_cnn_model()
    model.summary()
    print(f"\nTotal parameters: {model.count_params():,}")
    
    print("\n\nDS-CNN Small Model:")
    print("=" * 50)
    model_small = create_ds_cnn_small()
    model_small.summary()
    print(f"\nTotal parameters: {model_small.count_params():,}")
