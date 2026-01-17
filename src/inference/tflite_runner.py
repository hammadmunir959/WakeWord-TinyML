"""
TFLite Inference Runner for WakeWord
Handles loading and running TFLite models for keyword spotting.
"""

import numpy as np
try:
    import tensorflow as tf
except ImportError:
    try:
        # Try new Google LiteRT (ai-edge-litert)
        import ai_edge_litert.interpreter as tflite
        # Create a mock object that mimics tf.lite for our usage
        class MockTFLite:
            Interpreter = tflite.Interpreter
        class MockTF:
            lite = MockTFLite()
        tf = MockTF()
    except ImportError:
        try:
            # Try legacy tflite-runtime
            import tflite_runtime.interpreter as tflite
            class MockTFLite:
                Interpreter = tflite.Interpreter
            class MockTF:
                lite = MockTFLite()
            tf = MockTF()
        except ImportError:
            raise ImportError("Neither 'tensorflow', 'ai-edge-litert', nor 'tflite-runtime' were found.")

from pathlib import Path
from typing import Tuple, List, Optional
import time


class TFLiteRunner:
    """TFLite model inference runner."""
    
    def __init__(self, model_path: str, labels: Optional[List[str]] = None):
        """
        Initialize TFLite runner.
        
        Args:
            model_path: Path to TFLite model
            labels: List of class labels
        """
        self.model_path = Path(model_path)
        self.labels = labels or []
        
        # Load interpreter
        self.interpreter = tf.lite.Interpreter(model_path=str(self.model_path))
        self.interpreter.allocate_tensors()
        
        # Get input/output details
        self.input_details = self.interpreter.get_input_details()
        self.output_details = self.interpreter.get_output_details()
        
        # Get input info
        self.input_shape = self.input_details[0]['shape'][1:]  # Exclude batch dim
        self.input_dtype = self.input_details[0]['dtype']
        
        # Get quantization info
        self.input_scale = 1.0
        self.input_zero_point = 0
        self.output_scale = 1.0
        self.output_zero_point = 0
        
        if self.input_dtype == np.int8:
            quant = self.input_details[0].get('quantization', (1.0, 0))
            if quant[0] != 0:
                self.input_scale = quant[0]
                self.input_zero_point = quant[1]
            
            quant = self.output_details[0].get('quantization', (1.0, 0))
            if quant[0] != 0:
                self.output_scale = quant[0]
                self.output_zero_point = quant[1]
        
        print(f"Loaded TFLite model: {self.model_path.name}")
        print(f"Input shape: {self.input_shape}, dtype: {self.input_dtype}")
    
    def preprocess(self, features: np.ndarray) -> np.ndarray:
        """
        Preprocess features for inference.
        
        Args:
            features: Input features (time_steps, n_mfcc) or (batch, time_steps, n_mfcc)
            
        Returns:
            Preprocessed input tensor
        """
        # Add batch dim if needed
        if len(features.shape) == 2:
            features = features[np.newaxis, ...]
        
        # Add channel dim if needed
        if len(features.shape) == 3:
            features = features[..., np.newaxis]
        
        # Normalize
        mean = np.mean(features)
        std = np.std(features) + 1e-8
        features = (features - mean) / std
        
        # Quantize if needed
        if self.input_dtype == np.int8:
            features = (features / self.input_scale + self.input_zero_point)
            features = np.clip(features, -128, 127).astype(np.int8)
        else:
            features = features.astype(np.float32)
        
        return features
    
    def predict(self, features: np.ndarray) -> Tuple[np.ndarray, float]:
        """
        Run inference on features.
        
        Args:
            features: Input features
            
        Returns:
            Tuple of (predictions, inference_time_ms)
        """
        # Preprocess
        input_data = self.preprocess(features)
        
        # Run inference
        start = time.perf_counter()
        self.interpreter.set_tensor(self.input_details[0]['index'], input_data)
        self.interpreter.invoke()
        output_data = self.interpreter.get_tensor(self.output_details[0]['index'])
        inference_time = (time.perf_counter() - start) * 1000
        
        # Dequantize if needed
        if self.output_details[0]['dtype'] == np.int8:
            output_data = (output_data.astype(np.float32) - self.output_zero_point) * self.output_scale
        
        return output_data, inference_time
    
    def predict_class(
        self,
        features: np.ndarray,
        threshold: float = 0.5
    ) -> Tuple[Optional[str], float, float]:
        """
        Predict class label.
        
        Args:
            features: Input features
            threshold: Confidence threshold
            
        Returns:
            Tuple of (class_label, confidence, inference_time_ms)
        """
        predictions, inference_time = self.predict(features)
        
        # Get top prediction
        confidence = np.max(predictions)
        class_idx = np.argmax(predictions)
        
        if confidence >= threshold:
            label = self.labels[class_idx] if class_idx < len(self.labels) else str(class_idx)
        else:
            label = None
        
        return label, float(confidence), inference_time
    
    def get_model_info(self) -> dict:
        """Get model information."""
        return {
            'model_path': str(self.model_path),
            'input_shape': list(self.input_shape),
            'input_dtype': str(self.input_dtype),
            'num_classes': self.output_details[0]['shape'][-1],
            'is_quantized': self.input_dtype == np.int8
        }
