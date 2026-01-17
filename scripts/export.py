"""
TFLite Export Script for WakeWord
Converts trained models to TFLite format with optional quantization.
"""

import os
import sys
import argparse
from pathlib import Path
import json
import time

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import numpy as np
import tensorflow as tf
import yaml


def load_config(config_path: str) -> dict:
    """Load configuration from YAML file."""
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


def representative_dataset_gen(data_path: str, num_samples: int = 100):
    """
    Generator for representative dataset used in quantization.
    
    Args:
        data_path: Path to processed data directory
        num_samples: Number of samples to use
        
    Yields:
        List containing a sample input tensor
    """
    features = np.load(Path(data_path) / 'train' / 'features.npy')
    
    # Add channel dimension if needed
    if len(features.shape) == 3:
        features = features[..., np.newaxis]
    
    # Normalize
    mean = np.mean(features)
    std = np.std(features) + 1e-8
    features = (features - mean) / std
    
    # Select random samples
    indices = np.random.choice(len(features), min(num_samples, len(features)), replace=False)
    
    for i in indices:
        sample = features[i:i+1].astype(np.float32)
        yield [sample]


def convert_to_tflite(
    model_path: str,
    output_path: str,
    quantization: str = 'none',
    data_path: str = None
) -> dict:
    """
    Convert Keras model to TFLite.
    
    Args:
        model_path: Path to Keras model
        output_path: Path to save TFLite model
        quantization: Quantization type ('none', 'float16', 'int8')
        data_path: Path to data for int8 quantization
        
    Returns:
        Dictionary with conversion info
    """
    print(f"\nLoading model from: {model_path}")
    model = tf.keras.models.load_model(model_path)
    
    # Create converter
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    
    # Apply quantization
    if quantization == 'float16':
        print("Applying float16 quantization...")
        converter.optimizations = [tf.lite.Optimize.DEFAULT]
        converter.target_spec.supported_types = [tf.float16]
        
    elif quantization == 'int8':
        print("Applying int8 quantization...")
        if data_path is None:
            raise ValueError("Data path required for int8 quantization")
        
        converter.optimizations = [tf.lite.Optimize.DEFAULT]
        converter.representative_dataset = lambda: representative_dataset_gen(data_path)
        converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
        converter.inference_input_type = tf.int8
        converter.inference_output_type = tf.int8
    
    # Convert
    print("Converting to TFLite...")
    tflite_model = converter.convert()
    
    # Save model
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, 'wb') as f:
        f.write(tflite_model)
    
    # Get model size
    model_size = len(tflite_model)
    keras_size = os.path.getsize(model_path)
    
    print(f"\nSaved TFLite model to: {output_path}")
    print(f"Original size: {keras_size / 1024:.2f} KB")
    print(f"TFLite size: {model_size / 1024:.2f} KB")
    print(f"Compression ratio: {keras_size / model_size:.2f}x")
    
    return {
        'original_size_kb': keras_size / 1024,
        'tflite_size_kb': model_size / 1024,
        'compression_ratio': keras_size / model_size,
        'quantization': quantization
    }


def benchmark_tflite_model(
    tflite_path: str,
    input_shape: tuple,
    num_runs: int = 100,
    warmup_runs: int = 10
) -> dict:
    """
    Benchmark TFLite model inference time.
    
    Args:
        tflite_path: Path to TFLite model
        input_shape: Input shape for test data
        num_runs: Number of benchmark runs
        warmup_runs: Number of warmup runs
        
    Returns:
        Dictionary with benchmark results
    """
    print(f"\nBenchmarking: {tflite_path}")
    
    # Load interpreter
    interpreter = tf.lite.Interpreter(model_path=tflite_path)
    interpreter.allocate_tensors()
    
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    
    # Get input dtype
    input_dtype = input_details[0]['dtype']
    
    # Generate test input
    test_input = np.random.randn(1, *input_shape).astype(np.float32)
    
    # Normalize
    test_input = (test_input - test_input.mean()) / (test_input.std() + 1e-8)
    
    # Convert to int8 if needed
    if input_dtype == np.int8:
        scale = input_details[0]['quantization'][0]
        zero_point = input_details[0]['quantization'][1]
        test_input = (test_input / scale + zero_point).astype(np.int8)
    
    # Warmup
    print(f"Warming up ({warmup_runs} runs)...")
    for _ in range(warmup_runs):
        interpreter.set_tensor(input_details[0]['index'], test_input)
        interpreter.invoke()
    
    # Benchmark
    print(f"Benchmarking ({num_runs} runs)...")
    times = []
    for _ in range(num_runs):
        start = time.perf_counter()
        interpreter.set_tensor(input_details[0]['index'], test_input)
        interpreter.invoke()
        times.append(time.perf_counter() - start)
    
    times = np.array(times) * 1000  # Convert to ms
    
    results = {
        'mean_ms': float(np.mean(times)),
        'std_ms': float(np.std(times)),
        'min_ms': float(np.min(times)),
        'max_ms': float(np.max(times)),
        'p50_ms': float(np.percentile(times, 50)),
        'p95_ms': float(np.percentile(times, 95)),
        'p99_ms': float(np.percentile(times, 99))
    }
    
    print(f"\nBenchmark Results:")
    print(f"  Mean:  {results['mean_ms']:.2f} ms")
    print(f"  Std:   {results['std_ms']:.2f} ms")
    print(f"  Min:   {results['min_ms']:.2f} ms")
    print(f"  Max:   {results['max_ms']:.2f} ms")
    print(f"  P50:   {results['p50_ms']:.2f} ms")
    print(f"  P95:   {results['p95_ms']:.2f} ms")
    print(f"  P99:   {results['p99_ms']:.2f} ms")
    
    # Check target
    target_ms = 50
    if results['p95_ms'] < target_ms:
        print(f"\n[PASS] P95 latency ({results['p95_ms']:.2f}ms) < target ({target_ms}ms)")
    else:
        print(f"\n[FAIL] P95 latency ({results['p95_ms']:.2f}ms) >= target ({target_ms}ms)")
    
    return results


def main():
    """Entry point."""
    parser = argparse.ArgumentParser(description='Export WakeWord model to TFLite')
    parser.add_argument('--model', type=str, required=True,
                        help='Path to Keras model')
    parser.add_argument('--output', type=str, default=None,
                        help='Output path for TFLite model')
    parser.add_argument('--quantization', type=str, default='none',
                        choices=['none', 'float16', 'int8'],
                        help='Quantization type')
    parser.add_argument('--config', type=str, default='configs/config.yaml',
                        help='Path to config file')
    parser.add_argument('--benchmark', action='store_true',
                        help='Run benchmark after conversion')
    parser.add_argument('--num_runs', type=int, default=100,
                        help='Number of benchmark runs')
    
    args = parser.parse_args()
    
    # Change to project root
    os.chdir(project_root)
    
    # Load config
    config = load_config(args.config)
    
    # Set output path
    if args.output is None:
        model_name = Path(args.model).stem
        quant_suffix = f"_{args.quantization}" if args.quantization != 'none' else ""
        args.output = f"{config['paths']['tflite']}/{model_name}{quant_suffix}.tflite"
    
    print("=" * 50)
    print("WakeWord TFLite Export")
    print("=" * 50)
    
    # Convert
    data_path = config['paths']['data_processed'] if args.quantization == 'int8' else None
    conversion_info = convert_to_tflite(
        args.model,
        args.output,
        args.quantization,
        data_path
    )
    
    # Benchmark
    if args.benchmark:
        input_shape = tuple(config['model']['input_shape'])
        benchmark_results = benchmark_tflite_model(
            args.output,
            input_shape,
            args.num_runs
        )
        
        # Save results
        results = {**conversion_info, 'benchmark': benchmark_results}
        results_path = Path(args.output).with_suffix('.json')
        with open(results_path, 'w') as f:
            json.dump(results, f, indent=2)
        print(f"\nResults saved to: {results_path}")
    
    print("\nExport complete!")


if __name__ == "__main__":
    main()
