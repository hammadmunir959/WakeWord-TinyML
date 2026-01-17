"""
Benchmark Script for WakeWord
Comprehensive benchmarking of model performance.
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

import yaml
import numpy as np
import tensorflow as tf
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report
)

from src.inference.tflite_runner import TFLiteRunner
from src.data.dataloader import WakeWordDataset
from src.utils.audio import extract_mfcc


def load_config(config_path: str) -> dict:
    """Load configuration from YAML file."""
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


def benchmark_accuracy(runner: TFLiteRunner, dataset: WakeWordDataset) -> dict:
    """Benchmark model accuracy on test set."""
    print("\nBenchmarking accuracy...")
    
    # Load test data
    features, labels = dataset.load_split('test')
    
    # Add channel dimension if needed
    if len(features.shape) == 3:
        features = features[..., np.newaxis]
    
    # Normalize
    mean = np.mean(features, axis=(1, 2, 3), keepdims=True)
    std = np.std(features, axis=(1, 2, 3), keepdims=True) + 1e-8
    features = (features - mean) / std
    
    # Run inference
    predictions = []
    latencies = []
    
    for i in range(len(features)):
        pred, latency = runner.predict(features[i])
        predictions.append(np.argmax(pred))
        latencies.append(latency)
        
        if (i + 1) % 1000 == 0:
            print(f"  Processed {i + 1}/{len(features)}")
    
    predictions = np.array(predictions)
    latencies = np.array(latencies)
    
    # Calculate metrics
    accuracy = accuracy_score(labels, predictions)
    precision = precision_score(labels, predictions, average='weighted', zero_division=0)
    recall = recall_score(labels, predictions, average='weighted', zero_division=0)
    f1 = f1_score(labels, predictions, average='weighted', zero_division=0)
    
    # Per-class metrics
    class_names = dataset.classes
    report = classification_report(labels, predictions, target_names=class_names, output_dict=True)
    
    results = {
        'accuracy': float(accuracy),
        'precision': float(precision),
        'recall': float(recall),
        'f1_score': float(f1),
        'latency': {
            'mean_ms': float(np.mean(latencies)),
            'std_ms': float(np.std(latencies)),
            'min_ms': float(np.min(latencies)),
            'max_ms': float(np.max(latencies)),
            'p50_ms': float(np.percentile(latencies, 50)),
            'p95_ms': float(np.percentile(latencies, 95)),
            'p99_ms': float(np.percentile(latencies, 99))
        },
        'per_class': {}
    }
    
    for class_name in class_names:
        if class_name in report:
            results['per_class'][class_name] = {
                'precision': report[class_name]['precision'],
                'recall': report[class_name]['recall'],
                'f1_score': report[class_name]['f1-score'],
                'support': report[class_name]['support']
            }
    
    # Print results
    print(f"\nAccuracy Results:")
    print(f"  Accuracy:  {accuracy:.4f}")
    print(f"  Precision: {precision:.4f}")
    print(f"  Recall:    {recall:.4f}")
    print(f"  F1 Score:  {f1:.4f}")
    
    return results


def benchmark_latency(
    runner: TFLiteRunner,
    input_shape: tuple,
    num_runs: int = 1000,
    warmup_runs: int = 100
) -> dict:
    """Benchmark inference latency."""
    print(f"\nBenchmarking latency ({num_runs} runs)...")
    
    # Generate random input
    test_input = np.random.randn(1, *input_shape).astype(np.float32)
    test_input = (test_input - test_input.mean()) / (test_input.std() + 1e-8)
    
    # Warmup
    print(f"  Warming up ({warmup_runs} runs)...")
    for _ in range(warmup_runs):
        runner.predict(test_input[0])
    
    # Benchmark
    latencies = []
    for _ in range(num_runs):
        _, latency = runner.predict(test_input[0])
        latencies.append(latency)
    
    latencies = np.array(latencies)
    
    results = {
        'num_runs': num_runs,
        'mean_ms': float(np.mean(latencies)),
        'std_ms': float(np.std(latencies)),
        'min_ms': float(np.min(latencies)),
        'max_ms': float(np.max(latencies)),
        'p50_ms': float(np.percentile(latencies, 50)),
        'p95_ms': float(np.percentile(latencies, 95)),
        'p99_ms': float(np.percentile(latencies, 99))
    }
    
    print(f"\nLatency Results:")
    print(f"  Mean:  {results['mean_ms']:.2f} ms")
    print(f"  Std:   {results['std_ms']:.2f} ms")
    print(f"  P50:   {results['p50_ms']:.2f} ms")
    print(f"  P95:   {results['p95_ms']:.2f} ms")
    print(f"  P99:   {results['p99_ms']:.2f} ms")
    
    # Check target
    target_ms = 50
    if results['p95_ms'] < target_ms:
        print(f"\n  [PASS] P95 latency < {target_ms}ms target")
    else:
        print(f"\n  [FAIL] P95 latency >= {target_ms}ms target")
    
    return results


def benchmark_throughput(
    runner: TFLiteRunner,
    input_shape: tuple,
    duration_sec: float = 5.0
) -> dict:
    """Benchmark inference throughput."""
    print(f"\nBenchmarking throughput ({duration_sec}s)...")
    
    # Generate random input
    test_input = np.random.randn(1, *input_shape).astype(np.float32)
    test_input = (test_input - test_input.mean()) / (test_input.std() + 1e-8)
    
    # Run for specified duration
    count = 0
    start_time = time.perf_counter()
    end_time = start_time + duration_sec
    
    while time.perf_counter() < end_time:
        runner.predict(test_input[0])
        count += 1
    
    actual_duration = time.perf_counter() - start_time
    throughput = count / actual_duration
    
    results = {
        'duration_sec': actual_duration,
        'total_inferences': count,
        'throughput_hz': throughput,
        'avg_latency_ms': 1000 / throughput
    }
    
    print(f"\nThroughput Results:")
    print(f"  Total inferences: {count}")
    print(f"  Throughput: {throughput:.1f} Hz")
    print(f"  Avg latency: {1000/throughput:.2f} ms")
    
    return results


def get_model_size(model_path: str) -> dict:
    """Get model file size."""
    size_bytes = os.path.getsize(model_path)
    
    return {
        'size_bytes': size_bytes,
        'size_kb': size_bytes / 1024,
        'size_mb': size_bytes / 1024 / 1024
    }


def main():
    """Entry point."""
    parser = argparse.ArgumentParser(description='Benchmark WakeWord model')
    parser.add_argument('--model', type=str, required=True,
                        help='Path to TFLite model')
    parser.add_argument('--config', type=str, default='configs/config.yaml',
                        help='Path to config file')
    parser.add_argument('--runs', type=int, default=1000,
                        help='Number of benchmark runs')
    parser.add_argument('--skip_accuracy', action='store_true',
                        help='Skip accuracy benchmark')
    parser.add_argument('--output', type=str, default=None,
                        help='Output JSON path')
    
    args = parser.parse_args()
    
    # Change to project root
    os.chdir(project_root)
    
    # Load config
    config = load_config(args.config)
    
    print("=" * 50)
    print("WakeWord Model Benchmark")
    print("=" * 50)
    
    # Load model
    print(f"\nLoading model: {args.model}")
    runner = TFLiteRunner(args.model, config['dataset']['classes'])
    print(f"Model info: {runner.get_model_info()}")
    
    # Get model size
    size_info = get_model_size(args.model)
    print(f"Model size: {size_info['size_kb']:.2f} KB")
    
    # Get input shape
    input_shape = tuple(config['model']['input_shape'])
    
    # Initialize results
    results = {
        'model_path': args.model,
        'model_info': runner.get_model_info(),
        'model_size': size_info
    }
    
    # Benchmark latency
    results['latency'] = benchmark_latency(runner, input_shape, args.runs)
    
    # Benchmark throughput
    results['throughput'] = benchmark_throughput(runner, input_shape)
    
    # Benchmark accuracy
    if not args.skip_accuracy:
        try:
            data_path = config['paths']['data_processed']
            dataset = WakeWordDataset(data_path)
            results['accuracy'] = benchmark_accuracy(runner, dataset)
        except Exception as e:
            print(f"\nSkipping accuracy benchmark: {e}")
    
    # Save results
    if args.output:
        output_path = Path(args.output)
    else:
        output_path = Path(args.model).with_suffix('.benchmark.json')
    
    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\nResults saved to: {output_path}")
    print("\nBenchmark complete!")


if __name__ == "__main__":
    main()
