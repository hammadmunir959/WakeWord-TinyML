#!/usr/bin/env python3
"""Simple TFLite benchmark - no TensorFlow required"""

import numpy as np
import time
import json
import sys

# Try tflite_runtime first, fall back to tensorflow
try:
    from tflite_runtime.interpreter import Interpreter
except ImportError:
    try:
        from tensorflow.lite.python.interpreter import Interpreter
    except ImportError:
        print("ERROR: Install tflite-runtime: pip install tflite-runtime")
        sys.exit(1)

def benchmark_model(model_path, num_runs=100, warmup=10):
    """Benchmark TFLite model inference latency."""
    print(f"\n{'='*50}")
    print(f"Benchmarking: {model_path}")
    print(f"{'='*50}")
    
    # Load model
    interpreter = Interpreter(model_path=model_path)
    interpreter.allocate_tensors()
    
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    
    input_shape = input_details[0]['shape']
    input_dtype = input_details[0]['dtype']
    
    print(f"Input shape: {input_shape}")
    print(f"Input dtype: {input_dtype}")
    
    # Create random input
    if input_dtype == np.float32:
        test_input = np.random.randn(*input_shape).astype(np.float32)
    else:
        test_input = np.random.randint(-128, 127, size=input_shape).astype(input_dtype)
    
    # Warmup
    print(f"\nWarmup ({warmup} runs)...")
    for _ in range(warmup):
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
    
    # Get output
    output = interpreter.get_tensor(output_details[0]['index'])
    
    # Results
    print(f"\n--- Results ---")
    print(f"Mean latency:   {np.mean(times):.2f} ms")
    print(f"Std latency:    {np.std(times):.2f} ms")
    print(f"Min latency:    {np.min(times):.2f} ms")
    print(f"Max latency:    {np.max(times):.2f} ms")
    print(f"P50 latency:    {np.percentile(times, 50):.2f} ms")
    print(f"P95 latency:    {np.percentile(times, 95):.2f} ms")
    print(f"P99 latency:    {np.percentile(times, 99):.2f} ms")
    print(f"Throughput:     {1000 / np.mean(times):.1f} inferences/sec")
    
    return {
        'model': model_path,
        'mean_ms': float(np.mean(times)),
        'std_ms': float(np.std(times)),
        'p50_ms': float(np.percentile(times, 50)),
        'p95_ms': float(np.percentile(times, 95)),
        'p99_ms': float(np.percentile(times, 99)),
        'throughput': float(1000 / np.mean(times))
    }

if __name__ == "__main__":
    import os
    
    models_dir = "models"
    results = []
    
    for model_file in ["wakeword_float.tflite", "wakeword_int8.tflite"]:
        model_path = os.path.join(models_dir, model_file)
        if os.path.exists(model_path):
            size_kb = os.path.getsize(model_path) / 1024
            print(f"\nModel size: {size_kb:.2f} KB")
            result = benchmark_model(model_path)
            result['size_kb'] = size_kb
            results.append(result)
    
    # Save results
    with open('models/benchmark_results.json', 'w') as f:
        json.dump(results, f, indent=2)
    
    print("\n" + "="*50)
    print("BENCHMARK SUMMARY")
    print("="*50)
    for r in results:
        print(f"\n{r['model'].split('/')[-1]}:")
        print(f"  Size: {r['size_kb']:.2f} KB")
        print(f"  Latency (P95): {r['p95_ms']:.2f} ms")
        print(f"  Throughput: {r['throughput']:.1f} inf/sec")
