"""
Demo Application for WakeWord
Real-time keyword detection with visualization.
"""

import os
import sys
import argparse
from pathlib import Path
import time

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import yaml
import numpy as np

from src.inference.tflite_runner import TFLiteRunner
from src.inference.realtime import AudioBuffer, RealtimeDetector
from src.utils.audio import extract_mfcc


def load_config(config_path: str) -> dict:
    """Load configuration from YAML file."""
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


def create_preprocessor(config: dict):
    """Create MFCC preprocessor function."""
    audio_config = config['audio']
    
    def preprocessor(audio, sr):
        return extract_mfcc(
            audio,
            sr=sr,
            n_mfcc=audio_config['n_mfcc'],
            n_fft=audio_config['n_fft'],
            hop_length=audio_config['hop_length'],
            n_mels=audio_config['n_mels'],
            fmin=audio_config['fmin'],
            fmax=audio_config['fmax']
        )
    
    return preprocessor


def print_detection(label: str, confidence: float, latency: float):
    """Print detection event."""
    print(f"\n{'=' * 50}")
    print(f"  DETECTED: {label.upper()}")
    print(f"  Confidence: {confidence:.2%}")
    print(f"  Latency: {latency:.1f}ms")
    print(f"{'=' * 50}\n")


def print_audio_level(energy: float, bar_width: int = 40):
    """Print audio level bar."""
    level = min(energy * 10, 1.0)  # Scale energy
    filled = int(level * bar_width)
    bar = '|' + '#' * filled + '-' * (bar_width - filled) + '|'
    print(f"\rAudio: {bar} {energy:.4f}", end='', flush=True)


def run_demo(args):
    """Run the demo application."""
    # Load config
    config = load_config(args.config)
    
    # Load model
    print(f"\nLoading model: {args.model}")
    runner = TFLiteRunner(args.model, config['dataset']['classes'])
    print(f"Model info: {runner.get_model_info()}")
    
    # Create preprocessor
    preprocessor = create_preprocessor(config)
    
    # List available devices
    buffer = AudioBuffer()
    devices = buffer.list_devices()
    print("\nAvailable audio devices:")
    for dev in devices:
        print(f"  [{dev['index']}] {dev['name']}")
    
    # Create detector
    detector = RealtimeDetector(
        runner=runner,
        preprocessor=preprocessor,
        sample_rate=config['audio']['sample_rate'],
        chunk_duration=config['inference']['chunk_duration'],
        overlap=config['inference']['overlap'],
        threshold=args.threshold,
        smoothing_window=config['inference']['smoothing_window']
    )
    
    # Set callbacks
    detector.on_detection = print_detection
    if args.show_level:
        detector.on_audio_level = print_audio_level
    
    # Start detection
    print(f"\nStarting real-time detection...")
    print(f"Threshold: {args.threshold}")
    print(f"Listening for: {', '.join(config['dataset']['classes'])}")
    print("\nPress Ctrl+C to stop\n")
    
    try:
        detector.start(device_index=args.device)
        
        # Keep running until interrupted
        while True:
            time.sleep(0.1)
            
    except KeyboardInterrupt:
        print("\n\nStopping...")
    finally:
        detector.stop()
    
    print("Demo ended.")


def run_file_test(args):
    """Test detection on a single audio file."""
    from src.utils.audio import load_audio
    
    # Load config
    config = load_config(args.config)
    
    # Load model
    print(f"\nLoading model: {args.model}")
    runner = TFLiteRunner(args.model, config['dataset']['classes'])
    
    # Create preprocessor
    preprocessor = create_preprocessor(config)
    
    # Load audio file
    print(f"\nLoading audio: {args.file}")
    audio, sr = load_audio(
        args.file,
        sr=config['audio']['sample_rate'],
        duration=config['audio']['duration']
    )
    
    # Extract features
    features = preprocessor(audio, sr)
    print(f"Features shape: {features.shape}")
    
    # Run inference
    label, confidence, latency = runner.predict_class(features, args.threshold)
    
    print(f"\nResults:")
    print(f"  Prediction: {label or 'No detection'}")
    print(f"  Confidence: {confidence:.2%}")
    print(f"  Latency: {latency:.1f}ms")
    
    # Show top predictions
    predictions, _ = runner.predict(features)
    top_indices = np.argsort(predictions[0])[::-1][:5]
    
    print(f"\nTop 5 predictions:")
    for idx in top_indices:
        label = config['dataset']['classes'][idx]
        conf = predictions[0][idx]
        print(f"  {label}: {conf:.2%}")


def main():
    """Entry point."""
    parser = argparse.ArgumentParser(description='WakeWord Demo')
    parser.add_argument('--model', type=str, required=True,
                        help='Path to TFLite model')
    parser.add_argument('--config', type=str, default='configs/config.yaml',
                        help='Path to config file')
    parser.add_argument('--threshold', type=float, default=0.7,
                        help='Detection threshold')
    parser.add_argument('--device', type=int, default=None,
                        help='Audio device index')
    parser.add_argument('--show_level', action='store_true',
                        help='Show audio level meter')
    parser.add_argument('--file', type=str, default=None,
                        help='Test on audio file instead of live')
    
    args = parser.parse_args()
    
    # Change to project root
    os.chdir(project_root)
    
    print("=" * 50)
    print("WakeWord Demo")
    print("=" * 50)
    
    if args.file:
        run_file_test(args)
    else:
        run_demo(args)


if __name__ == "__main__":
    main()
