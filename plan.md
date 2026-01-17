# WakeWord - Audio Keyword Spotting (TinyML)

> A real-time voice-activated trigger system that listens for custom wake words and triggers actions.

---

## Project Overview

**Objective**: Build an end-to-end audio keyword spotting system that can detect custom wake words (e.g., "Hey Jarvis", "Lumos") in real-time, optimized for edge deployment.

**Core Challenge**: Audio processing pipeline with spectrograms, real-time buffering, and low-latency inference (<50ms for 1-second audio chunks on a 1.5GHz CPU).

---

## Phase 1: Environment Setup & Data Acquisition

### 1.1 Project Structure
```
WakeWord/
├── data/
│   ├── raw/                    # Original audio files
│   ├── processed/              # MFCC/Spectrogram features
│   └── splits/                 # train/val/test splits
├── notebooks/
│   ├── 01_data_exploration.ipynb
│   ├── 02_preprocessing.ipynb
│   ├── 03_model_training.ipynb
│   └── 04_optimization.ipynb
├── src/
│   ├── data/
│   │   ├── __init__.py
│   │   ├── downloader.py       # Dataset download utilities
│   │   └── preprocessor.py     # Audio → MFCC/Spectrogram
│   ├── model/
│   │   ├── __init__.py
│   │   ├── cnn.py              # Standard CNN architecture
│   │   └── ds_cnn.py           # Depthwise Separable CNN
│   ├── inference/
│   │   ├── __init__.py
│   │   ├── realtime.py         # Real-time audio capture
│   │   └── tflite_runner.py    # TFLite inference engine
│   └── utils/
│       ├── __init__.py
│       ├── audio.py            # Audio utilities
│       └── visualization.py    # Plotting functions
├── models/
│   ├── checkpoints/            # Training checkpoints
│   ├── exported/               # Full models
│   └── tflite/                 # Quantized TFLite models
├── configs/
│   └── config.yaml             # Hyperparameters & settings
├── scripts/
│   ├── download_data.py
│   ├── train.py
│   ├── export.py
│   ├── benchmark.py
│   └── demo.py
├── tests/
│   ├── test_preprocessing.py
│   ├── test_model.py
│   └── test_inference.py
├── requirements.txt
├── README.md
└── plan.md
```

### 1.2 Dataset: Google Speech Commands v2
- **Source**: [Google Speech Commands Dataset](https://www.tensorflow.org/datasets/catalog/speech_commands)
- **Size**: ~105,000 one-second audio clips
- **Classes**: 35 words including "yes", "no", "up", "down", "left", "right", "on", "off", "stop", "go", etc.
- **Format**: 16kHz WAV files
- **Strategy**: Use existing keywords first, then add custom wake word support

### 1.3 Dependencies
```txt
# Core
numpy>=1.21.0
scipy>=1.7.0
librosa>=0.9.0
soundfile>=0.10.0

# Deep Learning
tensorflow>=2.10.0
tensorflow-model-optimization>=0.7.0

# Real-time Audio
pyaudio>=0.2.12
sounddevice>=0.4.6

# Visualization
matplotlib>=3.5.0
seaborn>=0.12.0

# Utilities
pyyaml>=6.0
tqdm>=4.64.0
kaggle>=1.5.0
python-dotenv>=0.21.0

# Testing
pytest>=7.0.0
```

---

## Phase 2: Audio Preprocessing Pipeline

### 2.1 Audio Feature Extraction

#### MFCC (Mel-Frequency Cepstral Coefficients)
- **Why**: Compact representation mimicking human auditory perception
- **Parameters**:
  - Sample Rate: 16kHz
  - Frame Length: 25ms (400 samples)
  - Frame Stride: 10ms (160 samples)
  - Num MFCCs: 40
  - Num Mel Filters: 80
  - FFT Size: 512

```python
# Pseudo-code for MFCC extraction
def extract_mfcc(audio, sr=16000):
    mfccs = librosa.feature.mfcc(
        y=audio, 
        sr=sr,
        n_mfcc=40,
        n_fft=512,
        hop_length=160,
        n_mels=80
    )
    return mfccs.T  # Shape: (time_steps, n_mfcc)
```

#### Mel Spectrogram (Alternative)
- **Why**: Richer frequency information, better for complex patterns
- **Parameters**:
  - Same as MFCC but output raw Mel energies
  - Apply log compression

### 2.2 Data Augmentation
- **Time Shift**: Random shift ±100ms
- **Speed Perturbation**: 0.9x - 1.1x speed
- **Background Noise**: Mix with ambient sounds at SNR 5-20dB
- **Volume Perturbation**: ±3dB gain
- **Pitch Shift**: ±2 semitones

### 2.3 Preprocessing Pipeline Steps
1. Load WAV file (16kHz mono)
2. Normalize amplitude to [-1, 1]
3. Pad/trim to exactly 1 second
4. Apply augmentation (training only)
5. Extract MFCC features
6. Normalize features (per-channel mean/std)
7. Save as `.npy` files

---

## Phase 3: Model Architecture

### 3.1 CNN Baseline
```
Input: (98, 40, 1) - 98 time frames, 40 MFCCs

Conv2D(64, 3x3, ReLU) → BatchNorm → MaxPool(2x2)
Conv2D(64, 3x3, ReLU) → BatchNorm → MaxPool(2x2)
Conv2D(128, 3x3, ReLU) → BatchNorm → MaxPool(2x2)
Conv2D(128, 3x3, ReLU) → BatchNorm
GlobalAveragePooling2D
Dense(128, ReLU) → Dropout(0.5)
Dense(num_classes, Softmax)

Estimated Parameters: ~250K
```

### 3.2 DS-CNN (Depthwise Separable CNN) - Optimized
```
Input: (98, 40, 1)

Conv2D(64, 3x3, ReLU) → BatchNorm
DepthwiseSeparableConv(64) → BatchNorm → MaxPool
DepthwiseSeparableConv(64) → BatchNorm → MaxPool
DepthwiseSeparableConv(128) → BatchNorm → MaxPool
DepthwiseSeparableConv(128) → BatchNorm
GlobalAveragePooling2D
Dense(128, ReLU) → Dropout(0.3)
Dense(num_classes, Softmax)

Estimated Parameters: ~80K
```

### 3.3 Training Configuration
- **Optimizer**: Adam with learning rate 0.001
- **LR Schedule**: ReduceLROnPlateau or Cosine Annealing
- **Batch Size**: 64-128
- **Epochs**: 50-100 with early stopping
- **Loss**: Categorical Cross-Entropy
- **Metrics**: Accuracy, Precision, Recall, F1

---

## Phase 4: Training on Kaggle

### 4.1 Kaggle Notebook Setup
1. Create new Kaggle notebook
2. Enable GPU accelerator (P100)
3. Mount Google Speech Commands dataset
4. Upload preprocessing & model scripts
5. Train with full dataset

### 4.2 Training Script Workflow
```bash
# Local development
python scripts/train.py --config configs/config.yaml --debug

# Kaggle execution
python scripts/train.py --config configs/config.yaml --epochs 100 --batch_size 128
```

### 4.3 Model Checkpointing
- Save best model by validation accuracy
- Save every 10 epochs
- Export training logs for visualization

---

## Phase 5: Model Optimization & Export

### 5.1 TensorFlow Lite Conversion
```python
# Standard conversion
converter = tf.lite.TFLiteConverter.from_keras_model(model)
tflite_model = converter.convert()

# Quantized conversion (INT8)
converter.optimizations = [tf.lite.Optimize.DEFAULT]
converter.representative_dataset = representative_dataset_gen
converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
converter.inference_input_type = tf.int8
converter.inference_output_type = tf.int8
quantized_model = converter.convert()
```

### 5.2 Optimization Targets
| Metric | Target | Method |
|--------|--------|--------|
| Model Size | <500KB | INT8 Quantization |
| Inference Time | <50ms | DS-CNN + Quantization |
| Accuracy Drop | <2% | Quantization-Aware Training |

### 5.3 Profiling & Benchmarking
```python
# Benchmark script
def benchmark_model(tflite_path, num_runs=100):
    interpreter = tf.lite.Interpreter(model_path=tflite_path)
    interpreter.allocate_tensors()
    
    times = []
    for _ in range(num_runs):
        start = time.perf_counter()
        interpreter.invoke()
        times.append(time.perf_counter() - start)
    
    print(f"Mean: {np.mean(times)*1000:.2f}ms")
    print(f"Std: {np.std(times)*1000:.2f}ms")
    print(f"P95: {np.percentile(times, 95)*1000:.2f}ms")
```

---

## Phase 6: Real-Time Inference Application

### 6.1 Audio Capture Pipeline
```python
class AudioBuffer:
    def __init__(self, sample_rate=16000, chunk_duration=1.0):
        self.sr = sample_rate
        self.chunk_size = int(sample_rate * chunk_duration)
        self.buffer = np.zeros(self.chunk_size)
        self.stream = pyaudio.PyAudio().open(
            rate=sample_rate,
            channels=1,
            format=pyaudio.paFloat32,
            input=True,
            frames_per_buffer=1024,
            stream_callback=self._callback
        )
    
    def _callback(self, data, frame_count, time_info, status):
        audio = np.frombuffer(data, dtype=np.float32)
        self.buffer = np.roll(self.buffer, -len(audio))
        self.buffer[-len(audio):] = audio
        return (None, pyaudio.paContinue)
```

### 6.2 Inference Loop
```python
def run_detection():
    buffer = AudioBuffer()
    interpreter = load_tflite_model()
    
    while True:
        audio = buffer.get_chunk()
        features = extract_mfcc(audio)
        
        start = time.perf_counter()
        prediction = interpreter.predict(features)
        latency = (time.perf_counter() - start) * 1000
        
        if prediction.max() > THRESHOLD:
            detected_word = LABELS[prediction.argmax()]
            print(f"Detected: {detected_word} ({latency:.1f}ms)")
            trigger_action(detected_word)
        
        time.sleep(0.1)  # 100ms sliding window
```

### 6.3 Action Triggers
- Print detection to console
- Play confirmation sound
- Send HTTP webhook
- Execute custom callback

---

## Phase 7: Testing & Validation

### 7.1 Unit Tests
- `test_preprocessing.py`: MFCC extraction, normalization
- `test_model.py`: Model architecture, forward pass
- `test_inference.py`: TFLite loading, prediction

### 7.2 Integration Tests
- End-to-end pipeline from WAV → prediction
- Real-time buffer accuracy
- Latency under load

### 7.3 Performance Benchmarks
```bash
# Run benchmarks
python scripts/benchmark.py --model models/tflite/ds_cnn_int8.tflite --runs 1000
```

**Target Metrics**:
- Inference latency: <50ms (P95)
- Accuracy: >95% on test set
- False positive rate: <5%
- Model size: <500KB

---

## Phase 8: Documentation & Demo

### 8.1 README Structure
1. Project overview
2. Quick start guide
3. Installation instructions
4. Usage examples
5. Model architecture details
6. Performance benchmarks
7. Custom wake word training
8. Troubleshooting

### 8.2 Demo Application
```bash
# Run live demo
python scripts/demo.py --model models/tflite/ds_cnn_int8.tflite --threshold 0.8
```

Features:
- Live microphone visualization
- Real-time waveform display
- Detection confidence meter
- Latency statistics

---

## Timeline & Milestones

| Phase | Duration | Deliverables |
|-------|----------|--------------|
| Phase 1 | Day 1-2 | Project setup, data downloaded |
| Phase 2 | Day 3-4 | Preprocessing pipeline complete |
| Phase 3 | Day 5-6 | Model architectures implemented |
| Phase 4 | Day 7-9 | Training complete on Kaggle |
| Phase 5 | Day 10-11 | TFLite optimized, benchmarked |
| Phase 6 | Day 12-14 | Real-time app functional |
| Phase 7 | Day 15 | Tests passing, validated |
| Phase 8 | Day 16 | Documentation & demo ready |

**Total Estimated Duration**: 16 days

---

## Success Criteria

- [ ] Model trained with >95% test accuracy
- [ ] TFLite model <500KB
- [ ] Inference <50ms on 1.5GHz CPU
- [ ] Real-time detection working with <200ms end-to-end latency
- [ ] False positive rate <5%
- [ ] Clean, documented codebase
- [ ] Comprehensive test coverage

---

## Future Enhancements

1. **Custom Wake Word Training**: Few-shot learning for custom words
2. **Noise Robustness**: Train with diverse background noises
3. **Multi-Language Support**: Extend to other languages
4. **Microcontroller Deployment**: TensorFlow Lite Micro for ESP32/Arduino
5. **Streaming Inference**: Process audio in smaller chunks
6. **Wake Word + Command**: Two-stage detection system

---

## Resources

- [Google Speech Commands Dataset](https://www.tensorflow.org/datasets/catalog/speech_commands)
- [TensorFlow Lite for Microcontrollers](https://www.tensorflow.org/lite/microcontrollers)
- [Librosa Documentation](https://librosa.org/doc/)
- [PyAudio Tutorial](https://people.csail.mit.edu/hubert/pyaudio/)
- [Keyword Spotting Paper](https://arxiv.org/abs/1711.07128)
