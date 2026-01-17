# WakeWord - Audio Keyword Spotting (TinyML)

A real-time voice-activated trigger system for custom wake words, optimized for edge deployment with low-latency inference.

## Training Results

| Metric | Value |
|--------|-------|
| **Test Accuracy** | 93.82% |
| **Val Accuracy** | 93.32% |
| **Keywords** | yes, no, up, down, left, right, on, off, stop, go + unknown |
| **Architecture** | DS-CNN (57,356 params) |

### Model Sizes

| Model | Size | Quantization |
|-------|------|--------------|
| `wakeword_float.tflite` | 222 KB | Float32 |
| `wakeword_int8.tflite` | 86 KB | INT8 |
| `best_model.keras` | 806 KB | Full Keras |

### Training Details

- **Dataset**: Google Speech Commands v0.02
- **Epochs**: 35 (early stopped)
- **Platform**: Kaggle GPU (Tesla P100)
- **Training Time**: ~7 minutes

## Quick Start

### Option 1: Use Pre-trained Models

```bash
# Models are in models/ directory
ls models/*.tflite
# wakeword_float.tflite (222 KB)
# wakeword_int8.tflite (86 KB)
```

### Option 2: Train from Scratch (Kaggle)

1. Push notebook to Kaggle:
```bash
export KAGGLE_API_TOKEN=your_token
kaggle kernels push -p notebooks/kaggle
```

2. Monitor training:
```bash
kaggle kernels status hammadmunir959/wakeword-training-v2
```

3. Download models:
```bash
kaggle kernels output hammadmunir959/wakeword-training-v2 -p models/
```

## Project Structure

```
WakeWord/
├── configs/config.yaml       # Hyperparameters
├── models/
│   ├── wakeword_float.tflite # Float32 model (222 KB)
│   ├── wakeword_int8.tflite  # INT8 quantized (86 KB)
│   ├── best_model.keras      # Full Keras model
│   └── kaggle_output/        # Training artifacts
├── notebooks/kaggle/         # Kaggle training notebook
├── scripts/
│   ├── download_data.py      # Dataset download
│   ├── preprocess.py         # Audio preprocessing
│   ├── train.py              # Local training
│   ├── export.py             # TFLite export
│   ├── benchmark.py          # Performance benchmarks
│   └── demo.py               # Real-time demo
├── src/
│   ├── data/                 # Data loading
│   ├── model/                # CNN architectures
│   ├── inference/            # TFLite inference
│   └── utils/                # Audio utilities
└── plan.md                   # Development plan
```

## Model Architecture

### DS-CNN (Depthwise Separable CNN)
- Optimized for edge/mobile deployment
- 57,356 parameters
- 4 DS-Conv blocks + GlobalAveragePooling
- BatchNormalization + Dropout (0.3)

```
Input: (101, 40, 1) - MFCC features
├── Conv2D(64) + BN + ReLU
├── DSConv(64) + MaxPool
├── DSConv(64) + MaxPool
├── DSConv(128)
├── DSConv(128)
├── GlobalAveragePooling
├── Dense(128) + ReLU + Dropout
└── Dense(11) + Softmax
```

## Audio Configuration

| Parameter | Value |
|-----------|-------|
| Sample Rate | 16 kHz |
| Duration | 1.0 sec |
| MFCC Coefficients | 40 |
| FFT Size | 512 |
| Hop Length | 160 |
| Mel Bands | 80 |

## Keywords

The model recognizes 11 classes:

| ID | Keyword | ID | Keyword |
|----|---------|----|---------| 
| 0 | yes | 6 | on |
| 1 | no | 7 | off |
| 2 | up | 8 | stop |
| 3 | down | 9 | go |
| 4 | left | 10 | _unknown_ |
| 5 | right | | |

## Real-Time Demo

```bash
# Requires: pyaudio, numpy, tensorflow/tflite-runtime
python scripts/demo.py --model models/wakeword_int8.tflite --threshold 0.7
```

## Training on Kaggle

The notebook `notebooks/kaggle/wakeword-training.ipynb` includes:

1. Dataset download and extraction
2. MFCC feature extraction
3. DS-CNN model creation
4. Training with callbacks (EarlyStopping, ReduceLROnPlateau)
5. TFLite export (Float32 + INT8)
6. Automatic cleanup of dataset before output

## Performance

Trained on Kaggle with GPU acceleration:

```
Epoch 25: val_accuracy = 0.9332 (best)
Test Accuracy: 0.9382
Float model: 221.70 KB
INT8 model: 85.70 KB
```

## Deployment to Render

This project is optimized for deployment on [Render](https://render.com/).

### One-Click Deploy
1. Create a new **Web Service** on Render.
2. Connect your GitHub repository.
3. Render will automatically detect `render.yaml` and configure the service:
   - **Environment**: Python
   - **Build Command**: `pip install -r requirements.txt`
   - **Start Command**: `uvicorn web_app.app:app --host 0.0.0.0 --port $PORT`

### Manual Configuration
- Use `tflite-runtime` instead of `tensorflow` for a smaller footprint (<512MB RAM).
- Set Environment Variable `PYTHONPATH=.`.
