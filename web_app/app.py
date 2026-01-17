import os
import sys
import json
import numpy as np
import librosa
import time
from fastapi import FastAPI, WebSocket, WebSocketDisconnect, Request
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from pathlib import Path

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.append(str(PROJECT_ROOT))

from src.inference.tflite_runner import TFLiteRunner

app = FastAPI(title="WakeWord Real-time Demo")

# Model setup
MODEL_PATH = PROJECT_ROOT / "models" / "wakeword_int8.tflite"
METADATA_PATH = PROJECT_ROOT / "models" / "kaggle_output" / "metadata.json"

# Fallback if specific metadata not found
if not METADATA_PATH.exists():
    classes = ["yes", "no", "up", "down", "left", "right", "on", "off", "stop", "go", "_unknown_"]
else:
    with open(METADATA_PATH, "r") as f:
        metadata = json.load(f)
        classes = metadata["classes"]

# Initialize runner
runner = TFLiteRunner(str(MODEL_PATH), labels=classes)

# Static and Templates
app.mount("/static", StaticFiles(directory="web_app/static"), name="static")
templates = Jinja2Templates(directory="web_app/templates")

@app.get("/", response_class=HTMLResponse)
async def get_index(request: Request):
    return templates.TemplateResponse("index.html", {"request": request, "classes": classes})

@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    
    # Audio buffer: 1 second at 16kHz
    buffer_size = 16000
    audio_buffer = np.zeros(buffer_size, dtype=np.float32)
    
    try:
        while True:
            # Receive binary data (Float32 PCM)
            data = await websocket.receive_bytes()
            chunk = np.frombuffer(data, dtype=np.float32)
            
            # Update rolling buffer
            audio_buffer = np.roll(audio_buffer, -len(chunk))
            audio_buffer[-len(chunk):] = chunk
            
            # Simple energy threshold to skip silence
            energy = np.sqrt(np.mean(audio_buffer**2))
            if energy < 0.005:
                # Still send heartbeat for visualization
                await websocket.send_json({
                    "label": None,
                    "confidence": 0.0,
                    "energy": float(energy)
                })
                continue
                
            # MFCC Extraction (same as training)
            mfccs = librosa.feature.mfcc(
                y=audio_buffer, 
                sr=16000, 
                n_mfcc=40, 
                n_fft=512, 
                hop_length=160, 
                n_mels=80
            ).T
            
            # Prediction
            label, confidence, latency = runner.predict_class(mfccs, threshold=0.7)
            
            if label and label != "_unknown_":
                await websocket.send_json({
                    "label": label,
                    "confidence": float(confidence),
                    "latency": float(latency),
                    "energy": float(energy)
                })
            else:
                await websocket.send_json({
                    "label": None,
                    "confidence": 0.0,
                    "latency": float(latency),
                    "energy": float(energy)
                })
                
    except WebSocketDisconnect:
        pass
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
