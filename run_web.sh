#!/bin/bash
# WakeWord AI Web Demo Runner

echo "🚀 Starting WakeWord AI Web Demo..."

# Check for FastAPI and Uvicorn
if ! python3 -c "import fastapi, uvicorn" &> /dev/null; then
    echo "📦 Installing web dependencies..."
    pip install fastapi uvicorn jinja2 websockets librosa tensorflow-cpu python-multipart --break-system-packages
fi

# Run the app
export PYTHONPATH=$PYTHONPATH:.
python3 web_app/app.py
