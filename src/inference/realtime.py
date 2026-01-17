"""
Real-time Audio Capture for WakeWord
Handles microphone input and audio buffering for live detection.
"""

import numpy as np
import threading
import time
from typing import Callable, Optional
from collections import deque

try:
    import pyaudio
    PYAUDIO_AVAILABLE = True
except ImportError:
    PYAUDIO_AVAILABLE = False
    print("Warning: PyAudio not available. Real-time capture disabled.")

try:
    import sounddevice as sd
    SOUNDDEVICE_AVAILABLE = True
except ImportError:
    SOUNDDEVICE_AVAILABLE = False


class AudioBuffer:
    """
    Circular audio buffer for real-time capture.
    
    Continuously captures audio from microphone and maintains
    a rolling buffer of the most recent audio.
    """
    
    def __init__(
        self,
        sample_rate: int = 16000,
        chunk_duration: float = 1.0,
        buffer_chunks: int = 3,
        device_index: Optional[int] = None
    ):
        """
        Initialize audio buffer.
        
        Args:
            sample_rate: Audio sample rate in Hz
            chunk_duration: Duration of each chunk in seconds
            buffer_chunks: Number of chunks to keep in buffer
            device_index: Audio device index (None for default)
        """
        self.sample_rate = sample_rate
        self.chunk_size = int(sample_rate * chunk_duration)
        self.buffer_chunks = buffer_chunks
        self.device_index = device_index
        
        # Initialize buffer
        self.buffer = np.zeros(self.chunk_size * buffer_chunks, dtype=np.float32)
        self.buffer_lock = threading.Lock()
        
        # Capture state
        self.is_running = False
        self.capture_thread = None
        
        # PyAudio setup
        self.pa = None
        self.stream = None
    
    def _audio_callback(self, in_data, frame_count, time_info, status):
        """PyAudio callback for audio capture."""
        audio = np.frombuffer(in_data, dtype=np.float32)
        
        with self.buffer_lock:
            # Shift buffer and add new audio
            self.buffer = np.roll(self.buffer, -len(audio))
            self.buffer[-len(audio):] = audio
        
        return (None, pyaudio.paContinue)
    
    def start(self) -> None:
        """Start audio capture."""
        if not PYAUDIO_AVAILABLE:
            raise RuntimeError("PyAudio is not available")
        
        if self.is_running:
            return
        
        self.pa = pyaudio.PyAudio()
        
        # Get device info
        if self.device_index is None:
            device_info = self.pa.get_default_input_device_info()
            self.device_index = device_info['index']
        
        # Open stream
        self.stream = self.pa.open(
            rate=self.sample_rate,
            channels=1,
            format=pyaudio.paFloat32,
            input=True,
            input_device_index=self.device_index,
            frames_per_buffer=1024,
            stream_callback=self._audio_callback
        )
        
        self.stream.start_stream()
        self.is_running = True
        print(f"Audio capture started (device: {self.device_index})")
    
    def stop(self) -> None:
        """Stop audio capture."""
        if not self.is_running:
            return
        
        self.is_running = False
        
        if self.stream:
            self.stream.stop_stream()
            self.stream.close()
            self.stream = None
        
        if self.pa:
            self.pa.terminate()
            self.pa = None
        
        print("Audio capture stopped")
    
    def get_chunk(self, duration: float = 1.0) -> np.ndarray:
        """
        Get the most recent audio chunk.
        
        Args:
            duration: Duration in seconds
            
        Returns:
            Audio array of specified duration
        """
        num_samples = int(self.sample_rate * duration)
        
        with self.buffer_lock:
            return self.buffer[-num_samples:].copy()
    
    def get_energy(self) -> float:
        """Get current audio energy level."""
        with self.buffer_lock:
            chunk = self.buffer[-self.chunk_size:]
            return float(np.sqrt(np.mean(chunk ** 2)))
    
    def list_devices(self) -> list:
        """List available audio devices."""
        if not PYAUDIO_AVAILABLE:
            return []
        
        pa = pyaudio.PyAudio()
        devices = []
        
        for i in range(pa.get_device_count()):
            info = pa.get_device_info_by_index(i)
            if info['maxInputChannels'] > 0:
                devices.append({
                    'index': i,
                    'name': info['name'],
                    'sample_rate': int(info['defaultSampleRate']),
                    'channels': info['maxInputChannels']
                })
        
        pa.terminate()
        return devices
    
    def __enter__(self):
        self.start()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.stop()


class RealtimeDetector:
    """
    Real-time keyword detector.
    
    Combines audio capture with model inference for
    continuous keyword detection.
    """
    
    def __init__(
        self,
        runner,  # TFLiteRunner
        preprocessor,  # Function to extract MFCC from audio
        sample_rate: int = 16000,
        chunk_duration: float = 1.0,
        overlap: float = 0.5,
        threshold: float = 0.7,
        smoothing_window: int = 3
    ):
        """
        Initialize detector.
        
        Args:
            runner: TFLite inference runner
            preprocessor: Function to convert audio to MFCC
            sample_rate: Audio sample rate
            chunk_duration: Detection chunk duration
            overlap: Overlap between detection windows (0-1)
            threshold: Detection confidence threshold
            smoothing_window: Number of frames for smoothing
        """
        self.runner = runner
        self.preprocessor = preprocessor
        self.sample_rate = sample_rate
        self.chunk_duration = chunk_duration
        self.overlap = overlap
        self.threshold = threshold
        self.smoothing_window = smoothing_window
        
        # State
        self.is_running = False
        self.detection_thread = None
        self.audio_buffer = None
        
        # Detection history for smoothing
        self.detection_history = deque(maxlen=smoothing_window)
        
        # Callbacks
        self.on_detection: Optional[Callable] = None
        self.on_audio_level: Optional[Callable] = None
    
    def start(self, device_index: Optional[int] = None) -> None:
        """Start real-time detection."""
        if self.is_running:
            return
        
        # Start audio capture
        self.audio_buffer = AudioBuffer(
            sample_rate=self.sample_rate,
            chunk_duration=self.chunk_duration,
            device_index=device_index
        )
        self.audio_buffer.start()
        
        # Start detection thread
        self.is_running = True
        self.detection_thread = threading.Thread(target=self._detection_loop)
        self.detection_thread.start()
        
        print("Real-time detection started")
    
    def stop(self) -> None:
        """Stop real-time detection."""
        if not self.is_running:
            return
        
        self.is_running = False
        
        if self.detection_thread:
            self.detection_thread.join()
            self.detection_thread = None
        
        if self.audio_buffer:
            self.audio_buffer.stop()
            self.audio_buffer = None
        
        print("Real-time detection stopped")
    
    def _detection_loop(self) -> None:
        """Main detection loop."""
        step_duration = self.chunk_duration * (1 - self.overlap)
        
        while self.is_running:
            try:
                # Get audio chunk
                audio = self.audio_buffer.get_chunk(self.chunk_duration)
                
                # Report audio level
                energy = float(np.sqrt(np.mean(audio ** 2)))
                if self.on_audio_level:
                    self.on_audio_level(energy)
                
                # Skip if too quiet
                if energy < 0.01:
                    time.sleep(step_duration)
                    continue
                
                # Extract features
                features = self.preprocessor(audio, self.sample_rate)
                
                # Run inference
                label, confidence, latency = self.runner.predict_class(
                    features, self.threshold
                )
                
                # Smooth predictions
                self.detection_history.append((label, confidence))
                
                # Check for consistent detection
                if label and self._check_consistent_detection(label):
                    if self.on_detection:
                        self.on_detection(label, confidence, latency)
                    
                    # Clear history to prevent repeated triggers
                    self.detection_history.clear()
                
                # Wait for next window
                time.sleep(step_duration)
                
            except Exception as e:
                print(f"Detection error: {e}")
                time.sleep(0.1)
    
    def _check_consistent_detection(self, label: str) -> bool:
        """Check if detection is consistent across history."""
        if len(self.detection_history) < self.smoothing_window:
            return False
        
        # Check if all recent detections match
        for hist_label, _ in self.detection_history:
            if hist_label != label:
                return False
        
        return True
    
    def __enter__(self):
        self.start()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.stop()
