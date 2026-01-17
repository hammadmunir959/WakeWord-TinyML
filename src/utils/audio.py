"""
Audio Utilities for WakeWord
Functions for loading, processing, and augmenting audio data.
"""

import numpy as np
import librosa
import soundfile as sf
from pathlib import Path
from typing import Tuple, Optional, Union


def load_audio(
    path: Union[str, Path],
    sr: int = 16000,
    duration: float = 1.0,
    mono: bool = True
) -> Tuple[np.ndarray, int]:
    """
    Load audio file and resample to target sample rate.
    
    Args:
        path: Path to audio file
        sr: Target sample rate
        duration: Target duration in seconds (None for full length)
        mono: Convert to mono if True
        
    Returns:
        Tuple of (audio_data, sample_rate)
    """
    audio, orig_sr = librosa.load(path, sr=sr, mono=mono)
    
    if duration is not None:
        audio = pad_or_trim(audio, int(sr * duration))
    
    return audio, sr


def save_audio(
    path: Union[str, Path],
    audio: np.ndarray,
    sr: int = 16000
) -> None:
    """Save audio to file."""
    sf.write(path, audio, sr)


def pad_or_trim(audio: np.ndarray, target_length: int) -> np.ndarray:
    """
    Pad or trim audio to target length.
    
    Args:
        audio: Audio array
        target_length: Target number of samples
        
    Returns:
        Audio array of exact target length
    """
    current_length = len(audio)
    
    if current_length < target_length:
        # Pad with zeros
        padding = target_length - current_length
        audio = np.pad(audio, (0, padding), mode='constant')
    elif current_length > target_length:
        # Trim
        audio = audio[:target_length]
    
    return audio


def normalize_audio(audio: np.ndarray, target_db: float = -20.0) -> np.ndarray:
    """
    Normalize audio to target dB level.
    
    Args:
        audio: Audio array
        target_db: Target dB level
        
    Returns:
        Normalized audio
    """
    # Compute RMS
    rms = np.sqrt(np.mean(audio ** 2))
    if rms == 0:
        return audio
    
    # Compute target RMS
    target_rms = 10 ** (target_db / 20)
    
    # Scale audio
    audio = audio * (target_rms / rms)
    
    # Clip to prevent clipping
    audio = np.clip(audio, -1.0, 1.0)
    
    return audio


def extract_mfcc(
    audio: np.ndarray,
    sr: int = 16000,
    n_mfcc: int = 40,
    n_fft: int = 512,
    hop_length: int = 160,
    n_mels: int = 80,
    fmin: int = 20,
    fmax: int = 8000
) -> np.ndarray:
    """
    Extract MFCC features from audio.
    
    Args:
        audio: Audio array
        sr: Sample rate
        n_mfcc: Number of MFCCs
        n_fft: FFT window size
        hop_length: Hop length between frames
        n_mels: Number of Mel filterbanks
        fmin: Minimum frequency
        fmax: Maximum frequency
        
    Returns:
        MFCC features of shape (time_steps, n_mfcc)
    """
    mfccs = librosa.feature.mfcc(
        y=audio,
        sr=sr,
        n_mfcc=n_mfcc,
        n_fft=n_fft,
        hop_length=hop_length,
        n_mels=n_mels,
        fmin=fmin,
        fmax=fmax
    )
    
    # Transpose to (time_steps, n_mfcc)
    return mfccs.T


def extract_mel_spectrogram(
    audio: np.ndarray,
    sr: int = 16000,
    n_fft: int = 512,
    hop_length: int = 160,
    n_mels: int = 80,
    fmin: int = 20,
    fmax: int = 8000,
    power: float = 2.0
) -> np.ndarray:
    """
    Extract Mel spectrogram from audio.
    
    Args:
        audio: Audio array
        sr: Sample rate
        n_fft: FFT window size
        hop_length: Hop length between frames
        n_mels: Number of Mel filterbanks
        fmin: Minimum frequency
        fmax: Maximum frequency
        power: Exponent for the magnitude spectrogram
        
    Returns:
        Log Mel spectrogram of shape (time_steps, n_mels)
    """
    mel_spec = librosa.feature.melspectrogram(
        y=audio,
        sr=sr,
        n_fft=n_fft,
        hop_length=hop_length,
        n_mels=n_mels,
        fmin=fmin,
        fmax=fmax,
        power=power
    )
    
    # Convert to log scale
    log_mel = librosa.power_to_db(mel_spec, ref=np.max)
    
    # Transpose to (time_steps, n_mels)
    return log_mel.T


# ============== Data Augmentation ==============

def time_shift(audio: np.ndarray, shift_ms: int, sr: int = 16000) -> np.ndarray:
    """
    Shift audio in time by random amount.
    
    Args:
        audio: Audio array
        shift_ms: Maximum shift in milliseconds
        sr: Sample rate
        
    Returns:
        Time-shifted audio
    """
    max_shift = int(sr * shift_ms / 1000)
    shift = np.random.randint(-max_shift, max_shift)
    
    if shift > 0:
        audio = np.pad(audio, (shift, 0), mode='constant')[:-shift]
    elif shift < 0:
        audio = np.pad(audio, (0, -shift), mode='constant')[-shift:]
    
    return audio


def speed_change(audio: np.ndarray, speed_range: Tuple[float, float] = (0.9, 1.1)) -> np.ndarray:
    """
    Change speed of audio.
    
    Args:
        audio: Audio array
        speed_range: Tuple of (min_speed, max_speed)
        
    Returns:
        Speed-changed audio
    """
    speed = np.random.uniform(speed_range[0], speed_range[1])
    return librosa.effects.time_stretch(audio, rate=speed)


def add_noise(
    audio: np.ndarray,
    snr_db_range: Tuple[int, int] = (5, 20)
) -> np.ndarray:
    """
    Add Gaussian noise to audio.
    
    Args:
        audio: Audio array
        snr_db_range: Tuple of (min_snr, max_snr) in dB
        
    Returns:
        Noisy audio
    """
    snr_db = np.random.uniform(snr_db_range[0], snr_db_range[1])
    
    # Compute signal power
    signal_power = np.mean(audio ** 2)
    
    # Compute noise power
    noise_power = signal_power / (10 ** (snr_db / 10))
    
    # Generate noise
    noise = np.random.normal(0, np.sqrt(noise_power), len(audio))
    
    # Add noise
    noisy_audio = audio + noise
    
    return np.clip(noisy_audio, -1.0, 1.0)


def volume_change(audio: np.ndarray, db_range: float = 3.0) -> np.ndarray:
    """
    Change volume of audio.
    
    Args:
        audio: Audio array
        db_range: Maximum dB change
        
    Returns:
        Volume-changed audio
    """
    db_change = np.random.uniform(-db_range, db_range)
    gain = 10 ** (db_change / 20)
    return np.clip(audio * gain, -1.0, 1.0)


def pitch_shift(
    audio: np.ndarray,
    sr: int = 16000,
    semitones_range: float = 2.0
) -> np.ndarray:
    """
    Shift pitch of audio.
    
    Args:
        audio: Audio array
        sr: Sample rate
        semitones_range: Maximum semitones shift
        
    Returns:
        Pitch-shifted audio
    """
    semitones = np.random.uniform(-semitones_range, semitones_range)
    return librosa.effects.pitch_shift(audio, sr=sr, n_steps=semitones)


def augment_audio(
    audio: np.ndarray,
    sr: int = 16000,
    time_shift_ms: int = 100,
    speed_range: Tuple[float, float] = (0.9, 1.1),
    noise_snr_db: Tuple[int, int] = (5, 20),
    volume_db: float = 3.0,
    pitch_semitones: float = 2.0,
    target_length: Optional[int] = None
) -> np.ndarray:
    """
    Apply random augmentations to audio.
    
    Args:
        audio: Audio array
        sr: Sample rate
        time_shift_ms: Maximum time shift in ms
        speed_range: Speed change range
        noise_snr_db: SNR range for noise
        volume_db: Max volume change in dB
        pitch_semitones: Max pitch shift in semitones
        target_length: Target length after augmentation
        
    Returns:
        Augmented audio
    """
    # Apply augmentations randomly
    if np.random.random() > 0.5:
        audio = time_shift(audio, time_shift_ms, sr)
    
    if np.random.random() > 0.5:
        audio = speed_change(audio, speed_range)
    
    if np.random.random() > 0.5:
        audio = add_noise(audio, noise_snr_db)
    
    if np.random.random() > 0.5:
        audio = volume_change(audio, volume_db)
    
    if np.random.random() > 0.5:
        audio = pitch_shift(audio, sr, pitch_semitones)
    
    # Ensure target length
    if target_length is not None:
        audio = pad_or_trim(audio, target_length)
    
    return audio
