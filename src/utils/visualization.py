"""
Visualization Utilities for WakeWord
Functions for plotting audio waveforms, spectrograms, and training metrics.
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Optional, List, Tuple
import librosa
import librosa.display


def plot_waveform(
    audio: np.ndarray,
    sr: int = 16000,
    title: str = "Waveform",
    figsize: Tuple[int, int] = (12, 4),
    save_path: Optional[str] = None
) -> None:
    """Plot audio waveform."""
    plt.figure(figsize=figsize)
    librosa.display.waveshow(audio, sr=sr)
    plt.title(title)
    plt.xlabel("Time (s)")
    plt.ylabel("Amplitude")
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.show()


def plot_spectrogram(
    audio: np.ndarray,
    sr: int = 16000,
    n_fft: int = 512,
    hop_length: int = 160,
    title: str = "Spectrogram",
    figsize: Tuple[int, int] = (12, 4),
    save_path: Optional[str] = None
) -> None:
    """Plot spectrogram."""
    plt.figure(figsize=figsize)
    
    D = librosa.amplitude_to_db(
        np.abs(librosa.stft(audio, n_fft=n_fft, hop_length=hop_length)),
        ref=np.max
    )
    
    librosa.display.specshow(
        D, sr=sr, hop_length=hop_length,
        x_axis='time', y_axis='hz'
    )
    plt.colorbar(format='%+2.0f dB')
    plt.title(title)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.show()


def plot_mel_spectrogram(
    audio: np.ndarray,
    sr: int = 16000,
    n_fft: int = 512,
    hop_length: int = 160,
    n_mels: int = 80,
    title: str = "Mel Spectrogram",
    figsize: Tuple[int, int] = (12, 4),
    save_path: Optional[str] = None
) -> None:
    """Plot Mel spectrogram."""
    plt.figure(figsize=figsize)
    
    mel_spec = librosa.feature.melspectrogram(
        y=audio, sr=sr, n_fft=n_fft,
        hop_length=hop_length, n_mels=n_mels
    )
    mel_spec_db = librosa.power_to_db(mel_spec, ref=np.max)
    
    librosa.display.specshow(
        mel_spec_db, sr=sr, hop_length=hop_length,
        x_axis='time', y_axis='mel'
    )
    plt.colorbar(format='%+2.0f dB')
    plt.title(title)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.show()


def plot_mfcc(
    mfcc: np.ndarray,
    sr: int = 16000,
    hop_length: int = 160,
    title: str = "MFCC",
    figsize: Tuple[int, int] = (12, 4),
    save_path: Optional[str] = None
) -> None:
    """Plot MFCC features."""
    plt.figure(figsize=figsize)
    
    # Transpose if needed (should be n_mfcc x time)
    if mfcc.shape[0] > mfcc.shape[1]:
        mfcc = mfcc.T
    
    librosa.display.specshow(
        mfcc, sr=sr, hop_length=hop_length,
        x_axis='time'
    )
    plt.colorbar()
    plt.ylabel("MFCC Coefficient")
    plt.title(title)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.show()


def plot_training_history(
    history: dict,
    metrics: List[str] = ['loss', 'accuracy'],
    figsize: Tuple[int, int] = (14, 5),
    save_path: Optional[str] = None
) -> None:
    """Plot training history."""
    n_metrics = len(metrics)
    fig, axes = plt.subplots(1, n_metrics, figsize=figsize)
    
    if n_metrics == 1:
        axes = [axes]
    
    for ax, metric in zip(axes, metrics):
        if metric in history:
            ax.plot(history[metric], label=f'Train {metric}')
        if f'val_{metric}' in history:
            ax.plot(history[f'val_{metric}'], label=f'Val {metric}')
        
        ax.set_xlabel('Epoch')
        ax.set_ylabel(metric.capitalize())
        ax.set_title(f'{metric.capitalize()} over Training')
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.show()


def plot_confusion_matrix(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    class_names: List[str],
    title: str = "Confusion Matrix",
    figsize: Tuple[int, int] = (10, 8),
    save_path: Optional[str] = None
) -> None:
    """Plot confusion matrix."""
    from sklearn.metrics import confusion_matrix
    
    cm = confusion_matrix(y_true, y_pred)
    cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    
    plt.figure(figsize=figsize)
    sns.heatmap(
        cm_normalized,
        annot=True,
        fmt='.2f',
        cmap='Blues',
        xticklabels=class_names,
        yticklabels=class_names
    )
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title(title)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.show()


def plot_class_distribution(
    labels: np.ndarray,
    class_names: List[str],
    title: str = "Class Distribution",
    figsize: Tuple[int, int] = (12, 6),
    save_path: Optional[str] = None
) -> None:
    """Plot class distribution."""
    unique, counts = np.unique(labels, return_counts=True)
    
    plt.figure(figsize=figsize)
    bars = plt.bar(range(len(counts)), counts, color='steelblue')
    plt.xticks(range(len(counts)), [class_names[i] for i in unique], rotation=45, ha='right')
    plt.xlabel('Class')
    plt.ylabel('Count')
    plt.title(title)
    
    # Add count labels on bars
    for bar, count in zip(bars, counts):
        plt.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 50,
            str(count),
            ha='center',
            va='bottom'
        )
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.show()
