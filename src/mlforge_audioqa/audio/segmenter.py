from __future__ import annotations

import librosa
import numpy as np


def load_audio(path: str, sample_rate: int) -> np.ndarray:
    audio, _ = librosa.load(path, sr=sample_rate, mono=True)
    return audio.astype(np.float32)


def iter_windows(audio: np.ndarray, sample_rate: int, window_seconds: float, hop_seconds: float):
    window_size = int(window_seconds * sample_rate)
    hop_size = int(hop_seconds * sample_rate)

    if len(audio) < window_size:
        padded = np.pad(audio, (0, window_size - len(audio)))
        yield 0.0, window_seconds, padded
        return

    for start_idx in range(0, len(audio) - window_size + 1, hop_size):
        end_idx = start_idx + window_size
        start_t = start_idx / sample_rate
        end_t = end_idx / sample_rate
        yield start_t, end_t, audio[start_idx:end_idx]
