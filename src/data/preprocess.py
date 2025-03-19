import numpy as np
import librosa
import audiomentations

def load_audio(file_path, sr=32000, mono=True):
    """Load an audio file with consistent parameters."""
    audio, _ = librosa.load(file_path, sr=sr, mono=mono)
    return audio

def trim_audio(audio, top_db=20):
    """Trim leading and trailing silence."""
    return librosa.effects.trim(audio, top_db=top_db)[0]

def pad_or_truncate(audio, target_length):
    """Standardize audio to target length in samples."""
    if len(audio) > target_length:
        audio = audio[:target_length]
    else:
        padding = np.zeros(target_length - len(audio))
        audio = np.concatenate((audio, padding))
    return audio

def segment_audio(audio, sr=32000, segment_length=5):
    """Split audio into fixed-length segments."""
    segment_samples = segment_length * sr
    segments = []

    for start in range(0, len(audio), segment_samples):
        end = min(start + segment_samples, len(audio))
        segment = audio[start:end]

        # only keep segments of sufficient length
        if len(segment) >= segment_samples * 0.5:
            segment = pad_or_truncate(segment, segment_samples)
            segments.append(segment)

    return segments