import numpy as np
import librosa
import scipy
import torch

def extract_harmonic_percussive(audio, sr=32000):
    """Extract harmonic and percussive components from audio."""
    harmonic, percussive = librosa.effects.hpss(audio)
    return harmonic, percussive

def extract_delta_features(spectrogram, order=2):
    """Extract delta features from spectrogram."""
    deltas = []
    spec = spectrogram.copy()
    
    for d in range(order):
        delta = librosa.feature.delta(spec, order=1)
        deltas.append(delta)
        spec = delta
        
    return deltas

def extract_chroma(audio, sr=32000, n_chroma=12, n_fft=1024, hop_length=512):
    """Extract chromagram from audio."""
    chroma = librosa.feature.chroma_stft(
        y=audio, sr=sr, n_chroma=n_chroma, n_fft=n_fft, hop_length=hop_length
    )
    return chroma

def extract_stack_deltas(spectrogram, order=2):
    """Stack spectrogram with its deltas."""
    result = [spectrogram]
    
    deltas = extract_delta_features(spectrogram, order)
    result.extend(deltas)
    
    return np.vstack(result)

def extract_combined_features(audio, sr=32000, n_mels=128, n_fft=1024, hop_length=512):
    """Combine multiple feature types into a rich representation."""
    # mel spectrogram
    mel_spec = librosa.feature.melspectrogram(
        y=audio, sr=sr, n_mels=n_mels, n_fft=n_fft, hop_length=hop_length
    )
    log_mel_spec = librosa.power_to_db(mel_spec, ref=np.max)
    
    # chromagram
    chroma = librosa.feature.chroma_stft(
        y=audio, sr=sr, n_fft=n_fft, hop_length=hop_length
    )
    
    # spectral contrast
    contrast = librosa.feature.spectral_contrast(
        y=audio, sr=sr, n_fft=n_fft, hop_length=hop_length
    )
    
    # MFCC
    mfcc = librosa.feature.mfcc(
        y=audio, sr=sr, n_mfcc=13, n_fft=n_fft, hop_length=hop_length
    )
    
    # Stack features
    feature_stack = np.vstack([log_mel_spec, chroma, contrast, mfcc])
    
    return feature_stack

def extract_tempogram(audio, sr=32000, hop_length=512, win_length=384):
    """Extract tempogram for rhythm analysis."""
    tempogram = librosa.feature.tempogram(
        y=audio, sr=sr, hop_length=hop_length, win_length=win_length
    )
    return tempogram

def extract_spectral_contrast_mel(audio, sr=32000, n_mels=128, n_fft=1024, hop_length=512, class_name=None):
    """Extract mel spectrogram with spectral contrast based on taxonomic class."""
    # base mel spectrogram
    mel_spec = librosa.feature.melspectrogram(
        y=audio, sr=sr, n_mels=n_mels, n_fft=n_fft, hop_length=hop_length
    )
    log_mel_spec = librosa.power_to_db(mel_spec, ref=np.max)
    
    # add spectral contrast
    contrast = librosa.feature.spectral_contrast(
        y=audio, sr=sr, n_fft=n_fft, hop_length=hop_length
    )
    
    # class-specific adaptations
    if class_name == 'Aves':
        # Birds: enhance mid-high frequency bands, more mel bands in higher frequencies
        fmin, fmax = 500, 10000
        mel_spec_bird = librosa.feature.melspectrogram(
            y=audio, sr=sr, n_mels=n_mels, fmin=fmin, fmax=fmax, n_fft=n_fft, hop_length=hop_length
        )
        log_mel_spec_bird = librosa.power_to_db(mel_spec_bird, ref=np.max)
        return log_mel_spec_bird
    
    elif class_name == 'Amphibia':
        # Amphibians: enhance lower frequencies, higher temporal resolution
        fmin, fmax = 100, 6000
        hop_length_amphibian = hop_length // 2  # Higher temporal resolution
        mel_spec_amphibian = librosa.feature.melspectrogram(
            y=audio, sr=sr, n_mels=n_mels, fmin=fmin, fmax=fmax, 
            n_fft=n_fft, hop_length=hop_length_amphibian
        )
        log_mel_spec_amphibian = librosa.power_to_db(mel_spec_amphibian, ref=np.max)
        return log_mel_spec_amphibian
    
    elif class_name == 'Insecta':
        # Insects: focus on high frequencies, check for periodicity
        fmin, fmax = 1000, 15000
        mel_spec_insect = librosa.feature.melspectrogram(
            y=audio, sr=sr, n_mels=n_mels, fmin=fmin, fmax=fmax, n_fft=n_fft, hop_length=hop_length
        )
        log_mel_spec_insect = librosa.power_to_db(mel_spec_insect, ref=np.max)
        return log_mel_spec_insect
    
    elif class_name == 'Mammalia':
        # Mammals: enhance low-mid frequencies
        fmin, fmax = 50, 8000
        mel_spec_mammal = librosa.feature.melspectrogram(
            y=audio, sr=sr, n_mels=n_mels, fmin=fmin, fmax=fmax, n_fft=n_fft, hop_length=hop_length
        )
        log_mel_spec_mammal = librosa.power_to_db(mel_spec_mammal, ref=np.max)
        return log_mel_spec_mammal
    
    # default return original mel spectrogram
    return log_mel_spec