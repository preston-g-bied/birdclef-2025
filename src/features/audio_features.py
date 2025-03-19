import numpy as np
import librosa

def extract_melspectrogram(audio, sr=32000, n_mels=128, n_fft=1024, hop_length=512):
    """Extract mel spectrogram from audio."""
    mel_spec = librosa.feature.melspectrogram(
        y=audio, sr=sr, n_mels=n_mels, n_fft=n_fft, hop_length=hop_length
    )
    # convert to log scale (dB)
    log_mel_spec = librosa.power_to_db(mel_spec, ref=np.max)
    return log_mel_spec

def extract_mfcc(audio, sr=32000, n_mfcc=13, n_fft=1024, hop_length=512):
    """Extract MFCCs from audio."""
    mfccs = librosa.feature.mfcc(
        y=audio, sr=sr, n_mfcc=n_mfcc, n_fft=n_fft, hop_length=hop_length
    )
    # add delta and delta-delta features
    delta_mfccs = librosa.feature.delta(mfccs)
    delta2_mfccs = librosa.feature.delta(mfccs, order=2)
    return np.concatenate([mfccs, delta_mfccs, delta2_mfccs])

def extract_spectral_features(audio, sr=32000, n_fft=1024, hop_length=512):
    """Extract various spectral features from audio."""
    spectral_centroid = librosa.feature.spectral_centroid(y=audio, sr=sr, n_fft=n_fft, hop_length=hop_length)
    spectral_bandwidth = librosa.feature.spectral_bandwidth(y=audio, sr=sr, n_fft=n_fft, hop_length=hop_length)
    spectral_contrast = librosa.feature.spectral_contrast(y=audio, sr=sr, n_fft=n_fft, hop_length=hop_length)
    spectral_rolloff = librosa.feature.spectral_rolloff(y=audio, sr=sr, n_fft=n_fft, hop_length=hop_length)
    
    return {
        'centroid': spectral_centroid,
        'bandwidth': spectral_bandwidth,
        'contrast': spectral_contrast,
        'rolloff': spectral_rolloff
    }

def extract_bioacoustic_features(audio, sr=32000, class_name=None):
    """Extract bioacoustic-specific features based on taxonomic class."""
    # base features for all classes
    features = {}

    # zero crossing rate
    features['zcr'] = librosa.feature.zero_crossing_rate(audio)

    # RMS energy
    features['rms'] = librosa.feature.rms(y=audio)

    # specialized features by taxonomic class
    if class_name == 'Aves':
        # birds often have distinctive pitch patterns
        features['pitch'] = librosa.yin(audio, fmin=500, fmax=8000, sr=sr)

    elif class_name == 'Amphibia':
        # amphibians often have repetitive patterns and lower frequencies
        features['tempogram'] = librosa.feature.tempogram(y=audio, sr=sr)

    elif class_name == 'Insecta':
        # insects often have high-frequency components
        features['spectral_flatness'] = librosa.feature.spectral_flatness(y=audio)
    
    elif class_name == 'Mammalia':
        # mammals often have distinctive formants
        features['pitch'] = librosa.yin(audio, fmin=80, fmax=3000, sr=sr)

    return features