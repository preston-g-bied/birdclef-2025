import numpy as np
import random
import audiomentations
from audiomentations import Compose, AddGaussianNoise, TimeStretch, PitchShift, Shift
from audiomentations import Gain, Normalize, LowPassFilter, HighPassFilter
import librosa
import torch

def get_strong_augmentation_pipeline(sr=32000, soundscapes_path=None):
    """Create a strong augmentation pipeline for audio training data."""
    augmentations = [
        # time-domain augmentations
        TimeStretch(min_rate=0.8, max_rate=1.25, p=0.5),
        PitchShift(min_semitones=-4, max_semitones=4, p=0.5),
        Shift(min_shift=-0.5, max_shift=0.5, p=0.5),
        
        # amplitude augmentations
        Gain(min_gain_db=-10, max_gain_db=10, p=0.5),
        
        # noise augmentations
        AddGaussianNoise(min_amplitude=0.001, max_amplitude=0.015, p=0.5),
        
        # frequency augmentations
        LowPassFilter(min_cutoff_freq=1000, max_cutoff_freq=7500, p=0.3),
        HighPassFilter(min_cutoff_freq=100, max_cutoff_freq=2000, p=0.3),
        
        # compound effects
        audiomentations.OneOf([
            audiomentations.TimeMask(min_band_part=0.1, max_band_part=0.3, p=1.0),
            audiomentations.BandPassFilter(min_center_freq=200, max_center_freq=4000, 
                               min_bandwidth_fraction=0.5, max_bandwidth_fraction=1.5, p=1.0)
        ], p=0.5),
    ]
    
    # safely add background noise augmentation if path is provided
    if soundscapes_path:
        try:
            import os
            if os.path.exists(soundscapes_path) and len(os.listdir(soundscapes_path)) > 0:
                # first option: add background noise from soundscapes
                background_noise = audiomentations.AddBackgroundNoise(
                    sounds_path=soundscapes_path,
                    min_snr_db=3,
                    max_snr_db=20,
                    p=0.5
                )
                augmentations.append(background_noise)
                
                # second option: add sound effects from soundscapes
                sound_effects = audiomentations.AddShortNoises(
                    sounds_path=soundscapes_path,
                    min_snr_db=0,
                    max_snr_db=15,
                    min_time_between_sounds=1.0,
                    max_time_between_sounds=8.0,
                    p=0.3
                )
                augmentations.append(sound_effects)
        except Exception as e:
            print(f"Error setting up soundscape augmentations: {e}")
    
    augment = Compose(augmentations)
    return augment

class SpecAugment:
    """Implementation of SpecAugment for spectrogram augmentation."""
    def __init__(self, freq_mask_param=10, time_mask_param=10, num_freq_masks=2, num_time_masks=2):
        self.freq_mask_param = freq_mask_param
        self.time_mask_param = time_mask_param
        self.num_freq_masks = num_freq_masks
        self.num_time_masks = num_time_masks
        
    def __call__(self, spec):
        """
        Apply SpecAugment to spectrogram
        
        Args:
            spec: Spectrogram [frequency, time]
        
        Returns:
            Augmented spectrogram
        """
        spec = spec.copy()
        
        # frequency masking
        for _ in range(self.num_freq_masks):
            freq_dim = spec.shape[0]
            f = min(random.randint(0, self.freq_mask_param), freq_dim - 1)
            f0 = random.randint(0, freq_dim - f)
            spec[f0:f0+f, :] = 0
            
        # time masking
        for _ in range(self.num_time_masks):
            time_dim = spec.shape[1]
            t = min(random.randint(0, self.time_mask_param), time_dim - 1)
            t0 = random.randint(0, time_dim - t)
            spec[:, t0:t0+t] = 0
            
        return spec

class MixupAugmentation:
    """Mixup augmentation for audio classification."""
    def __init__(self, alpha=0.2):
        self.alpha = alpha
        
    def __call__(self, batch_x, batch_y):
        """
        Apply Mixup to a batch of spectrograms and labels
        
        Args:
            batch_x: Batch of spectrograms [batch_size, channels, height, width]
            batch_y: Batch of labels [batch_size, num_classes]
            
        Returns:
            Mixed spectrograms and labels
        """
        batch_size = batch_x.size(0)
        
        # sample lambda from beta distribution
        lam = np.random.beta(self.alpha, self.alpha, batch_size)
        lam = torch.tensor(lam, dtype=torch.float).to(batch_x.device)
        
        # reshape lambda for broadcasting
        lam = lam.view(batch_size, 1, 1, 1)
        
        # get random indices for mixing
        index = torch.randperm(batch_size).to(batch_x.device)
        
        # mix the samples
        mixed_x = lam * batch_x + (1 - lam) * batch_x[index]
        
        # reshape lambda for labels
        label_lam = lam.view(batch_size, 1)
        
        # mix the labels
        mixed_y = label_lam * batch_y + (1 - label_lam) * batch_y[index]
        
        return mixed_x, mixed_y

def taxonomic_specific_augmentation(audio, sr=32000, class_name=None):
    """Apply taxonomic-specific augmentations based on class."""
    # basic augmentation pipeline
    augment_basic = Compose([
        Normalize(p=1.0),
        AddGaussianNoise(min_amplitude=0.001, max_amplitude=0.01, p=0.3),
    ])
    
    audio = augment_basic(audio, sr)
    
    # class-specific augmentations
    if class_name == 'Aves':
        # Birds often have distinctive pitch patterns and calls
        augment_birds = Compose([
            PitchShift(min_semitones=-2, max_semitones=2, p=0.5),
            TimeStretch(min_rate=0.85, max_rate=1.15, p=0.5),
            Shift(min_fraction=-0.3, max_fraction=0.3, p=0.5),
            HighPassFilter(min_cutoff_freq=500, max_cutoff_freq=1500, p=0.3),
        ])
        audio = augment_birds(audio, sr)
        
    elif class_name == 'Amphibia':
        # amphibians often have repetitive calls at lower frequencies
        augment_amphibians = Compose([
            TimeStretch(min_rate=0.9, max_rate=1.1, p=0.5),  # Less time stretch
            PitchShift(min_semitones=-1, max_semitones=1, p=0.5),  # Less pitch shift
            # enhance low frequencies
            LowPassFilter(min_cutoff_freq=2000, max_cutoff_freq=4000, p=0.4),
        ])
        audio = augment_amphibians(audio, sr)
        
    elif class_name == 'Insecta':
        # insects often have high-frequency components and regular patterns
        augment_insects = Compose([
            Gain(min_gain_in_db=-6, max_gain_in_db=6, p=0.5),
            # less time/pitch modification to preserve rhythmic patterns
            TimeStretch(min_rate=0.95, max_rate=1.05, p=0.3),
            # enhance high frequencies
            LowPassFilter(min_cutoff_freq=5000, max_cutoff_freq=10000, p=0.3),
        ])
        audio = augment_insects(audio, sr)
        
    elif class_name == 'Mammalia':
        # mammals have diverse calls but often with distinctive formants
        augment_mammals = Compose([
            PitchShift(min_semitones=-3, max_semitones=3, p=0.5),
            TimeStretch(min_rate=0.8, max_rate=1.2, p=0.5),
            # diverse frequency components
            audiomentations.OneOf([
                LowPassFilter(min_cutoff_freq=1000, max_cutoff_freq=4000, p=1.0),
                HighPassFilter(min_cutoff_freq=300, max_cutoff_freq=1000, p=1.0),
            ], p=0.4),
        ])
        audio = augment_mammals(audio, sr)
    
    return audio