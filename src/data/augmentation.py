import os
import numpy as np
import audiomentations

def get_augmentation_pipeline(sr=32000, soundscapes_path=None):
    """Create an augmentation pipeline for audio training data."""
    # create a list of augmentations
    augmentations = [
        # time-domain augmentations
        audiomentations.TimeStretch(min_rate=0.8, max_rate=1.2, p=0.5),
        audiomentations.PitchShift(min_semitones=2, max_semitones=2, p=0.5),
        audiomentations.Shift(min_shift=0.5, max_shift=0.5, p=0.5),

        # amplitude augmentations
        audiomentations.Gain(min_gain_db=6, max_gain_db=6, p=0.5),

        # environmental augmentations
        audiomentations.AddGaussianNoise(min_amplitude=0.001, max_amplitude=0.015, p=0.5),

        # frequency augmentations
        audiomentations.LowPassFilter(min_cutoff_freq=2000, max_cutoff_freq=7500, p=0.3),
        audiomentations.HighPassFilter(min_cutoff_freq=100, max_cutoff_freq=1000, p=0.3)
    ]

    # safely add background noise augmentation if path os provided
    if soundscapes_path is not None:
        try:
            # check if the path exists and contains files
            if os.path.exists(soundscapes_path) and len(os.listdir(soundscapes_path)) > 0:
                background_noise = audiomentations.AddBackgroundNoise(
                    sounds_path = soundscapes_path,
                    min_snr_db = 3,
                    max_snr_db = 30,
                    p = 0.5
                )
                augmentations.append(background_noise)
                print(f'Added background noise augmentation using {soundscapes_path}')
            else:
                print(f"Warning: Soundscapes path {soundscapes_path} doesn't exist or is empty")
                print(f"Current working directory: {os.getcwd()}")
        except Exception as e:
            print(f'Error setting up background noise augmentation: {e}')

    augment = audiomentations.Compose(augmentations)
    return augment

def apply_augmentation(audio, sr=32000, augment=None):
    """Apply audio augmentation to a single audio sample."""
    if augment is None:
        augment = get_augmentation_pipeline(sr)

    return augment(samples=audio, sample_rate=sr)