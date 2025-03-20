import os
import numpy as np
import pandas as pd
import librosa
import torch
from tqdm import tqdm
from pathlib import Path
import yaml
import pickle
import sys
sys.path.append('.')

# import feature extraction functions
from src.features.audio_features import extract_melspectrogram
from src.features.advanced_features import extract_spectral_contrast_mel

def cache_audio_data(config_path, cache_mode='audio'):
    """
    Cache either preprocessed audio or computed features
    
    Args:
        config_path: Path to config file
        cache_mode: Either 'audio' (preprocessed audio arrays) or 'mel' (mel spectrograms)
    """
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)

    # load metadata
    train_df = pd.read_csv(os.path.join(config['paths']['data']['raw'], 'train.csv'))
    taxonomy_df = pd.read_csv(os.path.join(config['paths']['data']['raw'], 'taxonomy.csv'))
    
    # merge to get taxonomic class
    train_df = pd.merge(
        train_df,
        taxonomy_df[['primary_label', 'class_name']],
        on='primary_label',
        how='left'
    )

    # create cache directory mased on mode
    cache_dir = Path(f'data/processed/{cache_mode}_cache')
    os.makedirs(cache_dir, exist_ok=True)

    # audio parameters
    audio_dir = Path(config['paths']['data']['raw']) / 'train_audio'
    sr = config['audio']['sample_rate']
    n_mels = config['audio']['n_mels']
    duration = config['audio']['audio_length']
    expected_length = int(duration * sr)

    # track processed files
    processed_files = []
    errors = []

    # process files
    for idx, row in tqdm(train_df.iterrows(), total=len(train_df), desc=f'Caching {cache_mode}'):
        filename = row['filename']
        audio_path = audio_dir / filename
        cache_filename = f"{filename.replace('/', '_').replace('.ogg', '.pt')}"
        cache_path = cache_dir / cache_filename

        # skip if already cached
        if os.path.exists(cache_path):
            processed_files.append({
                'original_path': str(audio_path),
                'cache_path': str(cache_path),
                'primary_label': row['primary_label'],
                'class_name': row['class_name'],
                'secondary_labels': row['secondary_labels']
            })
            continue

        try:
            # load and preprocess audio
            audio, _ = librosa.load(str(audio_path), sr=sr)

            # pad or truncate to expected length
            if len(audio) > expected_length:
                # for training data, select a random segment
                if len(audio) > expected_length * 2:    # if long enough, select random chunk
                    max_start = len(audio) - expected_length
                    start = np.random.randint(0, max_start)
                    audio = audio[start:start + expected_length]
                else:   # otherwise just take the beginning
                    audio = audio[:expected_length]

            else:
                # pad with zeros if too short
                padding = np.zeros(expected_length - len(audio))
                audio = np.concatenate((audio, padding))

            if cache_mode == 'audio':
                # save preprocessed audio
                torch.save(torch.FloatTensor(audio), cache_path)
            else:   # 'mel' mode
                # extract mel spectrogram
                mel_spec = extract_melspectrogram(
                    audio, sr=sr, n_mels=n_mels,
                    n_fft=config['audio']['n_fft'],
                    hop_length=config['audio']['hop_length']
                )
                # save spectrogram
                torch.save(torch.FloatTensor(mel_spec), cache_path)

            # record metadata
            processed_files.append({
                'original_path': str(audio_path),
                'cache_path': str(cache_path),
                'primary_label': row['primary_label'],
                'class_name': row['class_name'],
                'secondary_labels': row['secondary_labels']
            })

        except Exception as e:
            print(f"Error processing {audio_path}: {e}")
            errors.append({
                'filename': filename,
                'error': str(e)
            })

    # save metadata
    cache_metadata = pd.DataFrame(processed_files)
    cache_metadata.to_csv(cache_dir / 'metadata.csv', index=False)

    # save error log
    if errors:
        error_df = pd.DataFrame(errors)
        error_df.to_csv(cache_dir / 'errors.csv', index=False)
    
    print(f"Cached {len(processed_files)} files to {cache_dir}")
    print(f"Encountered {len(errors)} errors")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description='Cache audio features for faster training')
    parser.add_argument('--config', type=str, default='configs/advanced_config.yaml',
                       help='Path to config file')
    parser.add_argument('--mode', type=str, choices=['audio', 'mel'], default='audio',
                       help='What to cache: raw audio or mel spectrograms')
    args = parser.parse_args()
    
    cache_audio_data(args.config, args.mode)