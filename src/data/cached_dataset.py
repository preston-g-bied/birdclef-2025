import torch
from torch.utils.data import Dataset
import pandas as pd
import numpy as np
import os
from pathlib import Path
import librosa
import sys
sys.path.append('.')  # add project root to path

from src.features.audio_features import extract_melspectrogram
from src.features.advanced_features import extract_spectral_contrast_mel

class CachedAdvancedBirdCLEFDataset(Dataset):
    def __init__(self, df, cache_dir, sr=32000, duration=5, transforms=None,
                 target_columns=None, is_train=True, taxonomy_path=None,
                 taxonomic=False, config=None, cache_mode='audio'):
        """
        Dataset that loads cached audio data and applies transformations on-the-fly.
        
        Args:
            df: DataFrame with metadata
            cache_dir: Directory with cached features
            sr: Sample rate
            duration: Duration in seconds for each segment
            transforms: Audio augmentation transforms (for 'audio' mode)
            target_columns: List of target species columns
            is_train: Whether this is a training dataset
            taxonomy_path: Path to taxonomy CSV
            taxonomic: Whether to use taxonomic-specific features
            config: Configuration dictionary
            cache_mode: Either 'audio' (cached audio arrays) or 'mel' (cached spectrograms)
        """
        self.df = df
        self.cache_dir = Path(cache_dir)
        self.sr = sr
        self.duration = duration
        self.audio_length = duration * sr
        self.transforms = transforms
        self.is_train = is_train
        self.taxonomic = taxonomic
        self.config = config
        self.cache_mode = cache_mode
        
        # load all species targets
        if target_columns is None and taxonomy_path:
            # get all species columns from taxonomy
            taxonomy = pd.read_csv(taxonomy_path)
            self.target_columns = taxonomy['primary_label'].tolist()
        else:
            self.target_columns = target_columns
            
        # map from species labels to indices
        self.label_to_idx = {label: i for i, label in enumerate(self.target_columns)}
    
    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        filename = row['filename'].replace('/', '_').replace('.ogg', '.pt')
        cache_path = self.cache_dir / filename
        
        # handle case where file might not be cached
        if not os.path.exists(cache_path):
            raise FileNotFoundError(f"Cached file not found: {cache_path}. Run cache_features.py first.")
            
        # load cached data
        cached_data = torch.load(cache_path)
        
        if self.cache_mode == 'audio':
            # audio data was cached - apply transforms and extract features
            audio = cached_data
            
            # apply audio augmentation if provided and in training mode
            if self.transforms and self.is_train:
                # convert back to numpy for audiomentations
                audio_np = audio.numpy()
                audio_np = self.transforms(samples=audio_np, sample_rate=self.sr)
                audio = torch.FloatTensor(audio_np)
            
            # extract features (possibly taxonomic-specific)
            if self.taxonomic:
                audio_np = audio.numpy()
                class_name = row['class_name']
                mel_spec = extract_spectral_contrast_mel(
                    audio_np,
                    sr=self.sr,
                    n_mels=self.config['audio']['n_mels'],
                    class_name=class_name
                )
                mel_spec = torch.FloatTensor(mel_spec)
            else:
                # standard mel spectrogram
                audio_np = audio.numpy()
                mel_spec = extract_melspectrogram(
                    audio_np,
                    sr=self.sr,
                    n_mels=self.config['audio']['n_mels']
                )
                mel_spec = torch.FloatTensor(mel_spec)
        else:
            # mel spectrogram was cached directly
            mel_spec = cached_data
            # in this case, audio is not available, 
            # and we can only apply spectrogram augmentations
            audio = None
            
        # create multi-label target
        target = torch.zeros(len(self.target_columns))
        
        # set primary label
        if row['primary_label'] in self.label_to_idx:
            target[self.label_to_idx[row['primary_label']]] = 1.0
            
        # set secondary labels if available
        if 'secondary_labels' in row and row['secondary_labels'] != "['']":
            secondary_labels = eval(row['secondary_labels']) if isinstance(row['secondary_labels'], str) else row['secondary_labels']
            for label in secondary_labels:
                if label in self.label_to_idx and label != '':
                    target[self.label_to_idx[label]] = 1.0
                    
        # ensure consistent dimensions for spectrograms
        if isinstance(mel_spec, torch.Tensor):
            target_time_dim = 400  # you can adjust based on your needs
            current_shape = mel_spec.shape
            
            if len(current_shape) == 2:  # height, width
                if current_shape[1] > target_time_dim:
                    # truncate if longer
                    mel_spec = mel_spec[:, :target_time_dim]
                elif current_shape[1] < target_time_dim:
                    # pad with zeros if shorter
                    padding = torch.zeros(current_shape[0], target_time_dim - current_shape[1])
                    mel_spec = torch.cat([mel_spec, padding], dim=1)
        
        result = {
            'mel_spec': mel_spec,
            'target': target,
            'primary_label': row['primary_label'],
            'class_name': row.get('class_name', '')
        }
        
        # include audio data if available
        if audio is not None:
            result['audio'] = audio
            
        return result