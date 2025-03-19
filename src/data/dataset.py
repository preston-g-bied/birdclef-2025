import os
import torch
import pandas as pd
import numpy as np
from torch.utils.data import Dataset, DataLoader
from pathlib import Path
import librosa

from src.data.preprocess import load_audio, pad_or_truncate, segment_audio
from src.features.audio_features import extract_melspectrogram

class BirdCLEFDataset(Dataset):
    def __init__(self, df, audio_dir, sr=32000, duration=5, transforms=None,
                 target_columns=None, is_train=True, taxonomy_path=None):
        """
        Dataset for BirdCLEF audio classification.

        Args:
            df: DataFrame with metadata
            audio_dir: Directory containing audio files
            sr: Sample rate
            duration: Duration in seconds for each segment
            transforms: Audio augmentation transforms
            target_columns: List of target species columns
            is_train: Whether this is a training dataset
        """
        self.df = df
        self.audio_dir = Path(audio_dir)
        self.sr = sr
        self.duration = duration
        self.audio_length = duration * sr
        self.transforms = transforms
        self.is_train = is_train

        if target_columns is None:
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
        file_path = self.audio_dir / row['filename']

        # load and preprocess audio
        audio = load_audio(file_path, sr=self.sr)

        # for training data, select a random segment if long enough
        if self.is_train and len(audio) > self.audio_length:
            max_start = len(audio) - self.audio_length
            start = np.random.randint(0, max_start)
            audio = audio[start:start + self.audio_length]

        # ensure consistent length
        audio = pad_or_truncate(audio, self.audio_length)

        # apply augmentation if provided
        if self.transforms and self.is_train:
            audio = self.transforms(samples=audio, sample_rate=self.sr)

        # extract mel spectrogram
        mel_spec = extract_melspectrogram(audio, sr=self.sr)

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

        return {
            'mel_spec': torch.FloatTensor(mel_spec),
            'audio': torch.FloatTensor(audio),
            'target': target,
            'primary_label': row['primary_label'],
            'class_name': row.get('class_name', '')
        }