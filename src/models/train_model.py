"""
Training script for BirdCLEF+ 2025 models.
"""

import sys
import os
import argparse
import yaml
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
import matplotlib.pyplot as plt
from pathlib import Path
from tqdm import tqdm
import wandb
from sklearn.model_selection import StratifiedKFold
import audiomentations
import warnings
warnings.filterwarnings('ignore')

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

# local imports
from src.data.dataset import BirdCLEFDataset
from src.data.augmentation import get_augmentation_pipeline
from src.models.baseline_cnn import BaselineCNN
from src.utils.metrics import calculate_lwlrap, macro_roc_auc

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Train BirdCLEF+ model')
    parser.add_argument('--config', type=str, default='configs/config.yaml',
                        help='Path to config file')
    parser.add_argument('--fold', type=int, default=0,
                        help='Validation fold to use')
    parser.add_argument('--debug', action='store_true',
                        help='Run in debug mode with a small dataset')
    parser.add_argument('--no_wandb', action='store_true',
                        help='Disable wandb logging')
    return parser.parse_args()

def load_config(config_path):
    """Load configuration from YAML file."""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config

def prepare_data(config, fold=0, debug=False):
    """Prepare train and validation datasets."""
    # load metadata
    train_df = pd.read_csv(os.path.join(config['paths']['data']['raw'], 'train.csv'))
    taxonomy_df = pd.read_csv(os.path.join(config['paths']['data']['raw'], 'taxonomy.csv'))

    # merge to get taxonomic class
    train_df = pd.merge(
        train_df,
        taxonomy_df[['primary_label', 'class_name']],
        on = 'primary_label',
        how = 'left'
    )

    # optionally subsample for debugging
    if debug:
        print('Debug mode: using a small subset of data')
        samples_per_class = 10

        sampled_dfs = []

        for class_name, group in train_df.groupby('class_name'):
            sampled_dfs.append(group.sample(min(samples_per_class, len(group))))

        train_df = pd.concat(sampled_dfs).reset_index(drop=True)

        n_splits = 2
    else:
        n_splits = 5

    # create stratified folds based on primary label
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=config['training']['seed'])
    train_df['fold'] = -1

    try:
        for fold_idx, (train_idx, val_idx) in enumerate(skf.split(train_df, train_df['primary_label'])):
            train_df.loc[val_idx, 'fold'] = fold_idx
    except Exception as e:
        print(f'Error in cross-validation split: {e}')
        # fallback to simple train/val split
        from sklearn.model_selection import train_test_split
        train_indices, val_indices = train_test_split(
            range(len(train_df)), test_size=0.2,
            stratify=train_df['primary_label'].values if train_df['primary_label'].nunique() > 1 else None,
            random_state = config['training']['seed']
        )
        train_df.loc[val_indices, 'fold'] = 0

    # ensure fold is valid
    if fold >= n_splits:
        fold = 0

    # split the data into train and validation sets based on specified fold
    train_data = train_df[train_df['fold'] != fold].reset_index(drop=True)
    val_data = train_df[train_df['fold'] == fold].reset_index(drop=True)

    print(f'Train set: {len(train_data)} samples')
    print(f'Validation set: {len(val_data)} samples')

    # set up augmentations
    soundscapes_path = os.path.join(config['paths']['data']['raw'], 'train_soundscapes')
    augment = get_augmentation_pipeline(
        sr = config['audio']['sample_rate'],
        soundscapes_path = soundscapes_path
    )

    # create datasets
    train_dataset = BirdCLEFDataset(
        train_data,
        os.path.join(config['paths']['data']['raw'], 'train_audio'),
        sr = config['audio']['sample_rate'],
        duration = config['audio']['audio_length'],
        transforms = augment,
        is_train = True,
        taxonomy_path = os.path.join(config['paths']['data']['raw'], 'taxonomy.csv')
    )

    val_dataset = BirdCLEFDataset(
        val_data,
        os.path.join(config['paths']['data']['raw'], 'train_audio'),
        sr = config['audio']['sample_rate'],
        duration = config['audio']['audio_length'],
        transforms = None,
        is_train = False,
        taxonomy_path = os.path.join(config['paths']['data']['raw'], 'taxonomy.csv')
    )

    # create weighted sampler to address class imbalance
    class_counts = train_data['primary_label'].value_counts().to_dict()
    sample_weights = [1.0 / class_counts[label] for label in train_data['primary_label']]
    sampler = torch.utils.data.WeightedRandomSampler(
        weights = sample_weights,
        num_samples = len(train_data),
        replacement = True
    )

    # create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size = config['training']['batch_size'],
        sampler = sampler,
        num_workers = 0 if debug else config['training']['num_workers'],
        pin_memory = True
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size = config['training']['batch_size'],
        shuffle = False,
        num_workers = config['training']['num_workers'],
        pin_memory = True
    )

    return train_loader, val_loader, train_dataset.target_columns

def train_epoch(model, train_loader, criterion, optimizer, device):
    """Train the model for one epoch."""
    model.train()
    running_loss = 0.0

    pbar = tqdm(train_loader, desc='Training')
    for batch in pbar:
        # get the inputs and labels
        mel_specs = batch['mel_spec'].to(device)
        targets = batch['target'].to(device)

        # add channel dimension if needed
        if mel_specs.dim() == 3:
            mel_specs = mel_specs.unsqueeze(1)

        # zero the parameter gradients
        optimizer.zero_grad()

        # forward pass
        outputs = model(mel_specs)
        loss = criterion(outputs, targets)

        # backward pass and optimize
        loss.backward()
        optimizer.step()

        # update statistics
        running_loss += loss.item()
        pbar.set_postfix({'Loss': f'{loss.item():.4f}'})
    
    return running_loss / len(train_loader)

def validate(model, val_loader, criterion, device):
    """Validate the model."""
    model.eval()
    running_loss = 0.0
    all_targets = []
    all_outputs = []

    with torch.no_grad():
        for batch in tqdm(val_loader, desc='Validation'):
            # get the inputs and labels
            mel_specs = batch['mel_spec'].to(device)
            targets = batch['target'].to(device)

            # add channel dimension if needed
            if mel_specs.dim() == 3:
                mel_specs = mel_specs.unsqueeze(1)

            # forward pass
            outputs = model(mel_specs)
            loss = criterion(outputs, targets)

            # update statistics
            running_loss += loss.item()

            # store predictions and targets for metrics calculation
            all_targets.append(targets.cpu().numpy())
            all_outputs.append(torch.sigmoid(outputs).cpu().numpy())

    # concatenate batches
    all_targets = np.vstack(all_targets)
    all_outputs = np.vstack(all_outputs)

    # calculate metrics
    roc_auc = macro_roc_auc(all_targets, all_outputs)
    lwlrap_score = calculate_lwlrap(all_targets, all_outputs)

    return running_loss / len(val_loader), roc_auc, lwlrap_score

def train_model(config, fold=0, debug=False, use_wandb=True):
    """Train the model using the specified configuration."""
    # set random seeds for reproducability
    torch.manual_seed(config['training']['seed'])
    np.random.seed(config['training']['seed'])

    # set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Using device: {device}')

    # prepare data
    train_loader, val_loader, target_columns = prepare_data(config, fold, debug)

    # initialize model
    model = BaselineCNN(num_classes=len(target_columns))
    model = model.to(device)

    # define loss function and optimizer
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters(), lr=config['training']['learning_rate'])

    # learning rate scheduler
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=3
    )

    # initialize wandb if enabled
    if use_wandb:
        wandb.init(
            project = config['project']['name'],
            name = f'fold{fold}',
            config = config
        )
        wandb.watch(model, log='all')

    # training loop
    best_roc_auc = 0.0
    best_model_path = None
    patience_counter = 0

    for epoch in range(config['training']['num_epochs']):
        print(f"Epoch {epoch+1}/{config['training']['num_epochs']}")

        # train for one epoch
        train_loss = train_epoch(model, train_loader, criterion, optimizer, device)

        # validate
        val_loss, roc_auc, lwlrap = validate(model, val_loader, criterion, device)

        # update learning rate scheduler
        scheduler.step(val_loss)

        # log metrics
        print(f'Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}, ROC-AUC: {roc_auc:.4f}, LWLRAP: {lwlrap:.4f}')

        if use_wandb:
            wandb.log({
                "train_loss": train_loss,
                "val_loss": val_loss,
                "roc_auc": roc_auc,
                "lwlrap": lwlrap,
                "learning_rate": optimizer.param_groups[0]['lr']
            })

        # save best model
        if roc_auc > best_roc_auc:
            best_roc_auc = roc_auc
            patience_counter = 0

            # create output directory if it doesn't exist
            os.makedirs(config['paths']['models'], exist_ok=True)

            # save model
            best_model_path = os.path.join(
                config['paths']['models'],
                f'baseline_fold{fold}_roc{roc_auc:.4f}.pth'
            )
            torch.save(model.state_dict(), best_model_path)
            print(f'Saved best model to {best_model_path}')
        else:
            patience_counter += 1
            if patience_counter >= config['training']['patience']:
                print(f'Early stopping after {epoch+1} epochs')
                break

    print(f'Best ROC-AUC: {best_roc_auc:.4f}')
    return best_model_path

def main():
    """Main function."""
    args = parse_args()
    config = load_config(args.config)

    best_model_path = train_model(
        config,
        fold = args.fold,
        debug = args.debug,
        use_wandb = not args.no_wandb
    )

    print(f'Training complete. Best model saved to {best_model_path}')

if __name__ == "__main__":
    main()