import sys
import os
import argparse
import yaml
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
from pathlib import Path
from tqdm import tqdm
import wandb
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score
import time
import random
import json
from collections import defaultdict
import warnings
warnings.filterwarnings('ignore')

# ensure relative imports work
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

# local imports
from src.data.dataset import BirdCLEFDataset
from src.data.enhanced_augmentation import get_strong_augmentation_pipeline, MixupAugmentation
from src.features.advanced_features import extract_spectral_contrast_mel
from src.models.efficientnet_model import EfficientNetAudio
from src.models.attention_cnn import AttentionCNN
from src.utils.metrics import calculate_lwlrap, macro_roc_auc, calculate_per_class_metrics

# advanced
class AdvancedBirdCLEFDataset(BirdCLEFDataset):
    def __init__(self, df, audio_dir, sr=32000, duration=5, transforms=None,
                 target_columns=None, is_train=True, taxonomy_path=None,
                 taxonomic=False, config=None):
        super().__init__(df, audio_dir, sr, duration, transforms,
                         target_columns, is_train, taxonomy_path)
        self.taxonomic = taxonomic
        self.config = config
        
    def __getitem__(self, idx):
        result = super().__getitem__(idx)
        
        # If using taxonomic-specific features, replace mel_spec
        if self.taxonomic:
            class_name = result['class_name']
            audio = result['audio'].numpy()
            mel_spec = extract_spectral_contrast_mel(
                audio,
                sr=self.sr,
                n_mels=self.config['audio']['n_mels'] if self.config else 128,
                class_name=class_name
            )
            result['mel_spec'] = torch.FloatTensor(mel_spec)
            
        # Ensure consistent dimensions
        if isinstance(result['mel_spec'], torch.Tensor):
            target_time_dim = 400
            current_shape = result['mel_spec'].shape
            
            if current_shape[1] > target_time_dim:
                # Truncate if longer
                result['mel_spec'] = result['mel_spec'][:, :target_time_dim]
            elif current_shape[1] < target_time_dim:
                # Pad with zeros if shorter
                padding = torch.zeros(current_shape[0], target_time_dim - current_shape[1])
                result['mel_spec'] = torch.cat([result['mel_spec'], padding], dim=1)
                
        return result

def seed_everything(seed):
    """Set seeds for reproducability."""
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Train BirdCLEF+ model with advanced techniques')
    parser.add_argument('--config', type=str, default='configs/advanced_config.yaml',
                        help='Path to config file')
    parser.add_argument('--fold', type=int, default=0,
                        help='Validation fold to use')
    parser.add_argument('--model', type=str, default='efficientnet',
                        choices=['efficientnet', 'attention', 'baseline'],
                        help='Model architecture to use')
    parser.add_argument('--debug', action='store_true',
                        help='Run in debug mode with a small dataset')
    parser.add_argument('--no_wandb', action='store_true',
                        help='Disable wandb logging')
    parser.add_argument('--mixup', action='store_true',
                        help='Use mixup augmentation')
    parser.add_argument('--taxonomic', action='store_true',
                        help='Use taxonomy-specific features and training')
    return parser.parse_args()

def load_config(config_path):
    """Load configuration from YAML file."""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config

def prepare_data(config, fold=0, debug=False, taxonomic=False):
    """Prepare train and validation datasets with advanced techniques."""

    # load metadata
    train_df = pd.read_csv(os.path.join(config['paths']['data']['raw'], 'train.csv'))
    taxonomy_df = pd.read_csv(os.path.join(config['paths']['data']['raw'], 'taxonomy.csv'))

    # convert string representation of lists to actual lists for secondary labels
    train_df['secondary_labels'] = train_df['secondary_labels'].apply(
        lambda x: eval(x) if isinstance(x, str) else x
    )

    # merge to get taxonomic class
    train_df = pd.merge(
        train_df,
        taxonomy_df[['primary_label', 'class_name']],
        on = 'primary_label',
        how = 'left'
    )

    # handle imbalance by oversampling rare classes
    if taxonomic:
        print('Using taxonomic balancing...')
        min_samples_per_class = 50
        balanced_train_dfs = []

        for class_name, class_group in train_df.groupby('class_name'):
            # for each taxonomic class, ensure minimum samples per species
            species_dfs = []

            for species, species_group in class_group.groupby('primary_label'):
                if len(species_group) < min_samples_per_class:
                    # oversample species with fewer samples
                    oversampled = species_group.sample(
                        min_samples_per_class,
                        replace = True,
                        random_state = config['training']['seed']
                    )
                    species_dfs.append(oversampled)
                else:
                    species_dfs.append(species_group)
                
            # combine all species in this taxonomic class
            balanced_class_df = pd.concat(species_dfs)
            balanced_train_dfs.append(balanced_class_df)

        # combine all balanced data
        balanced_train_df = pd.concat(balanced_train_dfs)
        train_df = balanced_train_df.reset_index(drop=True)
        print(f'After balancing: {len(train_df)} total samples')

        # display class distribution after balancing
        class_counts = train_df['class_name'].value_counts()
        for class_name, count in class_counts.items():
            print(f'{class_name}: {count} samples')

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

    # create stratified folds based on primary label and taxonomic class
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=config['training']['seed'])
    train_df['fold'] = -1

    # create a stratification target that combines primary label and class
    train_df['strat_target'] = train_df['primary_label'] + '_' + train_df['class_name']

    try:
        for fold_idx, (train_idx, val_idx) in enumerate(skf.split(train_df, train_df['strat_target'])):
            train_df.loc[val_idx, 'fold'] = fold_idx
    except Exception as e:
        print(f'Error in cross-validation split: {e}')
        # fallback to simple train/val split
        from sklearn.model_selection import train_test_split
        train_indices, val_indices = train_test_split(
            range(len(train_df)), test_size=0.2,
            stratify=train_df['strat_target'].values if train_df['strat_target'].nunique() > 1 else None,
            random_state=config['training']['seed']
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
    augment = get_strong_augmentation_pipeline(
        sr = config['audio']['sample_rate'],
        soundscapes_path = soundscapes_path
    )
        
    # create datasets
    train_dataset = AdvancedBirdCLEFDataset(
        train_data,
        os.path.join(config['paths']['data']['raw'], 'train_audio'),
        sr = config['audio']['sample_rate'],
        duration = config['audio']['audio_length'],
        transforms = augment,
        is_train = True,
        taxonomy_path = os.path.join(config['paths']['data']['raw'], 'taxonomy.csv')
    )

    val_dataset = AdvancedBirdCLEFDataset(
        val_data,
        os.path.join(config['paths']['data']['raw'], 'train_audio'),
        sr = config['audio']['sample_rate'],
        duration = config['audio']['audio_length'],
        transforms = None,
        is_train = False,
        taxonomy_path = os.path.join(config['paths']['data']['raw'], 'taxonomy.csv')
    )

    # create weighted sampler to address class imbalance
    if taxonomic:
        # for taxonomic training, use a class-weighted sampler
        class_weights = {}
        for class_name in train_data['class_name'].unique():
            class_count = (train_data['class_name'] == class_name).sum()
            class_weights[class_name] = 1.0 / class_count

        sample_weights = [class_weights[label] for label in train_data['class_name']]
    else:
        # traditional species-based weighting
        species_counts = train_data['primary_label'].value_counts().to_dict()
        sample_weights = [1.0 / species_counts[label] for label in train_data['primary_label']]

    # normalize weights
    sample_weights = np.array(sample_weights) / sum(sample_weights) * len(sample_weights)

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
        num_workers = 4 if not debug else 0,
        pin_memory = True
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size = config['training']['batch_size'],
        shuffle = False,
        num_workers = 4 if not debug else 0,
        pin_memory = True
    )

    return train_loader, val_loader, train_dataset.target_columns, train_dataset.label_to_idx

def create_model(model_type, num_classes, device):
    """Create model based on specified architecture."""
    if model_type == 'efficientnet':
        model = EfficientNetAudio(num_classes=num_classes)
    elif model_type == 'attention':
        model = AttentionCNN(num_classes=num_classes)
    else:
        from src.models.baseline_cnn import BaselineCNN
        model = BaselineCNN(num_classes=num_classes)

    return model.to(device)

def train_epoch(model, train_loader, criterion, optimizer, device, use_mixup=False, scheduler=None):
    """Train the model for one epoch with advanced techniques."""
    model.train()
    running_loss = 0.0
    
    # mixup augmentation if enabled
    mixup = MixupAugmentation(alpha=0.2) if use_mixup else None
    
    pbar = tqdm(train_loader, desc='Training')
    for batch in pbar:
        # get the inputs and labels
        mel_specs = batch['mel_spec'].to(device)
        targets = batch['target'].to(device)
        
        # add channel dimension if needed
        if mel_specs.dim() == 3:
            mel_specs = mel_specs.unsqueeze(1)
            
        # apply mixup if enabled
        if mixup is not None:
            mel_specs, targets = mixup(mel_specs, targets)
            
        # zero the parameter gradients
        optimizer.zero_grad()
        
        # forward pass
        outputs = model(mel_specs)
        loss = criterion(outputs, targets)
        
        # backward pass and optimize
        loss.backward()
        optimizer.step()
        
        # update learning rate if using OneCycleLR
        if scheduler is not None and isinstance(scheduler, torch.optim.lr_scheduler.OneCycleLR):
            scheduler.step()
        
        # update statistics
        running_loss += loss.item()
        pbar.set_postfix({'Loss': f'{loss.item():.4f}'})

    return running_loss / len(train_loader)

def validate(model, val_loader, criterion, device, label_to_idx, class_names=None):
    """Validate the model with detailed metrics."""
    model.eval()
    running_loss = 0.0
    all_targets = []
    all_outputs = []
    all_primary_labels = []  # store primary labels for per-species analysis
    
    with torch.no_grad():
        for batch in tqdm(val_loader, desc='Validation'):
            # get the inputs and labels
            mel_specs = batch['mel_spec'].to(device)
            targets = batch['target'].to(device)
            primary_labels = batch['primary_label']  # store for later analysis
            
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
            all_primary_labels.extend(primary_labels)
    
    # concatenate batches
    all_targets = np.vstack(all_targets)
    all_outputs = np.vstack(all_outputs)
    
    # calculate overall metrics
    val_loss = running_loss / len(val_loader)
    roc_auc = macro_roc_auc(all_targets, all_outputs)
    lwlrap_score = calculate_lwlrap(all_targets, all_outputs)
    
    # calculate per-class metrics if class names are provided
    class_metrics = None
    if class_names is not None:
        idx_to_label = {v: k for k, v in label_to_idx.items()}
        class_names_list = [idx_to_label[i] for i in range(len(idx_to_label))]
        class_metrics = calculate_per_class_metrics(all_targets, all_outputs, class_names_list)
        
        # group metrics by taxonomic class
        if 'class_name' in class_metrics.columns:
            taxonomic_metrics = class_metrics.groupby('class_name').agg({
                'auc': 'mean',
                'positive_samples': 'sum',
                'precision@0.5': 'mean',
                'recall@0.5': 'mean'
            }).reset_index()
            
            print("\nPerformance by Taxonomic Class:")
            print(taxonomic_metrics)

    return val_loss, roc_auc, lwlrap_score, class_metrics

def advanced_training(config, args):
    """Implement advanced training pipeline with multiple strategies."""
    # set random seeds for reproducibility
    seed_everything(config['training']['seed'])
    
    # set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Using device: {device}')
    
    # prepare data
    train_loader, val_loader, target_columns, label_to_idx = prepare_data(
        config, args.fold, args.debug, args.taxonomic
    )
    
    # create model
    model = create_model(args.model, len(target_columns), device)
    print(f"Created {args.model} model with {len(target_columns)} output classes")
    
    # define loss function with class weighting if specified
    if config['training'].get('use_weighted_loss', False):
        # calculate class weights based on frequency in training data
        print("Using weighted BCE loss...")
        pos_weight = torch.ones(len(target_columns), device=device)
        criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    else:
        criterion = nn.BCEWithLogitsLoss()
    
    # optimizers with different strategies
    if config['training'].get('use_adam', True):
        optimizer = optim.Adam(
            model.parameters(), 
            lr=config['training']['learning_rate'],
            weight_decay=float(config['training'].get('weight_decay', 1e-5))
        )
    else:
        optimizer = optim.SGD(
            model.parameters(),
            lr=config['training']['learning_rate'],
            momentum=0.9,
            weight_decay=float(config['training'].get('weight_decay', 1e-4))
        )
    
    # ;earning rate scheduler
    if config['training'].get('scheduler', 'reduce_on_plateau') == 'one_cycle':
        print("Using OneCycleLR scheduler")
        scheduler = optim.lr_scheduler.OneCycleLR(
            optimizer,
            max_lr=config['training']['learning_rate'],
            epochs=config['training']['num_epochs'],
            steps_per_epoch=len(train_loader)
        )
    else:
        print("Using ReduceLROnPlateau scheduler")
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='min', factor=0.5, patience=3, verbose=True
        )

    # initialize wandb if enabled
    if not args.no_wandb:
        experiment_name = f"{args.model}_fold{args.fold}"
        if args.taxonomic:
            experiment_name += "_taxonomic"
        if args.mixup:
            experiment_name += "_mixup"
            
        wandb.init(
            project=config['project']['name'],
            name=experiment_name,
            config={**config, **vars(args)}
        )
        wandb.watch(model, log='all')
    
    # training loop with early stopping
    best_roc_auc = 0.0
    best_model_path = None
    patience_counter = 0
    training_stats = defaultdict(list)
    
    # create output directory
    os.makedirs(config['paths']['models'], exist_ok=True)
    
    print(f"Starting training for {config['training']['num_epochs']} epochs")
    for epoch in range(config['training']['num_epochs']):
        print(f"Epoch {epoch+1}/{config['training']['num_epochs']}")
        start_time = time.time()
        
        # train for one epoch
        train_loss = train_epoch(
            model, 
            train_loader, 
            criterion, 
            optimizer, 
            device, 
            use_mixup=args.mixup,
            scheduler=scheduler if config['training'].get('scheduler', '') == 'one_cycle' else None
        )
        
        # validate
        val_loss, roc_auc, lwlrap, class_metrics = validate(
            model, val_loader, criterion, device, label_to_idx, target_columns
        )
        
        # update learning rate scheduler if not OneCycleLR
        if config['training'].get('scheduler', '') != 'one_cycle':
            scheduler.step(val_loss)
        
        # calculate epoch time
        epoch_time = time.time() - start_time
        
        # log metrics
        print(f'Epoch {epoch+1} - {epoch_time:.1f}s - Train Loss: {train_loss:.4f}, '
              f'Val Loss: {val_loss:.4f}, ROC-AUC: {roc_auc:.4f}, LWLRAP: {lwlrap:.4f}')
        
        # store stats
        training_stats['epoch'].append(epoch + 1)
        training_stats['train_loss'].append(train_loss)
        training_stats['val_loss'].append(val_loss)
        training_stats['roc_auc'].append(roc_auc)
        training_stats['lwlrap'].append(lwlrap)
        training_stats['lr'].append(optimizer.param_groups[0]['lr'])

        if not args.no_wandb:
            log_data = {
                "train_loss": train_loss,
                "val_loss": val_loss,
                "roc_auc": roc_auc,
                "lwlrap": lwlrap,
                "learning_rate": optimizer.param_groups[0]['lr'],
                "epoch_time": epoch_time
            }
            
            # log per-class metrics if available
            if class_metrics is not None:
                for _, row in class_metrics.iterrows():
                    class_name = row['class_name']
                    log_data[f"auc/{class_name}"] = row['auc']
                
            wandb.log(log_data)
        
        # save best model
        if roc_auc > best_roc_auc:
            best_roc_auc = roc_auc
            patience_counter = 0
            
            # create model filename with key parameters
            model_filename = f"{args.model}_fold{args.fold}"
            if args.taxonomic:
                model_filename += "_taxonomic"
            if args.mixup:
                model_filename += "_mixup"
            model_filename += f"_roc{roc_auc:.4f}.pth"
            
            best_model_path = os.path.join(config['paths']['models'], model_filename)
            torch.save({
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'roc_auc': roc_auc,
                'lwlrap': lwlrap,
                'config': config,
                'args': vars(args)
            }, best_model_path)
            print(f'Saved best model to {best_model_path}')
        else:
            patience_counter += 1
            if patience_counter >= config['training']['patience']:
                print(f'Early stopping after {epoch+1} epochs')
                break
    
    # save training history
    history_file = os.path.join(
        config['paths']['models'], 
        f"training_history_{args.model}_fold{args.fold}.json"
    )
    with open(history_file, 'w') as f:
        json.dump(training_stats, f)

    print(f'Best ROC-AUC: {best_roc_auc:.4f}')
    return best_model_path

def main():
    """Main function."""
    args = parse_args()
    config = load_config(args.config)
    
    best_model_path = advanced_training(config, args)
    
    print(f'Training complete. Best model saved to {best_model_path}')
    
if __name__ == "__main__":
    main()