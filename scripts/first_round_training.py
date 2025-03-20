#!/usr/bin/env python
"""
First Round Training Script for BirdCLEF+ 2025

This script automates the first round of training for the BirdCLEF+ competition:
- Trains three model architectures: Baseline CNN, EfficientNet, and Attention CNN
- Uses 5-fold cross-validation (folds 0-4)
- Enables taxonomic features and mixup augmentation
- Results in 15 distinct models (3 architectures × 5 folds)

Example usage:
    python first_round_training.py --config configs/advanced_config.yaml
    python first_round_training.py --config configs/advanced_config.yaml --debug
    python first_round_training.py --config configs/advanced_config.yaml --no_wandb
    python first_round_training.py --config configs/advanced_config.yaml --models efficientnet attention
"""

import os
import sys
import argparse
import subprocess
import yaml
import time
import json
from datetime import datetime
from pathlib import Path
import pandas as pd

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='First round training for BirdCLEF+ 2025')
    parser.add_argument('--config', type=str, default='configs/advanced_config.yaml',
                        help='Path to config file')
    parser.add_argument('--models', nargs='+', 
                        choices=['baseline', 'efficientnet', 'attention'],
                        default=['baseline', 'efficientnet', 'attention'],
                        help='Models to train (default: all three)')
    parser.add_argument('--folds', type=int, nargs='+', default=[0, 1, 2, 3, 4],
                        help='Folds to use for cross-validation (default: 0-4)')
    parser.add_argument('--debug', action='store_true',
                        help='Run in debug mode with a small dataset and fewer folds')
    parser.add_argument('--no_wandb', action='store_true',
                        help='Disable wandb logging')
    parser.add_argument('--output_dir', type=str, default='outputs/results/first_round',
                        help='Directory to save results summary')
    return parser.parse_args()

def load_config(config_path):
    """Load configuration from YAML file."""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config

def create_run_command(args, model, fold):
    """Create the command to run advanced_train.py with appropriate arguments."""
    cmd = [
        'python', 'src/models/advanced_train.py',
        '--config', args.config,
        '--model', model,
        '--fold', str(fold),
        '--taxonomic',  # enable taxonomic features
        '--mixup'       # enable mixup augmentation
    ]
    
    # add optional flags
    if args.debug:
        cmd.append('--debug')
    if args.no_wandb:
        cmd.append('--no_wandb')
    
    return cmd

def run_training(cmd):
    """Execute the training command and return process result."""
    print(f"\n{'='*80}")
    print(f"Running command: {' '.join(cmd)}")
    print(f"{'='*80}\n")
    
    process = subprocess.run(cmd, capture_output=True, text=True)
    
    if process.returncode != 0:
        print(f"Error running command: {' '.join(cmd)}")
        print(f"Error output: {process.stderr}")
    
    return process

def extract_results(process_output):
    """Extract ROC-AUC and other metrics from the training output."""
    lines = process_output.stdout.split('\n')
    
    # extract metrics
    metrics = {
        'best_roc_auc': None,
        'best_model_path': None,
        'early_stopping_epoch': None
    }
    
    for line in lines:
        if 'Best ROC-AUC:' in line:
            try:
                metrics['best_roc_auc'] = float(line.split(':')[1].strip())
            except:
                pass
        elif 'Saved best model to' in line:
            try:
                metrics['best_model_path'] = line.split('Saved best model to')[1].strip()
            except:
                pass
        elif 'Early stopping after' in line:
            try:
                metrics['early_stopping_epoch'] = int(line.split('after')[1].split('epochs')[0].strip())
            except:
                pass
    
    return metrics

def main():
    """Main execution function."""
    start_time = time.time()
    args = parse_args()
    config = load_config(args.config)
    
    # adjust for debug mode
    if args.debug:
        print("Debug mode: using only first two folds")
        args.folds = args.folds[:2]  # Use only first two folds in debug mode
    
    # create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # track results
    results = []
    
    # training loop for all models and folds
    total_runs = len(args.models) * len(args.folds)
    current_run = 1
    
    for model in args.models:
        for fold in args.folds:
            print(f"\n\n{'#'*100}")
            print(f"# Training {model} with fold {fold} ({current_run}/{total_runs})")
            print(f"{'#'*100}\n")
            
            # create and run the training command
            cmd = create_run_command(args, model, fold)
            process = run_training(cmd)
            
            # extract and store results
            run_metrics = extract_results(process)
            
            results.append({
                'model': model,
                'fold': fold,
                'best_roc_auc': run_metrics['best_roc_auc'],
                'best_model_path': run_metrics['best_model_path'],
                'early_stopping_epoch': run_metrics['early_stopping_epoch'],
                'success': process.returncode == 0
            })
            
            # save intermediate results in case of failure
            results_df = pd.DataFrame(results)
            results_df.to_csv(os.path.join(args.output_dir, 'training_results.csv'), index=False)
            
            current_run += 1
    
    # create summary report
    results_df = pd.DataFrame(results)
    
    # calculate average ROC-AUC by model
    model_summary = results_df.groupby('model')['best_roc_auc'].agg(['mean', 'std', 'min', 'max']).reset_index()
    model_summary.columns = ['model', 'mean_roc_auc', 'std_roc_auc', 'min_roc_auc', 'max_roc_auc']
    
    # find best model-fold combination
    if len(results_df) > 0 and results_df['best_roc_auc'].notna().any():
        best_model_idx = results_df['best_roc_auc'].idxmax()
        best_model_data = results_df.iloc[best_model_idx]
        best_model = f"{best_model_data['model']} (fold {best_model_data['fold']})"
        best_roc_auc = best_model_data['best_roc_auc']
    else:
        best_model = "No valid models"
        best_roc_auc = float('nan')
    
    # calculate total training time
    total_time = time.time() - start_time
    hours, remainder = divmod(total_time, 3600)
    minutes, seconds = divmod(remainder, 60)
    
    # create summary report
    summary = {
        'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        'config_file': args.config,
        'models_trained': args.models,
        'folds_used': args.folds,
        'total_models': len(results),
        'successful_models': sum(results_df['success']),
        'best_model': best_model,
        'best_roc_auc': best_roc_auc,
        'total_training_time': f"{int(hours)}h {int(minutes)}m {int(seconds)}s",
        'model_performance': model_summary.to_dict('records')
    }
    
    # save summary to JSON
    with open(os.path.join(args.output_dir, 'training_summary.json'), 'w') as f:
        json.dump(summary, f, indent=4)
    
    # save detailed results to CSV
    results_df.to_csv(os.path.join(args.output_dir, 'training_results.csv'), index=False)
    
    # print summary
    print("\n\n" + "="*50)
    print(f"Training Summary")
    print("="*50)
    print(f"Total models trained: {len(results)}")
    print(f"Successful models: {sum(results_df['success'])}")
    print(f"Best model: {best_model} (ROC-AUC: {best_roc_auc:.4f})")
    print(f"Total training time: {int(hours)}h {int(minutes)}m {int(seconds)}s")
    print("\nModel Performance Summary:")
    print(model_summary.to_string(index=False))
    print("\nResults saved to:", args.output_dir)
    print("="*50)

if __name__ == "__main__":
    main()