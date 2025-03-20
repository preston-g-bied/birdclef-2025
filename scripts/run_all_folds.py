"""
Script to run multiple cross-validation fold sequentially."""

import os
import sys
import argparse
import subprocess
import time
from datetime import datetime

def parse_args():
    parser = argparse.ArgumentParser(description='Run multiple CV folds sequentially')
    parser.add_argument('--config', type=str, default='configs/config.yaml',
                        help='Path to config file')
    parser.add_argument('--num_folds', type=int, default=5,
                        help='Number of folds to run')
    parser.add_argument('--starting_fold', type=int, default=0,
                        help='Fold to start at')
    parser.add_argument('--no_wandb', action='store_true',
                        help='Disable wandb logging')
    parser.add_argument('--debug', action='store_true',
                        help='Run in debug mode with a small dataset')
    parser.add_argument('--sleep', type=int, default=10,
                        help='Seconds to sleep between folds')
    return parser.parse_args()

def main():
    args = parse_args()

    start_time = datetime.now()
    print(f'Starting multi-fold training at {start_time}')
    print(f'Will run folds {args.starting_fold}-{args.num_folds-1} sequentially')

    results = []

    for fold in range(args.starting_fold, args.num_folds):
        print(f'\n{'='*60}')
        print(f'Starting fold {fold} at {datetime.now()}')
        print(f'{'='*60}\n')

        # build the command
        cmd = [
            'python3', 'src/models/train_model.py',
            '--config', args.config,
            '--fold', str(fold)
        ]

        if args.no_wandb:
            cmd.append('--no_wandb')

        if args.debug:
            cmd.append('--debug')

        # run the command and capture output
        try:
            output = subprocess.check_output(cmd, stderr=subprocess.STDOUT, universal_newlines=True)

            # extract best ROC-AUC from output
            for line in output.split('\n'):
                if 'Best ROC-AUC:' in line:
                    roc_auc = float(line.split(':')[1].strip())
                    results.append((fold, roc_auc))
                    break

            print(output)
            print(f'Completed fold {fold}')

            # sleep between folds
            if fold < args.num_folds - 1:
                print(f'Sleepinf for {args.sleep} seconds before next fold...')
                time.sleep(args.sleep)

        except subprocess.CalledProcessError as e:
            print(f'Error in fold {fold}:')
            print(e.output)
            results.append((fold, 'FAILED'))

    # print summary
    end_time = datetime.now()
    duration = end_time - start_time
    
    print('\n' + '='*60)
    print('TRAINING COMPLETE')
    print(f'Started: {start_time}')
    print(f'Finished: {end_time}')
    print(f'Duration: {duration}')
    print('\nResults summary:')

    for fold, score in results:
        print(f'Fold {fold}: {score}')

    if all(isinstance(score, float) for _, score in results):
        avg_score = sum(score for _, score in results) / len(results)
        print(f'\nAverage ROC-AUC across all folds: {avg_score:.4f}')

    print('='*60)

if __name__ == "__main__":
    main()