"""
Script to process raw data into features for modeling.
"""

import os
import pandas as pd
import numpy as np
import librosa
import soundfile as sf
from pathlib import Path
import yaml
import logging
from tqdm import tqdm

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def load_config(config_path='configs/config.yaml'):
    """Load configuration from YAML file."""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config

def setup_directories(config):
    """Create necessary directories if they don't exist."""
    for path_name, path_value in config['paths'].items():
        if isinstance(path_value, dict):
            for subpath_name, subpath_value in path_value.items():
                os.makedirs(subpath_value, exist_ok=True)
        else:
            os.makedirs(path_value, exist_ok=True)
    logger.info("Directories created.")

def process_metadata(config):
    """Process train.csv and taxonomy.csv to create a unified metadata file."""
    raw_path = config['paths']['data']['raw']
    interim_path = config['paths']['data']['interim']

    # load train.csv
    train_df = pd.read_csv(os.path.join(raw_path, 'train.csv'))

    # load taxonomy.csv
    taxonomy_df = pd.read_csv(os.path.join(raw_path, 'taxonomy.csv'))

    # merge to get taxonomic information
    merged_df = pd.merge(
        train_df,
        taxonomy_df,
        left_on = 'primary_label',
        right_on = 'species_id',
        how = 'left'
    )

    # save processed metadata
    merged_df.to_csv(os.path.join(interim_path, 'metadata.csv'), index=False)
    logger.info(f"Processed metadata saved to {os.path.join(interim_path, 'metadata.csv')}")

    return merged_df

def main():
    """Main function to run the data processing pipeline."""
    logger.info('Starting data processing')

    # load configuration
    config = load_config()

    # setup directories
    setup_directories(config)

    # process metadata
    metadata_df = process_metadata(config)

    logger.info('Data processing completed successfully')

if __name__ == "__main__":
    main()