# BirdCLEF+ 2025 - Multi-Species Acoustic Monitoring

This repository contains our solution for the [BirdCLEF+ 2025](https://www.kaggle.com/competitions/birdclef-2025) competition, which focuses on identifying various taxonomic groups (birds, amphibians, mammals, insects) from soundscape recordings in Colombia's El Silencio Natural Reserve.

## Project Overview

The competition aims to develop machine learning models to identify species based on their acoustic signatures in passive acoustic monitoring data. The real-world impact is to support biodiversity monitoring for conservation efforts in Colombia's Magdalena Valley.

## Repository Structure

```
├── configs/            # Configuration files
├── data/               # Data directory
│   ├── raw/            # Raw data
│   ├── interim/        # Intermediate processed data
│   └── processed/      # Fully processed data
├── docs/               # Documentation
├── notebooks/          # Jupyter notebooks
├── outputs/            # Output files
│   ├── models/         # Saved models
│   └── submissions/    # Competition submissions
├── src/                # Source code
│   ├── data/           # Data processing scripts
│   ├── features/       # Feature engineering scripts
│   ├── models/         # Model implementation
│   └── visualization/  # Visualization utilities
└── tests/              # Tests for the code
```

## Getting Started

### Prerequisites

- Python 3.8+
- pip or conda

### Installation

1. Clone this repository

```bash
git clone https://github.com/preston-g-bied/birdclef-2025.git
cd birdclef-2025
```

2. Set up the environment

```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
pip install -r requirements.txt
```

### Data Setup

Download competition data from Kaggle

```bash
kaggle competitions download -c birdclef-2025
unzip birdclef-2025.zip -d data/raw/
```

## Baseline Model

This repository includes a baseline CNN model for multi-species audio classification:

### Model Architecture
- Simple CNN with 5 convolutional blocks
- Global average pooling and fully connected classification layer
- Multi-label classification with BCE loss

### Training the Model

To train the baseline model:

```bash
# Train on fold 0 with default settings
python src/models/train_model.py --config configs/config.yaml --fold 0

# Run with debugging on a small dataset
python src/models/train_model.py --config configs/config.yaml --debug

# Train without Weights & Biases logging
python src/models/train_model.py --config configs/config.yaml --no_wandb
```