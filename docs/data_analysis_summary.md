# BirdCLEF+ 2025: Data Analysis Summary

This document summarizes the findings from our exploratory data analysis for the BirdCLEF+ 2025 competition. The analysis focuses on understanding the dataset characteristics, identifying potential challenges, and informing our feature engineering and modeling strategies.

## 1. Dataset Overview

The training dataset consists of 28,564 audio recordings across 206 species from four taxonomic classes:
- **Aves (birds)**: 146 species
- **Amphibia**: 34 species
- **Insecta**: 17 species
- **Mammalia**: 9 species

Data is sourced from three collections: Xeno-canto (XC), iNaturalist (iNat), and Colombian Sound Archive (CSA).

### 1.1 File Counts

- **Training audio files**: 28,564 individual species recordings
- **Training soundscapes**: 9,726 unlabeled ambient recordings

## 2. Class Imbalance Analysis

We observed significant imbalance across taxonomic classes and individual species:

### 2.1 Taxonomic Class Distribution

| Class | Total Recordings | Species Count | Min Recordings | Max Recordings | Median Recordings | Mean Recordings |
|-------|------------------|---------------|----------------|----------------|-------------------|-----------------|
| Aves | 27,648 | 146 | 6 | 990 | 129.5 | 189.37 |
| Amphibia | 583 | 34 | 2 | 82 | 6.0 | 17.15 |
| Insecta | 155 | 17 | 2 | 33 | 5.0 | 9.12 |
| Mammalia | 178 | 9 | 2 | 108 | 5.0 | 19.78 |

Birds (Aves) comprise 96.8% of all recordings, highlighting a severe class imbalance at the taxonomic level.

### 2.2 Species-Level Distribution

- 39 underrepresented species have fewer than 10 recordings each
- Among underrepresented species:
  - 21 are amphibians
  - 11 are insects
  - 5 are mammals
  - 1 is a bird
- The most represented species have hundreds of recordings

**Implications**: We'll need robust strategies to handle class imbalance, including:
- Oversampling rare species
- Data augmentation techniques
- Class-weighted loss functions
- Hierarchical classification approaches

## 3. Audio Duration Analysis

### 3.1 Overall Duration Statistics (seconds)

- **Mean**: 35.35
- **Median**: 20.98
- **Min**: 0.54
- **Max**: 1774.39 (≈29.6 minutes)
- **Standard Deviation**: 50.61

### 3.2 Duration by Taxonomic Class

| Class | Mean Duration (s) | Median Duration (s) | Min Duration (s) | Max Duration (s) |
|-------|-------------------|---------------------|------------------|------------------|
| Amphibia | 30.36 | 16.56 | 0.54 | 389.77 |
| Aves | 35.03 | 21.01 | 0.55 | 1774.39 |
| Insecta | 113.76 | 99.42 | 0.99 | 896.58 |
| Mammalia | 33.83 | 21.76 | 1.02 | 218.78 |

Notably, insect recordings are significantly longer on average than those of other taxonomic groups.

### 3.3 Duration Edge Cases

- 24 very short recordings (<1 second)
- 4,282 very long recordings (>60 seconds)

**Implications**: Our preprocessing pipeline will need to:
- Handle variable-length audio effectively
- Standardize segment length for training
- Consider strategies for very long recordings (chunking, feature aggregation)
- Potentially filter or special processing for very short recordings

## 4. Secondary Label Analysis

### 4.1 Co-occurring Species

- 2,679 recordings (9.38%) have secondary labels
- 132 unique species appear as secondary labels
- Most recordings with secondary labels contain 1-2 additional species

### 4.2 Co-occurrence Patterns

We created a network visualization of species co-occurrences, which revealed:
- Natural ecological associations between species
- Potential confounding factors for model training
- Community structures that might inform habitat-based features

**Implications**: These findings suggest:
- Our model should account for multi-label classification
- Secondary labels can provide additional training signal
- Co-occurrence patterns may inform post-processing strategies

## 5. Acoustic Characteristics Analysis

We extracted several acoustic features to understand the sonic profiles across taxonomic groups:

### 5.1 Extracted Features

- **Temporal features**: Duration, zero-crossing rate, energy, tempo
- **Spectral features**: Spectral centroid, bandwidth, rolloff
- **Cepstral features**: 13 MFCCs (Mel-frequency cepstral coefficients)

### 5.2 Principal Component Analysis

PCA of acoustic features revealed:
- Distinct clustering patterns by taxonomic class
- Overlap between some bird and amphibian calls
- Insects showing the most acoustic distinctiveness
- Top contributing features to PC1 include specific MFCCs and spectral features

**Implications**: These findings will inform:
- Feature engineering strategies
- Model architecture design
- Potential for hierarchical or specialized classifiers by taxonomic group

## 6. Key Challenges and Recommendations

Based on our analysis, we've identified the following key challenges:

### 6.1 Data Challenges

1. **Severe class imbalance** across taxonomic groups and species
2. **Variable audio quality** and recording conditions
3. **Inconsistent audio durations** across species
4. **Limited samples** for many non-avian species
5. **Multiple species** in some recordings

### 6.2 Recommended Strategies

1. **Data Augmentation**:
   - Time shifting, pitch shifting
   - Noise injection, masking
   - Mixup and spectrogram augmentation

2. **Model Architecture**:
   - Hierarchical classification (taxonomy-aware)
   - Ensemble of specialized models by taxonomic group
   - Self-supervised pretraining on unlabeled soundscapes

3. **Training Approach**:
   - Class-weighted loss functions
   - Multi-label training incorporating secondary labels
   - Curriculum learning starting with well-represented species

4. **Feature Engineering**:
   - Tailored feature extraction for different taxonomic groups
   - Time-frequency representations at multiple scales
   - Bioacoustic-specific features relevant to each class

5. **Evaluation**:
   - Stratified cross-validation by taxonomic group
   - Separate performance tracking for rare species
   - Custom thresholds optimized by species