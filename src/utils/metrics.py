"""
Evaluation metrics for BirdCLEF+ 2025 competition.
"""

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

def calculate_lwlrap(truth, scores):
    """
    Calculate the label-weighted label-ranking average precision.
    
    This is the competition metric for BirdCLEF.
    
    Args:
        truth: Binary indicator matrix with shape (num_samples, num_classes)
        scores: Score matrix with shape (num_samples, num_classes)
    
    Returns:
        lwlrap score
    """
    # initialize variables
    samples_num = truth.shape[0]
    classes_num = truth.shape[1]

    # initialize lwlrap to 0
    lwlrap = 0

    # iterate over all samples
    for i in range(samples_num):
        # get indices of true classes for this sample
        truth_indices = np.where(truth[i, :] == 1)[0]

        # number of true classes for this sample
        num_true = len(truth_indices)

        # if no true classes, continue to next sample
        if num_true == 0:
            continue

        # sort the predicted scores in descending order
        sorted_indices = np.argsort(scores[i, :])[::-1]

        # initialize variables for this sample
        rank = 0
        true_positives = 0
        sum_precision = 0.0

        # iterate through predictions in order
        for j in range(classes_num):
            rank += 1

            # if this prediction is correct (true positive)
            if sorted_indices[j] in truth_indices:
                true_positives += 1
                precision_at_rank = true_positives / rank
                sum_precision += precision_at_rank

        # normalize by the number of true classes
        lwlrap += sum_precision / num_true

    # normalize by the number of samples
    lwlrap /= samples_num

    return lwlrap

def macro_roc_auc(truth, scores):
    """
    Calculate macro-averaged ROC-AUC score.
    
    This is similar to the competition metric but uses sklearn implementation.
    
    Args:
        truth: Binary indicator matrix with shape (num_samples, num_classes)
        scores: Score matrix with shape (num_samples, num_classes)
    
    Returns:
        Macro-averaged ROC-AUC score
    """
    try:
        # calculate ROC-AUC for classes that have at least one positive sample
        valid_classes = []
        aucs = []

        for i in range(truth.shape[1]):
            # check if this class has at least one positive and one negative sample
            if np.sum(truth[:, i] == 1) > 0 and np.sum(truth[:, i] == 0) > 0:
                valid_classes.append(i)
                class_auc = roc_auc_score(truth[:, i], scores[:, i])
                aucs.append(class_auc)

        # return macro-averaged AUC
        if len(aucs) > 0:
            return np.mean(aucs)
        else:
            return 0.0
        
    except ValueError as e:
        # handle edge cases
        print(f'Error calculating ROC-AUC: {e}')
        return 0.0
    
def calculate_per_class_metrics(truth, scores, classes=None):
    """
    Calculate per-class performance metrics.
    
    Args:
        truth: Binary indicator matrix with shape (num_samples, num_classes)
        scores: Score matrix with shape (num_samples, num_classes)
        classes: List of class names (optional)
    
    Returns:
        DataFrame with per-class metrics
    """
    results = []

    for i in range(truth.shape[1]):
        # skip classes with no positive samples
        if np.sum(truth[:, i]) == 0:
            continue

        try:
            class_auc = roc_auc_score(truth[:, i], scores[:, i])
        except ValueError:
            class_auc = np.nan

        # calculate precision at different thresholds
        thresholds = [0.3, 0.5, 0.7]
        precisions = {}
        recalls = {}

        for threshold in thresholds:
            preds = (scores[:, i] >= threshold).astype(int)

            # true positives
            tp = np.sum((preds == 1) & (truth[:, i] == 1))
            # false positives
            fp = np.sum((preds == 1) & (truth[:, i] == 0))
            # false negatives
            fn = np.sum((preds == 0) & (truth[:, i] == 1))

            # calculate precision and recall
            precision = tp / (tp + fp) if (tp + fp) > 0 else 0
            recall = tp / (tp + fn) if (tp + fn) > 0 else 0

            precisions[threshold] = precision
            recalls[threshold] = recall

        class_info = {
            'class_idx': i,
            'class_name': classes[i] if classes is not None else f'Class {i}',
            'auc': class_auc,
            'positive_samples': np.sum(truth[:, i])
        }

        # add precision and recall at different thresholds
        for threshold in thresholds:
            class_info[f'precision@{threshold}'] = precisions[threshold]
            class_info[f'recall@{threshold}'] = recalls[threshold]
        
        results.append(class_info)

    return pd.DataFrame(results)