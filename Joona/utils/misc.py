from sklearn.model_selection import  train_test_split
from collections import Counter
from torch.utils.data import Subset
import numpy as np

def stratified_split(dataset):
    """
    Splits dataset into train and validation dataset using stratified splitting.
    Small (<2 samples) are used in training set
    """
    
    # Count class occurences
    labels = dataset.labels
    label_counts = Counter(labels)
    
    # Find common and rare classes
    rare_samples = [label for label, count in label_counts.items() if count < 2]
    common_mask = np.array([label not in rare_samples for label in labels])
    rare_mask = ~common_mask

    # Get indices for common and rare samples
    common_indices = np.where(common_mask)[0]
    rare_indices = np.where(rare_mask)[0]

    # Find classes for common indices
    common_labels = [labels[i] for i in common_indices]
    train_common, valid_common = train_test_split(
        common_indices,
        test_size=0.2,
        stratify=common_labels,
        random_state=42
    )

    # Add rare classes to training set
    train_idx = np.concatenate([train_common, rare_indices])

    # Take subset from the original dataset
    train_dataset = Subset(dataset, train_idx)
    val_dataset = Subset(dataset, valid_common)
    return train_dataset, val_dataset