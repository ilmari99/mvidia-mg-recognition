from sklearn.model_selection import  train_test_split
from collections import Counter
from torch.utils.data import Subset
import torch
import numpy as np
import timm
import torch.nn as nn
from sklearn.metrics import confusion_matrix
import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt
from torchvision.models.video import swin3d_b
from torchvision.models import swin_b

def stratified_split(dataset, num_frames = None):
    """
    Splits dataset into train and validation dataset using stratified splitting.
    Small classes (<2 clips) are used only in the training set
    """
    
    # Count class occurrences
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
    
    # Stratified split for common classes
    train_common, valid_common = train_test_split(
        common_indices,
        test_size=0.2,
        stratify=common_labels,
        random_state=42
    )

    # Add rare classes to training set
    train_idx = np.concatenate([train_common, rare_indices])

    if num_frames == 1:
        clip_to_indices = dataset.get_clip_indices()
        train_indices = []
        for clip_idx in train_idx:
            train_indices.extend(clip_to_indices[clip_idx])
        
        valid_indices = []
        for clip_idx in valid_common:
            valid_indices.extend(clip_to_indices[clip_idx])
        
    else:
        train_indices = train_idx
        valid_indices = valid_common
    
    
    # Take subset from the original dataset
    train_dataset = Subset(dataset, train_indices)
    val_dataset = Subset(dataset, valid_indices)
    
    return train_dataset, val_dataset

def create_backbone_model(args):
    if args.late_fusion or args.num_frames == 1:
        # Load image swin
        model = swin_b(weights='DEFAULT')
        num_features = model.head.in_features
        model.head = nn.Identity()
    else:
        # Load Video swin
        model = swin3d_b("Swin3D_B_Weights.KINETICS400_IMAGENET22K_V1")
        num_features = model.head.in_features
        model.head = nn.Identity()
    
    return model, num_features


def createConfusionMatrix(preds, labels):
    # constant for classes
    unique_classes = np.unique(np.concatenate((labels, preds)))

    classes = [str(i) for i in unique_classes] 
    cf_matrix = confusion_matrix(labels, preds)
    
    cf_matrix = cf_matrix.astype('float') / np.sum(cf_matrix, axis=1)[:, None]  # Row-wise normalization
    cf_matrix = np.nan_to_num(cf_matrix * 100)
    # Ensure correct DataFrame indexing
    df_cm = pd.DataFrame(cf_matrix, index=classes, columns=classes)

    plt.figure(figsize=(14, 10))
    plt.xlabel("Predicted Label", fontsize=12)
    plt.ylabel("True Label", fontsize=12)
    sns.heatmap(df_cm, annot=True, fmt=".1f", cmap="Blues")
    return sns.heatmap(df_cm, annot=True, fmt=".1f", cmap="Blues").get_figure()