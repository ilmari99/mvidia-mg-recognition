from sklearn.model_selection import  train_test_split
from collections import Counter
from torch.utils.data import Subset
import numpy as np
import timm
from transformers import AutoImageProcessor, VivitConfig, VivitForVideoClassification
import torch.nn as nn
from sklearn.metrics import confusion_matrix
import seaborn as sn
import pandas as pd
import matplotlib.pyplot as plt
from torchvision.models.video import swin3d_b

def stratified_split(num_frames, dataset):
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



def create_model(args):
    if args.model == "google/vivit-b-16x2-kinetics400":
        
        config = VivitConfig.from_pretrained(args.model)
        config.num_frames = args.num_frames

        # First create model with new config (uninitialized)
        model = VivitForVideoClassification(config)

        # Now load pre-trained weights, allowing for mismatched sizes
        pretrained_dict = VivitForVideoClassification.from_pretrained(
            args.model, 
            ignore_mismatched_sizes=True
        ).state_dict()

        # Filter out position embeddings which depend on sequence length
        # but keep other weights that can be reused
        model_dict = model.state_dict()
        pretrained_dict = {k: v for k, v in pretrained_dict.items() if 'position_embeddings' not in k and k in model_dict and v.shape == model_dict[k].shape}

        # Update model with filtered pre-trained weights
        model_dict.update(pretrained_dict)
        model.load_state_dict(model_dict)

        # Replace classifier head
        # model.classifier = nn.Linear(model.config.hidden_size, args.num_classes)
        model.classifier = nn.Identity()
        
    else:
        # Load the model using timm
        model = timm.create_model(args.model, pretrained=True, num_classes=0, drop_path_rate=0.2)
        
        # Create transform using timm utilities
        # transforms = create_transform(**resolve_data_config(model.pretrained_cfg, model=model))
        transforms = 0
    
    return model


def createConfusionMatrix(preds, labels):
    # constant for classes
    classes = [str(i) for i in range(max(labels))]
    
    # Build confusion matrix
    cf_matrix = confusion_matrix(labels, preds)
    df_cm = pd.DataFrame(cf_matrix, index=[i for i in classes],
                         columns=[i for i in classes])
    plt.figure(figsize=(12, 7))    
    return sn.heatmap(df_cm, annot=True, cmap="Blues").get_figure()