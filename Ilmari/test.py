import os
import glob
import numpy as np
from collections import defaultdict
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import plotly.graph_objects as go
from ImigueDS import ImigueDS
from tqdm import tqdm

# Example usage
if __name__ == "__main__":
    dataset = ImigueDS(root_dir="./", frame_count=10)
    # Print a sample
    print(f"Number of videos: {len(dataset)}")
    video, label = dataset[0]
    print(f"Video shape: {video.shape}, label: {label}")
    class_distr = dataset.get_class_distribution()
    fig = go.Figure()
    fig.add_trace(go.Bar(x=list(class_distr.keys()), y=list(class_distr.values())))
    fig.update_layout(title="Class distribution", xaxis_title="Class", yaxis_title="Count")
    fig.show()

    dataloader = DataLoader(dataset, batch_size=4, shuffle=True, num_workers=4)



