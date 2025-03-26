from torch.utils.data import DataLoader
import plotly.graph_objects as go
from ImigueDS import ImigueDS
from torchvision import transforms
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import random

def plot_class_distribution(dataset):
    """
    Plots the class distribution of a dataset normalized to percents.
    
    Args:
        dataset (ImigueDS): The dataset to analyze
    """
    class_distr = dataset.get_class_distribution()
    total = sum(class_distr.values())
    percentages = {cls: (count / total) * 100 for cls, count in class_distr.items()}

    fig = go.Figure()
    fig.add_trace(go.Bar(x=list(percentages.keys()), y=list(percentages.values())))
    fig.update_layout(
        title={'text': 'Class distribution (%)', 'font': {'size': 32}},
        xaxis={'title': {'text': 'Class', 'font': {'size': 28}}, 'tickfont': {'size': 14}},
        yaxis={'title': {'text': 'Percentage', 'font': {'size': 28}}, 'tickfont': {'size': 14}}
    )
    fig.show()
    
def show_clips(dataset, num_clips=3):
    """ Load num_clips from the dataset, and display them as GIFs
    Args:
        dataset (ImigueDS): The dataset to analyze
        num_clips (int): Number of clips to display
    """
    
    # Select N random indices
    indices = random.sample(range(len(dataset)), num_clips)
    
    animations = []
    
    for idx in indices:
        # Load the clip
        clip, label = dataset[idx]
        
        # Convert the clip to a list of images
        images = [frame.squeeze().numpy() for frame in clip]  # Squeeze to remove channel dimension
        
        # Create separate figure for each clip
        fig, ax = plt.subplots(figsize=(5, 5))
        ax.set_title(f"Class: {label}")
        
        # Create the animation frames
        frames = []
        im = ax.imshow(images[0], cmap='gray', animated=True)
        for frame in images[1:]:
            frames.append([ax.imshow(frame, cmap='gray', animated=True)])
        
        ani = animation.ArtistAnimation(fig, [[im]] + frames)
        animations.append(ani)  # Keep reference to prevent garbage collection
    
    plt.show()
    return animations

def show_clip_length_distribution(dataset):
    """
    Plots the distribution of clip lengths in the dataset normalized to percents.
    
    Args:
        dataset (ImigueDS): The dataset to analyze
    """
    clip_lengths = dataset.get_clip_length_distribution()
    fig = go.Figure()
    fig.add_trace(go.Histogram(x=clip_lengths, histnorm="percent"))
    fig.update_layout(
        title={'text': "Clip length distribution", 'font': {'size': 32}},
        xaxis={
            'title': {'text': "Clip length", 'font': {'size': 28}},
            'range': [0, 100]
        },
        yaxis={'title': {'text': "Percentage (%)", 'font': {'size': 28}}}
    )
    fig.show()
    
# Example usage
if __name__ == "__main__":
    # Resize, normalize, and convert to grayscale
    transform = transforms.Compose([
        transforms.Resize((128,128)),
        transforms.Grayscale(num_output_channels=1),
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])
    dataset = ImigueDS(image_directory="./imigue", frame_count=20, transform=transform)
    plot_class_distribution(dataset)
    show_clips(dataset)
    show_clip_length_distribution(dataset)
