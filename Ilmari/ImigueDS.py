import os
import glob
import torch
import numpy as np
from collections import defaultdict
from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image


class ImigueDS(Dataset):
    def __init__(self, image_directory, transform=None, frame_count=10):
        """
        Args:
            root_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied on a sample.
            frame_count (int): Fixed number of frames per video.
        """
        self.image_directory = image_directory
        self.transform = transform or transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])
        self.frame_count = frame_count
        
        # Get all image paths
        path = os.path.join(image_directory, "*", "*.jpg")
        self.image_paths = glob.glob(path)
        
        # Group images by video
        self.videos = defaultdict(list)
        for img_path in self.image_paths:
            # Parse path to extract class, video number, and frame number
            parts = os.path.basename(img_path).split(".")
            if len(parts) >= 3:
                class_num = int(os.path.basename(os.path.dirname(img_path)))
                video_num = int(parts[0])
                frame_num = int(parts[1])
                
                # Use (class_num, video_num) as key
                key = (class_num, video_num)
                self.videos[key].append((frame_num, img_path))
        
        # Sort frames in each video and create a list of video keys
        self.video_keys = []
        for key, frames in self.videos.items():
            frames.sort(key=lambda x: x[0])  # Sort by frame number
            self.videos[key] = [path for _, path in frames]  # Keep only paths
            
            # Only add videos with frames
            if len(self.videos[key]) > 0:
                self.video_keys.append(key)
    
    def get_class_distribution(self):
        """
        Returns the distribution of classes in the dataset without loading videos.
        
        Returns:
            dict: A dictionary mapping class indices (0-indexed) to counts
        """
        distribution = defaultdict(int)
        for class_num, _ in self.video_keys:
            # Convert to 0-indexed class label as used in __getitem__
            class_label = class_num - 1
            distribution[class_label] += 1
        return dict(distribution)
    
    def get_clip_length_distribution(self):
        """
        Returns the distribution of clip lengths in the dataset without loading videos.
        
        Returns:
            list: A list of clip lengths
        """
        clip_lengths = []
        for key in self.video_keys:
            clip_lengths.append(len(self.videos[key]))
        return clip_lengths
    
    def __len__(self):
        return len(self.video_keys)
    
    def __getitem__(self, idx):
        class_num, video_num = self.video_keys[idx]
        frame_paths = self.videos[(class_num, video_num)]
        
        # Handle videos with different number of frames
        if len(frame_paths) == self.frame_count:
            # If the video has exactly the right number of frames, use all
            selected_paths = frame_paths
        elif len(frame_paths) > self.frame_count:
            # If more frames than needed, sample uniformly
            indices = np.linspace(0, len(frame_paths) - 1, self.frame_count, dtype=int)
            selected_paths = [frame_paths[i] for i in indices]
        else:
            # If fewer frames than needed, repeat frames
            selected_paths = []
            for i in range(self.frame_count):
                selected_paths.append(frame_paths[i % len(frame_paths)])
        
        # Load frames
        frames = []
        for frame_path in selected_paths:
            image = Image.open(frame_path).convert('RGB')
            if self.transform:
                image = self.transform(image)
            frames.append(image)
        
        # Convert list of frames to tensor [num_frames, channels, height, width]
        video_tensor = torch.stack(frames, dim=0)
        
        # Class labels are 1-indexed in the folder structure but PyTorch expects 0-indexed
        class_label = class_num - 1
        assert class_label >= 0 and class_label <= 31, f"Invalid class label: {class_label}"
        
        return video_tensor, class_label