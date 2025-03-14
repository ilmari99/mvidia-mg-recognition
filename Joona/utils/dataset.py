from pathlib import Path
import numpy as np
from collections import defaultdict
import torch
from torch.utils.data import Dataset
from torchvision.transforms import v2
from PIL import Image

class ImigueDS(Dataset):
    def __init__(self, image_directory, transform=None, frame_count=10):
        """
        Args: 
            data_dir
            transformations
            maximum frame count
        """
        self.image_directory = Path(image_directory)
        self.transform = transform or v2.Compose([
            v2.ToImage(),
            v2.ToDtype(torch.float32, scale=True),
            v2.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])
        self.frame_count = frame_count

        # Find all frames
        self.img_paths = np.array([f for f in self.image_directory.glob("**/*.jpg")])
        self.videos = defaultdict(list)
        
        # Group frames per clip
        for path in self.img_paths:    
            parts = path.name.split(".")
            class_num = int(path.parent.name) - 1
            clip_num = parts[0]
            frame_num = parts[1]
            
            # Use clas and clip number as key
            key = (class_num, clip_num)
            self.videos[key].append((frame_num, path))
            
        # Sort the frames
        self.labels = []
        self.video_keys = []
        for key, frames in self.videos.items():
            frames.sort(key=lambda x: int(x[0]))  # Sort by frame number
            self.videos[key] = [path for _, path in frames]  # Keep only paths
            
            # Only add videos with frames
            if len(self.videos[key]) > 0:
                self.video_keys.append(key)
                self.labels.append(int(key[0]))
                
    def __len__(self):
        return len(self.video_keys)

    def __getitem__(self, idx):
        class_num, video_num = self.video_keys[idx]
        frame_paths = self.videos[(class_num, video_num)]
        
        # Handle varying number of frames
        if len(frame_paths) == self.frame_count:
            selected_paths = frame_paths
        elif len(frame_paths) > self.frame_count:
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
        video_tensor = torch.stack(frames, dim=0).squeeze()
        
        class_label = class_num
        assert class_label >= 0 and class_label <= 31, f"Invalid class label: {class_label}"
        
        return video_tensor, class_label
