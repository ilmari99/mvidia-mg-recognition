from pathlib import Path
import numpy as np
from collections import defaultdict
import torch
from torch.utils.data import Dataset
from torchvision.transforms import v2
from PIL import Image

class ImigueVideoDS(Dataset):
    def __init__(self, image_source, transform=None, frame_count=10):
        """
        Args: 
            image_source: Directory path or training table
            transform: Transformations to apply
            frame_count: Maximum frame count per video
            from_table: Boolean flag indicating if image_source is a training table
        """
        self.transform = transform or v2.Compose([
            v2.ToImage(),
            v2.ToDtype(torch.float32, scale=True),
            v2.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])
        self.frame_count = frame_count
        self.videos = defaultdict(list)

        # Image paths are collected from a directory
        image_directory = Path(image_source)
        self.img_paths = np.array([f for f in image_directory.glob("**/*.jpg")])

        # Group frames per video clip
        for path in self.img_paths:    
            parts = path.name.split(".")
            class_num = int(path.parent.name) - 1
            clip_num = parts[0]
            frame_num = parts[1]
            key = (class_num, clip_num)
            self.videos[key].append((frame_num, path))
        
        # Sort frames and filter videos
        self.labels = []
        self.video_keys = []
        for key, frames in self.videos.items():
            frames.sort(key=lambda x: int(x[0]))  # Sort by frame number
            self.videos[key] = [path for _, path in frames]  # Keep only paths
            if self.videos[key]:
                self.video_keys.append(key)
                self.labels.append(int(key[0]))
    
    def __len__(self):
        return len(self.video_keys)

    def __getitem__(self, idx):
        class_num, video_num = self.video_keys[idx]
        frame_paths = self.videos[(class_num, video_num)]
        
        # Handle varying frame counts
        if len(frame_paths) == self.frame_count:
            selected_paths = frame_paths
        elif len(frame_paths) > self.frame_count:
            indices = np.linspace(0, len(frame_paths) - 1, self.frame_count, dtype=int)
            selected_paths = [frame_paths[i] for i in indices]
        else:
            selected_paths = [frame_paths[i % len(frame_paths)] for i in range(self.frame_count)]
        
        # Load frames
        frames = [self.transform(Image.open(path).convert('RGB')) for path in selected_paths]
        video_tensor = torch.stack(frames, dim=0).squeeze()
        
        return video_tensor, class_num



class ImigueVideoDS_3lc(Dataset):
    def __init__(self, table, transform=None, frame_count=10):
        """
        Args: 
            image_source: Directory path or training table
            transform: Transformations to apply
            frame_count: Maximum frame count per video
            from_table: Boolean flag indicating if image_source is a training table
        """
        self.transform = transform or v2.Compose([
            v2.ToImage(),
            v2.ToDtype(torch.float32, scale=True),
            v2.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])
        self.frame_count = frame_count
        self.videos = defaultdict(list)

        # Image paths are extracted from a training table
        self.img_paths = np.array(table.get_column("image"))
        self.labels = np.array(table.get_column("label"))
        self.clip_nums = np.array(table.get_column("clip_num"))
        self.frame_nums = np.array(table.get_column("frame_num"))

        assert len(self.img_paths) == len(self.labels) == len(self.clip_nums) == len(self.frame_nums), \
            "Table columns must have the same length."

        for i in range(len(self.img_paths)):
            key = (self.labels[i], self.clip_nums[i])
            self.videos[key].append((self.frame_nums[i], self.img_paths[i]))

        self.video_keys = []
        for key, frames in self.videos.items():
            # Sort frames by frame number
            frames.sort(key=lambda x: x[0]) 
            self.videos[key] = [path for _, path in frames]  # Keep only paths
            
            # Keep only non-empty videos
            if self.videos[key]:
                self.video_keys.append(key)
            
    def __len__(self):
        return len(self.video_keys)

    def __getitem__(self, idx):
        class_num, video_num = self.video_keys[idx]
        frame_paths = self.videos[(class_num, video_num)]
        
        # Handle varying frame counts
        if len(frame_paths) == self.frame_count:
            selected_paths = frame_paths
        elif len(frame_paths) > self.frame_count:
            indices = np.linspace(0, len(frame_paths) - 1, self.frame_count, dtype=int)
            selected_paths = [frame_paths[i] for i in indices]
        else:
            selected_paths = [frame_paths[i % len(frame_paths)] for i in range(self.frame_count)]
        
        # Load frames
        frames = [self.transform(Image.open(path).convert('RGB')) for path in selected_paths]
        video_tensor = torch.stack(frames, dim=0).squeeze()
        
        return video_tensor, torch.tensor(class_num, dtype=torch.long)

# Not used
class ImigueImageDS(Dataset):
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

        # Group frames per clip
        self.videos = defaultdict(list)
        for path in self.img_paths:    
            parts = path.name.split(".")
            class_num = int(path.parent.name) - 1
            clip_num = parts[0]
            frame_num = parts[1]
            
            # Use clas and clip number as key
            key = (class_num, clip_num)
            self.videos[key].append((frame_num, path))
            
        # Sort the frames
        self.images = []
        self.clips = []  # Store clip ID for each image
        self.labels = []  # Store labels for stratified split
        self.clip_to_class = {}  # Map clip ID to class
        
        clip_id = 0
        for key, frames in self.videos.items():
            class_num = key[0]
            self.clip_to_class[clip_id] = class_num
            self.labels.append(class_num)  # Add label for the clip
            
            for _, path in sorted(frames, key=lambda x: int(x[0])):
                self.images.append((path, class_num, clip_id))
            
            clip_id += 1
                
    def __len__(self):
        return len(self.images)
    
    def get_clip_indices(self):
        """
        Returns a mapping of clip IDs to image indices
        This is used to ensure all frames from the same clip stay together
        """
        clip_to_indices = defaultdict(list)
        for i, (_, _, clip_id) in enumerate(self.images):
            clip_to_indices[clip_id].append(i)
        return clip_to_indices

    def __getitem__(self, idx):
        image_path, class_label, _ = self.images[idx]
        
        # Load and transform image
        image = Image.open(image_path).convert('RGB')
        if self.transform:
            image = self.transform(image)
        
        assert class_label >= 0 and class_label <= 31, f"Invalid class label: {class_label}"
        return image, class_label
    
