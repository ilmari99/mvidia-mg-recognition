import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from torch.utils.tensorboard import SummaryWriter
from ImigueDS import ImigueDS
from torchvision import transforms
import torchvision.models.video as models
import random
from torchvision.transforms import InterpolationMode
from tqdm import tqdm

class VideoClassifier(nn.Module):
    def __init__(self, num_classes=32, freeze_backbone=True):
        super(VideoClassifier, self).__init__()
        # Load a pre-trained 3D ResNet-18 model
        #self.backbone = models.r3d_18(weights=models.R3D_18_Weights.DEFAULT)
        self.backbone = models.swin3d_t(weights=models.Swin3D_T_Weights.DEFAULT)
        # Replace the final head layer with one that has num_classes outputs.
        self.backbone.head = nn.Linear(self.backbone.head.in_features, num_classes)
        
        self.freeze_backbone = freeze_backbone
        
        if freeze_backbone:
            for param in self.backbone.parameters():
                param.requires_grad = False
            for param in self.backbone.head.parameters():
                param.requires_grad = True

    def forward(self, x):
        # The dataset returns input in the shape [batch, frames, channels, height, width].
        # Pre-trained models expect [batch, channels, frames, height, width], so we permute.
        x = x.permute(0, 2, 1, 3, 4)
        x = self.backbone(x)
        return x
    
    def eval(self):
        self.backbone.eval()
        return self
    
    def train(self, mode=True):
        # If back bone is frozen, only train the final layer
        self.backbone.eval()
        if self.freeze_backbone:
            self.backbone.head.train()
        else:
            self.backbone.train()
        return self

if __name__ == "__main__":
    torch.manual_seed(42)
    random.seed(42)
    transform = transforms.Compose([
        transforms.Resize(256, interpolation=InterpolationMode.BILINEAR),
        transforms.CenterCrop((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    
    dataset = ImigueDS(image_directory="./imigue", frame_count=12, transform=transform)
    dataset_size = len(dataset)
    train_size = int(0.85 * dataset_size)
    val_size = dataset_size - train_size
    train_set, val_set = random_split(dataset, [train_size, val_size])
    
    # Increase batch size to 32
    train_loader = DataLoader(train_set, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_set, batch_size=32, shuffle=False)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = VideoClassifier(num_classes=32).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-4)
    
    # Initialize TensorBoard writer
    writer = SummaryWriter(log_dir="runs", flush_secs=30)
    
    num_epochs = 20  # Adjust as needed
    global_step = 0

    best_val_accuracy = 0.0
    patience = 3
    early_stop_counter = 0
    
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        correct_train = 0
        total_train = 0
        for i, batch in tqdm(enumerate(train_loader), total=len(train_loader)):
            videos, labels = batch
            videos = videos.to(device)
            labels = labels.to(device)
            optimizer.zero_grad()
            outputs = model(videos)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            writer.add_scalar("Loss/train_batch", loss.item(), global_step)
            
            _, predicted = torch.max(outputs.data, 1)
            total_train += labels.size(0)
            correct_train += (predicted == labels).sum().item()
            
            global_step += 1

        avg_loss = running_loss / len(train_loader)
        train_accuracy = 100 * correct_train / total_train
        writer.add_scalar("Loss/train_epoch", avg_loss, epoch)
        writer.add_scalar("Accuracy/train", train_accuracy, epoch)
        print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {avg_loss:.4f}, Training Accuracy: {train_accuracy:.2f}%")
        
        # Validation loop
        model.eval()
        correct_val = 0
        total_val = 0
        with torch.no_grad():
            for videos, labels in val_loader:
                videos = videos.to(device)
                labels = labels.to(device)
                outputs = model(videos)
                _, predicted = torch.max(outputs.data, 1)
                total_val += labels.size(0)
                correct_val += (predicted == labels).sum().item()
        val_accuracy = 100 * correct_val / total_val
        writer.add_scalar("Accuracy/val", val_accuracy, epoch)
        print(f"Validation Accuracy: {val_accuracy:.2f}%")
        
        # Early stopping condition
        if val_accuracy > best_val_accuracy:
            best_val_accuracy = val_accuracy
            early_stop_counter = 0
        else:
            early_stop_counter += 1
        
        if early_stop_counter >= patience:
            print("Early stopping triggered. No improvement in validation accuracy.")
            break
    
    writer.close()
