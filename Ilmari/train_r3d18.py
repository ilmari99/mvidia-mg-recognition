import os
os.environ["TORCH_USE_CUDA_DSA"] = "1"
os.environ["CUDA_LAUNCH_BLOCKING"] = "1"
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from torch.utils.tensorboard import SummaryWriter
from ImigueDS import ImigueDS
from torchvision import transforms
import torchvision.models.video as models
import random

class VideoClassifier(nn.Module):
    def __init__(self, num_classes=32):
        super(VideoClassifier, self).__init__()
        # Load a pre-trained 3D ResNet-18 model
        self.backbone = models.r3d_18(weights=models.R3D_18_Weights)
        # Replace the final fully connected layer with one that has num_classes outputs.
        self.backbone.fc = nn.Linear(self.backbone.fc.in_features, num_classes)
        self.softmax = nn.Softmax(dim=1)

        #for param in self.backbone.parameters():
        #    param.requires_grad = False
        #for param in self.backbone.fc.parameters():
        #    param.requires_grad = True

    def forward(self, x):
        # The dataset returns input in the shape [batch, frames, channels, height, width].
        # Pre-trained models expect [batch, channels, frames, height, width], so we permute.
        x = x.permute(0, 2, 1, 3, 4)
        x = self.softmax(self.backbone(x))
        return x

if __name__ == "__main__":
    torch.manual_seed(42)
    random.seed(42)

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Resize((128, 171)),
        transforms.CenterCrop((112,112)),
        transforms.Normalize(mean=[0.43216, 0.394666, 0.37645], std=[0.22803, 0.22145, 0.216989]),
    ])
    
    dataset = ImigueDS(image_directory="./imigue", frame_count=7, transform=transform)
    dataset_size = len(dataset)
    train_size = int(0.7 * dataset_size)
    val_size = int(0.15 * dataset_size)
    test_size = dataset_size - train_size - val_size
    train_set, val_set, test_set = random_split(dataset, [train_size, val_size, test_size])
    
    # Increase batch size to 32
    train_loader = DataLoader(train_set, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_set, batch_size=32, shuffle=False)
    test_loader = DataLoader(test_set, batch_size=32, shuffle=False)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = VideoClassifier(num_classes=32).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    
    # Initialize TensorBoard writer
    writer = SummaryWriter(log_dir="./runs")
    
    num_epochs = 5  # Adjust as needed
    global_step = 0
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        correct_train = 0
        total_train = 0
        for i, batch in enumerate(train_loader):
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
    
    writer.close()
