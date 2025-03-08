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
import random

# Define a simple 3D CNN for video classification
class VideoClassifier(nn.Module):
    def __init__(self, num_classes=31):
        super(VideoClassifier, self).__init__()
        # The input format expected here is (batch, channels, frames, height, width).
        self.conv1 = nn.Conv3d(1, 8, kernel_size=3, padding=1)
        self.pool1 = nn.MaxPool3d(kernel_size=(1,2,2))
        self.conv2 = nn.Conv3d(8, 16, kernel_size=3, padding=1)
        self.pool2 = nn.MaxPool3d(kernel_size=(1,2,2))
        self.conv3 = nn.Conv3d(16, 32, kernel_size=3, padding=1)
        self.pool3 = nn.AdaptiveAvgPool3d((1,1,1))
        self.fc = nn.Linear(32, num_classes)
        self.softmax = nn.Softmax(dim=1)
    
    def forward(self, x):
        # x: [batch, frames, channels, height, width]
        x = x.permute(0, 2, 1, 3, 4)
        x = torch.relu(self.conv1(x))
        x = self.pool1(x)
        x = torch.relu(self.conv2(x))
        x = self.pool2(x)
        x = torch.relu(self.conv3(x))
        x = self.pool3(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        x = self.softmax(x)
        return x

if __name__ == "__main__":
    torch.manual_seed(42)
    random.seed(42)

    # Change frame size to 64x64 and convert to grayscale
    transform = transforms.Compose([
        transforms.Resize((64, 64)),
        transforms.Grayscale(num_output_channels=1),
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])
    
    # Use 7 frames instead of 5
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
    # Adjust num_classes if necessary (currently set to 32 in training)
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

            # Accumulate loss
            running_loss += loss.item()
            
            # Record batch loss to TensorBoard
            writer.add_scalar("Loss/train_batch", loss.item(), global_step)
            
            # Calculate batch training accuracy
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
