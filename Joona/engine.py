import torch
import torch.nn as nn
import timm
from lightning.pytorch import  LightningModule
import torch.optim as optim
from torchmetrics import Accuracy

class VideoClassificationModel(LightningModule):
    def __init__(self, model, num_classes=32, frame_count=1, lr=1e-4):
        super().__init__()

        self.encoder = model
        self.feature_dim = self.encoder.num_features
        self.frame_count = frame_count  # Number of frames per video
        
        # Positional Embeddings (Learnable)
        self.positional_embedding = nn.Parameter(torch.randn(1, self.frame_count, self.feature_dim))

        # Transformer Encoder for Temporal Modeling
        encoder_layer = nn.TransformerEncoderLayer(d_model=self.feature_dim, nhead=8, dim_feedforward=1024, dropout=0.1, batch_first=True)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=4)

        # Classification Head
        self.fc = nn.Linear(self.feature_dim, num_classes)

        # Loss function & Metrics
        self.criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
        self.accuracy = Accuracy(task="multiclass", num_classes=num_classes)

        # Learning rate
        self.lr = lr
        self.save_hyperparameters(ignore=['model', 'encoder'])

    def forward(self, x):
        """
        x: [batch_size, num_frames, C, H, W]
        """
        batch_size, num_frames, C, H, W = x.shape

        # Reshape to process all frames through CNN
        x = x.view(batch_size * num_frames, C, H, W)  # (batch_size * num_frames, C, H, W)
        features = self.encoder(x)  # Extract per-frame features
        features = features.view(batch_size, num_frames, -1)  # (batch_size, num_frames, feature_dim)
        
        # Apply positional embedding
        # features += self.positional_embedding[:, :num_frames, :]

        # Transformer expects (batch_size, num_frames, feature_dim)
        transformer_out = self.transformer(features)  # (num_frames, batch_size, feature_dim)

        # Aggregate over time dimension 
        out = transformer_out.mean(dim=1)

        # Final classification
        return self.fc(out)

    def training_step(self, batch, batch_idx):
        img, label = batch
        logits = self.forward(img)
        loss = self.criterion(logits, label)
        acc = self.accuracy(logits, label)

        self.log("train_loss", loss, prog_bar=True, on_step=True, on_epoch=True)
        self.log("train_acc", acc, prog_bar=True, on_step=True, on_epoch=True)
        return loss

    def validation_step(self, batch, batch_idx):
        img, label = batch
        logits = self.forward(img)
        loss = self.criterion(logits, label)
        acc = self.accuracy(logits, label)

        self.log("val_loss", loss, prog_bar=True, on_epoch=True)
        self.log("val_acc", acc, prog_bar=True, on_epoch=True)
        return loss

    def configure_optimizers(self):
        optimizer = optim.AdamW(self.parameters(), lr=self.lr, weight_decay=1e-4)
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=10)
        return {"optimizer": optimizer, "lr_scheduler": scheduler}



class FineTuner(LightningModule):
    """
    PyTorch Lightning class for training
    """
    def __init__(self, model, num_classes, lr=1e-3):
        super(FineTuner, self).__init__()
        self.model = model
        self.criterion = nn.CrossEntropyLoss()
        self.accuracy = Accuracy(task='multiclass', num_classes=num_classes)
        self.lr = lr
        self.save_hyperparameters(ignore=['model'])
    
    def forward(self, x):
        return self.model(x)
    
    def training_step(self, batch, batch_idx):
        img, label = batch
        pred = self.forward(img)
        
        loss = self.criterion(pred, label)
        self.log("train_loss", loss, batch_size=img.size(0))
        return loss
    
    def validation_step(self, batch, batch_idx):
        img, label = batch
        pred = self.forward(img)
        loss = self.criterion(pred, label)
        
        acc = self.accuracy(pred, label)
        self.log("val_loss", loss, batch_size=img.size(0))
        self.log("val_acc", acc, on_step=False, on_epoch=True)
        return loss
    
    def on_validation_epoch_end(self):
        # Reset metrics at the end of validation epoch
        self.accuracy.reset()
        
    def configure_optimizers(self):
        optimizer = optim.Adam(self.model.parameters(), lr=self.lr, weight_decay=0.01)
        return optimizer