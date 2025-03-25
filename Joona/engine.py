import math
import torch
import torch.nn as nn
from torchmetrics import Accuracy
from utils.misc import createConfusionMatrix
from lightning.pytorch import  LightningModule

class FineTuner(LightningModule):
    """
    Fine tuner for 3D video models
    """
    def __init__(self, model, num_features, num_classes, lr=1e-3, frame_count=None):
        super(FineTuner, self).__init__()
        self.model = model
        self.classifier = nn.Sequential(
            nn.Linear(num_features, 512),
            nn.LayerNorm(512),
            nn.GELU(),
            nn.Dropout(0.2),
            nn.Linear(512, num_classes)
        )
        
        self.criterion = nn.CrossEntropyLoss()
        self.accuracy = Accuracy(task='multiclass', num_classes=num_classes)
        self.lr = lr
        self.weight_decay = 0.05
        self.save_hyperparameters(ignore=['model'])
        
        # Validation outputs for confusion matrix
        self.valid_step_outputs = []
        self.valid_step_labels = []
    
    def forward(self, x):
        # For timm models
        if self.model.__class__.__name__ == "SwinTransformer3d":
            x = x.permute(0, 2, 1, 3, 4)
        output = self.model(x)
        pred = self.classifier(output)
        return pred
    
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
        
        # As preds is just logits, take max as prediction
        predicted = pred.argmax(dim=1).cpu().detach().numpy()
        labels = label.cpu().detach().numpy()
        
        self.valid_step_outputs.extend(predicted)
        self.valid_step_labels.extend(labels)
        return loss
    
    def on_validation_epoch_end(self):
        # Get logger (tensorboard)
        logger = self.logger.experiment
        logger.add_figure("Confusion matrix", createConfusionMatrix(self.valid_step_outputs, self.valid_step_labels), self.current_epoch + 1)
        
        # Reset metrics at the end of validation epoch
        self.accuracy.reset()
        self.valid_step_outputs.clear()
        self.valid_step_labels.clear()
        
    def configure_optimizers(self):
        
        # Get layer-wise learning rate decay parameters
        optimizer = torch.optim.AdamW(self.parameters(), lr=self.lr, weight_decay=0.05)
        
        # The lr scheduler is effectively CosineAnnealingLR
        # Has a wamrup period of 5 epochs
        warm_up = 5
        epochs = 30
        lr_scheduler = torch.optim.lr_scheduler.LambdaLR(
            optimizer,
            lr_lambda=lambda epoch: min(
                (epoch + 1) / (warm_up + 1e-8),
                0.5 * (math.cos(epoch / epochs * math.pi) + 1),
            ),
        )
     
        return {"optimizer": optimizer, "lr_scheduler": lr_scheduler} 


# Late fusion engine

class LateFusion(LightningModule):
    """
    Fine tuner for late fusion approaches: First encoder each frame and then pass to transformer encoder.
    """
    def __init__(self, model, num_classes=32, frame_count=5, lr=1e-4, num_features=None):
        super().__init__()
        
        # Frame-level feature extractor
        self.model = model
        self.feature_dim = num_features if num_features else self.model.num_features
        self.frame_count = frame_count
        
        # Enhanced transformer with more parameters
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=self.feature_dim, 
            nhead=8, 
            dim_feedforward=1024,
            dropout=0.1,
            batch_first=True,
            activation='gelu'
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=1)  # More layers
        
        # Multi-level feature aggregation
        self.pool_types = ['mean']
        self.classifier = nn.Sequential(
            nn.Dropout(0.2),
            nn.Linear(self.feature_dim * len(self.pool_types), num_classes)
        )
        
        # Loss function & Metrics with focal loss component for imbalanced classes
        self.criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
        self.accuracy = Accuracy(task="multiclass", num_classes=num_classes)
        
        # Learning rate
        self.lr = lr
        self.save_hyperparameters(ignore=['model', 'transformer'])
        
        # Validation outputs for confusion matrix
        self.valid_step_outputs = []
        self.valid_step_labels = []
    
    def multi_pool(self, x):
        # Apply different pooling strategies and concatenate
        pools = []
        if 'mean' in self.pool_types:
            pools.append(x.mean(dim=1))
        if 'max' in self.pool_types:
            pools.append(x[:, -1, :] )
        return torch.cat(pools, dim=1)
    
    def forward(self, x):
        batch_size, num_frames, C, H, W = x.shape
        
        # Extract per-frame features
        x = x.view(batch_size * num_frames, C, H, W)
        features = self.model(x)
        features = features.view(batch_size, num_frames, -1)
        
        # Apply transformer for temporal modeling
        transformer_out = self.transformer(features)
        
        # Multi-level feature aggregation
        pooled_features = self.multi_pool(transformer_out)
        
        # Final classification
        return self.classifier(pooled_features)
    
    def training_step(self, batch, batch_idx):
        img, label = batch
        logits = self.forward(img)
        loss = self.criterion(logits, label)
        self.log("train_loss", loss, batch_size=img.size(0))
        return loss
            
    def validation_step(self, batch, batch_idx):
        img, label = batch
        pred = self.forward(img)
        loss = self.criterion(pred, label)
        
        acc = self.accuracy(pred, label)
        self.log("val_loss", loss, batch_size=img.size(0))
        self.log("val_acc", acc, on_step=False, on_epoch=True)
        
        # As preds is just logits, take max as prediction
        predicted = pred.argmax(dim=1).cpu().detach().numpy()
        labels = label.cpu().detach().numpy()
        
        self.valid_step_outputs.extend(predicted)
        self.valid_step_labels.extend(labels)
        return loss
    
    def on_validation_epoch_end(self):
        # Get logger (tensorboard)
        logger = self.logger.experiment
        logger.add_figure("Confusion matrix", createConfusionMatrix(self.valid_step_outputs, self.valid_step_labels), self.current_epoch + 1)
        
        # Reset metrics at the end of validation epoch
        self.accuracy.reset()
        self.valid_step_outputs.clear()
        self.valid_step_labels.clear()
        
    
    def configure_optimizers(self):
        # Create parameter groups with different learning rates
        optimizer = torch.optim.AdamW(self.parameters(), lr=self.lr, weight_decay=0.05)
        
        # The lr scheduler is effectively CosineAnnealingLR
        lr_scheduler = torch.optim.lr_scheduler.LambdaLR(
            optimizer,
            lr_lambda=lambda epoch: min(
                (epoch + 1) / (5 + 1e-8),
                0.5 * (math.cos(epoch / 30 * math.pi) + 1),
            ),
        )
        
        return {"optimizer": optimizer, "lr_scheduler": lr_scheduler}