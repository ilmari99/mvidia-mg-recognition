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
    def __init__(self, model, num_features, num_classes, lr=1e-3, frame_count=None, epochs=30):
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
        self.accuracy_topk = Accuracy(task='multiclass', num_classes=num_classes, top_k=5)
        self.lr = lr
        self.epochs = epochs
        self.weight_decay = 0.05
        self.save_hyperparameters(ignore=['model', 'classifier'])
        
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
        top5_acc = self.accuracy_topk(pred, label)
        
        self.log("val_loss", loss, batch_size=img.size(0))
        self.log("val_acc", acc, on_step=False, on_epoch=True)
        self.log("val_top5_acc", top5_acc, on_step=False, on_epoch=True)
        
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
        self.accuracy_topk.reset()
        self.valid_step_outputs.clear()
        self.valid_step_labels.clear()

    def test_step(self, batch, batch_idx):
        img, label = batch
        pred = self.forward(img)
        loss = self.criterion(pred, label)
        
        acc = self.accuracy(pred, label)
        top5_acc = self.accuracy_topk(pred, label)
        
        self.log("test_loss", loss, batch_size=img.size(0))
        self.log("test_acc", acc, on_step=False, on_epoch=True)
        self.log("test_top5_acc", top5_acc, on_step=False, on_epoch=True)
        
        # As preds is just logits, take max as prediction
        predicted = pred.argmax(dim=1).cpu().detach().numpy()
        labels = label.cpu().detach().numpy()
        
        self.valid_step_outputs.extend(predicted)
        self.valid_step_labels.extend(labels)
        return loss
    
    def on_test_epoch_end(self):
        # Reset metrics at the end of validation epoch
        self.accuracy.reset()
        self.accuracy_topk.reset()
        self.valid_step_outputs.clear()
        self.valid_step_labels.clear()

    def configure_optimizers(self):
        
        # Get layer-wise learning rate decay parameters
        optimizer = torch.optim.AdamW(self.parameters(), lr=self.lr, weight_decay=0.05)
        
        # The lr scheduler is effectively CosineAnnealingLR
        # Has a wamrup period of 5 epochs
        warm_up = 3
        lr_scheduler = torch.optim.lr_scheduler.LambdaLR(
            optimizer,
            lr_lambda=lambda epoch: min(
                (epoch + 1) / (warm_up + 1e-8),
                0.5 * (math.cos(epoch / self.epochs * math.pi) + 1),
            ),
        )
     
        return {"optimizer": optimizer, "lr_scheduler": lr_scheduler} 


class LateFusion(LightningModule):
    """
    Fine tuner for video clips utilizing frames
    """
    def __init__(self, model, num_features, num_classes, lr=1e-3, frame_count=None, epochs=30):
        super(LateFusion, self).__init__()
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
        self.accuracy_topk = Accuracy(task='multiclass', num_classes=num_classes, top_k=5)
        self.lr = lr
        self.epochs = epochs
        self.weight_decay = 0.05
        self.save_hyperparameters(ignore=['model', 'classifier'])
        
        # Validation outputs for confusion matrix
        self.valid_step_outputs = []
        self.valid_step_labels = []
    
    def forward(self, x):
        batch_size, num_frames, C, H, W = x.shape
        
        # Extract per-frame features
        x = x.view(batch_size * num_frames, C, H, W)
        features = self.model(x)
        pred = self.classifier(features)
        
        # Reshape predictions: [batch_size, num_frames, 32]
        pred = pred.reshape(batch_size, num_frames, 32)

        # Compute mean logits per clip
        clip_pred = pred.mean(dim=1)  # Shape: [batch_size, 32]

        # Final classification
        return clip_pred
    
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
        top5_acc = self.accuracy_topk(pred, label)
        
        self.log("val_loss", loss, batch_size=img.size(0))
        self.log("val_acc", acc, on_step=False, on_epoch=True)
        self.log("val_top5_acc", top5_acc, on_step=False, on_epoch=True)
        
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
        self.accuracy_topk.reset()
        self.valid_step_outputs.clear()
        self.valid_step_labels.clear()
        
    def configure_optimizers(self):
        
        # Get layer-wise learning rate decay parameters
        optimizer = torch.optim.AdamW(self.parameters(), lr=self.lr, weight_decay=0.05)
        
        # The lr scheduler is effectively CosineAnnealingLR
        # Has a wamrup period of 5 epochs
        warm_up = 3
        lr_scheduler = torch.optim.lr_scheduler.LambdaLR(
            optimizer,
            lr_lambda=lambda epoch: min(
                (epoch + 1) / (warm_up + 1e-8),
                0.5 * (math.cos(epoch / self.epochs * math.pi) + 1),
            ),
        )
     
        return {"optimizer": optimizer, "lr_scheduler": lr_scheduler} 
