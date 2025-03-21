import torch
import torch.nn as nn
import timm
from lightning.pytorch import  LightningModule
import torch.optim as optim
import math
from torchmetrics import Accuracy
from utils.misc import createConfusionMatrix

class FineTuner(LightningModule):
    """
    PyTorch Lightning class for training
    """
    def __init__(self, model, num_classes, lr=1e-3):
        super(FineTuner, self).__init__()
        self.model = model
        self.classifier = nn.Sequential(
            nn.Linear(768, 512),
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
        
        self.valid_step_outputs = []
        self.valid_step_labels = []
    
    def forward(self, x):
        # For timm models
        output = self.model(x)
        if "logits" in dict(output):
            output = output["logits"]
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
        
        # # The lr scheduler is effectively CosineAnnealingLR
        lr_scheduler = torch.optim.lr_scheduler.LambdaLR(
            optimizer,
            lr_lambda=lambda epoch: min(
                (epoch + 1) / (5 + 1e-8),
                0.5 * (math.cos(epoch / 30 * math.pi) + 1),
            ),
        )
     
        return {"optimizer": optimizer, "lr_scheduler": lr_scheduler} 
