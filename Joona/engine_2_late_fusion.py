
class VideoClassificationModel(LightningModule):
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
            dim_feedforward=2048,  # Larger feedforward dimension
            dropout=0.1,
            batch_first=True,
            activation='gelu'  # Use GELU like in modern transformers
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=4)  # More layers
        
        # Multi-level feature aggregation
        self.pool_types = ['max']
        self.classifier = nn.Sequential(
            nn.Linear(self.feature_dim * len(self.pool_types), 512),
            nn.LayerNorm(512),
            nn.GELU(),
            nn.Dropout(0.2),
            nn.Linear(512, num_classes)
        )
        
        # Loss function & Metrics with focal loss component for imbalanced classes
        self.criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
        self.accuracy = Accuracy(task="multiclass", num_classes=num_classes)
        
        # Learning rate
        self.lr = lr
        self.save_hyperparameters(ignore=['model', 'encoder'])
    
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
        
        # Add positional embedding
        # features = features + self.pos_embedding
        
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
        acc = self.accuracy(logits, label)
        self.log("train_loss", loss, prog_bar=True, on_step=True, on_epoch=True)
        self.log("train_acc", acc, prog_bar=True, on_step=True, on_epoch=True)
        return loss
    
    def on_training_epoch_end(self):
        # Reset metrics at the end of validation epoch
        self.accuracy.reset()
            
    def validation_step(self, batch, batch_idx):
        img, label = batch
        logits = self.forward(img)
        loss = self.criterion(logits, label)
        acc = self.accuracy(logits, label)
        self.log("val_loss", loss, prog_bar=True, on_epoch=True)
        self.log("val_acc", acc, prog_bar=True, on_epoch=True)
        return loss
    
    def on_validation_epoch_end(self):
        # Reset metrics at the end of validation epoch
        self.accuracy.reset()
    
    def configure_optimizers(self):
        # Create parameter groups with different learning rates
        encoder_params = list(self.model.parameters())
        other_params = [p for p in self.parameters() if p not in set(encoder_params)]
        
        param_groups = [
            {"params": encoder_params, "lr": self.lr * 0.1},  # Lower LR for pretrained encoder
            {"params": other_params, "lr": self.lr}           # Higher LR for new parameters
        ]
        
        optimizer = optim.AdamW(param_groups, weight_decay=0.05)
        
        # Add warmup and cosine decay
        scheduler = {
            "scheduler": optim.lr_scheduler.OneCycleLR(
                optimizer,
                max_lr=[self.lr * 0.1, self.lr],
                total_steps=self.trainer.estimated_stepping_batches,
                pct_start=0.1,
                div_factor=25,
                final_div_factor=1000,
            ),
            "interval": "step",
        }
        
        return {"optimizer": optimizer, "lr_scheduler": scheduler}
    
