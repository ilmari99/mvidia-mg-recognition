import torch
import timm
import argparse
from lightning.pytorch import Trainer
from torch.utils.data import DataLoader
from lightning.pytorch.loggers import TensorBoardLogger
from lightning.pytorch.callbacks import ModelCheckpoint
from timm.data import resolve_data_config
from timm.data.transforms_factory import create_transform
from utils.dataset import ImigueDS
from utils.misc import stratified_split
from engine import VideoClassificationModel

def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_path', type=str, default="./data/training", help='Path to dataset')
    parser.add_argument('--num_classes', type=int, default=32, help='Number of classes in dataset')
    parser.add_argument('--num_frames', type=int, default=5, help='Number of frames to take')
    
    parser.add_argument('--model', type=str, default="resnet18", help='Model from timm')
    parser.add_argument('--batch_size', type=int, default=64, help='Batch size')
    parser.add_argument('--epochs', type=int, default=30, help='Number of epochs')
    parser.add_argument('--gpu_id', type=int, default=1, help='GPU ID')
    parser.add_argument('--accum_grad_steps', type=int, default=2, help='Gradient accumulation steps')
    return parser.parse_args()

    
def main(args):
    # Load the model
    model = timm.create_model(args.model, pretrained=True, num_classes=0)
    transform = create_transform(**resolve_data_config(model.pretrained_cfg, model=model))
    
    # Load dataset
    dataset = ImigueDS(args.data_path, transform=transform, frame_count=args.num_frames)
    train_dataset, valid_dataset = stratified_split(dataset)
    
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=4)
    valid_loader = DataLoader(valid_dataset, batch_size=args.batch_size, shuffle=False, num_workers=4)
    
    # Define logger and callbacks
    logger = TensorBoardLogger(save_dir="logs", name=f"{args.model}_imigue")
    checkpoint_callback = ModelCheckpoint(
        monitor="val_loss", 
        mode="min", 
        save_top_k=1, 
        filename="model--{epoch}-{val_loss:.2f}")
    
    # Utilize medium precision for faster training
    torch.set_float32_matmul_precision('medium')
    
    # Define trainer
    trainer = Trainer(
        max_epochs=args.epochs,
        accelerator="gpu" if torch.cuda.is_available() else "cpu",
        devices=[args.gpu_id],
        strategy="auto",
        accumulate_grad_batches=args.accum_grad_steps,
        log_every_n_steps=1,
        logger=logger,
        callbacks=[checkpoint_callback]
    )
    
    # Initialize and train model
    fine_tuner = VideoClassificationModel(model, frame_count=args.num_frames, num_classes=args.num_classes)
    trainer.fit(fine_tuner, train_dataloaders=train_loader, val_dataloaders=valid_loader)
    torch.cuda.empty_cache()

if __name__ == "__main__":
    args = parse_arguments()
    main(args)
