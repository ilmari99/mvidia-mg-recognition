import torch
import argparse
from lightning.pytorch import Trainer
from torch.utils.data import DataLoader
from lightning.pytorch.loggers import TensorBoardLogger
from lightning.pytorch.callbacks import ModelCheckpoint
from utils.dataset import ImigueVideoDS, ImigueImageDS
from utils.misc import stratified_split, create_backbone_model
from engine import FineTuner, LateFusion
from torchvision.transforms import v2

def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_path', type=str, default="./data/training", help='Path to dataset')
    parser.add_argument('--num_classes', type=int, default=32, help='Number of classes in dataset')
    parser.add_argument('--num_frames', type=int, default=5, help='Number of frames to take')
    parser.add_argument('--lr', type=float, default=1e-4, help='Learning rate')
    parser.add_argument('--model', type=str, default="swin", help='Model from timm')
    
    parser.add_argument('--batch_size', type=int, default=64, help='Batch size')
    parser.add_argument('--epochs', type=int, default=10, help='Number of epochs')
    parser.add_argument('--gpu_id', type=int, default=0, help='GPU ID')
    parser.add_argument('--accum_grad_steps', type=int, default=2, help='Gradient accumulation steps')
    parser.add_argument('--late_fusion', action="store_true", help="Use late fusion")
    return parser.parse_args()

    
def main(args):
    transform = v2.Compose([
        v2.CenterCrop((224,224)),
        v2.ToImage(), 
        v2.ToDtype(torch.float32, scale=True),  
        v2.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    
    # Load dataset
    if args.num_frames == 1:
        print("Using image dataset")
        dataset = ImigueImageDS(args.data_path, transform=transform, frame_count=args.num_frames)
    else:
        dataset = ImigueVideoDS(args.data_path, transform=transform, frame_count=args.num_frames)
    train_dataset, valid_dataset = stratified_split(dataset, args.num_frames)
 
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=10)
    valid_loader = DataLoader(valid_dataset, batch_size=args.batch_size, shuffle=False, num_workers=10)
    
    # Define logger and callbacks
    logger = TensorBoardLogger(save_dir="logs", name=f"{args.model}_imigue")
    checkpoint_callback = ModelCheckpoint(
        monitor="val_acc", 
        mode="max", 
        save_top_k=1, 
        filename="model--{epoch}-{val_acc:.2f}")
    
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
        callbacks=[checkpoint_callback],
        gradient_clip_val=0.5
    )
    
    # Load the model
    model, num_features = create_backbone_model(args)
    if args.late_fusion:
        fine_tuner = LateFusion(model, frame_count=args.num_frames, num_features=num_features, num_classes=args.num_classes, lr = args.lr, epochs=args.epochs)
    else:
        fine_tuner = FineTuner(model, num_features=num_features, num_classes=args.num_classes, lr = args.lr, frame_count=args.num_frames, epochs=args.epochs)
    
    trainer.fit(fine_tuner, train_dataloaders=train_loader, val_dataloaders=valid_loader)
    torch.cuda.empty_cache()

if __name__ == "__main__":
    args = parse_arguments()
    main(args)
