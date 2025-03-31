import torch
import argparse
from lightning.pytorch import Trainer
from torch.utils.data import DataLoader
from utils.dataset import ImigueVideoDS
from torchvision.models.video import swin3d_b
from utils.misc import stratified_split
from engine import FineTuner
import torch.nn as nn
from torchvision.transforms import v2

def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_path', type=str, default="./data/training", help='Path to test dataset')
    parser.add_argument('--num_classes', type=int, default=32, help='Number of classes in dataset')
    parser.add_argument('--num_frames', type=int, default=5, help='Number of frames to take')
    
    parser.add_argument('--batch_size', type=int, default=64, help='Batch size')
    parser.add_argument('--lr', type=float, default=1e-4, help='Learning rate')
    parser.add_argument('--epochs', type=int, default=10, help='Number of epochs')
    parser.add_argument('--gpu_id', type=int, default=0, help='GPU ID')
    parser.add_argument('--late_fusion', action="store_true", help="Use late fusion")
    parser.add_argument('--checkpoint_path', type=str, default=None, help='Path to checkpoint file')
    return parser.parse_args()

    
def main(args):
    transform = v2.Compose([
        v2.CenterCrop((224,224)),
        v2.ToImage(), 
        v2.ToDtype(torch.float32, scale=True),  
        v2.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    
    # Load dataset
    dataset = ImigueVideoDS(args.data_path, transform=transform, frame_count=args.num_frames)
    test_loader = DataLoader(dataset, batch_size=args.batch_size, num_workers=10)
    
    # Utilize medium precision for faster testing
    torch.set_float32_matmul_precision('medium')
    
    # Define trainer
    trainer = Trainer(
        max_epochs=args.epochs,
        accelerator="gpu" if torch.cuda.is_available() else "cpu",
        devices=[args.gpu_id],
        strategy="auto",
    )
    
    # Explicitly load the Swin3D model
    model = swin3d_b()
    num_features = model.head.in_features
    model.head = nn.Identity()

    # Load the lightning module that utilizes the Swin3D as backbone
    fine_tuner = FineTuner(model, num_features=num_features, num_classes=args.num_classes, frame_count=args.num_frames)

    # Load model from checkpoint if provided
    if args.checkpoint_path:
        checkpoint = torch.load(args.checkpoint_path, map_location='cpu', weights_only=True)
        fine_tuner.load_state_dict(checkpoint['state_dict'])
        print(f"Loaded model from checkpoint: {args.checkpoint_path}")

    # Test the model
    trainer.test(fine_tuner, dataloaders=test_loader)
    torch.cuda.empty_cache()

if __name__ == "__main__":
    args = parse_arguments()
    main(args)
