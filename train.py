"""
Training Script for Drivable Area Segmentation
"""
import os
import argparse
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim import Adam, SGD
from torch.optim.lr_scheduler import ReduceLROnPlateau, CosineAnnealingLR
from tqdm import tqdm
import numpy as np
import json

from data.dataset import BDD100KDrivableDataset, get_train_transforms, get_val_transforms
from models.unet import UNet
from models.baseline_cnn import BaselineCNN
from utils.losses import BCEDiceLoss, DiceLoss, FocalLoss
from utils.metrics import calculate_iou, calculate_miou, calculate_pixel_accuracy, calculate_dice_coefficient
from utils.visualization import plot_training_curves, visualize_predictions


class Trainer:
    def __init__(self, config):
        self.config = config
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"Using device: {self.device}")
        
        # Create save directories
        os.makedirs(config.save_dir, exist_ok=True)
        os.makedirs(os.path.join(config.save_dir, 'checkpoints'), exist_ok=True)
        os.makedirs(os.path.join(config.save_dir, 'visualizations'), exist_ok=True)
        
        # Initialize model
        self.model = self._build_model()
        self.model.to(self.device)
        
        # Initialize data loaders
        self.train_loader, self.val_loader = self._build_dataloaders()
        
        # Initialize loss function and optimizer
        self.criterion = self._build_criterion()
        self.optimizer = self._build_optimizer()
        self.scheduler = self._build_scheduler()
        
        # Training history
        self.history = {
            'train_loss': [], 'val_loss': [],
            'train_iou': [], 'val_iou': [],
            'train_dice': [], 'val_dice': [],
            'train_acc': [], 'val_acc': []
        }
        
        self.best_val_iou = 0.0
        self.best_epoch = 0
    
    def _build_model(self):
        """Build model"""
        if self.config.model == 'unet':
            model = UNet(n_channels=3, n_classes=1, bilinear=True)
            print("Using U-Net model")
        elif self.config.model == 'baseline':
            model = BaselineCNN(n_channels=3, n_classes=1)
            print("Using Baseline CNN model")
        else:
            raise ValueError(f"Unknown model: {self.config.model}")
        
        # Print model parameters
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"Total parameters: {total_params:,}")
        print(f"Trainable parameters: {trainable_params:,}")
        
        return model
    
    def _build_dataloaders(self):
        """Build data loaders"""
        train_dataset = BDD100KDrivableDataset(
            image_dir=self.config.image_dir,
            mask_dir=self.config.mask_dir,
            split='train',
            transform=get_train_transforms(self.config.image_size),
            binary=True
        )
        
        val_dataset = BDD100KDrivableDataset(
            image_dir=self.config.image_dir,
            mask_dir=self.config.mask_dir,
            split='val',
            transform=get_val_transforms(self.config.image_size),
            binary=True
        )
        
        train_loader = DataLoader(
            train_dataset,
            batch_size=self.config.batch_size,
            shuffle=True,
            num_workers=self.config.num_workers,
            pin_memory=True
        )
        
        val_loader = DataLoader(
            val_dataset,
            batch_size=self.config.batch_size,
            shuffle=False,
            num_workers=self.config.num_workers,
            pin_memory=True
        )
        
        print(f"Train samples: {len(train_dataset)}")
        print(f"Val samples: {len(val_dataset)}")
        
        return train_loader, val_loader
    
    def _build_criterion(self):
        """Build loss function"""
        if self.config.loss == 'bce_dice':
            criterion = BCEDiceLoss(bce_weight=0.5, dice_weight=0.5)
        elif self.config.loss == 'dice':
            criterion = DiceLoss()
        elif self.config.loss == 'focal':
            criterion = FocalLoss()
        elif self.config.loss == 'bce':
            criterion = nn.BCEWithLogitsLoss()
        else:
            raise ValueError(f"Unknown loss: {self.config.loss}")
        
        print(f"Using loss: {self.config.loss}")
        return criterion
    
    def _build_optimizer(self):
        """Build optimizer"""
        if self.config.optimizer == 'adam':
            optimizer = Adam(self.model.parameters(), lr=self.config.lr, weight_decay=self.config.weight_decay)
        elif self.config.optimizer == 'sgd':
            optimizer = SGD(self.model.parameters(), lr=self.config.lr, momentum=0.9, weight_decay=self.config.weight_decay)
        else:
            raise ValueError(f"Unknown optimizer: {self.config.optimizer}")
        
        print(f"Using optimizer: {self.config.optimizer} with lr={self.config.lr}")
        return optimizer
    
    def _build_scheduler(self):
        """Build learning rate scheduler"""
        if self.config.scheduler == 'plateau':
            scheduler = ReduceLROnPlateau(self.optimizer, mode='max', factor=0.5, patience=5)
        elif self.config.scheduler == 'cosine':
            scheduler = CosineAnnealingLR(self.optimizer, T_max=self.config.epochs, eta_min=1e-6)
        else:
            scheduler = None
        
        if scheduler:
            print(f"Using scheduler: {self.config.scheduler}")
        return scheduler
    
    def train_epoch(self):
        """Train one epoch"""
        self.model.train()
        
        running_loss = 0.0
        running_iou = 0.0
        running_dice = 0.0
        running_acc = 0.0
        
        pbar = tqdm(self.train_loader, desc='Training')
        for images, masks in pbar:
            images = images.to(self.device)
            masks = masks.to(self.device)
            
            # Forward pass
            self.optimizer.zero_grad()
            logits = self.model(images)
            loss = self.criterion(logits, masks)
            
            # Backward pass
            loss.backward()
            self.optimizer.step()
            
            # Calculate metrics
            with torch.no_grad():
                probs = torch.sigmoid(logits)
                iou = calculate_iou(probs, masks)
                dice = calculate_dice_coefficient(probs, masks)
                acc = calculate_pixel_accuracy(probs, masks)
            
            running_loss += loss.item()
            running_iou += iou
            running_dice += dice
            running_acc += acc
            
            pbar.set_postfix({
                'loss': f'{loss.item():.4f}',
                'iou': f'{iou:.4f}'
            })
        
        epoch_loss = running_loss / len(self.train_loader)
        epoch_iou = running_iou / len(self.train_loader)
        epoch_dice = running_dice / len(self.train_loader)
        epoch_acc = running_acc / len(self.train_loader)
        
        return epoch_loss, epoch_iou, epoch_dice, epoch_acc
    
    @torch.no_grad()
    def validate_epoch(self):
        """Validate one epoch"""
        self.model.eval()
        
        running_loss = 0.0
        running_iou = 0.0
        running_dice = 0.0
        running_acc = 0.0
        
        # Save some samples for visualization
        sample_images = []
        sample_masks = []
        sample_preds = []
        
        pbar = tqdm(self.val_loader, desc='Validation')
        for images, masks in pbar:
            images = images.to(self.device)
            masks = masks.to(self.device)
            
            # Forward pass
            logits = self.model(images)
            loss = self.criterion(logits, masks)
            
            # Calculate metrics
            probs = torch.sigmoid(logits)
            iou = calculate_iou(probs, masks)
            dice = calculate_dice_coefficient(probs, masks)
            acc = calculate_pixel_accuracy(probs, masks)
            
            running_loss += loss.item()
            running_iou += iou
            running_dice += dice
            running_acc += acc
            
            # Save samples
            if len(sample_images) < 8:
                sample_images.append(images.cpu())
                sample_masks.append(masks.cpu())
                sample_preds.append(probs.cpu())
            
            pbar.set_postfix({
                'loss': f'{loss.item():.4f}',
                'iou': f'{iou:.4f}'
            })
        
        epoch_loss = running_loss / len(self.val_loader)
        epoch_iou = running_iou / len(self.val_loader)
        epoch_dice = running_dice / len(self.val_loader)
        epoch_acc = running_acc / len(self.val_loader)
        
        # Concatenate samples
        if sample_images:
            sample_images = torch.cat(sample_images, dim=0)
            sample_masks = torch.cat(sample_masks, dim=0)
            sample_preds = torch.cat(sample_preds, dim=0)
        
        return epoch_loss, epoch_iou, epoch_dice, epoch_acc, (sample_images, sample_masks, sample_preds)
    
    def train(self):
        """Complete training loop"""
        print(f"\nStarting training for {self.config.epochs} epochs...")
        print("="*80)
        
        for epoch in range(1, self.config.epochs + 1):
            print(f"\nEpoch {epoch}/{self.config.epochs}")
            print("-"*80)
            
            # Train
            train_loss, train_iou, train_dice, train_acc = self.train_epoch()
            
            # Validate
            val_loss, val_iou, val_dice, val_acc, samples = self.validate_epoch()
            
            # Record history
            self.history['train_loss'].append(train_loss)
            self.history['val_loss'].append(val_loss)
            self.history['train_iou'].append(train_iou)
            self.history['val_iou'].append(val_iou)
            self.history['train_dice'].append(train_dice)
            self.history['val_dice'].append(val_dice)
            self.history['train_acc'].append(train_acc)
            self.history['val_acc'].append(val_acc)
            
            # Print results
            print(f"\nTrain - Loss: {train_loss:.4f}, IoU: {train_iou:.4f}, Dice: {train_dice:.4f}, Acc: {train_acc:.4f}")
            print(f"Val   - Loss: {val_loss:.4f}, IoU: {val_iou:.4f}, Dice: {val_dice:.4f}, Acc: {val_acc:.4f}")
            
            # Update learning rate
            if self.scheduler:
                if isinstance(self.scheduler, ReduceLROnPlateau):
                    self.scheduler.step(val_iou)
                else:
                    self.scheduler.step()
            
            # Save best model
            if val_iou > self.best_val_iou:
                self.best_val_iou = val_iou
                self.best_epoch = epoch
                self.save_checkpoint('best_model.pth')
                print(f"✓ Best model saved! (IoU: {val_iou:.4f})")
            
            # Save checkpoints periodically
            if epoch % self.config.save_interval == 0:
                self.save_checkpoint(f'checkpoint_epoch_{epoch}.pth')
            
            # Visualize predictions
            if epoch % self.config.vis_interval == 0:
                vis_path = os.path.join(self.config.save_dir, 'visualizations', f'epoch_{epoch}.png')
                visualize_predictions(samples[0][:4], samples[1][:4], samples[2][:4], 
                                    num_samples=4, save_path=vis_path)
        
        print("\n" + "="*80)
        print(f"Training completed! Best IoU: {self.best_val_iou:.4f} at epoch {self.best_epoch}")
        
        # Save training history
        self.save_history()
        
        # Plot training curves
        plot_path = os.path.join(self.config.save_dir, 'training_curves.png')
        plot_training_curves(self.history, save_path=plot_path)
    
    def save_checkpoint(self, filename):
        """Save checkpoint"""
        checkpoint = {
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'history': self.history,
            'best_val_iou': self.best_val_iou,
            'best_epoch': self.best_epoch,
            'config': vars(self.config)
        }
        
        if self.scheduler:
            checkpoint['scheduler_state_dict'] = self.scheduler.state_dict()
        
        save_path = os.path.join(self.config.save_dir, 'checkpoints', filename)
        torch.save(checkpoint, save_path)
    
    def save_history(self):
        """Save training history"""
        history_path = os.path.join(self.config.save_dir, 'history.json')
        with open(history_path, 'w') as f:
            json.dump(self.history, f, indent=4)
        print(f"Training history saved to {history_path}")


def parse_args():
    parser = argparse.ArgumentParser(description='Train Drivable Area Segmentation Model')
    
    # Data related
    parser.add_argument('--image_dir', type=str, 
                       default='/root/bdd100k_data/bdd100k_images/bdd100k/images/10k',
                       help='Image directory path')
    parser.add_argument('--mask_dir', type=str,
                       default='/root/bdd100k_data/bdd100k_drivable_maps/bdd100k/drivable_maps/labels',
                       help='Mask directory path')
    parser.add_argument('--image_size', type=int, default=256, help='Input image size')
    
    # Model related
    parser.add_argument('--model', type=str, default='unet', choices=['unet', 'baseline'],
                       help='Model type')
    
    # Training related
    parser.add_argument('--batch_size', type=int, default=16, help='Batch size')
    parser.add_argument('--epochs', type=int, default=50, help='Number of epochs')
    parser.add_argument('--lr', type=float, default=1e-4, help='Learning rate')
    parser.add_argument('--weight_decay', type=float, default=1e-5, help='Weight decay')
    parser.add_argument('--optimizer', type=str, default='adam', choices=['adam', 'sgd'],
                       help='Optimizer')
    parser.add_argument('--scheduler', type=str, default='plateau', choices=['plateau', 'cosine', 'none'],
                       help='Learning rate scheduler')
    parser.add_argument('--loss', type=str, default='bce_dice', 
                       choices=['bce_dice', 'dice', 'focal', 'bce'],
                       help='Loss function')
    
    # Other
    parser.add_argument('--num_workers', type=int, default=4, help='Number of data loading workers')
    parser.add_argument('--save_dir', type=str, default='experiments/unet_default', help='Save directory')
    parser.add_argument('--save_interval', type=int, default=10, help='Checkpoint save interval')
    parser.add_argument('--vis_interval', type=int, default=5, help='Visualization interval')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    
    return parser.parse_args()


def set_seed(seed):
    """Set random seed"""
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def main():
    # Parse arguments
    config = parse_args()
    
    # Set random seed
    set_seed(config.seed)
    
    # Create trainer and train
    trainer = Trainer(config)
    trainer.train()


if __name__ == '__main__':
    main()
