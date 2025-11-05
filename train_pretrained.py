"""
Training Script with Pretrained Models and Improved Techniques
"""
import os
import argparse
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim import Adam, AdamW
from torch.optim.lr_scheduler import ReduceLROnPlateau, CosineAnnealingLR, OneCycleLR
from tqdm import tqdm
import numpy as np
import json

from data.dataset import BDD100KDrivableDataset, get_train_transforms, get_val_transforms
from models.unet_pretrained import PretrainedUNet, PretrainedUNetPlusPlus, PretrainedDeepLabV3Plus
from models.baseline_cnn import BaselineCNN
from utils.losses_improved import WeightedBCEDiceLoss, TverskyLoss, FocalTverskyLoss, ComboLoss
from utils.metrics import calculate_iou, calculate_dice_coefficient, calculate_pixel_accuracy
from utils.visualization import plot_training_curves, visualize_predictions
from train import Trainer as BaseTrainer


class ImprovedTrainer(BaseTrainer):
    """Improved trainer with pretrained models and better techniques"""
    
    def _build_model(self):
        """Build model (pretrained or baseline)"""
        model_type = self.config.model_type
        
        # Baseline CNN for comparison
        if model_type == 'baseline_cnn':
            model = BaselineCNN(n_channels=3, n_classes=1)
            print(f"Using Baseline CNN (from scratch, no pretrained weights)")
        # Pretrained models
        else:
            encoder_name = self.config.encoder_name
            
            if model_type == 'unet':
                model = PretrainedUNet(encoder_name=encoder_name, encoder_weights='imagenet')
                print(f"Using Pretrained U-Net with {encoder_name} encoder")
            elif model_type == 'unetplusplus':
                model = PretrainedUNetPlusPlus(encoder_name=encoder_name, encoder_weights='imagenet')
                print(f"Using Pretrained U-Net++ with {encoder_name} encoder")
            elif model_type == 'deeplabv3plus':
                model = PretrainedDeepLabV3Plus(encoder_name=encoder_name, encoder_weights='imagenet')
                print(f"Using Pretrained DeepLabV3+ with {encoder_name} encoder")
            else:
                raise ValueError(f"Unknown model type: {model_type}")
        
        # Print model parameters
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"Total parameters: {total_params:,}")
        print(f"Trainable parameters: {trainable_params:,}")
        
        return model
    
    def _build_criterion(self):
        """Build improved loss function"""
        if self.config.loss == 'weighted_bce_dice':
            criterion = WeightedBCEDiceLoss(
                bce_weight=0.5, 
                dice_weight=0.5, 
                pos_weight=self.config.pos_weight
            )
        elif self.config.loss == 'tversky':
            criterion = TverskyLoss(alpha=0.7, beta=0.3)
        elif self.config.loss == 'focal_tversky':
            criterion = FocalTverskyLoss(alpha=0.7, beta=0.3, gamma=0.75)
        elif self.config.loss == 'combo':
            criterion = ComboLoss(alpha=0.5, beta=0.5, pos_weight=self.config.pos_weight)
        else:
            # Fall back to base implementation
            return super()._build_criterion()
        
        print(f"Using loss: {self.config.loss}")
        return criterion
    
    def _build_optimizer(self):
        """Build optimizer with different learning rates for encoder and decoder"""
        # For baseline CNN, don't use differential LR (no pretrained encoder)
        if self.config.model_type == 'baseline_cnn':
            optimizer = AdamW(
                self.model.parameters(), 
                lr=self.config.lr, 
                weight_decay=self.config.weight_decay
            )
            print(f"Using AdamW optimizer with lr={self.config.lr}")
        elif self.config.use_differential_lr:
            # Lower learning rate for pretrained encoder
            encoder_params = []
            decoder_params = []
            
            for name, param in self.model.named_parameters():
                if 'encoder' in name:
                    encoder_params.append(param)
                else:
                    decoder_params.append(param)
            
            optimizer = AdamW([
                {'params': encoder_params, 'lr': self.config.lr * 0.1},  # 10x smaller LR for encoder
                {'params': decoder_params, 'lr': self.config.lr}
            ], weight_decay=self.config.weight_decay)
            
            print(f"Using differential learning rates:")
            print(f"  Encoder LR: {self.config.lr * 0.1}")
            print(f"  Decoder LR: {self.config.lr}")
        else:
            optimizer = AdamW(
                self.model.parameters(), 
                lr=self.config.lr, 
                weight_decay=self.config.weight_decay
            )
            print(f"Using AdamW optimizer with lr={self.config.lr}")
        
        return optimizer
    
    def _build_scheduler(self):
        """Build learning rate scheduler"""
        if self.config.scheduler == 'onecycle':
            scheduler = OneCycleLR(
                self.optimizer,
                max_lr=self.config.lr,
                epochs=self.config.epochs,
                steps_per_epoch=len(self.train_loader),
                pct_start=0.3,
                anneal_strategy='cos'
            )
            print(f"Using OneCycleLR scheduler")
        else:
            # Fall back to base implementation
            return super()._build_scheduler()
        
        return scheduler


def parse_args():
    parser = argparse.ArgumentParser(description='Train with Pretrained Models')
    
    # Data related
    parser.add_argument('--image_dir', type=str, 
                       default='/root/bdd100k_data/bdd100k_images/bdd100k/images/10k',
                       help='Image directory path')
    parser.add_argument('--mask_dir', type=str,
                       default='/root/bdd100k_data/bdd100k_drivable_maps/bdd100k/drivable_maps/labels',
                       help='Mask directory path')
    parser.add_argument('--image_size', type=int, default=320, help='Input image size')
    
    # Model related
    parser.add_argument('--model_type', type=str, default='unet', 
                       choices=['baseline_cnn', 'unet', 'unetplusplus', 'deeplabv3plus'],
                       help='Model architecture (baseline_cnn for simple CNN baseline)')
    parser.add_argument('--encoder_name', type=str, default='resnet34',
                       choices=['resnet34', 'resnet50', 'efficientnet-b0', 'efficientnet-b3'],
                       help='Encoder backbone (only for pretrained models, ignored for baseline_cnn)')
    
    # Training related
    parser.add_argument('--batch_size', type=int, default=8, help='Batch size')
    parser.add_argument('--epochs', type=int, default=50, help='Number of epochs')
    parser.add_argument('--lr', type=float, default=1e-4, help='Learning rate')
    parser.add_argument('--weight_decay', type=float, default=1e-4, help='Weight decay')
    parser.add_argument('--use_differential_lr', action='store_true', default=True,
                       help='Use different LR for encoder and decoder')
    
    # Loss function
    parser.add_argument('--loss', type=str, default='weighted_bce_dice',
                       choices=['weighted_bce_dice', 'tversky', 'focal_tversky', 'combo'],
                       help='Loss function')
    parser.add_argument('--pos_weight', type=float, default=10.0,
                       help='Positive class weight for handling imbalance')
    
    # Scheduler
    parser.add_argument('--scheduler', type=str, default='onecycle',
                       choices=['plateau', 'cosine', 'onecycle', 'none'],
                       help='Learning rate scheduler')
    
    # Other
    parser.add_argument('--num_workers', type=int, default=4, help='Number of data loading workers')
    parser.add_argument('--save_dir', type=str, default='experiments/pretrained_unet', help='Save directory')
    parser.add_argument('--save_interval', type=int, default=10, help='Checkpoint save interval')
    parser.add_argument('--vis_interval', type=int, default=5, help='Visualization interval')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    
    # For compatibility with base Trainer
    parser.add_argument('--model', type=str, default='pretrained')
    parser.add_argument('--optimizer', type=str, default='adamw')
    
    return parser.parse_args()


def main():
    from train import set_seed
    
    # Parse arguments
    config = parse_args()
    
    # Set random seed
    set_seed(config.seed)
    
    # Create trainer and train
    trainer = ImprovedTrainer(config)
    trainer.train()


if __name__ == '__main__':
    main()
