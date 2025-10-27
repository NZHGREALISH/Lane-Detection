"""
Evaluation Script for Drivable Area Segmentation
"""
import os
import argparse
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
import numpy as np
import json

from data.dataset import BDD100KDrivableDataset, get_val_transforms
from models.unet import UNet
from models.baseline_cnn import BaselineCNN
from utils.metrics import (calculate_iou, calculate_miou, calculate_pixel_accuracy, 
                          calculate_dice_coefficient, calculate_precision_recall_f1)
from utils.visualization import visualize_predictions, visualize_overlay


class Evaluator:
    def __init__(self, config):
        self.config = config
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"Using device: {self.device}")
        
        # Load model
        self.model = self._load_model()
        self.model.to(self.device)
        self.model.eval()
        
        # Load data
        self.val_loader = self._build_dataloader()
        
        # Result storage
        self.results = {
            'iou': [],
            'miou': [],
            'dice': [],
            'pixel_acc': [],
            'precision': [],
            'recall': [],
            'f1': []
        }
    
    def _load_model(self):
        """Load model"""
        if self.config.model == 'unet':
            model = UNet(n_channels=3, n_classes=1, bilinear=True)
        elif self.config.model == 'baseline':
            model = BaselineCNN(n_channels=3, n_classes=1)
        else:
            raise ValueError(f"Unknown model: {self.config.model}")
        
        # Load weights
        checkpoint = torch.load(self.config.checkpoint, map_location=self.device)
        model.load_state_dict(checkpoint['model_state_dict'])
        
        print(f"Model loaded from {self.config.checkpoint}")
        if 'best_val_iou' in checkpoint:
            print(f"Best validation IoU: {checkpoint['best_val_iou']:.4f}")
        
        return model
    
    def _build_dataloader(self):
        """Build data loader"""
        val_dataset = BDD100KDrivableDataset(
            image_dir=self.config.image_dir,
            mask_dir=self.config.mask_dir,
            split='val',
            transform=get_val_transforms(self.config.image_size),
            binary=True
        )
        
        val_loader = DataLoader(
            val_dataset,
            batch_size=self.config.batch_size,
            shuffle=False,
            num_workers=self.config.num_workers,
            pin_memory=True
        )
        
        print(f"Val samples: {len(val_dataset)}")
        
        return val_loader
    
    @torch.no_grad()
    def evaluate(self):
        """Evaluate model"""
        print("\nStarting evaluation...")
        print("="*80)
        
        # Save samples for visualization
        sample_images = []
        sample_masks = []
        sample_preds = []
        
        pbar = tqdm(self.val_loader, desc='Evaluating')
        for images, masks in pbar:
            images = images.to(self.device)
            masks = masks.to(self.device)
            
            # Forward pass
            logits = self.model(images)
            probs = torch.sigmoid(logits)
            
            # Calculate metrics
            iou = calculate_iou(probs, masks, threshold=self.config.threshold)
            miou, _ = calculate_miou(probs, masks, num_classes=2, threshold=self.config.threshold)
            dice = calculate_dice_coefficient(probs, masks, threshold=self.config.threshold)
            acc = calculate_pixel_accuracy(probs, masks, threshold=self.config.threshold)
            precision, recall, f1 = calculate_precision_recall_f1(probs, masks, threshold=self.config.threshold)
            
            # Record results
            self.results['iou'].append(iou)
            self.results['miou'].append(miou)
            self.results['dice'].append(dice)
            self.results['pixel_acc'].append(acc)
            self.results['precision'].append(precision)
            self.results['recall'].append(recall)
            self.results['f1'].append(f1)
            
            # Save samples
            if len(sample_images) < self.config.num_vis_samples:
                sample_images.append(images.cpu())
                sample_masks.append(masks.cpu())
                sample_preds.append(probs.cpu())
            
            pbar.set_postfix({
                'IoU': f'{iou:.4f}',
                'Dice': f'{dice:.4f}'
            })
        
        # Calculate average metrics
        avg_results = {k: np.mean(v) for k, v in self.results.items()}
        std_results = {k: np.std(v) for k, v in self.results.items()}
        
        # Print results
        print("\n" + "="*80)
        print("Evaluation Results:")
        print("-"*80)
        print(f"IoU:           {avg_results['iou']:.4f} ± {std_results['iou']:.4f}")
        print(f"mIoU:          {avg_results['miou']:.4f} ± {std_results['miou']:.4f}")
        print(f"Dice:          {avg_results['dice']:.4f} ± {std_results['dice']:.4f}")
        print(f"Pixel Acc:     {avg_results['pixel_acc']:.4f} ± {std_results['pixel_acc']:.4f}")
        print(f"Precision:     {avg_results['precision']:.4f} ± {std_results['precision']:.4f}")
        print(f"Recall:        {avg_results['recall']:.4f} ± {std_results['recall']:.4f}")
        print(f"F1 Score:      {avg_results['f1']:.4f} ± {std_results['f1']:.4f}")
        print("="*80)
        
        # Save results
        if self.config.save_results:
            self.save_results(avg_results, std_results)
        
        # Visualization
        if sample_images and self.config.visualize:
            sample_images = torch.cat(sample_images, dim=0)
            sample_masks = torch.cat(sample_masks, dim=0)
            sample_preds = torch.cat(sample_preds, dim=0)
            
            self.visualize_results(sample_images, sample_masks, sample_preds)
        
        return avg_results
    
    def save_results(self, avg_results, std_results):
        """Save evaluation results"""
        results = {
            'average': avg_results,
            'std': std_results,
            'config': vars(self.config)
        }
        
        save_path = os.path.join(self.config.save_dir, 'evaluation_results.json')
        os.makedirs(self.config.save_dir, exist_ok=True)
        
        with open(save_path, 'w') as f:
            json.dump(results, f, indent=4)
        
        print(f"\nResults saved to {save_path}")
    
    def visualize_results(self, images, masks, preds):
        """Visualize results"""
        print("\nGenerating visualizations...")
        
        os.makedirs(os.path.join(self.config.save_dir, 'visualizations'), exist_ok=True)
        
        # Standard visualization
        vis_path = os.path.join(self.config.save_dir, 'visualizations', 'predictions.png')
        visualize_predictions(images[:8], masks[:8], preds[:8], 
                            num_samples=8, threshold=self.config.threshold, save_path=vis_path)
        
        # Overlay visualization
        overlay_path = os.path.join(self.config.save_dir, 'visualizations', 'overlay.png')
        visualize_overlay(images[:8], masks[:8], preds[:8], 
                         num_samples=8, threshold=self.config.threshold, save_path=overlay_path)
        
        print(f"Visualizations saved to {self.config.save_dir}/visualizations/")


def parse_args():
    parser = argparse.ArgumentParser(description='Evaluate Drivable Area Segmentation Model')
    
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
    parser.add_argument('--checkpoint', type=str, required=True, help='Model checkpoint path')
    
    # Evaluation related
    parser.add_argument('--batch_size', type=int, default=16, help='Batch size')
    parser.add_argument('--threshold', type=float, default=0.5, help='Binary threshold')
    parser.add_argument('--num_workers', type=int, default=4, help='Number of data loading workers')
    
    # Output related
    parser.add_argument('--save_dir', type=str, default='evaluation_results', help='Save directory')
    parser.add_argument('--save_results', action='store_true', default=True, help='Save results')
    parser.add_argument('--visualize', action='store_true', default=True, help='Visualize results')
    parser.add_argument('--num_vis_samples', type=int, default=16, help='Number of visualization samples')
    
    return parser.parse_args()


def main():
    # Parse arguments
    config = parse_args()
    
    # Create evaluator and evaluate
    evaluator = Evaluator(config)
    evaluator.evaluate()


if __name__ == '__main__':
    main()
