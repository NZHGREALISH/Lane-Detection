"""
Compare multiple models on test dataset
"""
import os
import argparse
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from tqdm import tqdm
import json

from models.unet import UNet
from models.baseline_cnn import BaselineCNN
from models.unet_pretrained import PretrainedUNet
from data.dataset import get_val_transforms
from utils.metrics import (
    calculate_iou, 
    calculate_dice_coefficient,
    calculate_pixel_accuracy,
    calculate_precision_recall_f1
)


class TestDataset(Dataset):
    """Test dataset for inference"""
    
    def __init__(self, image_dir, mask_dir=None, transform=None, binary=True):
        self.image_dir = image_dir
        self.mask_dir = mask_dir
        self.transform = transform
        self.binary = binary
        
        # Get all image files
        self.images = sorted([f for f in os.listdir(image_dir) if f.endswith('.jpg')])
        
        # Check if masks exist
        self.has_masks = False
        if mask_dir and os.path.exists(mask_dir):
            # Check if corresponding masks exist
            test_img = self.images[0]
            test_mask = test_img.replace('.jpg', '_drivable_id.png')
            if os.path.exists(os.path.join(mask_dir, test_mask)):
                self.has_masks = True
        
        print(f"Found {len(self.images)} test images")
        if self.has_masks:
            print(f"Ground truth masks available")
        else:
            print(f"No ground truth masks (inference only)")
    
    def __len__(self):
        return len(self.images)
    
    def __getitem__(self, idx):
        # Load image
        img_file = self.images[idx]
        img_path = os.path.join(self.image_dir, img_file)
        image = np.array(Image.open(img_path).convert('RGB'))
        
        # Load mask if available
        mask = None
        if self.has_masks:
            mask_file = img_file.replace('.jpg', '_drivable_id.png')
            mask_path = os.path.join(self.mask_dir, mask_file)
            mask = np.array(Image.open(mask_path))
            if self.binary:
                mask = (mask > 0).astype(np.uint8)
        
        # Apply transform
        if self.transform:
            if mask is not None:
                transformed = self.transform(image=image, mask=mask)
                image = transformed['image']
                mask = transformed['mask'].float().unsqueeze(0)
            else:
                transformed = self.transform(image=image)
                image = transformed['image']
        
        if mask is not None:
            return image, mask, img_file
        else:
            return image, img_file


def load_model(checkpoint_path, model_type='unet', encoder_name='resnet34'):
    """Load trained model"""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Create model
    if model_type == 'pretrained_unet' or model_type == 'unet':
        if 'resnet' in encoder_name or 'efficientnet' in encoder_name:
            model = PretrainedUNet(
                encoder_name=encoder_name,
                encoder_weights=None,  # Load from checkpoint
                n_classes=1,
                activation=None
            )
        else:
            model = UNet(n_channels=3, n_classes=1)
    elif model_type == 'baseline':
        model = BaselineCNN(n_channels=3, n_classes=1)
    else:
        raise ValueError(f"Unknown model type: {model_type}")
    
    # Load checkpoint
    checkpoint = torch.load(checkpoint_path, map_location=device)
    if 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
    else:
        model.load_state_dict(checkpoint)
    
    model = model.to(device)
    model.eval()
    
    print(f"Model loaded from {checkpoint_path}")
    return model, device


def denormalize(image, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]):
    """Denormalize image"""
    image = image.clone()
    for t, m, s in zip(image, mean, std):
        t.mul_(s).add_(m)
    return torch.clamp(image, 0, 1)


def run_inference(model, data_loader, device, model_name):
    """Run inference on test set"""
    all_images = []
    all_masks = []
    all_preds = []
    all_filenames = []
    
    has_masks = False
    
    print(f"\nRunning inference with {model_name}...")
    
    with torch.no_grad():
        for batch in tqdm(data_loader, desc='Testing'):
            if len(batch) == 3:
                images, masks, filenames = batch
                has_masks = True
                all_masks.append(masks.cpu())
            else:
                images, filenames = batch
            
            images = images.to(device)
            
            # Forward pass
            logits = model(images)
            probs = torch.sigmoid(logits)
            
            all_images.append(images.cpu())
            all_preds.append(probs.cpu())
            all_filenames.extend(filenames)
    
    # Concatenate
    all_images = torch.cat(all_images, dim=0)
    all_preds = torch.cat(all_preds, dim=0)
    if has_masks:
        all_masks = torch.cat(all_masks, dim=0)
    
    return all_images, all_masks if has_masks else None, all_preds, all_filenames


def calculate_metrics(preds, masks, threshold=0.5):
    """Calculate metrics"""
    pred_binary = (preds > threshold).float()
    
    iou = calculate_iou(pred_binary, masks, threshold=0)
    dice = calculate_dice_coefficient(pred_binary, masks, threshold=0)
    acc = calculate_pixel_accuracy(pred_binary, masks, threshold=0)
    precision, recall, f1 = calculate_precision_recall_f1(pred_binary, masks, threshold=0)
    
    return {
        'IoU': iou.item(),
        'Dice': dice.item(),
        'Accuracy': acc.item(),
        'Precision': precision.item(),
        'Recall': recall.item(),
        'F1': f1.item()
    }


def visualize_comparison(images, masks, preds_dict, filenames, save_dir, num_samples=8, threshold=0.5):
    """Visualize predictions from multiple models"""
    os.makedirs(save_dir, exist_ok=True)
    
    num_models = len(preds_dict)
    has_masks = masks is not None
    
    # Select samples
    num_samples = min(num_samples, len(images))
    indices = np.linspace(0, len(images)-1, num_samples, dtype=int)
    
    # Create figure
    cols = 3 + num_models if has_masks else 2 + num_models
    fig, axes = plt.subplots(num_samples, cols, figsize=(cols*3, num_samples*3))
    
    if num_samples == 1:
        axes = axes.reshape(1, -1)
    
    for idx, img_idx in enumerate(indices):
        image = images[img_idx]
        
        # Denormalize image
        image_vis = denormalize(image).permute(1, 2, 0).numpy()
        
        col = 0
        
        # Original image
        axes[idx, col].imshow(image_vis)
        axes[idx, col].set_title(f'Image\n{filenames[img_idx][:15]}...', fontsize=9)
        axes[idx, col].axis('off')
        col += 1
        
        # Ground truth mask
        if has_masks:
            mask = masks[img_idx, 0].numpy()
            axes[idx, col].imshow(mask, cmap='gray', vmin=0, vmax=1)
            axes[idx, col].set_title('Ground Truth', fontsize=9)
            axes[idx, col].axis('off')
            col += 1
        
        # Predictions from each model
        for model_name, preds in preds_dict.items():
            pred = preds[img_idx, 0].numpy()
            pred_binary = (pred > threshold).astype(float)
            
            axes[idx, col].imshow(pred_binary, cmap='gray', vmin=0, vmax=1)
            
            # Calculate IoU for this sample if masks available
            if has_masks:
                mask_flat = masks[img_idx].flatten()
                pred_flat = torch.tensor(pred_binary).flatten()
                sample_iou = calculate_iou(pred_flat.unsqueeze(0), mask_flat.unsqueeze(0), threshold=0)
                axes[idx, col].set_title(f'{model_name}\nIoU: {sample_iou:.3f}', fontsize=9)
            else:
                axes[idx, col].set_title(model_name, fontsize=9)
            
            axes[idx, col].axis('off')
            col += 1
        
        # Overlay comparison
        axes[idx, col].imshow(image_vis)
        
        # Overlay predictions with different colors
        colors = [(1, 0, 0, 0.5), (0, 1, 0, 0.5), (0, 0, 1, 0.5), (1, 1, 0, 0.5)]
        for (model_name, preds), color in zip(preds_dict.items(), colors):
            pred = preds[img_idx, 0].numpy()
            pred_binary = (pred > threshold).astype(float)
            
            overlay = np.zeros((*pred_binary.shape, 4))
            overlay[pred_binary > 0.5] = color
            axes[idx, col].imshow(overlay)
        
        axes[idx, col].set_title('Overlay Comparison', fontsize=9)
        axes[idx, col].axis('off')
    
    plt.tight_layout()
    save_path = os.path.join(save_dir, 'test_comparison.png')
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"\nComparison visualization saved to: {save_path}")
    plt.close()


def plot_metrics_comparison(metrics_dict, save_dir):
    """Plot metrics comparison bar chart"""
    import pandas as pd
    
    df = pd.DataFrame(metrics_dict).T
    
    # Create bar plot
    fig, ax = plt.subplots(figsize=(12, 6))
    
    x = np.arange(len(df.columns))
    width = 0.8 / len(df)
    
    colors = plt.cm.Set2(np.linspace(0, 1, len(df)))
    
    for i, (model_name, row) in enumerate(df.iterrows()):
        offset = (i - len(df)/2 + 0.5) * width
        bars = ax.bar(x + offset, row.values, width, label=model_name, color=colors[i])
        
        # Add value labels
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{height:.3f}',
                   ha='center', va='bottom', fontsize=9)
    
    ax.set_xlabel('Metrics', fontsize=12, fontweight='bold')
    ax.set_ylabel('Score', fontsize=12, fontweight='bold')
    ax.set_title('Test Set Performance Comparison', fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(df.columns, rotation=0)
    ax.legend(loc='lower right', fontsize=10)
    ax.set_ylim([0, 1.05])
    ax.grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    save_path = os.path.join(save_dir, 'metrics_comparison.png')
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"Metrics comparison saved to: {save_path}")
    plt.close()
    
    # Print table
    print("\n" + "="*80)
    print("Test Set Performance Comparison")
    print("="*80)
    print(df.to_string())
    print("="*80)
    
    # Save to CSV
    csv_path = os.path.join(save_dir, 'test_metrics.csv')
    df.to_csv(csv_path)
    print(f"Metrics saved to: {csv_path}")


def main():
    parser = argparse.ArgumentParser(description='Compare models on test dataset')
    
    # Data
    parser.add_argument('--image_dir', type=str, 
                       default='/root/bdd100k_data/bdd100k_images/bdd100k/images/10k/test',
                       help='Test image directory')
    parser.add_argument('--mask_dir', type=str,
                       default='/root/bdd100k_data/bdd100k_drivable_maps/bdd100k/drivable_maps/labels/test',
                       help='Test mask directory (optional)')
    
    # Models to compare
    parser.add_argument('--models', nargs='+', required=True,
                       help='Model checkpoint paths')
    parser.add_argument('--model_names', nargs='+', required=True,
                       help='Model names for display')
    parser.add_argument('--model_types', nargs='+', required=True,
                       help='Model types (pretrained_unet, unet, baseline)')
    parser.add_argument('--encoder_names', nargs='+', default=['resnet34'],
                       help='Encoder names (for pretrained models)')
    
    # Test settings
    parser.add_argument('--batch_size', type=int, default=8, help='Batch size')
    parser.add_argument('--image_size', type=int, default=320, help='Input image size')
    parser.add_argument('--threshold', type=float, default=0.5, help='Binary threshold')
    parser.add_argument('--num_samples', type=int, default=8, help='Number of samples to visualize')
    
    # Output
    parser.add_argument('--save_dir', type=str, default='test_comparison',
                       help='Directory to save results')
    
    args = parser.parse_args()
    
    # Validate inputs
    if len(args.models) != len(args.model_names) or len(args.models) != len(args.model_types):
        raise ValueError("Number of models, model_names, and model_types must match!")
    
    # Expand encoder_names if needed
    if len(args.encoder_names) == 1 and len(args.models) > 1:
        args.encoder_names = args.encoder_names * len(args.models)
    
    os.makedirs(args.save_dir, exist_ok=True)
    
    # Create test dataset
    test_dataset = TestDataset(
        image_dir=args.image_dir,
        mask_dir=args.mask_dir if os.path.exists(args.mask_dir) else None,
        transform=get_val_transforms(args.image_size),
        binary=True
    )
    
    # Custom collate function
    def collate_fn(batch):
        if test_dataset.has_masks:
            images = torch.stack([item[0] for item in batch])
            masks = torch.stack([item[1] for item in batch])
            filenames = [item[2] for item in batch]
            return images, masks, filenames
        else:
            images = torch.stack([item[0] for item in batch])
            filenames = [item[1] for item in batch]
            return images, filenames
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=True,
        collate_fn=collate_fn
    )
    
    # Load all models and run inference
    all_preds = {}
    metrics_dict = {}
    
    for checkpoint, model_name, model_type, encoder_name in zip(
        args.models, args.model_names, args.model_types, args.encoder_names
    ):
        # Load model
        model, device = load_model(checkpoint, model_type, encoder_name)
        
        # Run inference
        images, masks, preds, filenames = run_inference(model, test_loader, device, model_name)
        all_preds[model_name] = preds
        
        # Calculate metrics if masks available
        if masks is not None:
            metrics = calculate_metrics(preds, masks, args.threshold)
            metrics_dict[model_name] = metrics
            print(f"\n{model_name} Metrics:")
            for metric_name, value in metrics.items():
                print(f"  {metric_name}: {value:.4f}")
    
    # Visualize comparison
    visualize_comparison(images, masks, all_preds, filenames, 
                        args.save_dir, args.num_samples, args.threshold)
    
    # Plot metrics comparison
    if metrics_dict:
        plot_metrics_comparison(metrics_dict, args.save_dir)
        
        # Save metrics to JSON
        json_path = os.path.join(args.save_dir, 'test_metrics.json')
        with open(json_path, 'w') as f:
            json.dump(metrics_dict, f, indent=4)
        print(f"Metrics saved to: {json_path}")
    
    print(f"\nAll results saved to: {args.save_dir}")


if __name__ == '__main__':
    main()
