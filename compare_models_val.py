"""
Compare multiple models on validation dataset (using custom train/val split)
"""
import os
import argparse
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import json

from models.unet import UNet
from models.baseline_cnn import BaselineCNN
from models.unet_pretrained import PretrainedUNet
from data.dataset import BDD100KDrivableDataset, get_val_transforms
from utils.metrics import (
    calculate_iou, 
    calculate_dice_coefficient,
    calculate_pixel_accuracy,
    calculate_precision_recall_f1
)


def load_model(checkpoint_path, model_type='unet', encoder_name='resnet34'):
    """Load trained model"""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Create model
    if model_type == 'pretrained_unet':
        model = PretrainedUNet(
            encoder_name=encoder_name,
            encoder_weights=None,
            n_classes=1,
            activation=None
        )
    elif model_type == 'unet':
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
    
    print(f"✓ Model loaded from {checkpoint_path}")
    return model, device


def denormalize(image, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]):
    """Denormalize image"""
    image = image.clone()
    for t, m, s in zip(image, mean, std):
        t.mul_(s).add_(m)
    return torch.clamp(image, 0, 1)


def run_inference(model, data_loader, device, model_name):
    """Run inference and collect predictions"""
    all_images = []
    all_masks = []
    all_preds = []
    
    print(f"\n{'='*80}")
    print(f"Running inference: {model_name}")
    print(f"{'='*80}")
    
    with torch.no_grad():
        for images, masks in tqdm(data_loader, desc='Inference'):
            images = images.to(device)
            masks = masks.to(device)
            
            # Forward pass
            logits = model(images)
            probs = torch.sigmoid(logits)
            
            all_images.append(images.cpu())
            all_masks.append(masks.cpu())
            all_preds.append(probs.cpu())
    
    # Concatenate
    all_images = torch.cat(all_images, dim=0)
    all_masks = torch.cat(all_masks, dim=0)
    all_preds = torch.cat(all_preds, dim=0)
    
    return all_images, all_masks, all_preds


def calculate_metrics(preds, masks, threshold=0.5):
    """Calculate evaluation metrics"""
    pred_binary = (preds > threshold).float()
    
    iou = calculate_iou(pred_binary, masks, threshold=0)
    dice = calculate_dice_coefficient(pred_binary, masks, threshold=0)
    acc = calculate_pixel_accuracy(pred_binary, masks, threshold=0)
    precision, recall, f1 = calculate_precision_recall_f1(pred_binary, masks, threshold=0)
    
    # Handle both tensor and float returns
    return {
        'IoU': iou.item() if hasattr(iou, 'item') else float(iou),
        'Dice': dice.item() if hasattr(dice, 'item') else float(dice),
        'Accuracy': acc.item() if hasattr(acc, 'item') else float(acc),
        'Precision': precision.item() if hasattr(precision, 'item') else float(precision),
        'Recall': recall.item() if hasattr(recall, 'item') else float(recall),
        'F1': f1.item() if hasattr(f1, 'item') else float(f1)
    }


def visualize_comparison(images, masks, preds_dict, save_dir, num_samples=12, threshold=0.5):
    """Visualize predictions from multiple models"""
    os.makedirs(save_dir, exist_ok=True)
    
    num_models = len(preds_dict)
    
    # Select samples (evenly distributed)
    num_samples = min(num_samples, len(images))
    indices = np.linspace(0, len(images)-1, num_samples, dtype=int)
    
    # Create figure: Original | GT | Model1 | Model2 | ... | Overlay
    cols = 3 + num_models
    fig, axes = plt.subplots(num_samples, cols, figsize=(cols*3, num_samples*3))
    
    if num_samples == 1:
        axes = axes.reshape(1, -1)
    
    for row_idx, img_idx in enumerate(indices):
        image = images[img_idx]
        mask = masks[img_idx, 0].numpy()
        
        # Denormalize image
        image_vis = denormalize(image).permute(1, 2, 0).numpy()
        
        col = 0
        
        # Original image
        axes[row_idx, col].imshow(image_vis)
        axes[row_idx, col].set_title('Original', fontsize=10, fontweight='bold')
        axes[row_idx, col].axis('off')
        col += 1
        
        # Ground truth
        axes[row_idx, col].imshow(mask, cmap='gray', vmin=0, vmax=1)
        axes[row_idx, col].set_title('Ground Truth', fontsize=10, fontweight='bold')
        axes[row_idx, col].axis('off')
        col += 1
        
        # Predictions from each model
        for model_name, preds in preds_dict.items():
            pred = preds[img_idx, 0].numpy()
            pred_binary = (pred > threshold).astype(float)
            
            # Calculate IoU for this sample
            mask_flat = torch.tensor(mask).flatten().unsqueeze(0)
            pred_flat = torch.tensor(pred_binary).flatten().unsqueeze(0)
            sample_iou = calculate_iou(pred_flat, mask_flat, threshold=0)
            
            axes[row_idx, col].imshow(pred_binary, cmap='gray', vmin=0, vmax=1)
            axes[row_idx, col].set_title(f'{model_name}\nIoU: {sample_iou:.3f}', 
                                        fontsize=10, fontweight='bold')
            axes[row_idx, col].axis('off')
            col += 1
        
        # Overlay comparison (GT in green, predictions in other colors)
        axes[row_idx, col].imshow(image_vis)
        
        # GT in green
        gt_overlay = np.zeros((*mask.shape, 4))
        gt_overlay[mask > 0.5] = [0, 1, 0, 0.3]  # Green with transparency
        axes[row_idx, col].imshow(gt_overlay)
        
        # Predictions in different colors
        colors = [(1, 0, 0, 0.4), (0, 0, 1, 0.4), (1, 1, 0, 0.4), (1, 0, 1, 0.4)]
        for (model_name, preds), color in zip(preds_dict.items(), colors):
            pred = preds[img_idx, 0].numpy()
            pred_binary = (pred > threshold).astype(float)
            
            overlay = np.zeros((*pred_binary.shape, 4))
            overlay[pred_binary > 0.5] = color
            axes[row_idx, col].imshow(overlay)
        
        axes[row_idx, col].set_title('Overlay\n(GT=Green)', fontsize=10, fontweight='bold')
        axes[row_idx, col].axis('off')
    
    plt.suptitle('Validation Set: Model Comparison', fontsize=16, fontweight='bold', y=0.995)
    plt.tight_layout()
    
    save_path = os.path.join(save_dir, 'validation_comparison.png')
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"\n✓ Visualization saved to: {save_path}")
    plt.close()


def plot_metrics_comparison(metrics_dict, save_dir):
    """Plot metrics comparison"""
    import pandas as pd
    
    df = pd.DataFrame(metrics_dict).T
    
    print("\n" + "="*80)
    print("📊 VALIDATION SET PERFORMANCE COMPARISON")
    print("="*80)
    print(df.to_string())
    print("="*80)
    
    # Save to CSV
    csv_path = os.path.join(save_dir, 'validation_metrics.csv')
    df.to_csv(csv_path)
    print(f"\n✓ Metrics table saved to: {csv_path}")
    
    # Create bar chart
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    axes = axes.flatten()
    
    colors = plt.cm.Set2(np.linspace(0, 1, len(df)))
    
    for idx, metric in enumerate(df.columns):
        ax = axes[idx]
        
        values = df[metric].values
        bars = ax.bar(range(len(df)), values, color=colors, edgecolor='black', linewidth=2)
        
        # Add value labels
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{height:.4f}',
                   ha='center', va='bottom', fontsize=11, fontweight='bold')
        
        ax.set_ylabel(metric, fontsize=13, fontweight='bold')
        ax.set_title(f'{metric} Comparison', fontsize=14, fontweight='bold')
        ax.set_xticks(range(len(df)))
        ax.set_xticklabels(df.index, rotation=15, ha='right')
        ax.set_ylim([0, min(1.05, max(values) * 1.1)])
        ax.grid(axis='y', alpha=0.3)
    
    plt.suptitle('Validation Metrics Comparison', fontsize=16, fontweight='bold')
    plt.tight_layout()
    
    metrics_path = os.path.join(save_dir, 'validation_metrics_chart.png')
    plt.savefig(metrics_path, dpi=150, bbox_inches='tight')
    print(f"✓ Metrics chart saved to: {metrics_path}")
    plt.close()
    
    # Create radar chart
    create_radar_chart(df, save_dir)


def create_radar_chart(df, save_dir):
    """Create radar chart for comparison"""
    import matplotlib.pyplot as plt
    from math import pi
    
    categories = list(df.columns)
    N = len(categories)
    
    # Compute angle for each category
    angles = [n / float(N) * 2 * pi for n in range(N)]
    angles += angles[:1]
    
    fig, ax = plt.subplots(figsize=(10, 10), subplot_kw=dict(projection='polar'))
    
    colors = plt.cm.Set2(np.linspace(0, 1, len(df)))
    
    for idx, (model_name, row) in enumerate(df.iterrows()):
        values = row.tolist()
        values += values[:1]
        
        ax.plot(angles, values, 'o-', linewidth=2, label=model_name, color=colors[idx])
        ax.fill(angles, values, alpha=0.15, color=colors[idx])
    
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(categories, size=12, fontweight='bold')
    ax.set_ylim(0, 1)
    ax.set_title('Performance Radar Chart', size=16, fontweight='bold', pad=20)
    ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1.1), fontsize=11)
    ax.grid(True)
    
    plt.tight_layout()
    radar_path = os.path.join(save_dir, 'validation_radar_chart.png')
    plt.savefig(radar_path, dpi=150, bbox_inches='tight')
    print(f"✓ Radar chart saved to: {radar_path}")
    plt.close()


def main():
    parser = argparse.ArgumentParser(description='Compare models on validation dataset')
    
    # Data
    parser.add_argument('--image_dir', type=str,
                       default='/root/bdd100k_data/bdd100k_images/bdd100k/images/10k',
                       help='Image directory')
    parser.add_argument('--mask_dir', type=str,
                       default='/root/bdd100k_data/bdd100k_drivable_maps/bdd100k/drivable_maps/labels',
                       help='Mask directory')
    
    # Models
    parser.add_argument('--models', nargs='+', required=True, help='Model checkpoint paths')
    parser.add_argument('--model_names', nargs='+', required=True, help='Model names')
    parser.add_argument('--model_types', nargs='+', required=True, help='Model types')
    parser.add_argument('--encoder_names', nargs='+', default=['resnet34'], help='Encoder names')
    
    # Settings
    parser.add_argument('--batch_size', type=int, default=16, help='Batch size')
    parser.add_argument('--image_size', type=int, default=320, help='Image size')
    parser.add_argument('--threshold', type=float, default=0.5, help='Binary threshold')
    parser.add_argument('--num_samples', type=int, default=12, help='Samples to visualize')
    parser.add_argument('--val_ratio', type=float, default=0.2, help='Validation ratio')
    
    # Output
    parser.add_argument('--save_dir', type=str, default='validation_comparison',
                       help='Save directory')
    
    args = parser.parse_args()
    
    # Validate
    if len(args.models) != len(args.model_names) or len(args.models) != len(args.model_types):
        raise ValueError("Number of models, names, and types must match!")
    
    if len(args.encoder_names) == 1:
        args.encoder_names = args.encoder_names * len(args.models)
    
    os.makedirs(args.save_dir, exist_ok=True)
    
    print("\n" + "="*80)
    print("🔍 MODEL COMPARISON ON VALIDATION SET")
    print("="*80)
    
    # Create validation dataset
    print("\nLoading validation dataset...")
    val_dataset = BDD100KDrivableDataset(
        image_dir=args.image_dir,
        mask_dir=args.mask_dir,
        split='val',
        transform=get_val_transforms(args.image_size),
        binary=True,
        val_ratio=args.val_ratio
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=True
    )
    
    # Run inference for all models
    all_preds = {}
    metrics_dict = {}
    images, masks = None, None
    
    for checkpoint, name, model_type, encoder in zip(
        args.models, args.model_names, args.model_types, args.encoder_names
    ):
        # Load model
        model, device = load_model(checkpoint, model_type, encoder)
        
        # Run inference
        images, masks, preds = run_inference(model, val_loader, device, name)
        all_preds[name] = preds
        
        # Calculate metrics
        metrics = calculate_metrics(preds, masks, args.threshold)
        metrics_dict[name] = metrics
        
        print(f"\n📈 {name} Metrics:")
        for metric_name, value in metrics.items():
            print(f"  {metric_name:12s}: {value:.4f}")
    
    # Visualize
    visualize_comparison(images, masks, all_preds, args.save_dir, 
                        args.num_samples, args.threshold)
    
    # Plot metrics
    plot_metrics_comparison(metrics_dict, args.save_dir)
    
    # Save metrics JSON
    json_path = os.path.join(args.save_dir, 'validation_metrics.json')
    with open(json_path, 'w') as f:
        json.dump(metrics_dict, f, indent=4)
    print(f"✓ Metrics JSON saved to: {json_path}")
    
    print(f"\n{'='*80}")
    print(f"✅ All results saved to: {args.save_dir}/")
    print(f"{'='*80}\n")


if __name__ == '__main__':
    main()
