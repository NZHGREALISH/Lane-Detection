"""
Visualization Utilities
"""
import torch
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import os


def denormalize(image, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]):
    """
    Denormalize image for visualization
    
    Args:
        image: Normalized image [C, H, W] or [B, C, H, W]
        mean: Normalization mean
        std: Normalization standard deviation
    
    Returns:
        denormalized image
    """
    if isinstance(image, torch.Tensor):
        image = image.clone()
    
    if image.dim() == 4:
        # Batch images
        for i in range(len(mean)):
            image[:, i, :, :] = image[:, i, :, :] * std[i] + mean[i]
    else:
        # Single image
        for i in range(len(mean)):
            image[i, :, :] = image[i, :, :] * std[i] + mean[i]
    
    return image


def visualize_predictions(images, masks, preds, num_samples=4, threshold=0.5, save_path=None):
    """
    Visualize prediction results
    
    Args:
        images: Input images [B, C, H, W]
        masks: Ground truth masks [B, 1, H, W]
        preds: Predicted probabilities [B, 1, H, W]
        num_samples: Number of samples to display
        threshold: Binary threshold
        save_path: Path to save visualization
    """
    num_samples = min(num_samples, images.shape[0])
    
    fig, axes = plt.subplots(num_samples, 4, figsize=(16, 4*num_samples))
    
    if num_samples == 1:
        axes = axes.reshape(1, -1)
    
    for idx in range(num_samples):
        # Prepare images
        image = images[idx].cpu()
        mask = masks[idx].squeeze().cpu().numpy()
        pred = preds[idx].squeeze().cpu().numpy()
        pred_binary = (pred > threshold).astype(np.float32)
        
        # Denormalize image
        image = denormalize(image)
        image = image.permute(1, 2, 0).numpy()
        image = np.clip(image, 0, 1)
        
        # Show original image
        axes[idx, 0].imshow(image)
        axes[idx, 0].set_title('Original Image')
        axes[idx, 0].axis('off')
        
        # Show ground truth mask (ensure it's in 0-1 range)
        mask_vis = np.clip(mask, 0, 1)  # Ensure mask is in valid range
        axes[idx, 1].imshow(mask_vis, cmap='gray', vmin=0, vmax=1)
        axes[idx, 1].set_title(f'Ground Truth (range: {mask.min():.2f}-{mask.max():.2f})')
        axes[idx, 1].axis('off')
        
        # Show prediction probabilities
        axes[idx, 2].imshow(pred, cmap='jet', vmin=0, vmax=1)
        axes[idx, 2].set_title('Prediction (Prob)')
        axes[idx, 2].axis('off')
        
        # Show binarized prediction
        axes[idx, 3].imshow(pred_binary, cmap='gray', vmin=0, vmax=1)
        axes[idx, 3].set_title(f'Prediction (Binary, th={threshold})')
        axes[idx, 3].axis('off')
    
    plt.tight_layout()
    
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Visualization saved to {save_path}")
    else:
        plt.show()
    
    plt.close()


def visualize_overlay(images, masks, preds, num_samples=4, threshold=0.5, alpha=0.5, save_path=None):
    """
    Visualize predictions overlaid on original images
    
    Args:
        images: Input images [B, C, H, W]
        masks: Ground truth masks [B, 1, H, W]
        preds: Predicted probabilities [B, 1, H, W]
        num_samples: Number of samples to display
        threshold: Binary threshold
        alpha: Overlay transparency
        save_path: Path to save visualization
    """
    num_samples = min(num_samples, images.shape[0])
    
    fig, axes = plt.subplots(num_samples, 3, figsize=(12, 4*num_samples))
    
    if num_samples == 1:
        axes = axes.reshape(1, -1)
    
    for idx in range(num_samples):
        # Prepare images
        image = images[idx].cpu()
        mask = masks[idx].squeeze().cpu().numpy()
        pred = preds[idx].squeeze().cpu().numpy()
        pred_binary = (pred > threshold).astype(np.float32)
        
        # Denormalize image
        image = denormalize(image)
        image = image.permute(1, 2, 0).numpy()
        image = np.clip(image, 0, 1)
        
        # Show original image
        axes[idx, 0].imshow(image)
        axes[idx, 0].set_title('Original Image')
        axes[idx, 0].axis('off')
        
        # Show GT overlay
        axes[idx, 1].imshow(image)
        axes[idx, 1].imshow(mask, cmap='Greens', alpha=alpha * mask)
        axes[idx, 1].set_title('Ground Truth Overlay')
        axes[idx, 1].axis('off')
        
        # Show prediction overlay
        axes[idx, 2].imshow(image)
        axes[idx, 2].imshow(pred_binary, cmap='Reds', alpha=alpha * pred_binary)
        axes[idx, 2].set_title('Prediction Overlay')
        axes[idx, 2].axis('off')
    
    plt.tight_layout()
    
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Overlay visualization saved to {save_path}")
    else:
        plt.show()
    
    plt.close()


def plot_training_curves(history, save_path=None):
    """
    Plot training curves
    
    Args:
        history: Training history dictionary containing loss and metrics
        save_path: Path to save plot
    """
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # Loss curve
    axes[0, 0].plot(history['train_loss'], label='Train Loss')
    axes[0, 0].plot(history['val_loss'], label='Val Loss')
    axes[0, 0].set_xlabel('Epoch')
    axes[0, 0].set_ylabel('Loss')
    axes[0, 0].set_title('Training and Validation Loss')
    axes[0, 0].legend()
    axes[0, 0].grid(True)
    
    # IoU curve
    if 'train_iou' in history:
        axes[0, 1].plot(history['train_iou'], label='Train IoU')
        axes[0, 1].plot(history['val_iou'], label='Val IoU')
        axes[0, 1].set_xlabel('Epoch')
        axes[0, 1].set_ylabel('IoU')
        axes[0, 1].set_title('Training and Validation IoU')
        axes[0, 1].legend()
        axes[0, 1].grid(True)
    
    # Dice coefficient curve
    if 'train_dice' in history:
        axes[1, 0].plot(history['train_dice'], label='Train Dice')
        axes[1, 0].plot(history['val_dice'], label='Val Dice')
        axes[1, 0].set_xlabel('Epoch')
        axes[1, 0].set_ylabel('Dice Coefficient')
        axes[1, 0].set_title('Training and Validation Dice')
        axes[1, 0].legend()
        axes[1, 0].grid(True)
    
    # Pixel accuracy curve
    if 'train_acc' in history:
        axes[1, 1].plot(history['train_acc'], label='Train Accuracy')
        axes[1, 1].plot(history['val_acc'], label='Val Accuracy')
        axes[1, 1].set_xlabel('Epoch')
        axes[1, 1].set_ylabel('Pixel Accuracy')
        axes[1, 1].set_title('Training and Validation Accuracy')
        axes[1, 1].legend()
        axes[1, 1].grid(True)
    
    plt.tight_layout()
    
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Training curves saved to {save_path}")
    else:
        plt.show()
    
    plt.close()


if __name__ == '__main__':
    # Test visualization
    batch_size = 4
    images = torch.randn(batch_size, 3, 256, 256)
    masks = torch.randint(0, 2, (batch_size, 1, 256, 256)).float()
    preds = torch.sigmoid(torch.randn(batch_size, 1, 256, 256))
    
    visualize_predictions(images, masks, preds, num_samples=2)
    visualize_overlay(images, masks, preds, num_samples=2)
