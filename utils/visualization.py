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
    反归一化图像用于可视化
    
    Args:
        image: 归一化后的图像 [C, H, W] 或 [B, C, H, W]
        mean: 归一化均值
        std: 归一化标准差
    
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
    可视化预测结果
    
    Args:
        images: 输入图像 [B, C, H, W]
        masks: 真实mask [B, 1, H, W]
        preds: 预测概率 [B, 1, H, W]
        num_samples: 显示样本数
        threshold: 二值化阈值
        save_path: 保存路径
    """
    num_samples = min(num_samples, images.shape[0])
    
    fig, axes = plt.subplots(num_samples, 4, figsize=(16, 4*num_samples))
    
    if num_samples == 1:
        axes = axes.reshape(1, -1)
    
    for idx in range(num_samples):
        # 准备图像
        image = images[idx].cpu()
        mask = masks[idx].squeeze().cpu().numpy()
        pred = preds[idx].squeeze().cpu().numpy()
        pred_binary = (pred > threshold).astype(np.float32)
        
        # 反归一化图像
        image = denormalize(image)
        image = image.permute(1, 2, 0).numpy()
        image = np.clip(image, 0, 1)
        
        # 显示原图
        axes[idx, 0].imshow(image)
        axes[idx, 0].set_title('Original Image')
        axes[idx, 0].axis('off')
        
        # 显示真实mask
        axes[idx, 1].imshow(mask, cmap='gray')
        axes[idx, 1].set_title('Ground Truth')
        axes[idx, 1].axis('off')
        
        # 显示预测概率
        axes[idx, 2].imshow(pred, cmap='jet', vmin=0, vmax=1)
        axes[idx, 2].set_title('Prediction (Prob)')
        axes[idx, 2].axis('off')
        
        # 显示二值化预测
        axes[idx, 3].imshow(pred_binary, cmap='gray')
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
    将预测结果叠加到原图上可视化
    
    Args:
        images: 输入图像 [B, C, H, W]
        masks: 真实mask [B, 1, H, W]
        preds: 预测概率 [B, 1, H, W]
        num_samples: 显示样本数
        threshold: 二值化阈值
        alpha: 叠加透明度
        save_path: 保存路径
    """
    num_samples = min(num_samples, images.shape[0])
    
    fig, axes = plt.subplots(num_samples, 3, figsize=(12, 4*num_samples))
    
    if num_samples == 1:
        axes = axes.reshape(1, -1)
    
    for idx in range(num_samples):
        # 准备图像
        image = images[idx].cpu()
        mask = masks[idx].squeeze().cpu().numpy()
        pred = preds[idx].squeeze().cpu().numpy()
        pred_binary = (pred > threshold).astype(np.float32)
        
        # 反归一化图像
        image = denormalize(image)
        image = image.permute(1, 2, 0).numpy()
        image = np.clip(image, 0, 1)
        
        # 显示原图
        axes[idx, 0].imshow(image)
        axes[idx, 0].set_title('Original Image')
        axes[idx, 0].axis('off')
        
        # 显示GT叠加
        axes[idx, 1].imshow(image)
        axes[idx, 1].imshow(mask, cmap='Greens', alpha=alpha * mask)
        axes[idx, 1].set_title('Ground Truth Overlay')
        axes[idx, 1].axis('off')
        
        # 显示预测叠加
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
    绘制训练曲线
    
    Args:
        history: 训练历史字典，包含loss和metrics
        save_path: 保存路径
    """
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # Loss曲线
    axes[0, 0].plot(history['train_loss'], label='Train Loss')
    axes[0, 0].plot(history['val_loss'], label='Val Loss')
    axes[0, 0].set_xlabel('Epoch')
    axes[0, 0].set_ylabel('Loss')
    axes[0, 0].set_title('Training and Validation Loss')
    axes[0, 0].legend()
    axes[0, 0].grid(True)
    
    # IoU曲线
    if 'train_iou' in history:
        axes[0, 1].plot(history['train_iou'], label='Train IoU')
        axes[0, 1].plot(history['val_iou'], label='Val IoU')
        axes[0, 1].set_xlabel('Epoch')
        axes[0, 1].set_ylabel('IoU')
        axes[0, 1].set_title('Training and Validation IoU')
        axes[0, 1].legend()
        axes[0, 1].grid(True)
    
    # Dice系数曲线
    if 'train_dice' in history:
        axes[1, 0].plot(history['train_dice'], label='Train Dice')
        axes[1, 0].plot(history['val_dice'], label='Val Dice')
        axes[1, 0].set_xlabel('Epoch')
        axes[1, 0].set_ylabel('Dice Coefficient')
        axes[1, 0].set_title('Training and Validation Dice')
        axes[1, 0].legend()
        axes[1, 0].grid(True)
    
    # 像素准确率曲线
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
    # 测试可视化
    batch_size = 4
    images = torch.randn(batch_size, 3, 256, 256)
    masks = torch.randint(0, 2, (batch_size, 1, 256, 256)).float()
    preds = torch.sigmoid(torch.randn(batch_size, 1, 256, 256))
    
    visualize_predictions(images, masks, preds, num_samples=2)
    visualize_overlay(images, masks, preds, num_samples=2)
