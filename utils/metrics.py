"""
Evaluation Metrics for Segmentation
"""
import torch
import numpy as np
from sklearn.metrics import confusion_matrix


def calculate_iou(pred, target, threshold=0.5):
    """
    计算IoU (Intersection over Union)
    
    Args:
        pred: 预测概率 [B, 1, H, W] 或 [B, H, W]
        target: 目标mask [B, 1, H, W] 或 [B, H, W]
        threshold: 二值化阈值
    
    Returns:
        iou: IoU值
    """
    # 确保维度一致
    if pred.dim() == 4:
        pred = pred.squeeze(1)
    if target.dim() == 4:
        target = target.squeeze(1)
    
    # 二值化预测
    pred_binary = (pred > threshold).float()
    target_binary = target.float()
    
    # 计算交集和并集
    intersection = (pred_binary * target_binary).sum()
    union = pred_binary.sum() + target_binary.sum() - intersection
    
    # 避免除以0
    if union == 0:
        return 1.0 if intersection == 0 else 0.0
    
    iou = intersection / union
    return iou.item()


def calculate_miou(pred, target, num_classes=2, threshold=0.5):
    """
    计算mIoU (mean Intersection over Union)
    
    Args:
        pred: 预测概率 [B, 1, H, W]
        target: 目标mask [B, 1, H, W]
        num_classes: 类别数
        threshold: 二值化阈值
    
    Returns:
        miou: mIoU值
        iou_per_class: 每个类别的IoU
    """
    # 确保维度一致
    if pred.dim() == 4:
        pred = pred.squeeze(1)
    if target.dim() == 4:
        target = target.squeeze(1)
    
    # 二值化预测
    pred_binary = (pred > threshold).long()
    target_binary = target.long()
    
    # 展平
    pred_flat = pred_binary.view(-1).cpu().numpy()
    target_flat = target_binary.view(-1).cpu().numpy()
    
    iou_per_class = []
    
    for cls in range(num_classes):
        pred_cls = (pred_flat == cls)
        target_cls = (target_flat == cls)
        
        intersection = np.logical_and(pred_cls, target_cls).sum()
        union = np.logical_or(pred_cls, target_cls).sum()
        
        if union == 0:
            iou = 1.0 if intersection == 0 else 0.0
        else:
            iou = intersection / union
        
        iou_per_class.append(iou)
    
    miou = np.mean(iou_per_class)
    
    return miou, iou_per_class


def calculate_pixel_accuracy(pred, target, threshold=0.5):
    """
    计算像素准确率
    
    Args:
        pred: 预测概率 [B, 1, H, W]
        target: 目标mask [B, 1, H, W]
        threshold: 二值化阈值
    
    Returns:
        accuracy: 像素准确率
    """
    # 确保维度一致
    if pred.dim() == 4:
        pred = pred.squeeze(1)
    if target.dim() == 4:
        target = target.squeeze(1)
    
    # 二值化预测
    pred_binary = (pred > threshold).float()
    target_binary = target.float()
    
    # 计算准确率
    correct = (pred_binary == target_binary).sum()
    total = target_binary.numel()
    accuracy = correct / total
    
    return accuracy.item()


def calculate_dice_coefficient(pred, target, threshold=0.5):
    """
    计算Dice系数
    
    Args:
        pred: 预测概率 [B, 1, H, W]
        target: 目标mask [B, 1, H, W]
        threshold: 二值化阈值
    
    Returns:
        dice: Dice系数
    """
    # 确保维度一致
    if pred.dim() == 4:
        pred = pred.squeeze(1)
    if target.dim() == 4:
        target = target.squeeze(1)
    
    # 二值化预测
    pred_binary = (pred > threshold).float()
    target_binary = target.float()
    
    # 计算Dice系数
    intersection = (pred_binary * target_binary).sum()
    dice = (2. * intersection) / (pred_binary.sum() + target_binary.sum() + 1e-6)
    
    return dice.item()


def calculate_precision_recall_f1(pred, target, threshold=0.5):
    """
    计算Precision, Recall, F1 Score
    
    Args:
        pred: 预测概率 [B, 1, H, W]
        target: 目标mask [B, 1, H, W]
        threshold: 二值化阈值
    
    Returns:
        precision, recall, f1: 各项指标
    """
    # 确保维度一致
    if pred.dim() == 4:
        pred = pred.squeeze(1)
    if target.dim() == 4:
        target = target.squeeze(1)
    
    # 二值化预测
    pred_binary = (pred > threshold).float()
    target_binary = target.float()
    
    # 计算TP, FP, FN
    tp = (pred_binary * target_binary).sum()
    fp = (pred_binary * (1 - target_binary)).sum()
    fn = ((1 - pred_binary) * target_binary).sum()
    
    # 计算指标
    precision = tp / (tp + fp + 1e-6)
    recall = tp / (tp + fn + 1e-6)
    f1 = 2 * precision * recall / (precision + recall + 1e-6)
    
    return precision.item(), recall.item(), f1.item()


if __name__ == '__main__':
    # 测试指标
    batch_size = 4
    height, width = 256, 256
    
    pred = torch.sigmoid(torch.randn(batch_size, 1, height, width))
    target = torch.randint(0, 2, (batch_size, 1, height, width)).float()
    
    iou = calculate_iou(pred, target)
    print(f"IoU: {iou:.4f}")
    
    miou, iou_per_class = calculate_miou(pred, target)
    print(f"mIoU: {miou:.4f}")
    print(f"IoU per class: {iou_per_class}")
    
    accuracy = calculate_pixel_accuracy(pred, target)
    print(f"Pixel Accuracy: {accuracy:.4f}")
    
    dice = calculate_dice_coefficient(pred, target)
    print(f"Dice Coefficient: {dice:.4f}")
    
    precision, recall, f1 = calculate_precision_recall_f1(pred, target)
    print(f"Precision: {precision:.4f}, Recall: {recall:.4f}, F1: {f1:.4f}")
