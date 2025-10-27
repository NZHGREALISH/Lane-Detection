"""
Loss Functions for Segmentation
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class DiceLoss(nn.Module):
    """
    Dice Loss用于处理类别不平衡问题
    """
    def __init__(self, smooth=1e-6):
        super(DiceLoss, self).__init__()
        self.smooth = smooth
    
    def forward(self, logits, targets):
        """
        Args:
            logits: 模型输出 [B, 1, H, W]
            targets: 目标mask [B, 1, H, W]
        """
        # 应用sigmoid
        probs = torch.sigmoid(logits)
        
        # 展平
        probs = probs.view(-1)
        targets = targets.view(-1)
        
        # 计算Dice系数
        intersection = (probs * targets).sum()
        dice = (2. * intersection + self.smooth) / (probs.sum() + targets.sum() + self.smooth)
        
        # Dice Loss = 1 - Dice Coefficient
        return 1 - dice


class BCEDiceLoss(nn.Module):
    """
    组合BCE Loss和Dice Loss
    """
    def __init__(self, bce_weight=0.5, dice_weight=0.5):
        super(BCEDiceLoss, self).__init__()
        self.bce_weight = bce_weight
        self.dice_weight = dice_weight
        self.bce = nn.BCEWithLogitsLoss()
        self.dice = DiceLoss()
    
    def forward(self, logits, targets):
        """
        Args:
            logits: 模型输出 [B, 1, H, W]
            targets: 目标mask [B, 1, H, W]
        """
        bce_loss = self.bce(logits, targets)
        dice_loss = self.dice(logits, targets)
        
        combined_loss = self.bce_weight * bce_loss + self.dice_weight * dice_loss
        
        return combined_loss


class FocalLoss(nn.Module):
    """
    Focal Loss用于处理难样本
    """
    def __init__(self, alpha=0.25, gamma=2.0):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
    
    def forward(self, logits, targets):
        """
        Args:
            logits: 模型输出 [B, 1, H, W]
            targets: 目标mask [B, 1, H, W]
        """
        # BCE loss
        bce_loss = F.binary_cross_entropy_with_logits(logits, targets, reduction='none')
        
        # 概率
        probs = torch.sigmoid(logits)
        pt = torch.where(targets == 1, probs, 1 - probs)
        
        # Focal weight
        focal_weight = (1 - pt) ** self.gamma
        
        # Alpha weight
        alpha_weight = torch.where(targets == 1, self.alpha, 1 - self.alpha)
        
        # Focal loss
        focal_loss = alpha_weight * focal_weight * bce_loss
        
        return focal_loss.mean()


if __name__ == '__main__':
    # 测试损失函数
    batch_size = 4
    height, width = 256, 256
    
    logits = torch.randn(batch_size, 1, height, width)
    targets = torch.randint(0, 2, (batch_size, 1, height, width)).float()
    
    # 测试Dice Loss
    dice_loss = DiceLoss()
    loss = dice_loss(logits, targets)
    print(f"Dice Loss: {loss.item():.4f}")
    
    # 测试BCE+Dice Loss
    bce_dice_loss = BCEDiceLoss()
    loss = bce_dice_loss(logits, targets)
    print(f"BCE+Dice Loss: {loss.item():.4f}")
    
    # 测试Focal Loss
    focal_loss = FocalLoss()
    loss = focal_loss(logits, targets)
    print(f"Focal Loss: {loss.item():.4f}")
