"""
Improved Loss Functions with Class Weighting
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class WeightedBCEDiceLoss(nn.Module):
    """
    Weighted BCE + Dice Loss to handle class imbalance
    """
    def __init__(self, bce_weight=0.5, dice_weight=0.5, pos_weight=10.0):
        super(WeightedBCEDiceLoss, self).__init__()
        self.bce_weight = bce_weight
        self.dice_weight = dice_weight
        self.pos_weight = torch.tensor([pos_weight])
        
    def forward(self, logits, targets):
        # Move pos_weight to same device as logits
        if self.pos_weight.device != logits.device:
            self.pos_weight = self.pos_weight.to(logits.device)
        
        # Weighted BCE Loss (给正样本更高权重)
        bce_loss = F.binary_cross_entropy_with_logits(
            logits, targets, pos_weight=self.pos_weight
        )
        
        # Dice Loss
        probs = torch.sigmoid(logits)
        probs_flat = probs.view(-1)
        targets_flat = targets.view(-1)
        
        intersection = (probs_flat * targets_flat).sum()
        dice = (2. * intersection + 1e-6) / (probs_flat.sum() + targets_flat.sum() + 1e-6)
        dice_loss = 1 - dice
        
        combined_loss = self.bce_weight * bce_loss + self.dice_weight * dice_loss
        
        return combined_loss


class TverskyLoss(nn.Module):
    """
    Tversky Loss - generalization of Dice Loss
    Better for handling false negatives
    """
    def __init__(self, alpha=0.7, beta=0.3, smooth=1e-6):
        super(TverskyLoss, self).__init__()
        self.alpha = alpha  # weight for false positives
        self.beta = beta    # weight for false negatives
        self.smooth = smooth
        
    def forward(self, logits, targets):
        probs = torch.sigmoid(logits)
        
        # Flatten
        probs = probs.view(-1)
        targets = targets.view(-1)
        
        # True Positives, False Positives & False Negatives
        TP = (probs * targets).sum()
        FP = ((1 - targets) * probs).sum()
        FN = (targets * (1 - probs)).sum()
        
        Tversky = (TP + self.smooth) / (TP + self.alpha*FP + self.beta*FN + self.smooth)
        
        return 1 - Tversky


class FocalTverskyLoss(nn.Module):
    """
    Focal Tversky Loss - emphasizes hard examples
    """
    def __init__(self, alpha=0.7, beta=0.3, gamma=0.75, smooth=1e-6):
        super(FocalTverskyLoss, self).__init__()
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.smooth = smooth
        
    def forward(self, logits, targets):
        probs = torch.sigmoid(logits)
        
        # Flatten
        probs = probs.view(-1)
        targets = targets.view(-1)
        
        # True Positives, False Positives & False Negatives
        TP = (probs * targets).sum()
        FP = ((1 - targets) * probs).sum()
        FN = (targets * (1 - probs)).sum()
        
        Tversky = (TP + self.smooth) / (TP + self.alpha*FP + self.beta*FN + self.smooth)
        FocalTversky = (1 - Tversky) ** self.gamma
        
        return FocalTversky


class ComboLoss(nn.Module):
    """
    Combination of Weighted BCE + Dice + Focal Loss
    """
    def __init__(self, alpha=0.5, beta=0.5, pos_weight=10.0):
        super(ComboLoss, self).__init__()
        self.alpha = alpha
        self.beta = beta
        self.pos_weight = torch.tensor([pos_weight])
        
    def forward(self, logits, targets):
        if self.pos_weight.device != logits.device:
            self.pos_weight = self.pos_weight.to(logits.device)
        
        # Weighted BCE
        bce = F.binary_cross_entropy_with_logits(
            logits, targets, pos_weight=self.pos_weight
        )
        
        # Dice
        probs = torch.sigmoid(logits)
        probs_flat = probs.view(-1)
        targets_flat = targets.view(-1)
        intersection = (probs_flat * targets_flat).sum()
        dice = (2. * intersection + 1e-6) / (probs_flat.sum() + targets_flat.sum() + 1e-6)
        dice_loss = 1 - dice
        
        return self.alpha * bce + self.beta * dice_loss


if __name__ == '__main__':
    # Test loss functions
    batch_size = 4
    logits = torch.randn(batch_size, 1, 256, 256)
    targets = torch.randint(0, 2, (batch_size, 1, 256, 256)).float()
    
    print("Testing Weighted BCE+Dice Loss...")
    loss_fn = WeightedBCEDiceLoss(pos_weight=10.0)
    loss = loss_fn(logits, targets)
    print(f"Loss: {loss.item():.4f}\n")
    
    print("Testing Tversky Loss...")
    loss_fn = TverskyLoss(alpha=0.7, beta=0.3)
    loss = loss_fn(logits, targets)
    print(f"Loss: {loss.item():.4f}\n")
    
    print("Testing Focal Tversky Loss...")
    loss_fn = FocalTverskyLoss(alpha=0.7, beta=0.3, gamma=0.75)
    loss = loss_fn(logits, targets)
    print(f"Loss: {loss.item():.4f}\n")
