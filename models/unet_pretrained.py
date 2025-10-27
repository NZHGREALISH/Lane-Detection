"""
U-Net with Pretrained Encoder (ResNet/EfficientNet)
Using segmentation_models_pytorch library
"""
import torch
import torch.nn as nn
import segmentation_models_pytorch as smp


class PretrainedUNet(nn.Module):
    """
    U-Net with pretrained encoder for better generalization
    
    Args:
        encoder_name: Name of encoder (resnet34, resnet50, efficientnet-b0, etc.)
        encoder_weights: Pretrained weights ('imagenet' or None)
        n_classes: Number of output classes (1 for binary)
        activation: Output activation function
    """
    
    def __init__(self, encoder_name='resnet34', encoder_weights='imagenet', 
                 n_classes=1, activation=None):
        super(PretrainedUNet, self).__init__()
        
        self.model = smp.Unet(
            encoder_name=encoder_name,
            encoder_weights=encoder_weights,
            in_channels=3,
            classes=n_classes,
            activation=activation,
        )
    
    def forward(self, x):
        return self.model(x)


class PretrainedUNetPlusPlus(nn.Module):
    """
    U-Net++ with pretrained encoder for better performance
    """
    
    def __init__(self, encoder_name='resnet34', encoder_weights='imagenet', 
                 n_classes=1, activation=None):
        super(PretrainedUNetPlusPlus, self).__init__()
        
        self.model = smp.UnetPlusPlus(
            encoder_name=encoder_name,
            encoder_weights=encoder_weights,
            in_channels=3,
            classes=n_classes,
            activation=activation,
        )
    
    def forward(self, x):
        return self.model(x)


class PretrainedDeepLabV3Plus(nn.Module):
    """
    DeepLabV3+ with pretrained encoder
    """
    
    def __init__(self, encoder_name='resnet34', encoder_weights='imagenet', 
                 n_classes=1, activation=None):
        super(PretrainedDeepLabV3Plus, self).__init__()
        
        self.model = smp.DeepLabV3Plus(
            encoder_name=encoder_name,
            encoder_weights=encoder_weights,
            in_channels=3,
            classes=n_classes,
            activation=activation,
        )
    
    def forward(self, x):
        return self.model(x)


if __name__ == '__main__':
    # Test pretrained models
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    print("Testing Pretrained U-Net...")
    model = PretrainedUNet(encoder_name='resnet34', encoder_weights='imagenet').to(device)
    x = torch.randn(2, 3, 256, 256).to(device)
    output = model(x)
    print(f"Input shape: {x.shape}")
    print(f"Output shape: {output.shape}")
    
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}\n")
    
    print("Testing U-Net++...")
    model = PretrainedUNetPlusPlus(encoder_name='resnet34', encoder_weights='imagenet').to(device)
    output = model(x)
    print(f"Output shape: {output.shape}")
    
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Total parameters: {total_params:,}\n")
