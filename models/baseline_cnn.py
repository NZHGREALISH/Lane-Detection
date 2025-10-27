"""
Baseline CNN Model Implementation
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class BaselineCNN(nn.Module):
    """
    简单的Baseline CNN用于语义分割
    基本结构: 卷积下采样 -> 上采样
    
    Args:
        n_channels: 输入通道数 (RGB为3)
        n_classes: 输出类别数 (二分类为1)
    """
    
    def __init__(self, n_channels=3, n_classes=1):
        super(BaselineCNN, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        
        # Encoder - 卷积下采样
        self.conv1 = nn.Sequential(
            nn.Conv2d(n_channels, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2)  # 128x128
        )
        
        self.conv2 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2)  # 64x64
        )
        
        self.conv3 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2)  # 32x32
        )
        
        self.conv4 = nn.Sequential(
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2)  # 16x16
        )
        
        # Decoder - 上采样
        self.upconv1 = nn.Sequential(
            nn.Conv2d(256, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)  # 32x32
        )
        
        self.upconv2 = nn.Sequential(
            nn.Conv2d(128, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)  # 64x64
        )
        
        self.upconv3 = nn.Sequential(
            nn.Conv2d(64, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)  # 128x128
        )
        
        self.upconv4 = nn.Sequential(
            nn.Conv2d(32, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)  # 256x256
        )
        
        # Output layer
        self.outc = nn.Conv2d(32, n_classes, kernel_size=1)
    
    def forward(self, x):
        # Encoder
        x = self.conv1(x)  # 128x128
        x = self.conv2(x)  # 64x64
        x = self.conv3(x)  # 32x32
        x = self.conv4(x)  # 16x16
        
        # Decoder
        x = self.upconv1(x)  # 32x32
        x = self.upconv2(x)  # 64x64
        x = self.upconv3(x)  # 128x128
        x = self.upconv4(x)  # 256x256
        
        # Output
        logits = self.outc(x)
        return logits


if __name__ == '__main__':
    # 测试模型
    model = BaselineCNN(n_channels=3, n_classes=1)
    x = torch.randn(2, 3, 256, 256)
    output = model(x)
    print(f"Input shape: {x.shape}")
    print(f"Output shape: {output.shape}")
    
    # 计算参数量
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
