"""
快速测试脚本：验证数据加载和模型
"""
import torch
from data.dataset import BDD100KDrivableDataset, get_train_transforms, get_val_transforms
from models.unet import UNet
from models.baseline_cnn import BaselineCNN
from utils.losses import BCEDiceLoss
from utils.metrics import calculate_iou, calculate_dice_coefficient
from utils.visualization import visualize_predictions


def test_dataset():
    """测试数据集加载"""
    print("="*80)
    print("测试数据集加载")
    print("="*80)
    
    image_dir = '/home/grealish/APS360/bdd100k_data/bdd100k_images/bdd100k/images/10k'
    mask_dir = '/home/grealish/APS360/bdd100k_data/bdd100k_drivable_maps/bdd100k/drivable_maps/labels'
    
    # 测试训练集
    train_dataset = BDD100KDrivableDataset(
        image_dir=image_dir,
        mask_dir=mask_dir,
        split='train',
        transform=get_train_transforms(256),
        binary=True
    )
    
    print(f"✓ 训练集大小: {len(train_dataset)}")
    
    # 测试加载一个样本
    image, mask = train_dataset[0]
    print(f"✓ 图像形状: {image.shape}")
    print(f"✓ Mask形状: {mask.shape}")
    print(f"✓ Mask取值范围: [{mask.min():.2f}, {mask.max():.2f}]")
    print(f"✓ Mask唯一值: {torch.unique(mask).tolist()}")
    
    # 测试验证集
    val_dataset = BDD100KDrivableDataset(
        image_dir=image_dir,
        mask_dir=mask_dir,
        split='val',
        transform=get_val_transforms(256),
        binary=True
    )
    
    print(f"✓ 验证集大小: {len(val_dataset)}")
    print("✓ 数据集加载测试通过！\n")
    
    return train_dataset


def test_models():
    """测试模型"""
    print("="*80)
    print("测试模型")
    print("="*80)
    
    batch_size = 2
    x = torch.randn(batch_size, 3, 256, 256)
    
    # 测试U-Net
    print("测试U-Net...")
    unet = UNet(n_channels=3, n_classes=1, bilinear=True)
    output = unet(x)
    print(f"✓ U-Net输入形状: {x.shape}")
    print(f"✓ U-Net输出形状: {output.shape}")
    
    unet_params = sum(p.numel() for p in unet.parameters())
    print(f"✓ U-Net参数量: {unet_params:,}")
    
    # 测试Baseline CNN
    print("\n测试Baseline CNN...")
    baseline = BaselineCNN(n_channels=3, n_classes=1)
    output = baseline(x)
    print(f"✓ Baseline输入形状: {x.shape}")
    print(f"✓ Baseline输出形状: {output.shape}")
    
    baseline_params = sum(p.numel() for p in baseline.parameters())
    print(f"✓ Baseline参数量: {baseline_params:,}")
    
    print(f"\n✓ 参数量对比: U-Net vs Baseline = {unet_params/baseline_params:.2f}x")
    print("✓ 模型测试通过！\n")


def test_loss_and_metrics():
    """测试损失函数和指标"""
    print("="*80)
    print("测试损失函数和指标")
    print("="*80)
    
    batch_size = 4
    logits = torch.randn(batch_size, 1, 256, 256)
    targets = torch.randint(0, 2, (batch_size, 1, 256, 256)).float()
    
    # 测试损失函数
    criterion = BCEDiceLoss()
    loss = criterion(logits, targets)
    print(f"✓ BCE+Dice Loss: {loss.item():.4f}")
    
    # 测试指标
    probs = torch.sigmoid(logits)
    iou = calculate_iou(probs, targets)
    dice = calculate_dice_coefficient(probs, targets)
    
    print(f"✓ IoU: {iou:.4f}")
    print(f"✓ Dice: {dice:.4f}")
    print("✓ 损失函数和指标测试通过！\n")


def test_visualization(dataset):
    """测试可视化"""
    print("="*80)
    print("测试可视化")
    print("="*80)
    
    # 获取一些样本
    images = []
    masks = []
    for i in range(4):
        img, mask = dataset[i]
        images.append(img)
        masks.append(mask)
    
    images = torch.stack(images)
    masks = torch.stack(masks)
    
    # 生成随机预测用于测试
    preds = torch.sigmoid(torch.randn_like(masks))
    
    # 测试可视化
    print("生成可视化...")
    visualize_predictions(images, masks, preds, num_samples=4, save_path='test_visualization.png')
    print("✓ 可视化测试通过！\n")


def test_forward_pass():
    """测试完整前向传播"""
    print("="*80)
    print("测试完整前向传播")
    print("="*80)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"使用设备: {device}")
    
    # 创建模型
    model = UNet(n_channels=3, n_classes=1, bilinear=True).to(device)
    criterion = BCEDiceLoss()
    
    # 创建虚拟数据
    batch_size = 4
    images = torch.randn(batch_size, 3, 256, 256).to(device)
    masks = torch.randint(0, 2, (batch_size, 1, 256, 256)).float().to(device)
    
    # 前向传播
    model.train()
    logits = model(images)
    loss = criterion(logits, masks)
    
    # 反向传播
    loss.backward()
    
    print(f"✓ 前向传播输出形状: {logits.shape}")
    print(f"✓ 损失值: {loss.item():.4f}")
    print("✓ 反向传播成功")
    
    # 推理模式
    model.eval()
    with torch.no_grad():
        logits = model(images)
        probs = torch.sigmoid(logits)
    
    print(f"✓ 推理输出形状: {probs.shape}")
    print(f"✓ 推理输出范围: [{probs.min():.4f}, {probs.max():.4f}]")
    print("✓ 完整前向传播测试通过！\n")


def main():
    """主测试函数"""
    print("\n" + "="*80)
    print("Drivable Area Segmentation - 快速测试")
    print("="*80 + "\n")
    
    try:
        # 测试数据集
        dataset = test_dataset()
        
        # 测试模型
        test_models()
        
        # 测试损失函数和指标
        test_loss_and_metrics()
        
        # 测试可视化
        test_visualization(dataset)
        
        # 测试完整前向传播
        test_forward_pass()
        
        print("="*80)
        print("✓ 所有测试通过！")
        print("="*80 + "\n")
        
        print("提示: 现在可以运行 'python train.py' 开始训练")
        
    except Exception as e:
        print(f"\n❌ 测试失败: {str(e)}")
        import traceback
        traceback.print_exc()


if __name__ == '__main__':
    main()
