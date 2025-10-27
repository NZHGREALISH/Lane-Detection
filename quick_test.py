"""
Quick Test Script: Verify data loading and models
"""
import torch
from data.dataset import BDD100KDrivableDataset, get_train_transforms, get_val_transforms
from models.unet import UNet
from models.baseline_cnn import BaselineCNN
from utils.losses import BCEDiceLoss
from utils.metrics import calculate_iou, calculate_dice_coefficient
from utils.visualization import visualize_predictions


def test_dataset():
    """Test dataset loading"""
    print("="*80)
    print("Testing Dataset Loading")
    print("="*80)
    
    image_dir = '/root/bdd100k_data/bdd100k_images/bdd100k/images/10k'
    mask_dir = '/root/bdd100k_data/bdd100k_drivable_maps/bdd100k/drivable_maps/labels'
    
    # Test training set
    train_dataset = BDD100KDrivableDataset(
        image_dir=image_dir,
        mask_dir=mask_dir,
        split='train',
        transform=get_train_transforms(256),
        binary=True
    )
    
    print(f"✓ Training set size: {len(train_dataset)}")
    
    # Test loading one sample
    image, mask = train_dataset[0]
    print(f"✓ Image shape: {image.shape}")
    print(f"✓ Mask shape: {mask.shape}")
    print(f"✓ Mask value range: [{mask.min():.2f}, {mask.max():.2f}]")
    print(f"✓ Mask unique values: {torch.unique(mask).tolist()}")
    
    # Test validation set
    val_dataset = BDD100KDrivableDataset(
        image_dir=image_dir,
        mask_dir=mask_dir,
        split='val',
        transform=get_val_transforms(256),
        binary=True
    )
    
    print(f"✓ Validation set size: {len(val_dataset)}")
    print("✓ Dataset loading test passed!\n")
    
    return train_dataset


def test_models():
    """Test models"""
    print("="*80)
    print("Testing Models")
    print("="*80)
    
    batch_size = 2
    x = torch.randn(batch_size, 3, 256, 256)
    
    # Test U-Net
    print("Testing U-Net...")
    unet = UNet(n_channels=3, n_classes=1, bilinear=True)
    output = unet(x)
    print(f"✓ U-Net input shape: {x.shape}")
    print(f"✓ U-Net output shape: {output.shape}")
    
    unet_params = sum(p.numel() for p in unet.parameters())
    print(f"✓ U-Net parameters: {unet_params:,}")
    
    # Test Baseline CNN
    print("\nTesting Baseline CNN...")
    baseline = BaselineCNN(n_channels=3, n_classes=1)
    output = baseline(x)
    print(f"✓ Baseline input shape: {x.shape}")
    print(f"✓ Baseline output shape: {output.shape}")
    
    baseline_params = sum(p.numel() for p in baseline.parameters())
    print(f"✓ Baseline parameters: {baseline_params:,}")
    
    print(f"\n✓ Parameter comparison: U-Net vs Baseline = {unet_params/baseline_params:.2f}x")
    print("✓ Model test passed!\n")


def test_loss_and_metrics():
    """Test loss functions and metrics"""
    print("="*80)
    print("Testing Loss Functions and Metrics")
    print("="*80)
    
    batch_size = 4
    logits = torch.randn(batch_size, 1, 256, 256)
    targets = torch.randint(0, 2, (batch_size, 1, 256, 256)).float()
    
    # Test loss functions
    criterion = BCEDiceLoss()
    loss = criterion(logits, targets)
    print(f"✓ BCE+Dice Loss: {loss.item():.4f}")
    
    # Test metrics
    probs = torch.sigmoid(logits)
    iou = calculate_iou(probs, targets)
    dice = calculate_dice_coefficient(probs, targets)
    
    print(f"✓ IoU: {iou:.4f}")
    print(f"✓ Dice: {dice:.4f}")
    print("✓ Loss functions and metrics test passed!\n")


def test_visualization(dataset):
    """Test visualization"""
    print("="*80)
    print("Testing Visualization")
    print("="*80)
    
    # Get some samples
    images = []
    masks = []
    for i in range(4):
        img, mask = dataset[i]
        images.append(img)
        masks.append(mask)
    
    images = torch.stack(images)
    masks = torch.stack(masks)
    
    # Generate random predictions for testing
    preds = torch.sigmoid(torch.randn_like(masks))
    
    # Test visualization
    print("Generating visualizations...")
    visualize_predictions(images, masks, preds, num_samples=4, save_path='/root/Lane-Detection/save_visual/test_visualization.png')
    print("✓ Visualization test passed!\n")


def test_forward_pass():
    """Test complete forward pass"""
    print("="*80)
    print("Testing Complete Forward Pass")
    print("="*80)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Create model
    model = UNet(n_channels=3, n_classes=1, bilinear=True).to(device)
    criterion = BCEDiceLoss()
    
    # Create dummy data
    batch_size = 4
    images = torch.randn(batch_size, 3, 256, 256).to(device)
    masks = torch.randint(0, 2, (batch_size, 1, 256, 256)).float().to(device)
    
    # Forward pass
    model.train()
    logits = model(images)
    loss = criterion(logits, masks)
    
    # Backward pass
    loss.backward()
    
    print(f"✓ Forward pass output shape: {logits.shape}")
    print(f"✓ Loss value: {loss.item():.4f}")
    print("✓ Backward pass successful")
    
    # Inference mode
    model.eval()
    with torch.no_grad():
        logits = model(images)
        probs = torch.sigmoid(logits)
    
    print(f"✓ Inference output shape: {probs.shape}")
    print(f"✓ Inference output range: [{probs.min():.4f}, {probs.max():.4f}]")
    print("✓ Complete forward pass test passed!\n")


def main():
    """Main test function"""
    print("\n" + "="*80)
    print("Drivable Area Segmentation - Quick Test")
    print("="*80 + "\n")
    
    try:
        # Test dataset
        dataset = test_dataset()
        
        # Test models
        test_models()
        
        # Test loss functions and metrics
        test_loss_and_metrics()
        
        # Test visualization
        test_visualization(dataset)
        
        # Test complete forward pass
        test_forward_pass()
        
        print("="*80)
        print("✓ All tests passed!")
        print("="*80 + "\n")
        
        print("Tip: You can now run 'python train.py' to start training")
        
    except Exception as e:
        print(f"\n❌ Test failed: {str(e)}")
        import traceback
        traceback.print_exc()


if __name__ == '__main__':
    main()
