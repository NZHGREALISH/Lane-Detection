"""
Test on BDD100K Test Set
Inference and visualization on test images
"""
import os
import argparse
import torch
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

from data.dataset import get_val_transforms
from models.unet_pretrained import PretrainedUNet, PretrainedUNetPlusPlus, PretrainedDeepLabV3Plus
from models.unet import UNet
from models.baseline_cnn import BaselineCNN
from utils.metrics import calculate_iou, calculate_dice_coefficient, calculate_pixel_accuracy
from utils.visualization import denormalize


class TestDataset(Dataset):
    """Test dataset that may or may not have labels"""
    
    def __init__(self, image_dir, mask_dir=None, transform=None, binary=True):
        """
        Args:
            image_dir: Directory containing test images
            mask_dir: Directory containing test masks (optional)
            transform: Transformations to apply
            binary: Whether to use binary labels
        """
        self.image_dir = image_dir
        self.mask_dir = mask_dir
        self.transform = transform
        self.binary = binary
        self.has_masks = False
        
        # Get all image files
        self.image_files = sorted([f for f in os.listdir(image_dir) 
                                   if f.endswith('.jpg') or f.endswith('.png')])
        
        # Check if masks exist
        if mask_dir and os.path.exists(mask_dir):
            # Check how many images have corresponding masks
            self.image_mask_pairs = []
            for img_file in self.image_files:
                mask_file = img_file.replace('.jpg', '_drivable_id.png').replace('.png', '_drivable_id.png')
                mask_path = os.path.join(mask_dir, mask_file)
                if os.path.exists(mask_path):
                    self.image_mask_pairs.append((img_file, mask_file))
            
            if len(self.image_mask_pairs) > 0:
                self.has_masks = True
                print(f"Found {len(self.image_mask_pairs)} test images with masks")
            else:
                print(f"Found {len(self.image_files)} test images (no masks available)")
        else:
            print(f"Found {len(self.image_files)} test images (no mask directory provided)")
    
    def __len__(self):
        if self.has_masks:
            return len(self.image_mask_pairs)
        return len(self.image_files)
    
    def __getitem__(self, idx):
        if self.has_masks:
            img_file, mask_file = self.image_mask_pairs[idx]
            img_path = os.path.join(self.image_dir, img_file)
            mask_path = os.path.join(self.mask_dir, mask_file)
            
            image = np.array(Image.open(img_path).convert('RGB'))
            mask = np.array(Image.open(mask_path))
            
            if self.binary:
                mask = (mask > 0).astype(np.uint8)
            
            if self.transform:
                transformed = self.transform(image=image, mask=mask)
                image = transformed['image']
                mask = transformed['mask']
            
            mask = mask.float().unsqueeze(0)
            return image, mask, img_file
        else:
            img_file = self.image_files[idx]
            img_path = os.path.join(self.image_dir, img_file)
            image = np.array(Image.open(img_path).convert('RGB'))
            
            if self.transform:
                transformed = self.transform(image=image)
                image = transformed['image']
            
            return image, img_file


def load_model(checkpoint_path, model_type='unet', encoder_name='resnet34'):
    """Load trained model"""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Build model
    if model_type == 'pretrained_unet':
        model = PretrainedUNet(encoder_name=encoder_name, encoder_weights=None)
    elif model_type == 'pretrained_unetplusplus':
        model = PretrainedUNetPlusPlus(encoder_name=encoder_name, encoder_weights=None)
    elif model_type == 'pretrained_deeplabv3plus':
        model = PretrainedDeepLabV3Plus(encoder_name=encoder_name, encoder_weights=None)
    elif model_type == 'unet':
        model = UNet(n_channels=3, n_classes=1, bilinear=True)
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
    
    model.to(device)
    model.eval()
    
    print(f"Model loaded from {checkpoint_path}")
    return model, device


def test_with_evaluation(model, test_loader, device, save_dir, threshold=0.5):
    """Test with ground truth available (compute metrics)"""
    print("\nRunning evaluation on test set...")
    
    results = {
        'iou': [],
        'dice': [],
        'pixel_acc': []
    }
    
    all_images = []
    all_masks = []
    all_preds = []
    all_filenames = []
    
    with torch.no_grad():
        for images, masks, filenames in tqdm(test_loader, desc='Testing'):
            images = images.to(device)
            masks = masks.to(device)
            
            # Forward pass
            logits = model(images)
            probs = torch.sigmoid(logits)
            
            # Calculate metrics
            iou = calculate_iou(probs, masks, threshold=threshold)
            dice = calculate_dice_coefficient(probs, masks, threshold=threshold)
            acc = calculate_pixel_accuracy(probs, masks, threshold=threshold)
            
            results['iou'].append(iou)
            results['dice'].append(dice)
            results['pixel_acc'].append(acc)
            
            # Save for visualization
            if len(all_images) < 16:
                all_images.append(images.cpu())
                all_masks.append(masks.cpu())
                all_preds.append(probs.cpu())
                all_filenames.extend(filenames)
    
    # Print results
    print("\n" + "="*80)
    print("Test Results:")
    print("-"*80)
    print(f"IoU:           {np.mean(results['iou']):.4f} ± {np.std(results['iou']):.4f}")
    print(f"Dice:          {np.mean(results['dice']):.4f} ± {np.std(results['dice']):.4f}")
    print(f"Pixel Acc:     {np.mean(results['pixel_acc']):.4f} ± {np.std(results['pixel_acc']):.4f}")
    print("="*80)
    
    # Visualize
    if all_images:
        visualize_results(all_images, all_masks, all_preds, all_filenames, 
                         save_dir, threshold, with_gt=True)
    
    return results


def test_without_evaluation(model, test_loader, device, save_dir, threshold=0.5):
    """Test without ground truth (inference only)"""
    print("\nRunning inference on test set...")
    
    all_images = []
    all_preds = []
    all_filenames = []
    
    with torch.no_grad():
        for images, filenames in tqdm(test_loader, desc='Testing'):
            images = images.to(device)
            
            # Forward pass
            logits = model(images)
            probs = torch.sigmoid(logits)
            
            # Save for visualization
            if len(all_images) < 16:
                all_images.append(images.cpu())
                all_preds.append(probs.cpu())
                all_filenames.extend(filenames)
    
    print(f"\nProcessed {len(test_loader) * test_loader.batch_size} images")
    
    # Visualize
    if all_images:
        visualize_results(all_images, None, all_preds, all_filenames, 
                         save_dir, threshold, with_gt=False)


def visualize_results(images, masks, preds, filenames, save_dir, threshold, with_gt=True):
    """Visualize results"""
    os.makedirs(save_dir, exist_ok=True)
    
    images = torch.cat(images, dim=0)
    preds = torch.cat(preds, dim=0)
    if with_gt and masks:
        masks = torch.cat(masks, dim=0)
    
    num_samples = min(16, images.shape[0])
    
    if with_gt:
        fig, axes = plt.subplots(num_samples, 4, figsize=(16, 4*num_samples))
        if num_samples == 1:
            axes = axes.reshape(1, -1)
    else:
        fig, axes = plt.subplots(num_samples, 3, figsize=(12, 4*num_samples))
        if num_samples == 1:
            axes = axes.reshape(1, -1)
    
    for idx in range(num_samples):
        # Denormalize image
        image = images[idx].cpu()
        image = denormalize(image)
        image = image.permute(1, 2, 0).numpy()
        image = np.clip(image, 0, 1)
        
        pred = preds[idx].squeeze().cpu().numpy()
        pred_binary = (pred > threshold).astype(np.float32)
        
        # Original image
        axes[idx, 0].imshow(image)
        axes[idx, 0].set_title(f'{filenames[idx][:20]}...')
        axes[idx, 0].axis('off')
        
        if with_gt:
            mask = masks[idx].squeeze().cpu().numpy()
            
            # Ground truth
            axes[idx, 1].imshow(mask, cmap='gray', vmin=0, vmax=1)
            axes[idx, 1].set_title('Ground Truth')
            axes[idx, 1].axis('off')
            
            # Prediction probability
            axes[idx, 2].imshow(pred, cmap='jet', vmin=0, vmax=1)
            axes[idx, 2].set_title('Prediction (Prob)')
            axes[idx, 2].axis('off')
            
            # Binary prediction
            axes[idx, 3].imshow(pred_binary, cmap='gray', vmin=0, vmax=1)
            axes[idx, 3].set_title(f'Prediction (th={threshold})')
            axes[idx, 3].axis('off')
        else:
            # Prediction probability
            axes[idx, 1].imshow(pred, cmap='jet', vmin=0, vmax=1)
            axes[idx, 1].set_title('Prediction (Prob)')
            axes[idx, 1].axis('off')
            
            # Binary prediction overlay
            axes[idx, 2].imshow(image)
            axes[idx, 2].imshow(pred_binary, cmap='Reds', alpha=0.5*pred_binary, vmin=0, vmax=1)
            axes[idx, 2].set_title(f'Overlay (th={threshold})')
            axes[idx, 2].axis('off')
    
    plt.tight_layout()
    save_path = os.path.join(save_dir, 'test_predictions.png')
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"\nVisualization saved to {save_path}")
    plt.close()


def main():
    parser = argparse.ArgumentParser(description='Test on BDD100K Test Set')
    
    parser.add_argument('--checkpoint', type=str, required=True,
                       help='Path to model checkpoint')
    parser.add_argument('--model_type', type=str, default='pretrained_unet',
                       choices=['pretrained_unet', 'pretrained_unetplusplus', 
                               'pretrained_deeplabv3plus', 'unet', 'baseline'],
                       help='Model type')
    parser.add_argument('--encoder_name', type=str, default='resnet34',
                       help='Encoder name for pretrained models')
    
    parser.add_argument('--image_dir', type=str,
                       default='/root/bdd100k_data/bdd100k_images/bdd100k/images/10k/test',
                       help='Test image directory')
    parser.add_argument('--mask_dir', type=str,
                       default='/root/bdd100k_data/bdd100k_drivable_maps/bdd100k/drivable_maps/labels/test',
                       help='Test mask directory (optional)')
    
    parser.add_argument('--batch_size', type=int, default=8, help='Batch size')
    parser.add_argument('--image_size', type=int, default=320, help='Image size')
    parser.add_argument('--threshold', type=float, default=0.5, help='Binary threshold')
    parser.add_argument('--save_dir', type=str, default='test_results', help='Save directory')
    
    args = parser.parse_args()
    
    # Load model
    model, device = load_model(args.checkpoint, args.model_type, args.encoder_name)
    
    # Create dataset
    test_dataset = TestDataset(
        image_dir=args.image_dir,
        mask_dir=args.mask_dir if os.path.exists(args.mask_dir) else None,
        transform=get_val_transforms(args.image_size),
        binary=True
    )
    
    # Custom collate function for handling different return types
    def collate_fn(batch):
        if test_dataset.has_masks:
            # With masks: (image, mask, filename)
            images = torch.stack([item[0] for item in batch])
            masks = torch.stack([item[1] for item in batch])
            filenames = [item[2] for item in batch]
            return images, masks, filenames
        else:
            # Without masks: (image, filename)
            images = torch.stack([item[0] for item in batch])
            filenames = [item[1] for item in batch]
            return images, filenames
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=True,
        collate_fn=collate_fn
    )
    
    # Run test
    os.makedirs(args.save_dir, exist_ok=True)
    
    if test_dataset.has_masks:
        results = test_with_evaluation(model, test_loader, device, args.save_dir, args.threshold)
    else:
        test_without_evaluation(model, test_loader, device, args.save_dir, args.threshold)


if __name__ == '__main__':
    main()
