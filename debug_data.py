"""
Debug script to check data loading
"""
import torch
import numpy as np
from data.dataset import BDD100KDrivableDataset, get_train_transforms, get_val_transforms
import matplotlib.pyplot as plt
from PIL import Image

def check_dataset():
    """Check if dataset is loading correctly"""
    print("="*80)
    print("Debugging Dataset")
    print("="*80)
    
    image_dir = '/root/bdd100k_data/bdd100k_images/bdd100k/images/10k'
    mask_dir = '/root/bdd100k_data/bdd100k_drivable_maps/bdd100k/drivable_maps/labels'
    
    # Create dataset without transforms first
    print("\n1. Checking raw data (no transforms)...")
    dataset_raw = BDD100KDrivableDataset(
        image_dir=image_dir,
        mask_dir=mask_dir,
        split='train',
        transform=None,
        binary=False  # Check original labels
    )
    
    # Check first sample
    img_name = dataset_raw.images[0]
    print(f"\nFirst image: {img_name}")
    
    # Load raw mask
    mask_name = img_name.replace('.jpg', '_drivable_id.png')
    mask_path = f"{mask_dir}/train/{mask_name}"
    print(f"Mask path: {mask_path}")
    
    try:
        raw_mask = np.array(Image.open(mask_path))
        print(f"Raw mask shape: {raw_mask.shape}")
        print(f"Raw mask dtype: {raw_mask.dtype}")
        print(f"Raw mask unique values: {np.unique(raw_mask)}")
        print(f"Raw mask min/max: {raw_mask.min()}/{raw_mask.max()}")
        print(f"Raw mask value counts:")
        unique, counts = np.unique(raw_mask, return_counts=True)
        for val, count in zip(unique, counts):
            percentage = (count / raw_mask.size) * 100
            print(f"  Value {val}: {count} pixels ({percentage:.2f}%)")
    except Exception as e:
        print(f"Error loading mask: {e}")
        return
    
    # Now check with binary conversion
    print("\n2. Checking with binary conversion...")
    dataset_binary = BDD100KDrivableDataset(
        image_dir=image_dir,
        mask_dir=mask_dir,
        split='train',
        transform=get_train_transforms(256),
        binary=True
    )
    
    image, mask = dataset_binary[0]
    print(f"Processed image shape: {image.shape}")
    print(f"Processed mask shape: {mask.shape}")
    print(f"Processed mask dtype: {mask.dtype}")
    print(f"Processed mask min/max: {mask.min()}/{mask.max()}")
    print(f"Processed mask unique values: {torch.unique(mask)}")
    
    # Count pixels
    total_pixels = mask.numel()
    positive_pixels = (mask > 0).sum().item()
    print(f"\nPixel statistics:")
    print(f"  Total pixels: {total_pixels}")
    print(f"  Positive pixels (drivable): {positive_pixels} ({positive_pixels/total_pixels*100:.2f}%)")
    print(f"  Negative pixels (background): {total_pixels - positive_pixels} ({(total_pixels-positive_pixels)/total_pixels*100:.2f}%)")
    
    # Visualize a few samples
    print("\n3. Creating visualization...")
    fig, axes = plt.subplots(3, 3, figsize=(12, 12))
    
    for i in range(3):
        image, mask = dataset_binary[i]
        
        # Denormalize image
        img_np = image.permute(1, 2, 0).numpy()
        img_np = img_np * np.array([0.229, 0.224, 0.225]) + np.array([0.485, 0.456, 0.406])
        img_np = np.clip(img_np, 0, 1)
        
        mask_np = mask.squeeze().numpy()
        
        axes[i, 0].imshow(img_np)
        axes[i, 0].set_title(f'Sample {i}: Image')
        axes[i, 0].axis('off')
        
        axes[i, 1].imshow(mask_np, cmap='gray', vmin=0, vmax=1)
        axes[i, 1].set_title(f'Mask (min={mask_np.min():.2f}, max={mask_np.max():.2f})')
        axes[i, 1].axis('off')
        
        # Overlay
        axes[i, 2].imshow(img_np)
        axes[i, 2].imshow(mask_np, cmap='Reds', alpha=0.5 * mask_np, vmin=0, vmax=1)
        axes[i, 2].set_title(f'Overlay')
        axes[i, 2].axis('off')
    
    plt.tight_layout()
    plt.savefig('debug_samples.png', dpi=150, bbox_inches='tight')
    print(f"\nVisualization saved to debug_samples.png")
    
    # Check validation set too
    print("\n4. Checking validation set...")
    dataset_val = BDD100KDrivableDataset(
        image_dir=image_dir,
        mask_dir=mask_dir,
        split='val',
        transform=get_val_transforms(256),
        binary=True
    )
    
    val_image, val_mask = dataset_val[0]
    print(f"Val mask shape: {val_mask.shape}")
    print(f"Val mask min/max: {val_mask.min()}/{val_mask.max()}")
    print(f"Val mask unique values: {torch.unique(val_mask)}")
    val_positive = (val_mask > 0).sum().item()
    print(f"Val positive pixels: {val_positive} ({val_positive/val_mask.numel()*100:.2f}%)")


if __name__ == '__main__':
    check_dataset()
