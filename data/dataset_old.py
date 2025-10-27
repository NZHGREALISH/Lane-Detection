"""
BDD100K Drivable Area Dataset - Fixed Version
Handles the case where not all images have masks
"""
import os
import torch
from torch.utils.data import Dataset
from PIL import Image
import numpy as np
from torchvision import transforms
import albumentations as A
from albumentations.pytorch import ToTensorV2
from sklearn.model_selection import train_test_split


class BDD100KDrivableDataset(Dataset):
    """BDD100K Drivable Area Dataset with automatic train/val split"""
    
    def __init__(self, image_dir, mask_dir, split='train', transform=None, binary=True, 
                 val_ratio=0.2, random_state=42):
        """
        Args:
            image_dir: Image directory path (contains train/val/test folders)
            mask_dir: Mask directory path (contains train/val/test folders)
            split: 'train' or 'val' (we'll split the available data)
            transform: albumentations transforms
            binary: Whether to convert 3-class to binary
            val_ratio: Validation split ratio
            random_state: Random seed for splitting
        """
        self.binary = binary
        self.transform = transform
        
        # Collect all available image-mask pairs from all splits
        all_pairs = []
        
        # Check all possible splits (train, val, test)
        for data_split in ['train', 'val', 'test']:
            img_split_dir = os.path.join(image_dir, data_split)
            mask_split_dir = os.path.join(mask_dir, data_split)
            
            if not os.path.exists(img_split_dir) or not os.path.exists(mask_split_dir):
                continue
            
            # Get all mask files
            try:
                mask_files = [f for f in os.listdir(mask_split_dir) 
                             if f.endswith('_drivable_id.png')]
            except:
                continue
            
            # Check which images exist
            for mask_name in mask_files:
                img_name = mask_name.replace('_drivable_id.png', '.jpg')
                img_path = os.path.join(img_split_dir, img_name)
                mask_path = os.path.join(mask_split_dir, mask_name)
                
                if os.path.exists(img_path):
                    all_pairs.append((img_path, mask_path))
        
        print(f"Found {len(all_pairs)} image-mask pairs in total")
        
        # Split into train and val
        if len(all_pairs) == 0:
            raise ValueError("No valid image-mask pairs found!")
        
        train_pairs, val_pairs = train_test_split(
            all_pairs, 
            test_size=val_ratio, 
            random_state=random_state
        )
        
        # Select appropriate split
        if split == 'train':
            self.pairs = train_pairs
        elif split == 'val':
            self.pairs = val_pairs
        else:
            raise ValueError(f"Unknown split: {split}")
        
        print(f"Using {len(self.pairs)} pairs for {split} split")
    
    def __len__(self):
        return len(self.pairs)
    
    def __getitem__(self, idx):
        img_path, mask_path = self.pairs[idx]
        
        # Load image
        image = np.array(Image.open(img_path).convert('RGB'))
        
        # Load mask
        mask = np.array(Image.open(mask_path))
        
        # Convert 3-class to binary
        # BDD100K drivable area labels: 0=background, 1=direct drivable, 2=alternative drivable
        if self.binary:
            mask = (mask > 0).astype(np.uint8)
        
        # Apply data augmentation
        if self.transform:
            transformed = self.transform(image=image, mask=mask)
            image = transformed['image']
            mask = transformed['mask']
        
        # Ensure mask is float type and has correct shape
        mask = mask.float().unsqueeze(0)  # [1, H, W]
        
        return image, mask


def get_train_transforms(image_size=256):
    """Training data augmentation"""
    return A.Compose([
        A.Resize(image_size, image_size),
        A.HorizontalFlip(p=0.5),
        A.RandomBrightnessContrast(p=0.3),
        A.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1, p=0.3),
        A.ShiftScaleRotate(shift_limit=0.1, scale_limit=0.1, rotate_limit=10, p=0.3),
        A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ToTensorV2(),
    ])


def get_val_transforms(image_size=256):
    """Validation data preprocessing"""
    return A.Compose([
        A.Resize(image_size, image_size),
        A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ToTensorV2(),
    ])


if __name__ == '__main__':
    # Test dataset
    image_dir = '/root/bdd100k_data/bdd100k_images/bdd100k/images/10k'
    mask_dir = '/root/bdd100k_data/bdd100k_drivable_maps/bdd100k/drivable_maps/labels'
    
    train_dataset = BDD100KDrivableDataset(
        image_dir=image_dir,
        mask_dir=mask_dir,
        split='train',
        transform=get_train_transforms(256),
        val_ratio=0.2
    )
    
    val_dataset = BDD100KDrivableDataset(
        image_dir=image_dir,
        mask_dir=mask_dir,
        split='val',
        transform=get_val_transforms(256),
        val_ratio=0.2
    )
    
    print(f"\nTrain dataset size: {len(train_dataset)}")
    print(f"Val dataset size: {len(val_dataset)}")
    
    # Test loading one sample
    image, mask = train_dataset[0]
    print(f"\nImage shape: {image.shape}")
    print(f"Mask shape: {mask.shape}")
    print(f"Mask unique values: {torch.unique(mask)}")
    print(f"Mask value range: [{mask.min():.2f}, {mask.max():.2f}]")
