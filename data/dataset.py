"""
BDD100K Drivable Area Dataset Implementation
"""
import os
import torch
from torch.utils.data import Dataset
from PIL import Image
import numpy as np
from torchvision import transforms
import albumentations as A
from albumentations.pytorch import ToTensorV2


class BDD100KDrivableDataset(Dataset):
    """BDD100K Drivable Area Dataset"""
    
    def __init__(self, image_dir, mask_dir, split='train', transform=None, binary=True):
        """
        Args:
            image_dir: Image directory path
            mask_dir: Mask directory path
            split: 'train', 'val', or 'test'
            transform: albumentations transforms
            binary: Whether to convert 3-class to binary (True: drivable/non-drivable)
        """
        self.image_dir = os.path.join(image_dir, split)
        self.mask_dir = os.path.join(mask_dir, split)
        self.split = split
        self.binary = binary
        self.transform = transform
        
        # Get all mask filenames first (since not all images have masks)
        all_masks = os.listdir(self.mask_dir)
        
        # Extract image names from mask names
        # Mask format: {image_name}_drivable_id.png
        self.images = []
        for mask_name in all_masks:
            if mask_name.endswith('_drivable_id.png'):
                # Remove '_drivable_id.png' to get image name
                img_name = mask_name.replace('_drivable_id.png', '.jpg')
                img_path = os.path.join(self.image_dir, img_name)
                # Only add if image exists
                if os.path.exists(img_path):
                    self.images.append(img_name)
        
        self.images = sorted(self.images)
        
        print(f"Loaded {len(self.images)} images with masks for {split} split")
        
    def __len__(self):
        return len(self.images)
    
    def __getitem__(self, idx):
        # Load image
        img_name = self.images[idx]
        img_path = os.path.join(self.image_dir, img_name)
        image = np.array(Image.open(img_path).convert('RGB'))
        
        # Load mask (replace .jpg with _drivable_id.png)
        mask_name = img_name.replace('.jpg', '_drivable_id.png')
        mask_path = os.path.join(self.mask_dir, mask_name)
        mask = np.array(Image.open(mask_path))
        
        # Convert 3-class to binary
        # BDD100K drivable area labels: 0=background, 1=direct drivable, 2=alternative drivable
        if self.binary:
            # Treat both 1 and 2 as drivable area (1), keep 0 as background (0)
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
        transform=get_train_transforms(256)
    )
    
    print(f"Dataset size: {len(train_dataset)}")
    
    # Test loading one sample
    image, mask = train_dataset[0]
    print(f"Image shape: {image.shape}")
    print(f"Mask shape: {mask.shape}")
    print(f"Mask unique values: {torch.unique(mask)}")
