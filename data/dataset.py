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
            image_dir: 图像目录路径
            mask_dir: mask目录路径
            split: 'train', 'val', or 'test'
            transform: albumentations transforms
            binary: 是否将三分类转为二分类 (True: 可行驶/不可行驶)
        """
        self.image_dir = os.path.join(image_dir, split)
        self.mask_dir = os.path.join(mask_dir, split)
        self.split = split
        self.binary = binary
        self.transform = transform
        
        # 获取所有图像文件名
        self.images = sorted(os.listdir(self.image_dir))
        
        print(f"Loaded {len(self.images)} images for {split} split")
        
    def __len__(self):
        return len(self.images)
    
    def __getitem__(self, idx):
        # 加载图像
        img_name = self.images[idx]
        img_path = os.path.join(self.image_dir, img_name)
        image = np.array(Image.open(img_path).convert('RGB'))
        
        # 加载mask (将.jpg替换为_drivable_id.png)
        mask_name = img_name.replace('.jpg', '_drivable_id.png')
        mask_path = os.path.join(self.mask_dir, mask_name)
        
        if os.path.exists(mask_path):
            mask = np.array(Image.open(mask_path))
        else:
            # 如果mask不存在，创建全零mask
            mask = np.zeros((image.shape[0], image.shape[1]), dtype=np.uint8)
        
        # 三分类转二分类
        # BDD100K drivable area标签: 0=背景, 1=直接可行驶, 2=替代可行驶
        if self.binary:
            # 将1和2都视为可行驶区域(1), 0保持为背景(0)
            mask = (mask > 0).astype(np.uint8)
        
        # 应用数据增强
        if self.transform:
            transformed = self.transform(image=image, mask=mask)
            image = transformed['image']
            mask = transformed['mask']
        
        # 确保mask是float类型且形状正确
        mask = mask.float().unsqueeze(0)  # [1, H, W]
        
        return image, mask


def get_train_transforms(image_size=256):
    """训练数据增强"""
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
    """验证数据预处理"""
    return A.Compose([
        A.Resize(image_size, image_size),
        A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ToTensorV2(),
    ])


if __name__ == '__main__':
    # 测试数据集
    image_dir = '/home/grealish/APS360/bdd100k_data/bdd100k_images/bdd100k/images/10k'
    mask_dir = '/home/grealish/APS360/bdd100k_data/bdd100k_drivable_maps/bdd100k/drivable_maps/labels'
    
    train_dataset = BDD100KDrivableDataset(
        image_dir=image_dir,
        mask_dir=mask_dir,
        split='train',
        transform=get_train_transforms(256)
    )
    
    print(f"Dataset size: {len(train_dataset)}")
    
    # 测试加载一个样本
    image, mask = train_dataset[0]
    print(f"Image shape: {image.shape}")
    print(f"Mask shape: {mask.shape}")
    print(f"Mask unique values: {torch.unique(mask)}")
