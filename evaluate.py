"""
Evaluation Script for Drivable Area Segmentation
"""
import os
import argparse
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
import numpy as np
import json

from data.dataset import BDD100KDrivableDataset, get_val_transforms
from models.unet import UNet
from models.baseline_cnn import BaselineCNN
from utils.metrics import (calculate_iou, calculate_miou, calculate_pixel_accuracy, 
                          calculate_dice_coefficient, calculate_precision_recall_f1)
from utils.visualization import visualize_predictions, visualize_overlay


class Evaluator:
    def __init__(self, config):
        self.config = config
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"Using device: {self.device}")
        
        # 加载模型
        self.model = self._load_model()
        self.model.to(self.device)
        self.model.eval()
        
        # 加载数据
        self.val_loader = self._build_dataloader()
        
        # 结果存储
        self.results = {
            'iou': [],
            'miou': [],
            'dice': [],
            'pixel_acc': [],
            'precision': [],
            'recall': [],
            'f1': []
        }
    
    def _load_model(self):
        """加载模型"""
        if self.config.model == 'unet':
            model = UNet(n_channels=3, n_classes=1, bilinear=True)
        elif self.config.model == 'baseline':
            model = BaselineCNN(n_channels=3, n_classes=1)
        else:
            raise ValueError(f"Unknown model: {self.config.model}")
        
        # 加载权重
        checkpoint = torch.load(self.config.checkpoint, map_location=self.device)
        model.load_state_dict(checkpoint['model_state_dict'])
        
        print(f"Model loaded from {self.config.checkpoint}")
        if 'best_val_iou' in checkpoint:
            print(f"Best validation IoU: {checkpoint['best_val_iou']:.4f}")
        
        return model
    
    def _build_dataloader(self):
        """构建数据加载器"""
        val_dataset = BDD100KDrivableDataset(
            image_dir=self.config.image_dir,
            mask_dir=self.config.mask_dir,
            split='val',
            transform=get_val_transforms(self.config.image_size),
            binary=True
        )
        
        val_loader = DataLoader(
            val_dataset,
            batch_size=self.config.batch_size,
            shuffle=False,
            num_workers=self.config.num_workers,
            pin_memory=True
        )
        
        print(f"Val samples: {len(val_dataset)}")
        
        return val_loader
    
    @torch.no_grad()
    def evaluate(self):
        """评估模型"""
        print("\nStarting evaluation...")
        print("="*80)
        
        # 保存样本用于可视化
        sample_images = []
        sample_masks = []
        sample_preds = []
        
        pbar = tqdm(self.val_loader, desc='Evaluating')
        for images, masks in pbar:
            images = images.to(self.device)
            masks = masks.to(self.device)
            
            # 前向传播
            logits = self.model(images)
            probs = torch.sigmoid(logits)
            
            # 计算指标
            iou = calculate_iou(probs, masks, threshold=self.config.threshold)
            miou, _ = calculate_miou(probs, masks, num_classes=2, threshold=self.config.threshold)
            dice = calculate_dice_coefficient(probs, masks, threshold=self.config.threshold)
            acc = calculate_pixel_accuracy(probs, masks, threshold=self.config.threshold)
            precision, recall, f1 = calculate_precision_recall_f1(probs, masks, threshold=self.config.threshold)
            
            # 记录结果
            self.results['iou'].append(iou)
            self.results['miou'].append(miou)
            self.results['dice'].append(dice)
            self.results['pixel_acc'].append(acc)
            self.results['precision'].append(precision)
            self.results['recall'].append(recall)
            self.results['f1'].append(f1)
            
            # 保存样本
            if len(sample_images) < self.config.num_vis_samples:
                sample_images.append(images.cpu())
                sample_masks.append(masks.cpu())
                sample_preds.append(probs.cpu())
            
            pbar.set_postfix({
                'IoU': f'{iou:.4f}',
                'Dice': f'{dice:.4f}'
            })
        
        # 计算平均指标
        avg_results = {k: np.mean(v) for k, v in self.results.items()}
        std_results = {k: np.std(v) for k, v in self.results.items()}
        
        # 打印结果
        print("\n" + "="*80)
        print("Evaluation Results:")
        print("-"*80)
        print(f"IoU:           {avg_results['iou']:.4f} ± {std_results['iou']:.4f}")
        print(f"mIoU:          {avg_results['miou']:.4f} ± {std_results['miou']:.4f}")
        print(f"Dice:          {avg_results['dice']:.4f} ± {std_results['dice']:.4f}")
        print(f"Pixel Acc:     {avg_results['pixel_acc']:.4f} ± {std_results['pixel_acc']:.4f}")
        print(f"Precision:     {avg_results['precision']:.4f} ± {std_results['precision']:.4f}")
        print(f"Recall:        {avg_results['recall']:.4f} ± {std_results['recall']:.4f}")
        print(f"F1 Score:      {avg_results['f1']:.4f} ± {std_results['f1']:.4f}")
        print("="*80)
        
        # 保存结果
        if self.config.save_results:
            self.save_results(avg_results, std_results)
        
        # 可视化
        if sample_images and self.config.visualize:
            sample_images = torch.cat(sample_images, dim=0)
            sample_masks = torch.cat(sample_masks, dim=0)
            sample_preds = torch.cat(sample_preds, dim=0)
            
            self.visualize_results(sample_images, sample_masks, sample_preds)
        
        return avg_results
    
    def save_results(self, avg_results, std_results):
        """保存评估结果"""
        results = {
            'average': avg_results,
            'std': std_results,
            'config': vars(self.config)
        }
        
        save_path = os.path.join(self.config.save_dir, 'evaluation_results.json')
        os.makedirs(self.config.save_dir, exist_ok=True)
        
        with open(save_path, 'w') as f:
            json.dump(results, f, indent=4)
        
        print(f"\nResults saved to {save_path}")
    
    def visualize_results(self, images, masks, preds):
        """可视化结果"""
        print("\nGenerating visualizations...")
        
        os.makedirs(os.path.join(self.config.save_dir, 'visualizations'), exist_ok=True)
        
        # 标准可视化
        vis_path = os.path.join(self.config.save_dir, 'visualizations', 'predictions.png')
        visualize_predictions(images[:8], masks[:8], preds[:8], 
                            num_samples=8, threshold=self.config.threshold, save_path=vis_path)
        
        # 叠加可视化
        overlay_path = os.path.join(self.config.save_dir, 'visualizations', 'overlay.png')
        visualize_overlay(images[:8], masks[:8], preds[:8], 
                         num_samples=8, threshold=self.config.threshold, save_path=overlay_path)
        
        print(f"Visualizations saved to {self.config.save_dir}/visualizations/")


def parse_args():
    parser = argparse.ArgumentParser(description='Evaluate Drivable Area Segmentation Model')
    
    # 数据相关
    parser.add_argument('--image_dir', type=str,
                       default='/home/grealish/APS360/bdd100k_data/bdd100k_images/bdd100k/images/10k',
                       help='图像目录路径')
    parser.add_argument('--mask_dir', type=str,
                       default='/home/grealish/APS360/bdd100k_data/bdd100k_drivable_maps/bdd100k/drivable_maps/labels',
                       help='mask目录路径')
    parser.add_argument('--image_size', type=int, default=256, help='输入图像大小')
    
    # 模型相关
    parser.add_argument('--model', type=str, default='unet', choices=['unet', 'baseline'],
                       help='模型类型')
    parser.add_argument('--checkpoint', type=str, required=True, help='模型检查点路径')
    
    # 评估相关
    parser.add_argument('--batch_size', type=int, default=16, help='批次大小')
    parser.add_argument('--threshold', type=float, default=0.5, help='二值化阈值')
    parser.add_argument('--num_workers', type=int, default=4, help='数据加载线程数')
    
    # 输出相关
    parser.add_argument('--save_dir', type=str, default='evaluation_results', help='保存目录')
    parser.add_argument('--save_results', action='store_true', default=True, help='保存结果')
    parser.add_argument('--visualize', action='store_true', default=True, help='可视化结果')
    parser.add_argument('--num_vis_samples', type=int, default=16, help='可视化样本数')
    
    return parser.parse_args()


def main():
    # 解析参数
    config = parse_args()
    
    # 创建评估器并评估
    evaluator = Evaluator(config)
    evaluator.evaluate()


if __name__ == '__main__':
    main()
