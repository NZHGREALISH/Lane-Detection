#!/bin/bash
# Train baseline models for comparison with pretrained ResNet34

echo "=========================================="
echo "Training Baseline Models for Comparison"
echo "=========================================="

# Baseline 1: Original U-Net (no pretrained encoder)
echo ""
echo "1/3: Training Original U-Net (no pretrained encoder)..."
echo "This shows the effect of pretraining"
python train.py \
    --model unet \
    --loss bce_dice \
    --batch_size 16 \
    --epochs 50 \
    --lr 1e-4 \
    --image_size 320 \
    --save_dir experiments/baseline_unet_no_pretrain \
    --save_interval 10 \
    --vis_interval 5

# Baseline 2: Simple Baseline CNN
echo ""
echo "2/3: Training Baseline CNN (simple architecture)..."
echo "This shows the effect of architecture complexity"
python train.py \
    --model baseline \
    --loss bce_dice \
    --batch_size 16 \
    --epochs 50 \
    --lr 1e-4 \
    --image_size 320 \
    --save_dir experiments/baseline_simple_cnn \
    --save_interval 10 \
    --vis_interval 5

# Baseline 3: Lightweight pretrained model (EfficientNet-B0)
echo ""
echo "3/3: Training EfficientNet-B0 U-Net (lightweight pretrained)..."
echo "This shows efficiency vs performance tradeoff"
python train_pretrained.py \
    --model_type unet \
    --encoder_name efficientnet-b0 \
    --loss weighted_bce_dice \
    --pos_weight 10.0 \
    --batch_size 8 \
    --epochs 50 \
    --lr 1e-4 \
    --image_size 320 \
    --save_dir experiments/baseline_efficientnet_b0 \
    --use_differential_lr

echo ""
echo "=========================================="
echo "All baseline models trained!"
echo "=========================================="
echo ""
echo "Models trained:"
echo "1. Original U-Net (no pretrain) - experiments/baseline_unet_no_pretrain"
echo "2. Simple Baseline CNN - experiments/baseline_simple_cnn"
echo "3. EfficientNet-B0 U-Net - experiments/baseline_efficientnet_b0"
echo ""
echo "Compare with:"
echo "   Pretrained ResNet34 U-Net - experiments/pretrained_resnet34_fixed"
