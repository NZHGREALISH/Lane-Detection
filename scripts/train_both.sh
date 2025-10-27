#!/bin/bash
# Train both U-Net and Baseline CNN models

echo "=================================="
echo "Training U-Net and Baseline CNN"
echo "=================================="

# Train U-Netwdawdawdawd
echo "Training U-Net..."
python train.py \
    --model unet \
    --batch_size 16 \
    --epochs 50 \
    --lr 1e-4 \
    --save_dir experiments/unet \
    --loss bce_dice


# Train Baseline CNN
echo "Training Baseline CNN..."
python train.py \
    --model baseline \
    --batch_size 16 \
    --epochs 50 \
    --lr 1e-4 \
    --save_dir experiments/baseline \
    --loss bce_dice

echo "=================================="
echo "Training complete!"
echo "=================================="
