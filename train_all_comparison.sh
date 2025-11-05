#!/bin/bash
# Train all models for comprehensive comparison
# Baseline CNN vs Pretrained ResNet vs Pretrained EfficientNet

echo "=========================================="
echo "Training All Models for Comparison"
echo "=========================================="
echo ""
echo "This will train 3 models:"
echo "  1. Baseline CNN (no pretrained weights)"
echo "  2. U-Net with ResNet34 (pretrained on ImageNet)"
echo "  3. U-Net with EfficientNet-B0 (pretrained on ImageNet)"
echo ""

# Common hyperparameters
EPOCHS=50
IMAGE_SIZE=320
LR=1e-4
LOSS=weighted_bce_dice
POS_WEIGHT=10.0

# ===========================================
# 1. Baseline CNN (no pretrained weights)
# ===========================================
echo ""
echo "=========================================="
echo "1/3: Training Baseline CNN"
echo "=========================================="
echo "  Architecture: Simple CNN encoder-decoder"
echo "  Pretrained: NO"
echo "  Batch Size: 16"
echo ""

python train_pretrained.py \
    --model_type baseline_cnn \
    --epochs $EPOCHS \
    --batch_size 16 \
    --image_size $IMAGE_SIZE \
    --lr $LR \
    --loss $LOSS \
    --pos_weight $POS_WEIGHT \
    --scheduler plateau \
    --save_dir experiments/comparison_baseline_cnn \
    --save_interval 10 \
    --vis_interval 5

# ===========================================
# 2. U-Net with ResNet34 (pretrained)
# ===========================================
echo ""
echo "=========================================="
echo "2/3: Training U-Net with ResNet34"
echo "=========================================="
echo "  Architecture: U-Net"
echo "  Encoder: ResNet34 (pretrained on ImageNet)"
echo "  Batch Size: 8"
echo "  Differential LR: YES"
echo ""

python train_pretrained.py \
    --model_type unet \
    --encoder_name resnet34 \
    --epochs $EPOCHS \
    --batch_size 8 \
    --image_size $IMAGE_SIZE \
    --lr $LR \
    --loss $LOSS \
    --pos_weight $POS_WEIGHT \
    --scheduler onecycle \
    --use_differential_lr \
    --save_dir experiments/comparison_resnet34 \
    --save_interval 10 \
    --vis_interval 5

# ===========================================
# 3. U-Net with EfficientNet-B0 (pretrained)
# ===========================================
echo ""
echo "=========================================="
echo "3/3: Training U-Net with EfficientNet-B0"
echo "=========================================="
echo "  Architecture: U-Net"
echo "  Encoder: EfficientNet-B0 (pretrained on ImageNet)"
echo "  Batch Size: 8"
echo "  Differential LR: YES"
echo ""

python train_pretrained.py \
    --model_type unet \
    --encoder_name efficientnet-b0 \
    --epochs $EPOCHS \
    --batch_size 8 \
    --image_size $IMAGE_SIZE \
    --lr $LR \
    --loss $LOSS \
    --pos_weight $POS_WEIGHT \
    --scheduler onecycle \
    --use_differential_lr \
    --save_dir experiments/comparison_efficientnet_b0 \
    --save_interval 10 \
    --vis_interval 5

# ===========================================
# Summary
# ===========================================
echo ""
echo "=========================================="
echo "All Models Trained Successfully!"
echo "=========================================="
echo ""
echo "Results:"
echo "  1. Baseline CNN:      experiments/comparison_baseline_cnn"
echo "  2. ResNet34 U-Net:    experiments/comparison_resnet34"
echo "  3. EfficientNet U-Net: experiments/comparison_efficientnet_b0"
echo ""
echo "To compare results, check the history.json files in each directory"
echo "Or use the compare_models_val.py script"
echo ""

