#!/bin/bash
# Train Baseline CNN for comparison with pretrained models

echo "=========================================="
echo "Training Baseline CNN"
echo "=========================================="

# Set hyperparameters
EPOCHS=50
BATCH_SIZE=16
IMAGE_SIZE=320
LR=1e-4
LOSS=weighted_bce_dice
POS_WEIGHT=10.0

echo ""
echo "Configuration:"
echo "  Model: Baseline CNN (no pretrained weights)"
echo "  Epochs: $EPOCHS"
echo "  Batch Size: $BATCH_SIZE"
echo "  Image Size: $IMAGE_SIZE"
echo "  Learning Rate: $LR"
echo "  Loss: $LOSS"
echo "  Pos Weight: $POS_WEIGHT"
echo ""

# Train Baseline CNN
python train_pretrained.py \
    --model_type baseline_cnn \
    --epochs $EPOCHS \
    --batch_size $BATCH_SIZE \
    --image_size $IMAGE_SIZE \
    --lr $LR \
    --loss $LOSS \
    --pos_weight $POS_WEIGHT \
    --scheduler plateau \
    --save_dir experiments/baseline_cnn \
    --save_interval 10 \
    --vis_interval 5

echo ""
echo "=========================================="
echo "Baseline CNN training completed!"
echo "=========================================="
echo ""
echo "Results saved to: experiments/baseline_cnn"
echo ""
echo "Compare with pretrained models:"
echo "  ResNet34:       experiments/pretrained_resnet34_*"
echo "  EfficientNet:   experiments/pretrained_efficientnet_*"
echo ""

