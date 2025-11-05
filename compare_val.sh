#!/bin/bash
# Compare pretrained models on VALIDATION dataset (with ground truth)

echo "=========================================="
echo "Comparing Models on Validation Dataset"
echo "=========================================="

# Define models to compare
MODEL1="experiments/pretrained_resnet34_fixed/checkpoints/best_model.pth"
MODEL2="experiments/baseline_efficientnet_b0/checkpoints/best_model.pth"

# Check if models exist
if [ ! -f "$MODEL1" ]; then
    echo "Error: $MODEL1 not found!"
    exit 1
fi

if [ ! -f "$MODEL2" ]; then
    echo "Error: $MODEL2 not found!"
    exit 1
fi

# Run comparison on VALIDATION set (has ground truth)
python compare_on_test.py \
    --models \
        "$MODEL1" \
        "$MODEL2" \
    --model_names \
        "ResNet34-UNet" \
        "EfficientNet-B0-UNet" \
    --model_types \
        pretrained_unet \
        pretrained_unet \
    --encoder_names \
        resnet34 \
        efficientnet-b0 \
    --image_dir /root/bdd100k_data/bdd100k_images/bdd100k/images/10k/val \
    --mask_dir /root/bdd100k_data/bdd100k_drivable_maps/bdd100k/drivable_maps/labels/val \
    --batch_size 8 \
    --image_size 320 \
    --num_samples 12 \
    --save_dir val_comparison_with_metrics

echo ""
echo "=========================================="
echo "Comparison complete!"
echo "Results saved to: val_comparison_with_metrics/"
echo "  - Quantitative metrics (IoU, Dice, etc.)"
echo "  - Visual comparisons"
echo "  - Metrics charts"
echo "=========================================="
