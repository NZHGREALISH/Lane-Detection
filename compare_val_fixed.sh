#!/bin/bash
# Compare pretrained models on validation dataset (with ground truth)

echo "=========================================="
echo "Comparing Models on Validation Dataset"
echo "=========================================="

MODEL1="experiments/pretrained_resnet34_fixed/checkpoints/best_model.pth"
MODEL2="experiments/baseline_efficientnet_b0/checkpoints/best_model.pth"

# Check models exist
if [ ! -f "$MODEL1" ]; then
    echo "Error: $MODEL1 not found!"
    exit 1
fi

if [ ! -f "$MODEL2" ]; then
    echo "Error: $MODEL2 not found!"
    exit 1
fi

# Run comparison
python compare_models_val.py \
    --models "$MODEL1" "$MODEL2" \
    --model_names "ResNet34-UNet" "EfficientNet-B0-UNet" \
    --model_types pretrained_unet pretrained_unet \
    --encoder_names resnet34 efficientnet-b0 \
    --image_dir /root/bdd100k_data/bdd100k_images/bdd100k/images/10k \
    --mask_dir /root/bdd100k_data/bdd100k_drivable_maps/bdd100k/drivable_maps/labels \
    --batch_size 16 \
    --image_size 320 \
    --num_samples 12 \
    --val_ratio 0.2 \
    --save_dir validation_comparison_pretrained

echo ""
echo "=========================================="
echo "✅ Comparison complete!"
echo "Results saved to: validation_comparison_pretrained/"
echo "=========================================="
