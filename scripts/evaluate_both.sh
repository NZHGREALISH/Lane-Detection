#!/bin/bash
# Evaluate both models

echo "=================================="
echo "Evaluating models"
echo "=================================="

# Evaluate U-Net
echo "Evaluating U-Net..."
python evaluate.py \
    --model unet \
    --checkpoint experiments/unet/checkpoints/best_model.pth \
    --save_dir evaluation_results/unet \
    --visualize

# Evaluate Baseline CNN
echo "Evaluating Baseline CNN..."
python evaluate.py \
    --model baseline \
    --checkpoint experiments/baseline/checkpoints/best_model.pth \
    --save_dir evaluation_results/baseline \
    --visualize

echo "=================================="
echo "Evaluation complete!"
echo "=================================="
