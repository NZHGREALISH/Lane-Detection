#!/bin/bash
# Setup script for Drivable Area Segmentation project

echo "=================================="
echo "Setting up environment..."
echo "=================================="

# Install dependencies
echo "Installing Python dependencies..."
pip install -r requirements.txt

# Create necessary directories
echo "Creating directories..."
mkdir -p experiments
mkdir -p evaluation_results
mkdir -p comparison_results

echo "=================================="
echo "Setup complete!"
echo "=================================="
echo ""
echo "Next steps:"
echo "1. Run quick test: python quick_test.py"
echo "2. Train model: python train.py --model unet --epochs 50"
echo "3. Evaluate model: python evaluate.py --model unet --checkpoint <path_to_checkpoint>"
echo ""
