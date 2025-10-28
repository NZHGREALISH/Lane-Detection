# Drivable Area Segmentation

A drivable area segmentation project based on the BDD100K dataset.

## Project Overview

This project implements drivable area segmentation for autonomous driving scenarios, training deep learning models to identify drivable areas on roads using the BDD100K dataset.
<img width="1743" height="9583" alt="test_predictions" src="https://github.com/user-attachments/assets/baeead3c-ebb2-4984-a0ca-c30a13dd2e29" />

## Project Structure

```
Lane-Detection/
├── data/
│   └── dataset.py          # Data loading and preprocessing
├── models/
│   ├── unet.py            # U-Net model
│   └── baseline_cnn.py    # Baseline CNN model
├── utils/
│   ├── losses.py          # Loss functions
│   ├── metrics.py         # Evaluation metrics
│   └── visualization.py   # Visualization tools
├── train.py               # Training script
├── evaluate.py            # Evaluation script
├── config.py              # Configuration file
├── run_experiments.py     # Experiment comparison script
├── quick_test.py          # Quick test script
├── requirements.txt       # Dependencies
└── README.md             # Project documentation
```

## Environment Setup

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Dataset

Dataset location: `/root/bdd100k_data`

Dataset structure:
- Images: `bdd100k_images/bdd100k/images/10k/{train,val,test}`
- Labels: `bdd100k_drivable_maps/bdd100k/drivable_maps/labels/{train,val}`

## Quick Start

### 1. Quick Test

Run a quick test to verify environment configuration:

```bash
python quick_test.py
```

### 2. Train Models

#### Train U-Net model:

```bash
python train.py \
    --model unet \
    --batch_size 16 \
    --epochs 50 \
    --lr 1e-4 \
    --save_dir experiments/unet
```

#### Train Baseline CNN model:

```bash
python train.py \
    --model baseline \
    --batch_size 16 \
    --epochs 50 \
    --lr 1e-4 \
    --save_dir experiments/baseline
```

### 3. Evaluate Models

```bash
python evaluate.py \
    --model unet \
    --checkpoint experiments/unet/checkpoints/best_model.pth \
    --save_dir evaluation_results/unet \
    --visualize
```

### 4. Run Comparison Experiments

Automatically train and compare U-Net vs Baseline CNN:

```bash
python run_experiments.py
```

## Model Description

### U-Net
- **Architecture**: Encoder-Decoder with Skip Connections
- **Features**: Captures multi-scale features, suitable for semantic segmentation
- **Parameters**: ~31M

### Baseline CNN
- **Architecture**: Simple convolutional encoder + upsampling decoder
- **Features**: Lightweight, fast training
- **Parameters**: ~3M

## Training Configuration

### Loss Functions
- `bce_dice`: BCE Loss + Dice Loss (recommended)
- `dice`: Dice Loss
- `focal`: Focal Loss
- `bce`: BCE Loss

### Optimizers
- `adam`: Adam optimizer (recommended)
- `sgd`: SGD with momentum

### Learning Rate Schedulers
- `plateau`: ReduceLROnPlateau
- `cosine`: CosineAnnealingLR

## Evaluation Metrics

- **IoU (Intersection over Union)**: Intersection over Union
- **mIoU (mean IoU)**: Mean Intersection over Union
- **Dice Coefficient**: Dice coefficient
- **Pixel Accuracy**: Pixel-level accuracy
- **Precision/Recall/F1**: Precision, Recall, and F1 score

## Main Parameters

| Parameter | Description | Default Value |
|-----------|-------------|---------------|
| `--model` | Model type (unet/baseline) | unet |
| `--batch_size` | Batch size | 16 |
| `--epochs` | Number of epochs | 50 |
| `--lr` | Learning rate | 1e-4 |
| `--image_size` | Image size | 256 |
| `--loss` | Loss function | bce_dice |
| `--optimizer` | Optimizer | adam |
| `--save_dir` | Save directory | experiments/unet_default |

## Output Results

### Training Outputs
- `checkpoints/`: Model checkpoints
- `visualizations/`: Training visualization
- `training_curves.png`: Training curves
- `history.json`: Training history

### Evaluation Outputs
- `evaluation_results.json`: Evaluation results
- `visualizations/`: Prediction visualizations

## Data Augmentation

Data augmentation techniques used during training:
- Random horizontal flip
- Random brightness/contrast adjustment
- Color jitter
- Translation/scaling/rotation

## Performance Comparison

After running `run_experiments.py`, performance comparison will be generated:
- `comparison_results/comparison_table.csv`: Comparison table
- `comparison_results/comparison_chart.png`: Comparison chart

## Notes

1. **GPU Memory**: If GPU memory is insufficient, reduce `batch_size` or `image_size`
2. **Training Time**: Complete 50-epoch training takes several hours (hardware dependent)
3. **Data Path**: Ensure dataset paths are correctly configured

## Implementation Features

### Data Processing
- ✅ 3-class to binary conversion (background vs drivable area)
- ✅ Image normalization (ImageNet standards)
- ✅ Albumentations data augmentation
- ✅ Dynamic handling of missing masks

### Model Architecture
- ✅ U-Net: Complete encoder-decoder + skip connections
- ✅ Baseline CNN: Lightweight convolutional network
- ✅ BatchNorm + ReLU activation
- ✅ Bilinear interpolation upsampling

### Training Strategy
- ✅ BCE + Dice combined loss
- ✅ Adam optimizer + learning rate scheduling
- ✅ Early stopping (saves best model)
- ✅ Training/validation metrics monitoring
- ✅ Periodic visualization

### Evaluation System
- ✅ Multiple evaluation metrics (IoU, Dice, Accuracy, F1)
- ✅ Prediction visualization
- ✅ Overlay display
- ✅ Failure case analysis

## Usage Examples

### Example 1: Train High-Resolution U-Net

```bash
python train.py \
    --model unet \
    --image_size 512 \
    --batch_size 8 \
    --epochs 50 \
    --lr 5e-5 \
    --save_dir experiments/unet_512
```

### Example 2: Use Focal Loss

```bash
python train.py \
    --model unet \
    --loss focal \
    --save_dir experiments/unet_focal
```

### Example 3: Evaluate and Generate Visualizations

```bash
python evaluate.py \
    --model unet \
    --checkpoint experiments/unet/checkpoints/best_model.pth \
    --threshold 0.5 \
    --visualize
```

## Project Progress

- [x] Dataset loading and preprocessing
- [x] U-Net model implementation
- [x] Baseline CNN model implementation
- [x] Training script
- [x] Evaluation script
- [x] Multiple loss functions
- [x] Evaluation metrics
- [x] Visualization tools
- [x] Experiment comparison system

## Future Improvements

- [ ] Add more model architectures (DeepLabV3+, SegFormer, etc.)
- [ ] Support multi-GPU training
- [ ] Add test set inference script
- [ ] Real-time inference demo
- [ ] Model compression and acceleration

## Acknowledgments

- BDD100K Dataset: https://bdd-data.berkeley.edu/
- U-Net Paper: https://arxiv.org/abs/1505.04597
