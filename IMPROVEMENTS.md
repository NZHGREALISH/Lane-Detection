# Improvements to Address Overfitting

## Problem Analysis

Your training showed severe overfitting:
- **Training IoU**: 43.27% (improving)
- **Validation IoU**: ~0% (no improvement)
- **Root Cause**: Model memorizing training data but not generalizing

## Solutions Implemented

### 1. **Pretrained Models** (Most Important!)
- Using ImageNet pretrained encoders (ResNet34/50, EfficientNet)
- Benefits:
  - Better feature extraction from pre-learned representations
  - Reduced overfitting through transfer learning
  - Faster convergence

### 2. **Improved Loss Functions**
- **WeightedBCEDiceLoss**: Handles class imbalance with `pos_weight=10`
- **TverskyLoss**: Better for false negative reduction
- **FocalTverskyLoss**: Emphasizes hard examples
- **ComboLoss**: Combination approach

### 3. **Differential Learning Rates**
- Encoder (pretrained): **0.1x** learning rate
- Decoder (random init): **1x** learning rate
- Prevents destroying pretrained features

### 4. **Better Optimization**
- AdamW optimizer (with weight decay)
- OneCycleLR scheduler for faster convergence
- Larger weight decay (1e-4) for regularization

## How to Use

### Quick Start (Recommended Settings)

```bash
# Install new dependencies first
pip install segmentation-models-pytorch timm

# Train with pretrained ResNet34 U-Net
python train_pretrained.py \
    --model_type unet \
    --encoder_name resnet34 \
    --loss weighted_bce_dice \
    --pos_weight 10.0 \
    --lr 1e-4 \
    --batch_size 8 \
    --image_size 320 \
    --epochs 50 \
    --save_dir experiments/pretrained_resnet34
```

### Advanced Options

#### Try Different Architectures:
```bash
# U-Net++ (better skip connections)
python train_pretrained.py --model_type unetplusplus --encoder_name resnet34

# DeepLabV3+ (better for large objects)
python train_pretrained.py --model_type deeplabv3plus --encoder_name resnet50

# EfficientNet encoder (more efficient)
python train_pretrained.py --model_type unet --encoder_name efficientnet-b0
```

#### Try Different Loss Functions:
```bash
# Tversky Loss (focus on reducing false negatives)
python train_pretrained.py --loss tversky

# Focal Tversky Loss (focus on hard examples)
python train_pretrained.py --loss focal_tversky --pos_weight 15.0

# Combo Loss
python train_pretrained.py --loss combo --pos_weight 10.0
```

## Expected Improvements

With these changes, you should see:
1. ✅ Validation IoU > 30% (instead of 0%)
2. ✅ Smoother training curves
3. ✅ Better generalization
4. ✅ Faster convergence

## Model Comparison

| Model | Parameters | Speed | Expected Val IoU |
|-------|-----------|-------|------------------|
| ResNet34 U-Net | ~24M | Fast | 35-45% |
| ResNet50 U-Net | ~30M | Medium | 40-50% |
| EfficientNet-B0 U-Net | ~8M | Fast | 35-45% |
| U-Net++ ResNet34 | ~28M | Medium | 40-50% |
| DeepLabV3+ ResNet34 | ~22M | Medium | 40-50% |

## Additional Tips

### If validation still shows 0% IoU:
1. **Increase pos_weight**: Try 15.0 or 20.0
2. **Check data**: Verify masks are loaded correctly
3. **Reduce learning rate**: Try 5e-5 or 1e-5
4. **Add more augmentation**: Increase augmentation probability

### If training is too slow:
1. Use `--encoder_name efficientnet-b0` (lighter model)
2. Reduce `--batch_size` to 4
3. Use `--image_size 256` instead of 320

### Monitor training:
```bash
# Watch loss curves
tensorboard --logdir experiments/

# Check visualizations
ls experiments/pretrained_resnet34/visualizations/
```

## Key Configuration Parameters

- `pos_weight=10.0`: Weight for positive class (drivable area)
  - Increase if model predicts all background
  - Decrease if model predicts all foreground
  
- `use_differential_lr=True`: Use 10x smaller LR for pretrained encoder
  - Prevents destroying pretrained features
  
- `weight_decay=1e-4`: Regularization strength
  - Increase to 1e-3 if overfitting persists
  
- `scheduler=onecycle`: Fast convergence schedule
  - Alternative: `plateau` for more stable training

## Troubleshooting

**Problem**: Still getting 0% validation IoU
**Solution**: 
```bash
python train_pretrained.py --pos_weight 20.0 --lr 5e-5 --loss focal_tversky
```

**Problem**: Out of memory
**Solution**:
```bash
python train_pretrained.py --batch_size 4 --image_size 256
```

**Problem**: Training too slow
**Solution**:
```bash
python train_pretrained.py --encoder_name efficientnet-b0 --batch_size 8
```
