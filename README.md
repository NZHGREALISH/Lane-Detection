# Drivable Area Segmentation

基于BDD100K数据集的可行驶区域分割项目。

## 项目简介

本项目实现了用于自动驾驶场景的可行驶区域分割任务，使用BDD100K数据集训练深度学习模型来识别道路上的可行驶区域。

## 项目结构

```
Lane-Detection/
├── data/
│   └── dataset.py          # 数据加载和预处理
├── models/
│   ├── unet.py            # U-Net模型
│   └── baseline_cnn.py    # Baseline CNN模型
├── utils/
│   ├── losses.py          # 损失函数
│   ├── metrics.py         # 评估指标
│   └── visualization.py   # 可视化工具
├── train.py               # 训练脚本
├── evaluate.py            # 评估脚本
├── config.py              # 配置文件
├── run_experiments.py     # 实验对比脚本
├── quick_test.py          # 快速测试脚本
├── requirements.txt       # 依赖包
└── README.md             # 项目说明
```

## 环境配置

### 1. 安装依赖

```bash
pip install -r requirements.txt
```

### 2. 数据集

数据集已下载至: `/home/grealish/APS360/bdd100k_data`

数据集结构:
- 图像: `bdd100k_images/bdd100k/images/10k/{train,val,test}`
- 标签: `bdd100k_drivable_maps/bdd100k/drivable_maps/labels/{train,val}`

## 快速开始

### 1. 快速测试

运行快速测试以验证环境配置:

```bash
python quick_test.py
```

### 2. 训练模型

#### 训练U-Net模型:

```bash
python train.py \
    --model unet \
    --batch_size 16 \
    --epochs 50 \
    --lr 1e-4 \
    --save_dir experiments/unet
```

#### 训练Baseline CNN模型:

```bash
python train.py \
    --model baseline \
    --batch_size 16 \
    --epochs 50 \
    --lr 1e-4 \
    --save_dir experiments/baseline
```

### 3. 评估模型

```bash
python evaluate.py \
    --model unet \
    --checkpoint experiments/unet/checkpoints/best_model.pth \
    --save_dir evaluation_results/unet \
    --visualize
```

### 4. 运行对比实验

自动训练和对比U-Net与Baseline CNN:

```bash
python run_experiments.py
```

## 模型说明

### U-Net
- **结构**: Encoder-Decoder架构 + Skip Connections
- **特点**: 能够捕获多尺度特征，适合语义分割任务
- **参数量**: ~31M

### Baseline CNN
- **结构**: 简单的卷积编码器 + 上采样解码器
- **特点**: 轻量级，训练速度快
- **参数量**: ~3M

## 训练配置

### 损失函数
- `bce_dice`: BCE Loss + Dice Loss (推荐)
- `dice`: Dice Loss
- `focal`: Focal Loss
- `bce`: BCE Loss

### 优化器
- `adam`: Adam优化器 (推荐)
- `sgd`: SGD with momentum

### 学习率调度
- `plateau`: ReduceLROnPlateau
- `cosine`: CosineAnnealingLR

## 评估指标

- **IoU (Intersection over Union)**: 交并比
- **mIoU (mean IoU)**: 平均交并比
- **Dice Coefficient**: Dice系数
- **Pixel Accuracy**: 像素准确率
- **Precision/Recall/F1**: 精确率/召回率/F1分数

## 主要参数

| 参数 | 说明 | 默认值 |
|------|------|--------|
| `--model` | 模型类型 (unet/baseline) | unet |
| `--batch_size` | 批次大小 | 16 |
| `--epochs` | 训练轮数 | 50 |
| `--lr` | 学习率 | 1e-4 |
| `--image_size` | 图像大小 | 256 |
| `--loss` | 损失函数 | bce_dice |
| `--optimizer` | 优化器 | adam |
| `--save_dir` | 保存目录 | experiments/unet_default |

## 输出结果

### 训练输出
- `checkpoints/`: 模型检查点
- `visualizations/`: 训练过程可视化
- `training_curves.png`: 训练曲线
- `history.json`: 训练历史

### 评估输出
- `evaluation_results.json`: 评估结果
- `visualizations/`: 预测可视化

## 数据增强

训练时使用以下数据增强技术:
- 随机水平翻转
- 随机亮度/对比度调整
- 颜色抖动
- 平移/缩放/旋转

## 性能对比

运行 `run_experiments.py` 后会自动生成性能对比:
- `comparison_results/comparison_table.csv`: 对比表格
- `comparison_results/comparison_chart.png`: 对比图表

## 注意事项

1. **GPU内存**: 如果GPU内存不足，可以减小 `batch_size` 或 `image_size`
2. **训练时间**: 完整训练50个epoch约需要数小时 (取决于硬件)
3. **数据路径**: 确保数据集路径正确配置

## 实现特点

### 数据处理
- ✅ 三分类转二分类 (背景 vs 可行驶区域)
- ✅ 图像归一化 (ImageNet标准)
- ✅ Albumentations数据增强
- ✅ 动态处理缺失mask

### 模型架构
- ✅ U-Net: 完整的encoder-decoder + skip connections
- ✅ Baseline CNN: 轻量级卷积网络
- ✅ BatchNorm + ReLU激活
- ✅ 双线性插值上采样

### 训练策略
- ✅ BCE + Dice组合损失
- ✅ Adam优化器 + 学习率调度
- ✅ 早停机制 (保存最佳模型)
- ✅ 训练/验证指标监控
- ✅ 定期可视化

### 评估系统
- ✅ 多种评估指标 (IoU, Dice, Accuracy, F1)
- ✅ 预测结果可视化
- ✅ 叠加显示 (overlay)
- ✅ 失败案例分析

## 使用示例

### 示例1: 训练高分辨率U-Net

```bash
python train.py \
    --model unet \
    --image_size 512 \
    --batch_size 8 \
    --epochs 50 \
    --lr 5e-5 \
    --save_dir experiments/unet_512
```

### 示例2: 使用Focal Loss

```bash
python train.py \
    --model unet \
    --loss focal \
    --save_dir experiments/unet_focal
```

### 示例3: 评估并生成可视化

```bash
python evaluate.py \
    --model unet \
    --checkpoint experiments/unet/checkpoints/best_model.pth \
    --threshold 0.5 \
    --visualize
```

## 项目进度

- [x] 数据集加载和预处理
- [x] U-Net模型实现
- [x] Baseline CNN模型实现
- [x] 训练脚本
- [x] 评估脚本
- [x] 多种损失函数
- [x] 评估指标
- [x] 可视化工具
- [x] 实验对比系统

## 未来改进

- [ ] 添加更多模型架构 (DeepLabV3+, SegFormer等)
- [ ] 支持多GPU训练
- [ ] 添加测试集推理脚本
- [ ] 实时推理demo
- [ ] 模型压缩和加速

## 联系方式

如有问题，请联系项目维护者。

## 致谢

- BDD100K数据集: https://bdd-data.berkeley.edu/
- U-Net论文: https://arxiv.org/abs/1505.04597
