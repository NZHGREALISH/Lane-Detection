"""
Configuration File for Drivable Area Segmentation
"""

class Config:
    """基础配置"""
    
    # 数据路径
    IMAGE_DIR = '/home/grealish/APS360/bdd100k_data/bdd100k_images/bdd100k/images/10k'
    MASK_DIR = '/home/grealish/APS360/bdd100k_data/bdd100k_drivable_maps/bdd100k/drivable_maps/labels'
    
    # 图像参数
    IMAGE_SIZE = 256
    NUM_CLASSES = 1  # 二分类（可行驶/不可行驶）
    
    # 训练参数
    BATCH_SIZE = 16
    EPOCHS = 50
    LEARNING_RATE = 1e-4
    WEIGHT_DECAY = 1e-5
    
    # 优化器和调度器
    OPTIMIZER = 'adam'  # 'adam' or 'sgd'
    SCHEDULER = 'plateau'  # 'plateau', 'cosine', or 'none'
    
    # 损失函数
    LOSS = 'bce_dice'  # 'bce_dice', 'dice', 'focal', or 'bce'
    
    # 数据加载
    NUM_WORKERS = 4
    
    # 保存
    SAVE_DIR = 'experiments'
    SAVE_INTERVAL = 10  # 保存检查点的epoch间隔
    VIS_INTERVAL = 5  # 可视化的epoch间隔
    
    # 评估
    THRESHOLD = 0.5  # 二值化阈值
    
    # 其他
    SEED = 42


class UNetConfig(Config):
    """U-Net配置"""
    MODEL = 'unet'
    SAVE_DIR = 'experiments/unet'


class BaselineCNNConfig(Config):
    """Baseline CNN配置"""
    MODEL = 'baseline'
    SAVE_DIR = 'experiments/baseline'


class UNetLargeConfig(UNetConfig):
    """U-Net大模型配置（更高分辨率）"""
    IMAGE_SIZE = 512
    BATCH_SIZE = 8
    LEARNING_RATE = 5e-5
    SAVE_DIR = 'experiments/unet_large'


class UNetFocalConfig(UNetConfig):
    """U-Net + Focal Loss配置"""
    LOSS = 'focal'
    SAVE_DIR = 'experiments/unet_focal'


# 可用配置
CONFIGS = {
    'unet': UNetConfig,
    'baseline': BaselineCNNConfig,
    'unet_large': UNetLargeConfig,
    'unet_focal': UNetFocalConfig,
}


def get_config(name='unet'):
    """获取配置"""
    if name not in CONFIGS:
        raise ValueError(f"Unknown config: {name}. Available: {list(CONFIGS.keys())}")
    return CONFIGS[name]()
