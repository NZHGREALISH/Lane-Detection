"""
Configuration File for Drivable Area Segmentation
"""

class Config:
    """Base configuration"""
    
    # Data paths
    IMAGE_DIR = '/root/bdd100k_data/bdd100k_images/bdd100k/images/10k'
    MASK_DIR = '/root/bdd100k_data/bdd100k_drivable_maps/bdd100k/drivable_maps/labels'
    
    # Image parameters
    IMAGE_SIZE = 256
    NUM_CLASSES = 1  # Binary classification (drivable/non-drivable)
    
    # Training parameters
    BATCH_SIZE = 16
    EPOCHS = 50
    LEARNING_RATE = 1e-4
    WEIGHT_DECAY = 1e-5
    
    # Optimizer and scheduler
    OPTIMIZER = 'adam'  # 'adam' or 'sgd'
    SCHEDULER = 'plateau'  # 'plateau', 'cosine', or 'none'
    
    # Loss function
    LOSS = 'bce_dice'  # 'bce_dice', 'dice', 'focal', or 'bce'
    
    # Data loading
    NUM_WORKERS = 4
    
    # Saving
    SAVE_DIR = 'experiments'
    SAVE_INTERVAL = 10  # Epoch interval for saving checkpoints
    VIS_INTERVAL = 5  # Epoch interval for visualization
    
    # Evaluation
    THRESHOLD = 0.5  # Binary threshold
    
    # Other
    SEED = 42


class UNetConfig(Config):
    """U-Net configuration"""
    MODEL = 'unet'
    SAVE_DIR = 'experiments/unet'


class BaselineCNNConfig(Config):
    """Baseline CNN configuration"""
    MODEL = 'baseline'
    SAVE_DIR = 'experiments/baseline'


class UNetLargeConfig(UNetConfig):
    """U-Net large model configuration (higher resolution)"""
    IMAGE_SIZE = 512
    BATCH_SIZE = 8
    LEARNING_RATE = 5e-5
    SAVE_DIR = 'experiments/unet_large'


class UNetFocalConfig(UNetConfig):
    """U-Net + Focal Loss configuration"""
    LOSS = 'focal'
    SAVE_DIR = 'experiments/unet_focal'


# Available configurations
CONFIGS = {
    'unet': UNetConfig,
    'baseline': BaselineCNNConfig,
    'unet_large': UNetLargeConfig,
    'unet_focal': UNetFocalConfig,
}


def get_config(name='unet'):
    """Get configuration by name"""
    if name not in CONFIGS:
        raise ValueError(f"Unknown config: {name}. Available: {list(CONFIGS.keys())}")
    return CONFIGS[name]()
