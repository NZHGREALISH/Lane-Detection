"""
Compare multiple trained models
"""
import os
import json
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

def load_history(exp_dir):
    """Load training history from experiment directory"""
    history_path = os.path.join(exp_dir, 'history.json')
    if os.path.exists(history_path):
        with open(history_path, 'r') as f:
            return json.load(f)
    return None

def extract_best_metrics(history):
    """Extract best metrics from history"""
    if not history:
        return None
    
    best_val_iou = max(history['val_iou']) if history['val_iou'] else 0
    best_val_dice = max(history['val_dice']) if history['val_dice'] else 0
    best_val_acc = max(history['val_acc']) if history['val_acc'] else 0
    final_train_iou = history['train_iou'][-1] if history['train_iou'] else 0
    
    # Find epoch of best IoU
    best_epoch = history['val_iou'].index(best_val_iou) + 1 if history['val_iou'] else 0
    
    return {
        'Best Val IoU': best_val_iou,
        'Best Val Dice': best_val_dice,
        'Best Val Acc': best_val_acc,
        'Final Train IoU': final_train_iou,
        'Best Epoch': best_epoch,
        'Overfitting Gap': final_train_iou - best_val_iou
    }

def count_parameters(model_name):
    """Estimate model parameters"""
    param_dict = {
        'ResNet34 U-Net': 24.4,
        'Original U-Net': 17.3,
        'Baseline CNN': 1.6,
        'EfficientNet-B0 U-Net': 8.2
    }
    return param_dict.get(model_name, 'N/A')

def compare_models(exp_dirs, model_names, save_dir='comparison_results'):
    """Compare multiple models"""
    os.makedirs(save_dir, exist_ok=True)
    
    print("="*80)
    print("Model Comparison")
    print("="*80)
    
    # Load all histories
    histories = {}
    metrics_data = {}
    
    for exp_dir, name in zip(exp_dirs, model_names):
        print(f"\nLoading {name} from {exp_dir}...")
        history = load_history(exp_dir)
        if history:
            histories[name] = history
            metrics = extract_best_metrics(history)
            metrics['Parameters (M)'] = count_parameters(name)
            metrics_data[name] = metrics
            print(f"  Best Val IoU: {metrics['Best Val IoU']:.4f}")
        else:
            print(f"  Warning: No history found for {name}")
    
    if not metrics_data:
        print("No valid data found!")
        return
    
    # Create comparison table
    df = pd.DataFrame(metrics_data).T
    df = df.round(4)
    
    print("\n" + "="*80)
    print("Performance Comparison Table")
    print("="*80)
    print(df.to_string())
    print("="*80)
    
    # Save table
    table_path = os.path.join(save_dir, 'comparison_table.csv')
    df.to_csv(table_path)
    print(f"\nTable saved to: {table_path}")
    
    # Create visualizations
    create_comparison_plots(histories, df, save_dir)

def create_comparison_plots(histories, metrics_df, save_dir):
    """Create comparison visualizations"""
    
    # 1. Training curves comparison
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    
    metrics = ['loss', 'iou', 'dice', 'acc']
    titles = ['Loss', 'IoU', 'Dice Coefficient', 'Pixel Accuracy']
    
    colors = plt.cm.Set2(np.linspace(0, 1, len(histories)))
    
    for idx, (metric, title) in enumerate(zip(metrics, titles)):
        ax = axes[idx // 2, idx % 2]
        
        for (name, history), color in zip(histories.items(), colors):
            if f'val_{metric}' in history and history[f'val_{metric}']:
                epochs = range(1, len(history[f'val_{metric}']) + 1)
                ax.plot(epochs, history[f'val_{metric}'], 
                       label=name, linewidth=2, color=color)
        
        ax.set_xlabel('Epoch', fontsize=12)
        ax.set_ylabel(title, fontsize=12)
        ax.set_title(f'Validation {title} Comparison', fontsize=14, fontweight='bold')
        ax.legend(loc='best', fontsize=10)
        ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    curves_path = os.path.join(save_dir, 'training_curves_comparison.png')
    plt.savefig(curves_path, dpi=150, bbox_inches='tight')
    print(f"Training curves saved to: {curves_path}")
    plt.close()
    
    # 2. Bar chart comparison
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    
    # Best Val IoU
    ax = axes[0]
    metrics_df['Best Val IoU'].plot(kind='bar', ax=ax, color=colors[:len(metrics_df)])
    ax.set_title('Best Validation IoU', fontsize=14, fontweight='bold')
    ax.set_ylabel('IoU', fontsize=12)
    ax.set_ylim([0, 1])
    ax.grid(axis='y', alpha=0.3)
    for i, v in enumerate(metrics_df['Best Val IoU']):
        ax.text(i, v + 0.02, f'{v:.3f}', ha='center', fontsize=10, fontweight='bold')
    
    # Parameters
    ax = axes[1]
    params = metrics_df['Parameters (M)'].astype(float)
    params.plot(kind='bar', ax=ax, color=colors[:len(metrics_df)])
    ax.set_title('Model Size (Parameters)', fontsize=14, fontweight='bold')
    ax.set_ylabel('Parameters (Millions)', fontsize=12)
    ax.grid(axis='y', alpha=0.3)
    for i, v in enumerate(params):
        ax.text(i, v + 0.5, f'{v:.1f}M', ha='center', fontsize=10, fontweight='bold')
    
    # Overfitting Gap
    ax = axes[2]
    metrics_df['Overfitting Gap'].plot(kind='bar', ax=ax, color=colors[:len(metrics_df)])
    ax.set_title('Overfitting Gap (Train - Val IoU)', fontsize=14, fontweight='bold')
    ax.set_ylabel('IoU Gap', fontsize=12)
    ax.axhline(y=0, color='black', linestyle='--', linewidth=1)
    ax.grid(axis='y', alpha=0.3)
    for i, v in enumerate(metrics_df['Overfitting Gap']):
        ax.text(i, v + 0.01, f'{v:.3f}', ha='center', fontsize=10, fontweight='bold')
    
    plt.tight_layout()
    bars_path = os.path.join(save_dir, 'metrics_comparison.png')
    plt.savefig(bars_path, dpi=150, bbox_inches='tight')
    print(f"Metrics comparison saved to: {bars_path}")
    plt.close()
    
    # 3. Efficiency vs Performance scatter plot
    fig, ax = plt.subplots(figsize=(10, 8))
    
    params = metrics_df['Parameters (M)'].astype(float).values
    iou = metrics_df['Best Val IoU'].values
    names = metrics_df.index.values
    
    scatter = ax.scatter(params, iou, s=300, c=range(len(params)), 
                        cmap='viridis', alpha=0.6, edgecolors='black', linewidth=2)
    
    for i, name in enumerate(names):
        ax.annotate(name, (params[i], iou[i]), 
                   xytext=(10, 10), textcoords='offset points',
                   fontsize=11, fontweight='bold',
                   bbox=dict(boxstyle='round,pad=0.5', facecolor='yellow', alpha=0.3))
    
    ax.set_xlabel('Model Size (Million Parameters)', fontsize=13, fontweight='bold')
    ax.set_ylabel('Best Validation IoU', fontsize=13, fontweight='bold')
    ax.set_title('Model Efficiency vs Performance', fontsize=15, fontweight='bold')
    ax.grid(True, alpha=0.3)
    ax.set_ylim([0, 1])
    
    plt.tight_layout()
    scatter_path = os.path.join(save_dir, 'efficiency_vs_performance.png')
    plt.savefig(scatter_path, dpi=150, bbox_inches='tight')
    print(f"Efficiency plot saved to: {scatter_path}")
    plt.close()

def main():
    # Define models to compare
    experiments = [
        ('experiments/pretrained_resnet34_fixed', 'ResNet34 U-Net'),
        ('experiments/baseline_unet_no_pretrain', 'Original U-Net'),
        ('experiments/baseline_simple_cnn', 'Baseline CNN'),
        ('experiments/baseline_efficientnet_b0', 'EfficientNet-B0 U-Net'),
    ]
    
    # Filter existing experiments
    exp_dirs = []
    model_names = []
    
    for exp_dir, name in experiments:
        if os.path.exists(exp_dir):
            exp_dirs.append(exp_dir)
            model_names.append(name)
        else:
            print(f"Warning: {exp_dir} not found, skipping {name}")
    
    if len(exp_dirs) < 2:
        print("Need at least 2 models to compare!")
        return
    
    # Compare models
    compare_models(exp_dirs, model_names)

if __name__ == '__main__':
    main()
