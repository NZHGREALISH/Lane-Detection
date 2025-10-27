"""
运行实验脚本：对比U-Net和Baseline CNN
"""
import os
import subprocess
import json
import pandas as pd
import matplotlib.pyplot as plt


def run_experiment(model_name, config_args):
    """运行单个实验"""
    print("\n" + "="*80)
    print(f"Running experiment: {model_name}")
    print("="*80 + "\n")
    
    # 构建命令
    cmd = [
        'python', 'train.py',
        '--model', config_args['model'],
        '--save_dir', config_args['save_dir'],
        '--batch_size', str(config_args.get('batch_size', 16)),
        '--epochs', str(config_args.get('epochs', 50)),
        '--lr', str(config_args.get('lr', 1e-4)),
        '--image_size', str(config_args.get('image_size', 256)),
        '--loss', config_args.get('loss', 'bce_dice'),
        '--optimizer', config_args.get('optimizer', 'adam'),
        '--scheduler', config_args.get('scheduler', 'plateau'),
    ]
    
    # 运行训练
    result = subprocess.run(cmd)
    
    if result.returncode != 0:
        print(f"❌ Experiment {model_name} failed!")
        return None
    
    print(f"✓ Experiment {model_name} completed!")
    return config_args['save_dir']


def evaluate_model(model_name, checkpoint_path, save_dir):
    """评估模型"""
    print(f"\n评估模型: {model_name}")
    
    cmd = [
        'python', 'evaluate.py',
        '--model', model_name,
        '--checkpoint', checkpoint_path,
        '--save_dir', save_dir,
        '--visualize',
    ]
    
    result = subprocess.run(cmd)
    
    if result.returncode != 0:
        print(f"❌ Evaluation {model_name} failed!")
        return None
    
    # 读取结果
    results_path = os.path.join(save_dir, 'evaluation_results.json')
    if os.path.exists(results_path):
        with open(results_path, 'r') as f:
            results = json.load(f)
        return results['average']
    
    return None


def compare_models(results_dict, save_path='comparison'):
    """对比模型性能"""
    os.makedirs(save_path, exist_ok=True)
    
    # 创建对比表格
    df = pd.DataFrame(results_dict).T
    df = df.round(4)
    
    print("\n" + "="*80)
    print("模型性能对比")
    print("="*80)
    print(df.to_string())
    
    # 保存表格
    table_path = os.path.join(save_path, 'comparison_table.csv')
    df.to_csv(table_path)
    print(f"\n对比表格已保存到: {table_path}")
    
    # 绘制对比图
    metrics = ['iou', 'dice', 'pixel_acc', 'f1']
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    axes = axes.flatten()
    
    for idx, metric in enumerate(metrics):
        if metric in df.columns:
            df[metric].plot(kind='bar', ax=axes[idx], color=['#3498db', '#e74c3c'])
            axes[idx].set_title(f'{metric.upper()} Comparison', fontsize=12, fontweight='bold')
            axes[idx].set_ylabel(metric.upper())
            axes[idx].set_xlabel('Model')
            axes[idx].grid(axis='y', alpha=0.3)
            axes[idx].set_ylim([0, 1])
            
            # 添加数值标签
            for i, v in enumerate(df[metric]):
                axes[idx].text(i, v + 0.02, f'{v:.4f}', ha='center', va='bottom')
    
    plt.tight_layout()
    chart_path = os.path.join(save_path, 'comparison_chart.png')
    plt.savefig(chart_path, dpi=150, bbox_inches='tight')
    print(f"对比图表已保存到: {chart_path}")
    plt.close()


def main():
    """主函数"""
    print("="*80)
    print("Drivable Area Segmentation - 实验对比")
    print("="*80)
    
    # 定义实验配置
    experiments = {
        'U-Net': {
            'model': 'unet',
            'save_dir': 'experiments/unet',
            'batch_size': 16,
            'epochs': 30,  # 可以根据需要调整
            'lr': 1e-4,
            'image_size': 256,
            'loss': 'bce_dice',
            'optimizer': 'adam',
            'scheduler': 'plateau',
        },
        'Baseline CNN': {
            'model': 'baseline',
            'save_dir': 'experiments/baseline',
            'batch_size': 16,
            'epochs': 30,
            'lr': 1e-4,
            'image_size': 256,
            'loss': 'bce_dice',
            'optimizer': 'adam',
            'scheduler': 'plateau',
        }
    }
    
    # 运行实验
    experiment_dirs = {}
    for exp_name, exp_config in experiments.items():
        exp_dir = run_experiment(exp_name, exp_config)
        if exp_dir:
            experiment_dirs[exp_name] = exp_dir
    
    # 评估模型
    print("\n" + "="*80)
    print("开始评估模型")
    print("="*80)
    
    results = {}
    for exp_name, exp_dir in experiment_dirs.items():
        checkpoint_path = os.path.join(exp_dir, 'checkpoints', 'best_model.pth')
        eval_dir = os.path.join('evaluation_results', exp_name.lower().replace(' ', '_'))
        
        if os.path.exists(checkpoint_path):
            model_type = experiments[exp_name]['model']
            result = evaluate_model(model_type, checkpoint_path, eval_dir)
            if result:
                results[exp_name] = result
    
    # 对比结果
    if len(results) > 0:
        compare_models(results, save_path='comparison_results')
    
    print("\n" + "="*80)
    print("所有实验完成！")
    print("="*80)


if __name__ == '__main__':
    main()
