#!/usr/bin/env python3
"""
Enhanced script to view comprehensive statistics from a saved PGGCN model checkpoint.

Usage:
    python view_model_stats.py <path_to_model.pth>
"""

import sys
import torch
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np


def format_bytes(bytes_val):
    """Format bytes into human-readable format."""
    for unit in ['B', 'KB', 'MB', 'GB', 'TB']:
        if bytes_val < 1024.0:
            return f"{bytes_val:.2f} {unit}"
        bytes_val /= 1024.0
    return f"{bytes_val:.2f} PB"


def print_training_history(history):
    """Print detailed training history with epoch-by-epoch breakdown."""
    if not history:
        print("  No training history available")
        return
    
    # Get number of epochs
    num_epochs = len(history.get('train_mae', []))
    if num_epochs == 0:
        print("  No training data recorded")
        return
    
    print(f"\n  Total Epochs Trained: {num_epochs}")
    
    # Print summary statistics
    if 'train_mae' in history:
        train_maes = history['train_mae']
        print(f"\n  Training MAE:")
        print(f"    Initial: {train_maes[0]:.4f}")
        print(f"    Final:   {train_maes[-1]:.4f}")
        print(f"    Best:    {min(train_maes):.4f} (epoch {train_maes.index(min(train_maes)) + 1})")
        print(f"    Improvement: {((train_maes[0] - train_maes[-1]) / train_maes[0] * 100):.1f}%")
    
    if 'val_mae' in history:
        val_maes = history['val_mae']
        print(f"\n  Validation MAE:")
        print(f"    Initial: {val_maes[0]:.4f}")
        print(f"    Final:   {val_maes[-1]:.4f}")
        print(f"    Best:    {min(val_maes):.4f} (epoch {val_maes.index(min(val_maes)) + 1})")
        print(f"    Improvement: {((val_maes[0] - val_maes[-1]) / val_maes[0] * 100):.1f}%")
    
    # Print loss components
    print(f"\n  Loss Components (Final Epoch):")
    if 'train_empirical' in history:
        print(f"    Train Empirical: {history['train_empirical'][-1]:.4f}")
    if 'train_physics' in history:
        print(f"    Train Physics:   {history['train_physics'][-1]:.4f}")
    if 'val_empirical' in history:
        print(f"    Val Empirical:   {history['val_empirical'][-1]:.4f}")
    if 'val_physics' in history:
        print(f"    Val Physics:     {history['val_physics'][-1]:.4f}")
    
    # Print epoch-by-epoch details (every 10 epochs + first and last)
    print(f"\n  Epoch-by-Epoch Summary (showing every 10 epochs):")
    print(f"  {'Epoch':>6} | {'Train MAE':>10} | {'Val MAE':>10} | {'Train Emp':>10} | {'Train Phys':>11} | {'Val Emp':>10} | {'Val Phys':>10}")
    print(f"  {'-'*6}-+-{'-'*10}-+-{'-'*10}-+-{'-'*10}-+-{'-'*11}-+-{'-'*10}-+-{'-'*10}")
    
    for epoch in range(num_epochs):
        # Show first, last, and every 10th epoch
        if epoch == 0 or epoch == num_epochs - 1 or (epoch + 1) % 10 == 0:
            train_mae = history.get('train_mae', [None])[epoch]
            val_mae = history.get('val_mae', [None])[epoch]
            train_emp = history.get('train_empirical', [None])[epoch]
            train_phys = history.get('train_physics', [None])[epoch]
            val_emp = history.get('val_empirical', [None])[epoch]
            val_phys = history.get('val_physics', [None])[epoch]
            
            print(f"  {epoch+1:6d} | {train_mae:10.4f} | {val_mae:10.4f} | "
                  f"{train_emp:10.4f} | {train_phys:11.4f} | "
                  f"{val_emp:10.4f} | {val_phys:10.4f}")


def plot_training_curves(history, save_path=None):
    """Generate comprehensive training curve plots."""
    if not history or len(history.get('train_mae', [])) == 0:
        print("\nNo training history to plot")
        return
    
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    epochs = range(1, len(history['train_mae']) + 1)
    
    # Plot 1: MAE over time
    ax = axes[0, 0]
    if 'train_mae' in history:
        ax.plot(epochs, history['train_mae'], 'b-', label='Train MAE', linewidth=2)
    if 'val_mae' in history:
        ax.plot(epochs, history['val_mae'], 'r-', label='Val MAE', linewidth=2)
    ax.set_xlabel('Epoch', fontsize=12)
    ax.set_ylabel('MAE', fontsize=12)
    ax.set_title('Mean Absolute Error', fontsize=14, fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Plot 2: Empirical Loss over time
    ax = axes[0, 1]
    if 'train_empirical' in history:
        ax.plot(epochs, history['train_empirical'], 'b-', label='Train Empirical', linewidth=2)
    if 'val_empirical' in history:
        ax.plot(epochs, history['val_empirical'], 'r-', label='Val Empirical', linewidth=2)
    ax.set_xlabel('Epoch', fontsize=12)
    ax.set_ylabel('Empirical Loss (RMSE)', fontsize=12)
    ax.set_title('Empirical Loss Component', fontsize=14, fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Plot 3: Physics Loss over time
    ax = axes[1, 0]
    if 'train_physics' in history:
        ax.plot(epochs, history['train_physics'], 'b-', label='Train Physics', linewidth=2)
    if 'val_physics' in history:
        ax.plot(epochs, history['val_physics'], 'r-', label='Val Physics', linewidth=2)
    ax.set_xlabel('Epoch', fontsize=12)
    ax.set_ylabel('Physics Loss (RMSE)', fontsize=12)
    ax.set_title('Physics Loss Component', fontsize=14, fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Plot 4: Loss ratio (Empirical / Physics)
    ax = axes[1, 1]
    if 'train_empirical' in history and 'train_physics' in history:
        train_ratio = np.array(history['train_empirical']) / (np.array(history['train_physics']) + 1e-8)
        ax.plot(epochs, train_ratio, 'b-', label='Train Emp/Phys', linewidth=2)
    if 'val_empirical' in history and 'val_physics' in history:
        val_ratio = np.array(history['val_empirical']) / (np.array(history['val_physics']) + 1e-8)
        ax.plot(epochs, val_ratio, 'r-', label='Val Emp/Phys', linewidth=2)
    ax.set_xlabel('Epoch', fontsize=12)
    ax.set_ylabel('Loss Ratio', fontsize=12)
    ax.set_title('Empirical/Physics Loss Ratio', fontsize=14, fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.axhline(y=1.0, color='gray', linestyle='--', alpha=0.5, label='Equal losses')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"\n✓ Training curves saved to: {save_path}")
    else:
        plt.savefig('training_curves.png', dpi=300, bbox_inches='tight')
        print(f"\n✓ Training curves saved to: training_curves.png")
    
    plt.close()


def print_model_stats(model_path):
    """
    Load and print comprehensive information from a saved model checkpoint.
    
    Args:
        model_path: Path to the saved .pth file
    """
    print(f"Loading checkpoint from: {model_path}")
    checkpoint = torch.load(model_path, map_location='cpu', weights_only=False)
    
    print("\n" + "=" * 80)
    print("SAVED MODEL INFORMATION")
    print("=" * 80)
    
    # Timestamp
    if 'timestamp' in checkpoint:
        print(f"\nTimestamp: {checkpoint['timestamp']}")
    
    # Configuration
    if 'config' in checkpoint:
        print("\nConfiguration:")
        for key, value in checkpoint['config'].items():
            print(f"  {key}: {value}")
    
    # Hyperparameters
    if 'hyperparameters' in checkpoint:
        print("\nHyperparameters:")
        for key, value in checkpoint['hyperparameters'].items():
            print(f"  {key}: {value}")
    
    # Dataset info
    if 'dataset_info' in checkpoint:
        print("\nDataset Information:")
        for key, value in checkpoint['dataset_info'].items():
            print(f"  {key}: {value}")
    
    # Test Metrics
    if 'metrics' in checkpoint:
        print("\nTest Set Metrics:")
        metrics = checkpoint['metrics']
        if 'rmse' in metrics:
            print(f"  RMSE: {metrics['rmse']:.4f} kcal/mol")
        if 'mae' in metrics:
            print(f"  MAE:  {metrics['mae']:.4f} kcal/mol")
        if 'r2' in metrics:
            print(f"  R²:   {metrics['r2']:.4f}")
        if 'sign_accuracy' in metrics:
            print(f"  Sign Accuracy: {metrics['sign_accuracy']*100:.1f}%")
    
    # Training History - ENHANCED
    if 'history' in checkpoint:
        print("\n" + "=" * 80)
        print("TRAINING HISTORY")
        print("=" * 80)
        print_training_history(checkpoint['history'])
        
        # Generate plots
        plot_training_curves(checkpoint['history'], 
                           save_path=model_path.replace('.pth', '_training_curves.png'))
    
    # Timing
    if 'timing' in checkpoint:
        print("\n" + "=" * 80)
        print("TIMING")
        print("=" * 80)
        timing = checkpoint['timing']
        print(f"  Data Loading: {timing.get('data_load_time_formatted', 'N/A')}")
        print(f"  Training: {timing.get('training_time_formatted', 'N/A')}")
        print(f"  Total: {timing.get('total_time_formatted', 'N/A')}")
    
    # Resource usage
    if 'resource_usage' in checkpoint:
        print("\n" + "=" * 80)
        print("RESOURCE USAGE (PEAK)")
        print("=" * 80)
        res = checkpoint['resource_usage']
        
        if 'cpu' in res:
            print(f"\n  CPU ({res['cpu']['cores']} cores):")
            print(f"    Mean: {res['cpu']['mean_percent']:.1f}%")
            print(f"    Max:  {res['cpu']['max_percent']:.1f}%")
        
        if 'ram' in res:
            print(f"\n  RAM:")
            print(f"    Mean: {res['ram']['mean_formatted']} ({res['ram']['mean_percent']:.1f}%)")
            print(f"    Max:  {res['ram']['max_formatted']} ({res['ram']['max_percent']:.1f}%)")
            print(f"    Total: {res['ram']['total_formatted']}")
        
        if 'gpu' in res:
            print(f"\n  GPU ({res['gpu']['gpu_name']}):")
            print(f"    Mean: {res['gpu']['allocated_mean_formatted']} ({res['gpu']['allocated_mean_percent']:.1f}%)")
            print(f"    Max:  {res['gpu']['allocated_max_formatted']} ({res['gpu']['allocated_max_percent']:.1f}%)")
            print(f"    Total: {res['gpu']['total_formatted']}")
    
    # System info
    if 'system_info' in checkpoint:
        print("\n" + "=" * 80)
        print("SYSTEM INFORMATION")
        print("=" * 80)
        sys_info = checkpoint['system_info']
        print(f"  Device: {sys_info.get('device', 'N/A')}")
        print(f"  CPU Cores: {sys_info.get('cpu_count', 'N/A')}")
        print(f"  Total RAM: {sys_info.get('total_ram_formatted', 'N/A')}")
        if sys_info.get('has_gpu', False):
            print(f"  GPU: {sys_info.get('gpu_name', 'N/A')}")
            print(f"  GPU Memory: {sys_info.get('total_gpu_memory_formatted', 'N/A')}")
    
    print("\n" + "=" * 80)


def main():
    if len(sys.argv) != 2:
        print("Usage: python view_model_stats.py <path_to_model.pth>")
        print("\nExample:")
        print("  python view_model_stats.py /path/to/pggcn_model.pth")
        sys.exit(1)
    
    model_path = sys.argv[1]
    
    try:
        print_model_stats(model_path)
    except FileNotFoundError:
        print(f"Error: File not found: {model_path}")
        sys.exit(1)
    except Exception as e:
        print(f"Error loading checkpoint: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()