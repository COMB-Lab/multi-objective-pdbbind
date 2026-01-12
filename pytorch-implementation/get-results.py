#!/usr/bin/env python3
"""
Simple script to view statistics from a saved PGGCN model checkpoint.

Usage:
    python view_model_stats.py <path_to_model.pth>
"""

import sys
import torch


def format_bytes(bytes_val):
    """Format bytes into human-readable format."""
    for unit in ['B', 'KB', 'MB', 'GB', 'TB']:
        if bytes_val < 1024.0:
            return f"{bytes_val:.2f} {unit}"
        bytes_val /= 1024.0
    return f"{bytes_val:.2f} PB"


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
    
    # Metrics
    if 'metrics' in checkpoint:
        print("\nTest Metrics:")
        for key, value in checkpoint['metrics'].items():
            if key == 'sign_accuracy':
                print(f"  {key}: {value*100:.1f}%")
            else:
                print(f"  {key}: {value:.4f}")
    
    # Timing
    if 'timing' in checkpoint:
        print("\nTiming:")
        timing = checkpoint['timing']
        print(f"  Data Loading: {timing.get('data_load_time_formatted', 'N/A')}")
        print(f"  Training: {timing.get('training_time_formatted', 'N/A')}")
        print(f"  Total: {timing.get('total_time_formatted', 'N/A')}")
    
    # Resource usage
    if 'resource_usage' in checkpoint:
        print("\nResource Usage (Peak):")
        res = checkpoint['resource_usage']
        
        if 'cpu' in res:
            print(f"  CPU ({res['cpu']['cores']} cores):")
            print(f"    Mean: {res['cpu']['mean_percent']:.1f}%")
            print(f"    Max:  {res['cpu']['max_percent']:.1f}%")
        
        if 'ram' in res:
            print(f"  RAM:")
            print(f"    Mean: {res['ram']['mean_formatted']} ({res['ram']['mean_percent']:.1f}%)")
            print(f"    Max:  {res['ram']['max_formatted']} ({res['ram']['max_percent']:.1f}%)")
            print(f"    Total: {res['ram']['total_formatted']}")
        
        if 'gpu' in res:
            print(f"  GPU ({res['gpu']['gpu_name']}):")
            print(f"    Mean: {res['gpu']['allocated_mean_formatted']} ({res['gpu']['allocated_mean_percent']:.1f}%)")
            print(f"    Max:  {res['gpu']['allocated_max_formatted']} ({res['gpu']['allocated_max_percent']:.1f}%)")
            print(f"    Total: {res['gpu']['total_formatted']}")
    
    # System info
    if 'system_info' in checkpoint:
        print("\nSystem Information:")
        sys_info = checkpoint['system_info']
        print(f"  Device: {sys_info.get('device', 'N/A')}")
        print(f"  CPU Cores: {sys_info.get('cpu_count', 'N/A')}")
        print(f"  Total RAM: {sys_info.get('total_ram_formatted', 'N/A')}")
        if sys_info.get('has_gpu', False):
            print(f"  GPU: {sys_info.get('gpu_name', 'N/A')}")
            print(f"  GPU Memory: {sys_info.get('total_gpu_memory_formatted', 'N/A')}")
    
    # Training curves info
    if 'train_losses' in checkpoint and 'val_losses' in checkpoint:
        print("\nTraining Progress:")
        print(f"  Total epochs: {len(checkpoint['train_losses'])}")
        print(f"  Final train loss: {checkpoint['train_losses'][-1]:.4f}")
        print(f"  Final val loss: {checkpoint['val_losses'][-1]:.4f}")
        print(f"  Best val loss: {min(checkpoint['val_losses']):.4f} (epoch {checkpoint['val_losses'].index(min(checkpoint['val_losses'])) + 1})")
    
    print("=" * 80)


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
        sys.exit(1)


if __name__ == "__main__":
    main()