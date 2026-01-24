"""
Empirical vs Physics Loss Plotting for PGGCN Training

This script provides two ways to create the plot:
1. Standalone: Load from saved model and create plots
2. Inline: Add function directly to training script

Author: Based on TensorFlow implementation
"""

import matplotlib.pyplot as plt
import numpy as np
from adjustText import adjust_text
import torch
import os


# ==============================================================================
# STANDALONE VERSION - Use with saved model files
# ==============================================================================

def plot_empirical_vs_physics_over_epochs(history, physics_weight, 
                                          save_path="empirical_vs_physics_by_epoch.png"):
    """
    Plots empirical vs physics loss at each epoch for a single training run.
    Matches the style of the Pareto front plots.
    
    Args:
        history: Dictionary containing training history with keys:
                 'train_empirical', 'train_physics', 'val_empirical', 'val_physics'
        physics_weight: The lambda value used for physics loss weighting
        save_path: Path to save the figure
    
    Returns:
        matplotlib Figure object
    """
    # Extract losses (use validation for cleaner visualization)
    empirical = history['val_empirical']
    physics = history['val_physics']
    epochs = list(range(1, len(empirical) + 1))
    
    # Match Pareto style: larger square figure
    plt.figure(figsize=(10, 10))
    
    # Line + scatter (matches your original code)
    plt.plot(empirical, physics, '-', color='deepskyblue', alpha=0.8, 
             linewidth=2, zorder=1)
    plt.scatter(empirical, physics, s=20, color='orchid', alpha=0.8,
                edgecolors='orchid', zorder=2)
    
    # Add epoch labels for selected points
    texts = []
    for i, w in enumerate(epochs):
        # Label every 25th epoch, plus first and last
        if w % 25 == 0 or i == 0 or i == len(epochs) - 1:
            texts.append(plt.text(empirical[i], physics[i], f"{w}",
                                  fontsize=10, zorder=4))
    
    # Adjust text positions to avoid overlap
    adjust_text(
        texts,
        arrowprops=dict(arrowstyle='-', color='gray', lw=0.5),
        expand_text=(1.4, 1.2),
        expand_points=(1.4, 1.4),
        force_points=1.5,
        force_text=1.0
    )
    
    # Same axis/label formatting as Pareto plot
    plt.xlabel("Empirical Loss", fontsize=14)
    plt.ylabel("Physics Loss", fontsize=14)
    plt.title(f"Empirical vs Physics Loss\n(λ = {physics_weight})",
              fontsize=16)
    plt.xscale('log')
    plt.yscale('log')
    plt.tick_params(axis='both', which='major', labelsize=15)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    # Save at same DPI
    plt.savefig(save_path, dpi=150)
    print(f"✓ Plot saved to: {save_path}")
    plt.show()
    
    return plt.gcf()


def load_and_plot_from_saved_model(model_path, output_dir=None):
    """
    Load a saved model and create the empirical vs physics plot.
    
    Args:
        model_path: Path to the saved .pth file (e.g., 'pggcn_pcgrad_final.pth')
        output_dir: Directory to save plot (defaults to same directory as model)
    
    Example:
        >>> load_and_plot_from_saved_model(
        ...     '/path/to/saved_models/pggcn_pcgrad_final.pth'
        ... )
    """
    print(f"\nLoading model from: {model_path}")
    
    # Load checkpoint (weights_only=False for PyTorch 2.6+ compatibility)
    checkpoint = torch.load(model_path, map_location='cpu', weights_only=False)
    history = checkpoint['history']
    config = checkpoint['config']
    physics_weight = config['physics_weight']
    
    print(f"✓ Loaded model")
    print(f"  Physics weight: {physics_weight}")
    print(f"  Total epochs: {len(history['val_empirical'])}")
    
    # Set output directory
    if output_dir is None:
        output_dir = os.path.dirname(model_path)
    os.makedirs(output_dir, exist_ok=True)
    
    # Create filename
    base_name = os.path.splitext(os.path.basename(model_path))[0]
    save_path = os.path.join(output_dir, f"{base_name}_empirical_vs_physics.png")
    
    # Create plot
    plot_empirical_vs_physics_over_epochs(history, physics_weight, save_path)
    
    return save_path


# ==============================================================================
# INLINE VERSION - Add this function to your training script
# ==============================================================================

def plot_pcgrad_inline(history, physics_weight, save_dir):
    """
    Simplified inline version to add directly to training script.
    
    Add this function to your training script, then call it after training:
    
        # In main(), after training completes:
        if use_pcgrad:
            plot_pcgrad_inline(history, config.PHYSICS_WEIGHT, save_dir)
    
    Args:
        history: Dictionary from train_model() with keys:
                 'train_empirical', 'train_physics', 'val_empirical', 'val_physics'
        physics_weight: The lambda value (e.g., 0.58)
        save_dir: Directory to save the plot
    """
    empirical = history['val_empirical']
    physics = history['val_physics']
    epochs = list(range(1, len(empirical) + 1))
    
    plt.figure(figsize=(10, 10))
    
    # Line + scatter
    plt.plot(empirical, physics, '-', color='deepskyblue', alpha=0.8, zorder=1)
    plt.scatter(empirical, physics, s=20, color='orchid', alpha=0.8,
                edgecolors='orchid', zorder=2)
    
    # Epoch labels
    texts = []
    for i, w in enumerate(epochs):
        if w % 25 == 0 or i == 0 or i == len(epochs) - 1:
            texts.append(plt.text(empirical[i], physics[i], f"{w}",
                                  fontsize=10, zorder=4))
    
    adjust_text(
        texts,
        arrowprops=dict(arrowstyle='-', color='gray', lw=0.5),
        expand_text=(1.4, 1.2),
        expand_points=(1.4, 1.4),
        force_points=1.5,
        force_text=1.0
    )
    
    # Formatting
    plt.xlabel("Empirical Loss", fontsize=14)
    plt.ylabel("Physics Loss", fontsize=14)
    plt.title(f"Empirical vs Physics Loss\n(λ = {physics_weight})", fontsize=16)
    plt.xscale('log')
    plt.yscale('log')
    plt.tick_params(axis='both', which='major', labelsize=15)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    # Save
    os.makedirs(save_dir, exist_ok=True)
    save_path = os.path.join(save_dir, 'pcgrad_empirical_vs_physics.png')
    plt.savefig(save_path, dpi=150)
    print(f"\n✓ Plot saved to: {save_path}")
    plt.close()


# ==============================================================================
# USAGE EXAMPLES
# ==============================================================================

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(
        description='Create empirical vs physics loss plot from saved model'
    )
    parser.add_argument(
        '--model-path', 
        type=str,
        default='/home/exouser/multi-objective-pdbbind/multi-objective-pdbbind/pytorch-implementation/hostguest/saved_models/pggcn_pcgrad_0.18.pth',
        help='Path to saved model .pth file'
    )
    parser.add_argument(
        '--output-dir', 
        type=str,
        default=None,
        help='Directory to save plots (default: same as model directory)'
    )
    
    args = parser.parse_args()
    
    print("=" * 80)
    print("EMPIRICAL VS PHYSICS LOSS PLOTTER")
    print("=" * 80)
    
    if not os.path.exists(args.model_path):
        print(f"\n✗ Error: Model file not found: {args.model_path}")
        print("\nExpected paths:")
        print("  For PCGrad:    .../saved_models/pggcn_pcgrad_final.pth")
        print("  For no-PCGrad: .../saved_models/pggcn_no_pcgrad_final.pth")
        exit(1)
    
    # Create plot from saved model
    output_path = load_and_plot_from_saved_model(args.model_path, args.output_dir)
    
    print("\n" + "=" * 80)
    print("PLOT CREATED SUCCESSFULLY")
    print("=" * 80)
    print(f"\nView your plot at: {output_path}")
