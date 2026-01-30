import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import sys
import os

# Set style
plt.style.use('seaborn-v0_8-paper')
sns.set_palette("husl")

# ============================================================================
# LOAD RESULTS
# ============================================================================

def load_saved_results(base_path):
    """Load saved model results for all three configurations"""
    
    results = {}
    
    # Configuration: (filename, key_name)
    configs = [
        ("pggcn_adam_0.18.pth", "adam"),
        ("pggcn_pcgrad_0.18.pth", "pcgrad"),
        ("pggcn_no_physics_0.18.pth", "no_physics")
    ]
    
    for filename, key in configs:
        filepath = Path(base_path) / filename
        if filepath.exists():
            checkpoint = torch.load(filepath, map_location='cpu', weights_only=False)
            results[key] = {
                'metrics': checkpoint.get('metrics', {}),
                'history': checkpoint.get('history', {}),
                'config': checkpoint.get('config', {})
            }
            print(f"✓ Loaded {key} results from {filepath}")
        else:
            print(f"✗ {key} results not found at {filepath}")
    
    return results


# ============================================================================
# CREATE COMPARISON TABLES
# ============================================================================

def create_training_comparison_table(results):
    """
    Create training metrics comparison table (like the paper's K-fold table)
    
    This mimics Table from the paper showing Mean Train Loss and Mean Test Loss
    """
    print("\n" + "=" * 110)
    print("TRAINING METRICS COMPARISON (Similar to Paper's K-fold Table)")
    print("=" * 110)
    
    # Define display names for each configuration
    display_names = {
        'pcgrad': 'ΔΔG with PCGrad + Multi-loss',
        'adam': 'ΔΔG with Adam + Multi-loss',
        'no_physics': 'ΔΔG without Multi-loss'
    }
    
    table_data = []
    
    for key in ['pcgrad', 'adam', 'no_physics']:
        if key not in results:
            continue
            
        data = results[key]
        history = data['history']
        
        # Get final losses (mean of last 10 epochs for stability)
        train_losses = history['train_empirical'][-10:]
        test_losses = history['val_empirical'][-10:]
        
        mean_train = np.mean(train_losses)
        std_train = np.std(train_losses)
        mean_test = np.mean(test_losses)
        std_test = np.std(test_losses)
        
        # MAE from final metrics
        mae = data['metrics']['mae']
        
        table_data.append({
            'model': display_names.get(key, key),
            'mean_train': mean_train,
            'std_train': std_train,
            'mean_test': mean_test,
            'std_test': std_test,
            'mae': mae
        })
    
    # Print ASCII table
    print(f"\n{'Model Type':<40} | {'Mean Train Loss':<25} | {'Mean Test Loss':<25} | {'MAE':<10}")
    print("-" * 110)
    
    for row in table_data:
        print(f"{row['model']:<40} | "
              f"{row['mean_train']:.2f} ± {row['std_train']:.2f}{'':<14} | "
              f"{row['mean_test']:.2f} ± {row['std_test']:.2f}{'':<14} | "
              f"{row['mae']:.2f}")
    
    print("=" * 110)
    
    # Print LaTeX version
    print("\n" + "=" * 110)
    print("LaTeX Table Format (for paper):")
    print("=" * 110)
    print(r"\begin{table}[ht!]")
    print(r"  \caption{Performance comparison for binding free energy ($\Delta \Delta G$) prediction (kcal/mol).}")
    print(r"  \label{tab:BFE_comparison}")
    print(r"  \centering")
    print(r"  \small")
    print(r"  \begin{tabular}{p{5.0cm} p{2.2cm} p{2.2cm} p{2.2cm}}")
    print(r"    \toprule")
    print(r"    \textbf{Model Type} & \textbf{Mean Train Loss} & \textbf{Mean Test Loss} & \textbf{Mean Absolute Error} \\")
    print(r"    \midrule")
    
    for row in table_data:
        # Clean up model name for LaTeX
        latex_name = row['model'].replace('ΔΔG', r'$\Delta \Delta G$')
        print(f"    {latex_name} & "
              f"{row['mean_train']:.2f}$\\pm${row['std_train']:.2f} & "
              f"{row['mean_test']:.2f}$\\pm${row['std_test']:.2f} & "
              f"{row['mae']:.2f} \\\\")
    
    print(r"    \bottomrule")
    print(r"  \end{tabular}")
    print(r"\end{table}")
    print("=" * 110)
    
    return table_data


def create_test_metrics_table(results):
    """
    Create comprehensive test metrics comparison table
    Shows RMSE, MAE, R², and Physics Loss for all three models
    """
    print("\n" + "=" * 110)
    print("TEST METRICS COMPARISON - ALL MODELS")
    print("=" * 110)
    
    # Define display names for each configuration
    display_names = {
        'pcgrad': 'ΔΔG with PCGrad + Multi-loss',
        'adam': 'ΔΔG with Adam + Multi-loss',
        'no_physics': 'ΔΔG without Multi-loss'
    }
    
    # Print header
    print(f"\n{'Model Type':<40} | {'RMSE':<12} | {'MAE':<12} | {'R²':<12} | {'Physics Loss':<12}")
    print("-" * 110)
    
    # Print each model's metrics
    table_data = []
    for key in ['pcgrad', 'adam', 'no_physics']:
        if key not in results:
            continue
            
        data = results[key]
        metrics = data['metrics']
        
        row_data = {
            'model': display_names.get(key, key),
            'rmse': metrics.get('rmse', 0),
            'mae': metrics.get('mae', 0),
            'r2': metrics.get('r2', 0),
            'physics_loss': metrics.get('physics_loss', 0)
        }
        table_data.append(row_data)
        
        print(f"{row_data['model']:<40} | "
              f"{row_data['rmse']:<12.4f} | "
              f"{row_data['mae']:<12.4f} | "
              f"{row_data['r2']:<12.4f} | "
              f"{row_data['physics_loss']:<12.4f}")
    
    print("=" * 110)
    
    # Calculate and print improvements (best vs others)
    print("\n" + "=" * 110)
    print("IMPROVEMENTS ANALYSIS")
    print("=" * 110)
    
    # Find best values for each metric
    best_rmse = min(row['rmse'] for row in table_data)
    best_mae = min(row['mae'] for row in table_data)
    best_r2 = max(row['r2'] for row in table_data)
    best_physics = min(row['physics_loss'] for row in table_data)
    
    print(f"\nBest Values:")
    print(f"  RMSE: {best_rmse:.4f}")
    print(f"  MAE: {best_mae:.4f}")
    print(f"  R²: {best_r2:.4f}")
    print(f"  Physics Loss: {best_physics:.4f}")
    
    print("\nRelative to Best Model:")
    for row in table_data:
        print(f"\n{row['model']}:")
        print(f"  RMSE: {row['rmse']:.4f} (Δ: {row['rmse'] - best_rmse:+.4f})")
        print(f"  MAE: {row['mae']:.4f} (Δ: {row['mae'] - best_mae:+.4f})")
        print(f"  R²: {row['r2']:.4f} (Δ: {row['r2'] - best_r2:+.4f})")
        print(f"  Physics Loss: {row['physics_loss']:.4f} (Δ: {row['physics_loss'] - best_physics:+.4f})")
    
    print("=" * 110)
    
    # LaTeX table for test metrics
    print("\n" + "=" * 110)
    print("LaTeX Table Format (Test Metrics):")
    print("=" * 110)
    print(r"\begin{table}[ht!]")
    print(r"  \caption{Test set performance metrics for different model configurations.}")
    print(r"  \label{tab:test_metrics_comparison}")
    print(r"  \centering")
    print(r"  \small")
    print(r"  \begin{tabular}{p{5.0cm} p{1.8cm} p{1.8cm} p{1.8cm} p{2.0cm}}")
    print(r"    \toprule")
    print(r"    \textbf{Model Type} & \textbf{RMSE} & \textbf{MAE} & \textbf{R²} & \textbf{Physics Loss} \\")
    print(r"    \midrule")
    
    for row in table_data:
        # Clean up model name for LaTeX
        latex_name = row['model'].replace('ΔΔG', r'$\Delta \Delta G$')
        print(f"    {latex_name} & "
              f"{row['rmse']:.2f} & "
              f"{row['mae']:.2f} & "
              f"{row['r2']:.2f} & "
              f"{row['physics_loss']:.2f} \\\\")
    
    print(r"    \bottomrule")
    print(r"  \end{tabular}")
    print(r"\end{table}")
    print("=" * 110)
    
    # Save tables to CSV
    df = pd.DataFrame(table_data)
    return df


def print_comparison_table(df, phys_weight=0.18):
    """Print formatted comparison table for PCGrad vs No-PCGrad"""
    
    if len(df) < 2:
        print("\nNeed at least 2 models for comparison")
        return
    
    print("\n" + "="*110)
    print(f"PCGRAD VS ADAM VS NO-PHYSICS COMPARISON (Physics Weight λ = {phys_weight})")
    print("="*110)
    print(df.to_string(index=False))
    print("="*110)


# ============================================================================
# TRAINING CURVES
# ============================================================================

def plot_training_curves(results, save_path):
    """Plot training and validation losses over epochs for all three models"""
    
    fig, axes = plt.subplots(2, 2, figsize=(16, 11))
    
    # Get physics weight from config
    phys_weight = results.get('adam', results.get('pcgrad', {})).get('config', {}).get('physics_weight', 0.18)
    
    fig.suptitle(f'Model Comparison: Training Dynamics (λ_phys = {phys_weight})', 
                 fontsize=18, fontweight='bold')
    
    # Define colors and labels
    colors = {'pcgrad': 'tab:orange', 'adam': 'tab:blue', 'no_physics': 'tab:green'}
    labels = {
        'pcgrad': 'PCGrad + Multi-loss',
        'adam': 'Adam + Multi-loss',
        'no_physics': 'No Multi-loss'
    }
    
    # 1. Total Loss
    ax = axes[0, 0]
    for key in ['pcgrad', 'adam', 'no_physics']:
        if key not in results:
            continue
        hist = results[key]['history']
        epochs = range(1, len(hist.get('train_losses', [])) + 1)
        ax.plot(epochs, hist.get('train_losses', []), 
                label=f'{labels[key]} Train', color=colors[key], alpha=0.7, linewidth=1.5)
        ax.plot(epochs, hist.get('val_losses', []), 
                label=f'{labels[key]} Val', color=colors[key], linestyle='--', alpha=0.7, linewidth=1.5)
    
    ax.set_xlabel('Epoch', fontsize=11)
    ax.set_ylabel('Total Loss', fontsize=11)
    ax.set_title('Total Loss (Empirical + Weighted Physics)', fontsize=12, fontweight='bold')
    ax.legend(fontsize=9, loc='best')
    ax.grid(True, alpha=0.3)
    
    # 2. Empirical Loss (RMSE)
    ax = axes[0, 1]
    for key in ['pcgrad', 'adam', 'no_physics']:
        if key not in results:
            continue
        hist = results[key]['history']
        epochs = range(1, len(hist.get('train_empirical', [])) + 1)
        ax.plot(epochs, hist.get('train_empirical', []), 
                label=f'{labels[key]} Train', color=colors[key], alpha=0.7, linewidth=1.5)
        ax.plot(epochs, hist.get('val_empirical', []), 
                label=f'{labels[key]} Val', color=colors[key], linestyle='--', alpha=0.7, linewidth=1.5)
    
    ax.set_xlabel('Epoch', fontsize=11)
    ax.set_ylabel('Empirical Loss (RMSE)', fontsize=11)
    ax.set_title('Empirical Loss', fontsize=12, fontweight='bold')
    ax.legend(fontsize=9, loc='best')
    ax.grid(True, alpha=0.3)
    
    # 3. Physics Loss
    ax = axes[1, 0]
    for key in ['pcgrad', 'adam', 'no_physics']:
        if key not in results:
            continue
        hist = results[key]['history']
        epochs = range(1, len(hist.get('train_physics', [])) + 1)
        ax.plot(epochs, hist.get('train_physics', []), 
                label=f'{labels[key]} Train', color=colors[key], alpha=0.7, linewidth=1.5)
        ax.plot(epochs, hist.get('val_physics', []), 
                label=f'{labels[key]} Val', color=colors[key], linestyle='--', alpha=0.7, linewidth=1.5)
    
    ax.set_xlabel('Epoch', fontsize=11)
    ax.set_ylabel('Physics Loss (Raw)', fontsize=11)
    ax.set_title('Physics Consistency Loss', fontsize=12, fontweight='bold')
    ax.legend(fontsize=9, loc='best')
    ax.grid(True, alpha=0.3)
    
    # 4. Final Metrics Table
    ax = axes[1, 1]
    ax.axis('tight')
    ax.axis('off')
    
    # Create table data
    table_data = [['Model', 'RMSE', 'MAE', 'R²', 'Physics Loss']]
    
    for key in ['pcgrad', 'adam', 'no_physics']:
        if key not in results:
            continue
        metrics = results[key]['metrics']
        table_data.append([
            labels[key],
            f"{metrics.get('rmse', 0):.3f}",
            f"{metrics.get('mae', 0):.3f}",
            f"{metrics.get('r2', 0):.3f}",
            f"{metrics.get('physics_loss', 0):.3f}"
        ])
    
    # Create table
    table = ax.table(cellText=table_data, cellLoc='center', loc='center',
                     colWidths=[0.25, 0.18, 0.18, 0.18, 0.21])
    
    # Style the table
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1, 2.2)
    
    # Color header row
    for i in range(5):
        cell = table[(0, i)]
        cell.set_facecolor('#34495e')
        cell.set_text_props(weight='bold', color='white')
    
    # Color rows with alternating colors
    for i in range(1, len(table_data)):
        for j in range(5):
            cell = table[(i, j)]
            if i % 2 == 0:
                cell.set_facecolor('#ecf0f1')
            else:
                cell.set_facecolor('#ffffff')
    
    ax.set_title('Final Test Metrics', fontsize=12, fontweight='bold', pad=20)
    
    # Add footnote
    fig.text(0.99, 0.01, f'Physics weight (λ_phys) = {phys_weight}', 
             ha='right', va='bottom', fontsize=9, style='italic', color='gray')
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"\n✓ Training curves saved to: {save_path}")
    
    return fig


# ============================================================================
# METRICS COMPARISON PLOT
# ============================================================================

def plot_metrics_comparison(results, save_path):
    """Create detailed metrics comparison plot for all three models"""
    
    fig, ax = plt.subplots(1, 1, figsize=(14, 7))
    
    # Get physics weight
    phys_weight = results.get('adam', results.get('pcgrad', {})).get('config', {}).get('physics_weight', 0.18)
    
    fig.suptitle(f'Model Performance Comparison (λ_phys = {phys_weight})', 
                 fontsize=18, fontweight='bold')
    
    # Define display order and labels
    model_order = ['pcgrad', 'adam', 'no_physics']
    labels_map = {
        'pcgrad': 'PCGrad +\nMulti-loss',
        'adam': 'Adam +\nMulti-loss',
        'no_physics': 'No\nMulti-loss'
    }
    colors_map = {'pcgrad': '#e74c3c', 'adam': '#3498db', 'no_physics': '#2ecc71'}
    
    # Prepare data
    metrics_names = ['MAE', 'RMSE', 'R²', 'Physics\nLoss']
    n_metrics = len(metrics_names)
    n_models = len([k for k in model_order if k in results])
    
    # Create grouped bar positions
    x = np.arange(n_metrics)
    width = 0.25
    
    # Plot bars for each model
    for i, key in enumerate(model_order):
        if key not in results:
            continue
        
        metrics = results[key]['metrics']
        values = [
            metrics.get('mae', 0),
            metrics.get('rmse', 0),
            metrics.get('r2', 0),
            metrics.get('physics_loss', 0)
        ]
        
        offset = (i - 1) * width
        bars = ax.bar(x + offset, values, width, label=labels_map[key],
                     color=colors_map[key], alpha=0.85, edgecolor='black', linewidth=1.2)
        
        # Add value labels on bars
        for bar, val in zip(bars, values):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{val:.2f}', ha='center', va='bottom', 
                   fontsize=9, fontweight='bold')
    
    ax.set_ylabel('Value', fontsize=14, fontweight='bold')
    ax.set_xlabel('Metric', fontsize=14, fontweight='bold')
    ax.set_title('Test Set Performance Metrics', fontsize=15, pad=15)
    ax.set_xticks(x)
    ax.set_xticklabels(metrics_names, fontsize=12)
    ax.legend(fontsize=12, framealpha=0.95, edgecolor='black', loc='upper left')
    ax.grid(True, alpha=0.3, axis='y', linestyle='--')
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"✓ Metrics comparison saved to: {save_path}")
    
    return fig


# ============================================================================
# MAIN
# ============================================================================

def main():
    print("="*110)
    print("COMPREHENSIVE MODEL COMPARISON: PCGrad vs Adam vs No-Physics")
    print("="*110)
    
    # Set paths
    base_path = "/home/exouser/multi-objective-pdbbind/multi-objective-pdbbind/pytorch-implementation/hostguest/saved_models"
    output_dir = "/home/exouser/multi-objective-pdbbind/multi-objective-pdbbind/pytorch-implementation/hostguest/comparison_plots"
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Load results
    print("\nLoading saved results...")
    results = load_saved_results(base_path)
    
    if len(results) < 2:
        print("\n✗ Error: Need at least 2 model results!")
        print(f"   Looking in: {base_path}")
        print("   Expected files:")
        print("     - pggcn_pcgrad_0.18.pth")
        print("     - pggcn_adam_0.18.pth")
        print("     - pggcn_no_physics_0.18.pth")
        return
    
    # Create training comparison table
    print("\nCreating training metrics comparison table...")
    train_table_data = create_training_comparison_table(results)
    
    # Create test metrics table
    print("\nCreating test metrics comparison table...")
    test_metrics_df = create_test_metrics_table(results)
    
    # Save tables to CSV
    train_csv_path = os.path.join(output_dir, "training_comparison_table.csv")
    test_csv_path = os.path.join(output_dir, "test_metrics_comparison_table.csv")
    
    pd.DataFrame(train_table_data).to_csv(train_csv_path, index=False)
    test_metrics_df.to_csv(test_csv_path, index=False)
    
    print(f"\n✓ Training comparison table saved to: {train_csv_path}")
    print(f"✓ Test metrics table saved to: {test_csv_path}")
    
    # Create plots
    print("\nGenerating visualizations...")
    
    # 1. Training curves
    plot_training_curves(results, os.path.join(output_dir, "all_models_training_curves.png"))
    
    # 2. Metrics comparison
    plot_metrics_comparison(results, os.path.join(output_dir, "all_models_metrics_comparison.png"))
    
    print("\n" + "="*110)
    print("COMPARISON COMPLETE!")
    print("="*110)
    print(f"\nAll outputs saved to: {output_dir}")
    print("\nFiles created:")
    print("  1. training_comparison_table.csv - Training metrics (like paper's K-fold table)")
    print("  2. test_metrics_comparison_table.csv - Comprehensive test metrics")
    print("  3. all_models_training_curves.png - Training dynamics for all models")
    print("  4. all_models_metrics_comparison.png - Final metrics comparison")
    print("="*110)

if __name__ == "__main__":
    main()