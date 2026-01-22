"""
PCGrad vs No-PCGrad Comparison and Visualization
Loads saved models and creates publication-quality plots and tables
"""

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
    """Load saved model results"""
    
    results = {}
    
    # Try to load no-PCGrad results
    no_pcgrad_path = Path(base_path) / "pggcn_no_pcgrad_0.2.pth"
    if no_pcgrad_path.exists():
        checkpoint = torch.load(no_pcgrad_path, map_location='cpu', weights_only=False)
        results['no_pcgrad'] = {
            'metrics': checkpoint.get('metrics', {}),
            'history': checkpoint.get('history', {}),
            'config': checkpoint.get('config', {})
        }
        print(f"✓ Loaded No-PCGrad results from {no_pcgrad_path}")
    else:
        print(f"✗ No-PCGrad results not found at {no_pcgrad_path}")
    
    # Try to load PCGrad results
    pcgrad_path = Path(base_path) / "pggcn_pcgrad_0.2.pth"
    if pcgrad_path.exists():
        checkpoint = torch.load(pcgrad_path, map_location='cpu', weights_only=False)
        results['pcgrad'] = {
            'metrics': checkpoint.get('metrics', {}),
            'history': checkpoint.get('history', {}),
            'config': checkpoint.get('config', {})
        }
        print(f"✓ Loaded PCGrad results from {pcgrad_path}")
    else:
        print(f"✗ PCGrad results not found at {pcgrad_path}")
    
    return results

# ============================================================================
# CREATE COMPARISON TABLE
# ============================================================================

def create_comparison_table(results):
    """Create detailed comparison table"""
    
    if 'no_pcgrad' not in results or 'pcgrad' not in results:
        print("Error: Missing results for comparison")
        return None
    
    no_pc = results['no_pcgrad']['metrics']
    pc = results['pcgrad']['metrics']
    
    # Create comparison data
    data = {
        'Metric': ['Empirical Loss (RMSE)', 'MAE', 'R²', 'Physics Loss'],
        'No PCGrad': [
            no_pc.get('rmse', 0),
            no_pc.get('mae', 0),
            no_pc.get('r2', 0),
            no_pc.get('physics_loss', 0)
        ],
        'PCGrad': [
            pc.get('rmse', 0),
            pc.get('mae', 0),
            pc.get('r2', 0),
            pc.get('physics_loss', 0)
        ]
    }
    
    df = pd.DataFrame(data)
    
    # Calculate absolute differences (improvement)
    df['Improvement'] = df['No PCGrad'] - df['PCGrad']
    
    # Format for display
    df['No PCGrad'] = df['No PCGrad'].round(4)
    df['PCGrad'] = df['PCGrad'].round(4)
    df['Improvement'] = df['Improvement'].round(4)
    
    return df

def print_comparison_table(df, phys_weight=0.58):
    """Print formatted comparison table"""
    
    print("\n" + "="*90)
    print(f"PCGRAD VS NO-PCGRAD COMPARISON (Physics Weight λ = {phys_weight})")
    print("="*90)
    print(df.to_string(index=False))
    print("="*90)
    
    # Interpretation
    print("\nINTERPRETATION:")
    print("-"*90)
    
    emp_improv = df[df['Metric'] == 'Empirical Loss (RMSE)']['Improvement'].values[0]
    mae_improv = df[df['Metric'] == 'MAE']['Improvement'].values[0]
    phys_improv = df[df['Metric'] == 'Physics Loss']['Improvement'].values[0]
    
    if emp_improv > 0:
        print(f"✓ PCGrad IMPROVES Empirical Loss (RMSE) by {emp_improv:.4f} (lower is better)")
    else:
        print(f"✗ PCGrad WORSENS Empirical Loss by {abs(emp_improv):.4f}")
    
    if mae_improv > 0:
        print(f"✓ PCGrad IMPROVES MAE by {mae_improv:.4f} (lower is better)")
    else:
        print(f"✗ PCGrad WORSENS MAE by {abs(mae_improv):.4f}")
    
    if phys_improv < 0:
        print(f"✓ Physics loss IMPROVED by {abs(phys_improv):.4f} (lower is better)")
    else:
        print(f"⚠ Physics loss INCREASED by {phys_improv:.4f} (acceptable trade-off)")
    
    print("\nPAPER REFERENCE (Table 2):")
    print("  Without PCGrad: MAE 3.06, Empirical 3.69, Physics 12.78")
    print("  With PCGrad:    MAE 2.94, Empirical 3.58, Physics 12.90")
    print("  Improvement:    MAE -0.12, Empirical -0.11 (PCGrad better)")
    print("="*90)

# ============================================================================
# TRAINING CURVES
# ============================================================================

def plot_training_curves(results, save_path):
    """Plot training and validation losses over epochs"""
    
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    
    # Get physics weight from config
    phys_weight = results['no_pcgrad']['config'].get('physics_weight', 0.58)
    
    fig.suptitle(f'PCGrad vs No-PCGrad: Training Dynamics (λ_phys = {phys_weight})', 
                 fontsize=16, fontweight='bold')
    
    # Extract histories
    no_pc_hist = results['no_pcgrad']['history']
    pc_hist = results['pcgrad']['history']
    
    epochs_no_pc = range(1, len(no_pc_hist.get('train_losses', [])) + 1)
    epochs_pc = range(1, len(pc_hist.get('train_losses', [])) + 1)
    
    # 1. Total Loss
    ax = axes[0, 0]
    ax.plot(epochs_no_pc, no_pc_hist.get('train_losses', []), 
            label='No PCGrad Train', color='tab:blue', alpha=0.7)
    ax.plot(epochs_no_pc, no_pc_hist.get('val_losses', []), 
            label='No PCGrad Val', color='tab:blue', linestyle='--', alpha=0.7)
    ax.plot(epochs_pc, pc_hist.get('train_losses', []), 
            label='PCGrad Train', color='tab:orange', alpha=0.7)
    ax.plot(epochs_pc, pc_hist.get('val_losses', []), 
            label='PCGrad Val', color='tab:orange', linestyle='--', alpha=0.7)
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Total Loss')
    ax.set_title('Total Loss (Empirical + Physics)')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # 2. Empirical Loss
    ax = axes[0, 1]
    ax.plot(epochs_no_pc, no_pc_hist.get('train_empirical', []), 
            label='No PCGrad Train', color='tab:blue', alpha=0.7)
    ax.plot(epochs_no_pc, no_pc_hist.get('val_empirical', []), 
            label='No PCGrad Val', color='tab:blue', linestyle='--', alpha=0.7)
    ax.plot(epochs_pc, pc_hist.get('train_empirical', []), 
            label='PCGrad Train', color='tab:orange', alpha=0.7)
    ax.plot(epochs_pc, pc_hist.get('val_empirical', []), 
            label='PCGrad Val', color='tab:orange', linestyle='--', alpha=0.7)
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Empirical Loss (RMSE)')
    ax.set_title('Empirical Loss')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # 3. Physics Loss
    ax = axes[1, 0]
    ax.plot(epochs_no_pc, no_pc_hist.get('train_physics', []), 
            label='No PCGrad Train', color='tab:blue', alpha=0.7)
    ax.plot(epochs_no_pc, no_pc_hist.get('val_physics', []), 
            label='No PCGrad Val', color='tab:blue', linestyle='--', alpha=0.7)
    ax.plot(epochs_pc, pc_hist.get('train_physics', []), 
            label='PCGrad Train', color='tab:orange', alpha=0.7)
    ax.plot(epochs_pc, pc_hist.get('val_physics', []), 
            label='PCGrad Val', color='tab:orange', linestyle='--', alpha=0.7)
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Physics Loss')
    ax.set_title('Physics Consistency Loss')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # 4. Final Metrics Table (instead of bar chart)
    ax = axes[1, 1]
    ax.axis('tight')
    ax.axis('off')
    
    # Create table data with empirical loss
    table_data = [
        ['Metric', 'No PCGrad', 'PCGrad', 'Δ (Improvement)'],
        ['Empirical Loss',
         f"{results['no_pcgrad']['metrics'].get('rmse', 0):.4f}",
         f"{results['pcgrad']['metrics'].get('rmse', 0):.4f}",
         f"{results['no_pcgrad']['metrics'].get('rmse', 0) - results['pcgrad']['metrics'].get('rmse', 0):.4f}"],
        ['MAE', 
         f"{results['no_pcgrad']['metrics'].get('mae', 0):.4f}",
         f"{results['pcgrad']['metrics'].get('mae', 0):.4f}",
         f"{results['no_pcgrad']['metrics'].get('mae', 0) - results['pcgrad']['metrics'].get('mae', 0):.4f}"],
        ['R²',
         f"{results['no_pcgrad']['metrics'].get('r2', 0):.4f}",
         f"{results['pcgrad']['metrics'].get('r2', 0):.4f}",
         f"{results['pcgrad']['metrics'].get('r2', 0) - results['no_pcgrad']['metrics'].get('r2', 0):.4f}"],
        ['Physics Loss',
         f"{results['no_pcgrad']['metrics'].get('physics_loss', 0):.4f}",
         f"{results['pcgrad']['metrics'].get('physics_loss', 0):.4f}",
         f"{results['pcgrad']['metrics'].get('physics_loss', 0) - results['no_pcgrad']['metrics'].get('physics_loss', 0):.4f}"]
    ]
    
    # Create table
    table = ax.table(cellText=table_data, cellLoc='center', loc='center',
                     colWidths=[0.25, 0.25, 0.25, 0.25])
    
    # Style the table
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1, 2)
    
    # Color header row
    for i in range(4):
        cell = table[(0, i)]
        cell.set_facecolor('#34495e')
        cell.set_text_props(weight='bold', color='white')
    
    # Color improvement column based on metric type
    for i in range(1, 5):  # Now we have 5 data rows (1-4 indices)
        cell = table[(i, 3)]
        value = float(table_data[i][3])
        # Empirical, MAE - positive delta is good (lower is better)
        # R² - positive delta is good (higher is better)  
        # Physics - depends on preference
        if i in [1, 2]:  # Empirical, MAE - positive improvement is good
            cell.set_facecolor('#d5f4e6' if value > 0 else '#fadbd8')
        elif i == 3:  # R² - positive improvement is good
            cell.set_facecolor('#d5f4e6' if value > 0 else '#fadbd8')
        else:  # Physics loss (i == 4) - negative improvement means lower loss
            cell.set_facecolor('#fadbd8' if value > 0 else '#d5f4e6')
    
    ax.set_title('Final Test Metrics', fontsize=12, fontweight='bold', pad=20)
    
    # Add footnote about physics weight
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
    """Create detailed metrics comparison plot"""
    
    fig, ax = plt.subplots(1, 1, figsize=(12, 6))
    
    # Get physics weight
    phys_weight = results['no_pcgrad']['config'].get('physics_weight', 0.58)
    
    fig.suptitle(f'PCGrad Impact on Model Performance (λ_phys = {phys_weight})', 
                 fontsize=16, fontweight='bold')
    
    no_pc = results['no_pcgrad']['metrics']
    pc = results['pcgrad']['metrics']
    
    # Side-by-side comparison
    metrics = ['MAE', 'RMSE', 'R²', 'Physics\nLoss']
    no_pc_vals = [no_pc.get('mae', 0), no_pc.get('rmse', 0), 
                  no_pc.get('r2', 0), no_pc.get('physics_loss', 0)]
    pc_vals = [pc.get('mae', 0), pc.get('rmse', 0), 
               pc.get('r2', 0), pc.get('physics_loss', 0)]
    
    x = np.arange(len(metrics))
    width = 0.35
    
    bars1 = ax.bar(x - width/2, no_pc_vals, width, label='No PCGrad', 
                   color='#3498db', alpha=0.8, edgecolor='black', linewidth=1.2)
    bars2 = ax.bar(x + width/2, pc_vals, width, label='PCGrad', 
                   color='#e74c3c', alpha=0.8, edgecolor='black', linewidth=1.2)
    
    ax.set_ylabel('Value', fontsize=13, fontweight='bold')
    ax.set_xlabel('Metric', fontsize=13, fontweight='bold')
    ax.set_title('Test Set Performance Metrics', fontsize=14, pad=15)
    ax.set_xticks(x)
    ax.set_xticklabels(metrics, fontsize=11)
    ax.legend(fontsize=12, framealpha=0.9, edgecolor='black')
    ax.grid(True, alpha=0.3, axis='y', linestyle='--')
    
    # Add value labels on bars
    for bars in [bars1, bars2]:
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{height:.3f}', ha='center', va='bottom', 
                   fontsize=9, fontweight='bold')
    
    # Add improvement annotations
    for i, (v1, v2) in enumerate(zip(no_pc_vals, pc_vals)):
        improvement = v1 - v2
        if i < 3:  # MAE, RMSE, R² (lower/higher is better)
            if (i < 2 and improvement > 0) or (i == 2 and improvement < 0):
                color = 'green'
                symbol = '▼' if i < 2 else '▲'
            else:
                color = 'red'
                symbol = '▲' if i < 2 else '▼'
        else:  # Physics loss (lower is better for pure accuracy)
            color = 'orange'
            symbol = '▲'
        
        ax.text(i, max(v1, v2) * 1.02, f'{symbol}', ha='center', va='bottom',
               fontsize=16, color=color, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"✓ Metrics comparison saved to: {save_path}")
    
    return fig

# ============================================================================
# CONVERGENCE ANALYSIS
# ============================================================================

def plot_convergence_analysis(results, save_path):
    """Analyze convergence behavior"""
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    # Get physics weight
    phys_weight = results['no_pcgrad']['config'].get('physics_weight', 0.58)
    
    fig.suptitle(f'Convergence Analysis (λ_phys = {phys_weight})', 
                 fontsize=16, fontweight='bold')
    
    no_pc_hist = results['no_pcgrad']['history']
    pc_hist = results['pcgrad']['history']
    
    # 1. Loss reduction over time
    ax = axes[0]
    
    # Calculate cumulative improvement
    no_pc_val = no_pc_hist.get('val_empirical', [])
    pc_val = pc_hist.get('val_empirical', [])
    
    if no_pc_val and pc_val:
        no_pc_improv = [(no_pc_val[0] - v) / no_pc_val[0] * 100 for v in no_pc_val]
        pc_improv = [(pc_val[0] - v) / pc_val[0] * 100 for v in pc_val]
        
        ax.plot(no_pc_improv, label='No PCGrad', color='tab:blue', linewidth=2)
        ax.plot(pc_improv, label='PCGrad', color='tab:orange', linewidth=2)
        ax.set_xlabel('Epoch', fontsize=12)
        ax.set_ylabel('% Improvement from Initial', fontsize=12)
        ax.set_title('Validation Loss Improvement', fontsize=14)
        ax.legend(fontsize=11)
        ax.grid(True, alpha=0.3)
    
    # 2. Final epochs comparison (last 50 epochs)
    ax = axes[1]
    
    if no_pc_val and pc_val:
        last_n = min(50, len(no_pc_val), len(pc_val))
        
        ax.plot(range(len(no_pc_val) - last_n, len(no_pc_val)), 
               no_pc_val[-last_n:], label='No PCGrad', 
               color='tab:blue', linewidth=2, marker='o', markersize=3)
        ax.plot(range(len(pc_val) - last_n, len(pc_val)), 
               pc_val[-last_n:], label='PCGrad', 
               color='tab:orange', linewidth=2, marker='s', markersize=3)
        
        ax.set_xlabel('Epoch', fontsize=12)
        ax.set_ylabel('Validation Empirical Loss', fontsize=12)
        ax.set_title(f'Final {last_n} Epochs (Convergence)', fontsize=14)
        ax.legend(fontsize=11)
        ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"✓ Convergence analysis saved to: {save_path}")
    
    return fig

# ============================================================================
# MAIN
# ============================================================================

def main():
    print("="*100)
    print("PCGRAD VS NO-PCGRAD: COMPREHENSIVE COMPARISON")
    print("="*100)
    
    # Set paths
    base_path = "/home/exouser/multi-objective-pdbbind/multi-objective-pdbbind/pytorch-implementation/saved_models"
    output_dir = "/home/exouser/multi-objective-pdbbind/multi-objective-pdbbind/pytorch-implementation/comparison_plots"
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Load results
    print("\nLoading saved results...")
    results = load_saved_results(base_path)
    
    if len(results) < 2:
        print("\n✗ Error: Need both PCGrad and No-PCGrad results!")
        print(f"   Looking in: {base_path}")
        print("   Expected files:")
        print("     - pggcn_no_pcgrad_final.pth")
        print("     - pggcn_pcgrad_final.pth")
        return
    
    # Create comparison table
    print("\nCreating comparison table...")
    df = create_comparison_table(results)
    if df is not None:
        # Get physics weight from config
        phys_weight = results['no_pcgrad']['config'].get('physics_weight', 0.58)
        print_comparison_table(df, phys_weight)
        
        # Save table to CSV
        table_path = os.path.join(output_dir, "new_pcgrad_comparison_table.csv")
        df.to_csv(table_path, index=False)
        print(f"\n✓ Table saved to: {table_path}")
    
    # Create plots
    print("\nGenerating visualizations...")
    
    # 1. Training curves
    plot_training_curves(results, os.path.join(output_dir, "new_training_curves.png"))
    
    # 2. Metrics comparison
    plot_metrics_comparison(results, os.path.join(output_dir, "new_metrics_comparison.png"))
    
    # 3. Convergence analysis
    plot_convergence_analysis(results, os.path.join(output_dir, "new_convergence_analysis.png"))
    
    print("\n" + "="*100)
    print("COMPARISON COMPLETE!")
    print("="*100)
    print(f"\nAll outputs saved to: {output_dir}")
    print("\nFiles created:")
    print("  1. pcgrad_comparison_table.csv - Detailed metrics table")
    print("  2. training_curves.png - Training dynamics over epochs")
    print("  3. metrics_comparison.png - Final metrics comparison")
    print("  4. convergence_analysis.png - Convergence behavior")
    print("="*100)

if __name__ == "__main__":
    main()