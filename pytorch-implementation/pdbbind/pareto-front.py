"""
Simple Pareto Front Plotter - Fixed for PDBBind Data
Easy to customize version with clear visual hierarchy
"""

import json
import numpy as np
import matplotlib.pyplot as plt
from adjustText import adjust_text


def is_pareto_optimal(costs):
    """Find Pareto optimal points (non-dominated solutions)"""
    is_pareto = np.ones(costs.shape[0], dtype=bool)
    for i, c in enumerate(costs):
        if is_pareto[i]:
            is_pareto[is_pareto] = np.any(costs[is_pareto] < c, axis=1)
            is_pareto[i] = True
    return is_pareto


def find_best_tradeoff(empirical, physics, pareto_mask):
    """Find the best trade-off point (knee of curve)"""
    # Get Pareto points
    pareto_emp = empirical[pareto_mask]
    pareto_phys = physics[pareto_mask]
    
    # Normalize to [0,1]
    emp_norm = (pareto_emp - pareto_emp.min()) / (pareto_emp.max() - pareto_emp.min())
    phys_norm = (pareto_phys - pareto_phys.min()) / (pareto_phys.max() - pareto_phys.min())
    
    # Find closest to ideal (0,0)
    distances = np.sqrt(emp_norm**2 + phys_norm**2)
    pareto_indices = np.where(pareto_mask)[0]
    
    return pareto_indices[np.argmin(distances)]

def smart_fmt(x: float, decimals: int = 3, small: float = 1e-3, large: float = 1e6) -> str:
    if x == 0:
        return f"{0:.{decimals}f}"
    ax = abs(x)
    if ax < small or ax >= large:
        return f"{x:.{decimals}e}"   # scientific with 3 decimals
    return f"{x:.{decimals}f}"

def plot_pareto_simple(results_path, output_path='pareto_front_recent.png'):
    """
    Create clean Pareto front plot.
    
    CUSTOMIZATION GUIDE:
    --------------------
    Line 82-87:   All points (size, transparency, colors)
    Line 89-93:   Pareto front line (style, width, color)
    Line 95-99:   Pareto points (marker, size, colors)
    Line 101-106: Single optimal point (marker, size, colors)
    Line 108-135: Label selection and styling
    Line 143-160: Axes, title, grid styling
    """
    
    # Load data
    print(f"Loading: {results_path}")
    with open(results_path, 'r') as f:
        results = json.load(f)
    
    # Extract arrays - FIXED to match your data structure
    empirical = np.array([r['test_empirical'] for r in results])
    physics = np.array([r['test_physics'] for r in results])
    weights = np.array([r['physics_weight'] for r in results])
    mae = np.array([r.get('test_mae', 0) for r in results])
    
    # Find Pareto optimal points
    costs = np.column_stack([empirical, physics])
    pareto_mask = is_pareto_optimal(costs)
    pareto_indices = np.where(pareto_mask)[0]
    
    # Sort Pareto points by empirical loss for smooth line
    sort_idx = np.argsort(empirical[pareto_mask])
    pareto_emp_sorted = empirical[pareto_mask][sort_idx]
    pareto_phys_sorted = physics[pareto_mask][sort_idx]
    pareto_weights_sorted = weights[pareto_mask][sort_idx]
    pareto_indices_sorted = pareto_indices[sort_idx]
    
    # Find best trade-off point
    best_idx = find_best_tradeoff(empirical, physics, pareto_mask)
    print(f"{len(results)} configurations, {len(pareto_indices)} Pareto optimal")
    print(f"Best trade-off: λ={weights[best_idx]:.3f}, "
          f"Emp={empirical[best_idx]:.4f}, Phys={physics[best_idx]:.4f}, MAE={mae[best_idx]:.4f}")
    
    # CREATE FIGURE
    fig, ax = plt.subplots(figsize=(12, 10))
    
    # ============================================================================
    # LAYER 1: All configurations (background, semi-transparent)
    # ============================================================================
    scatter_all = ax.scatter(
        empirical, physics,
        c=weights, cmap='viridis',
        s=100,              # Size of points
        alpha=0.35,         # Transparency (0=invisible, 1=opaque)
        edgecolors='none',
        label='All configurations',
        zorder=1
    )
    
    # ============================================================================
    # LAYER 2: Pareto front line
    # ============================================================================
    ax.plot(
        pareto_emp_sorted, pareto_phys_sorted,
        color='red',         # Line color
        linestyle='--',      # Line style: '-', '--', '-.', ':'
        linewidth=2.5,       # Line thickness
        alpha=0.85,          # Transparency
        label='Pareto Front',
        zorder=2
    )
    
    # ============================================================================
    # LAYER 3: Pareto optimal points
    # ============================================================================
    ax.scatter(
        pareto_emp_sorted, pareto_phys_sorted,
        marker='^',          # Marker: 'o', '^', 's', 'D', '*', 'v', '<', '>'
        s=180,               # Size
        facecolors='red',    # Fill color
        edgecolors='darkred', # Border color
        linewidths=2,        # Border width
        alpha=0.9,
        label='Pareto Optimal',
        zorder=3
    )
    
    # ============================================================================
    # LAYER 4: Single best point (THE STAR OF THE SHOW)
    # ============================================================================
    ax.scatter(
        [empirical[best_idx]], [physics[best_idx]],
        marker='*',          # Marker type (star is nice for "best")
        s=1000,              # Large size to stand out
        facecolors='gold',   # Bright color
        edgecolors='darkorange', # Border color
        linewidths=3,        # Thick border
        label=f'Best Trade-off (λ={weights[best_idx]:.3e})',
        zorder=4
    )
    
    # ============================================================================
    # LABELS: Weight values on Pareto points
    # ============================================================================
    texts = []
    
    # Labeling strategy: 
    # - First and last points
    # - Every 8th point
    # - The best point
    # - Key references (0.0, 0.5, 1.0)
    
    for i, idx in enumerate(pareto_indices_sorted):
        weight = weights[idx]
        
        # Decide if this point should be labeled
        should_label = (
            i == 0 or                              # First
            i == len(pareto_indices_sorted) - 1 or # Last
            i % 8 == 0 or                          # Every 8th
            idx == best_idx or                     # Best point
            weight in [0.0, 0.5, 1.0]              # Key references
        )
        
        if should_label:
            text = ax.text(
                empirical[idx], physics[idx],
                f"λ=" + f"{smart_fmt(weight)}",
                fontsize=11,           # Text size
                fontweight='bold',     # Text weight
                bbox=dict(
                    boxstyle='round,pad=0.5',
                    facecolor='yellow' if idx == best_idx else 'white',
                    edgecolor='darkorange' if idx == best_idx else 'black',
                    linewidth=2.5 if idx == best_idx else 1.2,
                    alpha=0.95
                ),
                zorder=5
            )
            texts.append(text)
    
    # Auto-adjust label positions to avoid overlap
    print("Adjusting labels...")
    adjust_text(
        texts, ax=ax,
        arrowprops=dict(arrowstyle='->', color='gray', lw=1.5),
        expand_text=(1.2, 1.2),
        expand_points=(1.2, 1.2),
        force_points=0.4,
        force_text=0.7
    )
    
    # ============================================================================
    # FORMATTING
    # ============================================================================
    
    # Axes labels
    ax.set_xlabel('Empirical Loss (RMSE)', fontsize=16, fontweight='bold')
    ax.set_ylabel('Physics Loss', fontsize=16, fontweight='bold')
    
    # Title
    ax.set_title(
        'Pareto Front: Empirical vs Physics Loss\nMulti-Objective Optimization Trade-off on PDBBind Dataset',
        fontsize=18, fontweight='bold', pad=20
    )
    
    # Log scale (better for visualizing wide ranges)
    ax.set_xscale('log')
    ax.set_yscale('log')
    
    # Grid
    ax.grid(True, alpha=0.3, which='both', linestyle='--', linewidth=0.8)
    
    # Colorbar for physics weight
    cbar = plt.colorbar(scatter_all, ax=ax, pad=0.02)
    cbar.set_label('Physics Weight (λ)', fontsize=14, fontweight='bold')
    cbar.ax.tick_params(labelsize=12)
    
    # Legend
    legend = ax.legend(
        fontsize=13,
        loc='best',
        framealpha=0.95,
        edgecolor='black',
        fancybox=True,
        shadow=True
    )
    legend.get_frame().set_linewidth(1.5)
    
    # Tick labels
    ax.tick_params(axis='both', which='major', labelsize=13)
    ax.tick_params(axis='both', which='minor', labelsize=10)
    
    plt.tight_layout()
    
    # Save
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"✓ Saved: {output_path}")
    
    # High-res version
    highres = output_path.replace('.png', '_highres.png')
    plt.savefig(highres, dpi=300, bbox_inches='tight')
    print(f"✓ High-res: {highres}")
    
    plt.show()
    
    return fig


# ============================================================================
# QUICK USAGE
# ============================================================================

if __name__ == "__main__":
    import os
    import sys
    
    # Check if paths provided as arguments
    if len(sys.argv) > 1:
        results_path = sys.argv[1]
        output_path = sys.argv[2] if len(sys.argv) > 2 else 'pareto_plot.png'
    else:
        # Default to uploaded file
        results_path = '/home/exouser/multi-objective-pdbbind/multi-objective-pdbbind/pytorch-implementation/pdbbind/grid_search/pdbbind_grid_search_results_20260202-174303.json'
        output_path = '/home/exouser/multi-objective-pdbbind/multi-objective-pdbbind/pytorch-implementation/pdbbind/comparison_plots/pareto_plot-test.png'
    
    # Check if results exist
    if not os.path.exists(results_path):
        print(f"Error: Results not found at {results_path}")
        print("\nUsage:")
        print("  python pareto_plot_fixed.py [results.json] [output.png]")
        exit(1)
    
    # Create output directory
    os.makedirs(os.path.dirname(output_path) if os.path.dirname(output_path) else '.', exist_ok=True)
    
    print("="*70)
    print("PARETO FRONT PLOTTER - PDBBind Edition")
    print("="*70)
    
    # Create plot
    plot_pareto_simple(results_path, output_path)
    
    print("="*70)
    print("DONE")
    print("="*70)
    
    print("\nCustomization tips:")
    print("  - Edit lines 82-106 to change markers and colors")
    print("  - Edit line 125 to change label frequency (e.g., i % 5)")
    print("  - Edit lines 143-160 to change axes and title")
    print("  - Remove ax.set_xscale('log') for linear scale")