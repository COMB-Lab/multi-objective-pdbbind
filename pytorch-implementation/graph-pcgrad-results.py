"""
Simple Pareto Front Plotter
For results with physics_weight, empirical_loss, and physics_loss
"""

import json
import matplotlib.pyplot as plt
import numpy as np
import argparse

def load_results(json_path):
    """Load results from JSON file."""
    with open(json_path, 'r') as f:
        results = json.load(f)
    return results

def find_pareto_optimal(empirical, physics):
    """Find Pareto optimal points."""
    points = np.array(list(zip(empirical, physics)))
    pareto_mask = np.ones(len(points), dtype=bool)
    
    for i, point in enumerate(points):
        # A point is Pareto optimal if no other point dominates it
        # (i.e., is better in both objectives)
        pareto_mask[i] = not np.any(np.all(points < point, axis=1))
    
    return pareto_mask

def plot_pareto_front(results, save_path='pareto_front.png'):
    """Plot Pareto front from results."""
    
    # Extract data
    weights = [r['physics_weight'] for r in results]
    empirical = [r['best_val_empirical_loss'] for r in results]
    physics = [r['best_val_physics_loss'] for r in results]
    
    # Find Pareto optimal points
    pareto_mask = find_pareto_optimal(empirical, physics)
    pareto_points = np.array(list(zip(empirical, physics)))[pareto_mask]
    pareto_weights = [weights[i] for i in range(len(weights)) if pareto_mask[i]]
    
    # Sort Pareto points by empirical loss for line plot
    sorted_indices = np.argsort(pareto_points[:, 0])
    pareto_sorted = pareto_points[sorted_indices]
    pareto_weights_sorted = [pareto_weights[i] for i in sorted_indices]
    
    # Create figure
    fig, ax = plt.subplots(figsize=(10, 8))
    
    # Plot all points
    scatter = ax.scatter(empirical, physics, 
                        c=weights, s=100, cmap='viridis',
                        edgecolors='black', linewidth=1.5, alpha=0.7,
                        label='All configurations')
    
    # Plot Pareto front line
    ax.plot(pareto_sorted[:, 0], pareto_sorted[:, 1], 
            'r--', linewidth=2.5, label='Pareto Front', alpha=0.8, zorder=10)
    
    # Highlight Pareto optimal points
    ax.scatter(pareto_sorted[:, 0], pareto_sorted[:, 1],
              c='red', s=200, marker='*', edgecolors='darkred',
              linewidth=2, label='Pareto Optimal', zorder=11)
    
    # Annotate Pareto points with their weights
    for i, (point, weight) in enumerate(zip(pareto_sorted, pareto_weights_sorted)):
        ax.annotate(f'{weight:.3f}', 
                   point,
                   xytext=(8, 8), textcoords='offset points',
                   fontsize=9, fontweight='bold',
                   bbox=dict(boxstyle='round,pad=0.4', 
                            facecolor='yellow', alpha=0.8, edgecolor='black'),
                   arrowprops=dict(arrowstyle='->', connectionstyle='arc3,rad=0',
                                  color='black', lw=1.5))
    
    # Labels and title
    ax.set_xlabel('Empirical Loss (RMSE)', fontsize=14, fontweight='bold')
    ax.set_ylabel('Physics Loss (RMSE)', fontsize=14, fontweight='bold')
    ax.set_title('Pareto Front: Empirical vs Physics Loss\nMulti-Objective Optimization Trade-off', 
                fontsize=16, fontweight='bold', pad=20)
    
    # Legend
    ax.legend(fontsize=11, loc='best', framealpha=0.9)
    
    # Grid
    ax.grid(True, alpha=0.3, linestyle='--', linewidth=0.5)
    
    # Colorbar
    cbar = plt.colorbar(scatter, ax=ax)
    cbar.set_label('Physics Weight', fontsize=12, fontweight='bold')
    
    # Tight layout
    plt.tight_layout()
    
    # Save
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"\n✓ Pareto front saved to: {save_path}")
    
    return fig, pareto_weights_sorted, pareto_sorted


def print_pareto_summary(results, pareto_weights, pareto_points):
    """Print summary of Pareto optimal configurations."""
    print("\n" + "="*80)
    print("PARETO OPTIMAL CONFIGURATIONS")
    print("="*80)
    print(f"\nFound {len(pareto_weights)} Pareto optimal points:")
    print(f"\n{'Weight':<12} {'Empirical Loss':<16} {'Physics Loss':<16}")
    print("-"*44)
    
    for weight, point in zip(pareto_weights, pareto_points):
        print(f"{weight:<12.6f} {point[0]:<16.4f} {point[1]:<16.4f}")
    
    print("\n" + "="*80)
    print("INTERPRETATION")
    print("="*80)
    print("\nPareto optimal points represent the best trade-offs:")
    print("  • Moving along the Pareto front, you can only improve one")
    print("    objective by sacrificing the other")
    print("  • Points NOT on the front are strictly dominated")
    print("    (worse in at least one objective, no better in the other)")
    print("\nThe 'best' point depends on your priorities:")
    print("  • Lowest empirical loss: Best prediction accuracy")
    print("  • Lowest physics loss: Best physics consistency")
    print("  • Middle of curve: Balanced trade-off")
    print("="*80)


def main():
    parser = argparse.ArgumentParser(description='Plot Pareto front from grid search results')
    parser.add_argument('--input', type=str, required=True,
                       help='Input JSON file with results')
    parser.add_argument('--output', type=str, default='pareto_front.png',
                       help='Output image file')
    
    args = parser.parse_args()
    
    print("="*80)
    print("PARETO FRONT PLOTTER")
    print("="*80)
    
    # Load results
    print(f"\nLoading results from: {args.input}")
    results = load_results(args.input)
    print(f"Loaded {len(results)} configurations")
    
    # Plot Pareto front
    print("\nGenerating Pareto front plot...")
    fig, pareto_weights, pareto_points = plot_pareto_front(results, args.output)
    
    # Print summary
    print_pareto_summary(results, pareto_weights, pareto_points)
    
    print("\n✓ Done!")


if __name__ == "__main__":
    main()