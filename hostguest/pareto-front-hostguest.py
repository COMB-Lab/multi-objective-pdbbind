"""
Pareto Front Plotter - Host-Guest Grid Search Results
Same structure as PDBBind plotter, adjusted for host-guest JSON key names.
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
    pareto_emp  = empirical[pareto_mask]
    pareto_phys = physics[pareto_mask]

    emp_norm  = (pareto_emp  - pareto_emp.min())  / (pareto_emp.max()  - pareto_emp.min())
    phys_norm = (pareto_phys - pareto_phys.min()) / (pareto_phys.max() - pareto_phys.min())

    distances      = np.sqrt(emp_norm**2 + phys_norm**2)
    pareto_indices = np.where(pareto_mask)[0]

    return pareto_indices[np.argmin(distances)]


def plot_pareto_hostguest(results_path, output_path='pareto_front_hostguest.png'):
    """
    Create clean Pareto front plot for host-guest grid search results.

    Host-guest JSON uses:
        best_val_empirical_loss  (empirical axis)
        best_val_physics_loss    (physics axis)
        physics_weight           (color)
        val_mae                  (optional)
    """

    print(f"Loading: {results_path}")
    with open(results_path, 'r') as f:
        results = json.load(f)

    # Extract arrays using host-guest key names
    empirical = np.array([r['best_val_empirical_loss'] for r in results])
    physics   = np.array([r['best_val_physics_loss']   for r in results])
    weights   = np.array([r['physics_weight']           for r in results])
    mae       = np.array([r.get('val_mae', 0)           for r in results])

    # Find Pareto optimal points
    costs       = np.column_stack([empirical, physics])
    pareto_mask = is_pareto_optimal(costs)
    pareto_indices = np.where(pareto_mask)[0]

    # Sort Pareto points by empirical loss for smooth line
    sort_idx              = np.argsort(empirical[pareto_mask])
    pareto_emp_sorted     = empirical[pareto_mask][sort_idx]
    pareto_phys_sorted    = physics[pareto_mask][sort_idx]
    pareto_weights_sorted = weights[pareto_mask][sort_idx]
    pareto_indices_sorted = pareto_indices[sort_idx]

    # Find best trade-off point
    best_idx = find_best_tradeoff(empirical, physics, pareto_mask)

    print(f" {len(results)} configurations, {len(pareto_indices)} Pareto optimal")
    print(f" Best trade-off: λ={weights[best_idx]:.3f}, "
          f"Emp={empirical[best_idx]:.4f}, Phys={physics[best_idx]:.4f}")

    # ── Figure ────────────────────────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(12, 10))

    # Layer 1: All configurations
    scatter_all = ax.scatter(
        empirical, physics,
        c=weights, cmap='viridis',
        s=100,
        alpha=0.35,
        edgecolors='none',
        label='All configurations',
        zorder=1
    )

    # Layer 2: Pareto front line
    ax.plot(
        pareto_emp_sorted, pareto_phys_sorted,
        color='red',
        linestyle='--',
        linewidth=2.5,
        alpha=0.85,
        label='Pareto Front',
        zorder=2
    )

    # Layer 3: Pareto optimal points
    ax.scatter(
        pareto_emp_sorted, pareto_phys_sorted,
        marker='^',
        s=180,
        facecolors='red',
        edgecolors='darkred',
        linewidths=2,
        alpha=0.9,
        label='Pareto Optimal',
        zorder=3
    )

    # Layer 4: Best trade-off point
    ax.scatter(
        [empirical[best_idx]], [physics[best_idx]],
        marker='*',
        s=300,
        facecolors='gold',
        edgecolors='darkorange',
        linewidths=3,
        label='Best Trade-off',
        zorder=4
    )

    # ── Labels ────────────────────────────────────────────────────────────────
    texts = []

    for i, idx in enumerate(pareto_indices_sorted):
        weight = weights[idx]

        should_label = (
            i == 0 or
            i == len(pareto_indices_sorted) - 1 or
            i % 8 == 0 or
            idx == best_idx or
            weight in [0.0, 0.5, 1.0]
        )

        if should_label:
            text = ax.text(
                empirical[idx], physics[idx],
                f"{weight:.3f}",
                fontsize=18,
                fontweight='bold',
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

    print("Adjusting labels...")
    adjust_text(
        texts, ax=ax,
        arrowprops=dict(arrowstyle='->', color='gray', lw=1.5),
        expand_text=(1.2, 1.2),
        expand_points=(1.2, 1.2),
        force_points=0.4,
        force_text=0.7
    )

    # ── Formatting ────────────────────────────────────────────────────────────
    ax.set_xlabel('Empirical Loss (Best Val RMSE)', fontsize=16, fontweight='bold')
    ax.set_ylabel('Physics Loss (Best Val Physics)', fontsize=16, fontweight='bold')
    ax.set_title(
        'Pareto Front: Empirical vs Physics Loss\n'
        'Host-Guest Multi-Objective Optimization Trade-off',
        fontsize=18, fontweight='bold', pad=20
    )

    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.grid(True, alpha=0.3, which='both', linestyle='--', linewidth=0.8)

    cbar = plt.colorbar(scatter_all, ax=ax, pad=0.02)
    cbar.set_label('Physics Weight (λ)', fontsize=14, fontweight='bold')
    cbar.ax.tick_params(labelsize=12)

    legend = ax.legend(
        fontsize=16,
        markerscale=1.5,
        loc='best',
        framealpha=0.95,
        edgecolor='black',
        fancybox=True,
        shadow=True
    )
    legend.get_frame().set_linewidth(1.5)

    ax.tick_params(axis='both', which='major', labelsize=13)
    ax.tick_params(axis='both', which='minor', labelsize=10)

    plt.tight_layout()

    # Save
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"Saved: {output_path}")

    highres = output_path.replace('.png', '_highres.png')
    plt.savefig(highres, dpi=300, bbox_inches='tight')
    print(f"High-res: {highres}")

    plt.show()
    return fig


# ============================================================================
# MAIN
# ============================================================================

if __name__ == "__main__":
    import os

    results_path = '/home/exouser/multi-objective-pdbbind/multi-objective-pdbbind/pytorch-implementation/grid_search_results/hostguest/hostguest_grid_search_batched_FINAL.json'
    output_path  = '/home/exouser/multi-objective-pdbbind/multi-objective-pdbbind/pytorch-implementation/plots/pareto_front_hostguest.png'

    if not os.path.exists(results_path):
        # Fall back to checkpoint if final doesn't exist yet
        checkpoint_dir = os.path.dirname(results_path)
        checkpoints = sorted([
            f for f in os.listdir(checkpoint_dir)
            if f.startswith('hostguest_grid_search') and f.endswith('.json')
        ])
        if checkpoints:
            results_path = os.path.join(checkpoint_dir, checkpoints[-1])
            print(f"Final not found — using latest checkpoint: {checkpoints[-1]}")
        else:
            print(f"Error: No results found in {checkpoint_dir}")
            exit(1)

    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    print("=" * 70)
    print("PARETO FRONT PLOTTER — HOST-GUEST")
    print("=" * 70)

    plot_pareto_hostguest(results_path, output_path)

    print("=" * 70)
    print("DONE")
    print("=" * 70)