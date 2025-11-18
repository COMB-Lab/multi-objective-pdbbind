'''
    The purpose of this file is to analyze and visualize the results from pcgrad-multi-objective-pdbbind.py
    It loads the saved pickle files containing training results and generates plots for loss curves,
    performance metrics, and other relevant visualizations to assess model performance.
'''

import pickle
import matplotlib.pyplot as plt
import os
import numpy as np

DIR = '/home/exouser/multi-objective-pdbbind/multi-objective-pdbbind/Results'

def load_results(pickle_filename):
    """Load results from a pickle file."""
    save_path = os.path.join(DIR, pickle_filename)
    with open(save_path, 'rb') as f:
        results = pickle.load(f)
    return results

# =========================
# Plot and analysis from pcgrad-multi-objective-pdbbind.py
# =========================
def plot_training_results(loss_tracker, config):
    """Plot training loss over epochs with configuration info."""
    if not loss_tracker.total_losses:
        print("No training loss data to plot.")
        return
    
    try:
        plt.figure(figsize=(15, 10))
        
        epoch_length = range(1, len(loss_tracker.total_losses) + 1)
        
        # Total loss
        plt.subplot(2, 3, 1)
        plt.plot(epoch_length, loss_tracker.total_losses, 'b-', label='Total Loss', linewidth=2)
        plt.title(f'Total Loss Over Epochs\n({config.dataset_size} structures)', fontsize=14)
        plt.xlabel('Epochs', fontsize=12)
        plt.ylabel('Loss', fontsize=12)
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # Learning rate plot
        if loss_tracker.learning_rates:
            plt.subplot(2, 3, 2)
            plt.plot(epoch_length, loss_tracker.learning_rates, 'r-', label='Learning Rate', linewidth=2)
            plt.title('Learning Rate Over Epochs', fontsize=14)
            plt.xlabel('Epochs', fontsize=12)
            plt.ylabel('Learning Rate', fontsize=12)
            plt.legend()
            plt.grid(True, alpha=0.3)
        
        # Loss trend (last 10 epochs)
        if len(loss_tracker.total_losses) > 10:
            plt.subplot(2, 3, 3)
            recent_losses = loss_tracker.total_losses[-10:]
            recent_epochs = list(range(len(loss_tracker.total_losses)-9, len(loss_tracker.total_losses)+1))
            plt.plot(recent_epochs, recent_losses, 'g-', label='Recent Loss', linewidth=2, marker='o')
            plt.title('Loss Trend (Last 10 Epochs)', fontsize=14)
            plt.xlabel('Epochs', fontsize=12)
            plt.ylabel('Loss', fontsize=12)
            plt.legend()
            plt.grid(True, alpha=0.3)
        
        # Configuration info
        plt.subplot(2, 3, 4)
        plt.text(0.1, 0.9, f"Dataset Size: {config.dataset_size}", transform=plt.gca().transAxes, fontsize=12)
        plt.text(0.1, 0.8, f"Batch Size: {config.batch_size}", transform=plt.gca().transAxes, fontsize=12)
        plt.text(0.1, 0.7, f"Max Padding: {config.max_padding}", transform=plt.gca().transAxes, fontsize=12)
        plt.text(0.1, 0.6, f"Epochs: {config.epochs}", transform=plt.gca().transAxes, fontsize=12)
        plt.text(0.1, 0.5, f"Memory Limit: {config.memory_limit_gb} GB", transform=plt.gca().transAxes, fontsize=12)
        plt.text(0.1, 0.4, f"Final Loss: {loss_tracker.total_losses[-1]:.6f}", transform=plt.gca().transAxes, fontsize=12)
        plt.title('Training Configuration', fontsize=14)
        plt.axis('off')
        
        # Save the plot
        plt.tight_layout()
        filename = f'training_results_{config.dataset_size}structures_new.png'
        plt.savefig(filename, dpi=300, bbox_inches='tight')
        print(f"Plot saved as '{filename}'")
        
        # Try to show the plot
        plt.show()
        
        # Print summary statistics
        print(f"\nTraining Summary ({config.dataset_size} structures):")
        print(f"  Initial Loss: {loss_tracker.total_losses[0]:.6f}")
        print(f"  Final Loss: {loss_tracker.total_losses[-1]:.6f}")
        print(f"  Loss Reduction: {((loss_tracker.total_losses[0] - loss_tracker.total_losses[-1]) / loss_tracker.total_losses[0] * 100):.2f}%")
        print(f"  Total Epochs: {len(loss_tracker.total_losses)}")
            
    except Exception as e:
        print(f"Error creating plot: {e}")
        print("Printing loss values instead:")
        for i, loss in enumerate(loss_tracker.total_losses):
            print(f"Epoch {i+1}: {loss:.6f}")


# ============================================================================
# RESULT ANALYSIS UTILITIES
# ============================================================================

def load_and_explore_results(pickle_filename):
    """
    Load saved results and print exploration summary (schema-agnostic).
    Returns the loaded dict (even if summary keys differ).
    """
    try:
        with open(pickle_filename, 'rb') as f:
            results_data = pickle.load(f)

        print("=" * 70)
        print(f"LOADED RESULTS FROM: {pickle_filename}")
        print("=" * 70)

        exp = results_data.get('experiment_info', {})
        print("Experiment Info:")
        print(f"  Timestamp: {exp.get('timestamp')}")
        print(f"  Dataset size: {exp.get('dataset_size')} structures")
        print(f"  Runtime: {exp.get('total_runtime_minutes')} minutes")

        # Print whatever summary keys exist, without assuming names
        summary = results_data.get('summary_metrics', {})
        if summary:
            print("\nSummary metrics (available keys):")
            for k, v in summary.items():
                print(f"  {k}: {v}")
        else:
            print("\nSummary metrics: <none present>")

        # Print availability of plots
        all_results = results_data.get('all_results', [])
        if not all_results:
            print("\nNo per-run results found in 'all_results'.")
        else:
            print("\nAvailable per-run entries:")
            for i, r in enumerate(all_results, 1):
                keys = ", ".join(sorted(r.keys()))
                print(f"  Result {i}: keys = [{keys}]")

        return results_data

    except Exception as e:
        print(f"Error loading pickle file: {e}")
        return None
    

def pick_best_run(all_results):
    """
    Pick a best run using a metric priority:
    1) test_rmse  (lower is better)
    2) test_loss
    3) mean_abs_diff or mean_abs_value
    4) test_mae
    Returns (best_run, metric_used).
    """
    if not all_results:
        return None, None

    # candidates: (metric_key, lower_is_better)
    priorities = [
        ("test_rmse", True),
        ("test_loss", True),
        ("mean_abs_diff", True),
        ("mean_abs_value", True),
        ("test_mae", True),
    ]
    for key, lower in priorities:
        present = [r for r in all_results if key in r and isinstance(r[key], (int, float))]
        if present:
            best = min(present, key=lambda r: r[key]) if lower else max(present, key=lambda r: r[key])
            return best, key
    # fallback: just return the first
    return all_results[0], "<no standard metric>"



def plots_from_pickle(pickle_path):
    data = load_and_explore_results(pickle_path)
    if not data:
        print("No data loaded.")
        return

    runs = data.get('all_results', [])
    if not runs:
        print("No results to plot (all_results is empty).")
        return

    best, metric = pick_best_run(runs)
    if not best:
        print("Could not determine a best run.")
        return

    print(f"\nUsing best run selected by '{metric}':")
    print(f"  physics_weight = {best.get('physics_weight')}")
    for k in ("test_rmse", "test_loss", "mean_abs_diff", "mean_abs_value", "test_mae", "final_train_loss"):
        if k in best:
            print(f"  {k} = {best[k]}")

    # 1) Parity plot (y_true vs y_pred)
    y_true = np.array(best.get('y_true_test', []), dtype=float).ravel()
    y_pred = np.array(best.get('y_pred_test', []), dtype=float).ravel()

    if y_true.size == 0 or y_pred.size == 0:
        print("Best run has no y_true_test / y_pred_test; skipping parity plot.")
    else:
        lo = float(min(y_true.min(), y_pred.min()))
        hi = float(max(y_true.max(), y_pred.max()))
        plt.figure(figsize=(6,6))
        plt.scatter(y_true, y_pred, s=16, alpha=0.7)
        plt.plot([lo, hi], [lo, hi], linewidth=2)
        plt.xlabel("True ΔG")
        plt.ylabel("Predicted ΔG")
        plt.title(f"Parity (best by {metric}, w={best.get('physics_weight')})")
        plt.grid(True, alpha=0.3)
        out1 = "parity_best_run.png"
        plt.tight_layout()
        plt.savefig(out1, dpi=200)
        print(f"Saved {out1}")

    # 2) Loss curve if available
    losses = best.get("training_history", {}).get("total_losses", [])
    if losses:
        plt.figure(figsize=(7,4))
        plt.plot(range(1, len(losses)+1), losses, linewidth=2)
        plt.xlabel("Epoch")
        plt.ylabel("Total loss")
        plt.title(f"Training loss over epochs (best by {metric})")
        plt.grid(True, alpha=0.3)
        out2 = "loss_curve_best_run.png"
        plt.tight_layout()
        plt.savefig(out2, dpi=200)
        print(f"Saved {out2}")
    else:
        print("Best run has no training_history.total_losses; skipping loss curve.")

# ============================================================================
# Pareto front graph
# ============================================================================
import numpy as np
import matplotlib.pyplot as plt

# ---- helpers ----
def _extract_emp_phy(result):
    """
    Try to get (empirical_loss, physics_loss) for a single run dict.
    Priority:
      1) result['test_emp_loss'], result['test_phy_loss']
      2) derive from totals if keys present (not always available)
    Returns (emp, phy) or (None, None) if unavailable.
    """
    # 1) direct keys (your grid_search setup)
    emp = result.get('test_emp_loss', None)
    phy = result.get('test_phy_loss', None)
    if emp is not None and phy is not None:
        return float(emp), float(phy)

    # 2) fallback: try to derive from training_history if present (avg of last epochs)
    hist = result.get('training_history', {})
    emp_hist = hist.get('empirical_losses', None)
    phy_hist = hist.get('physics_losses', None)
    if emp_hist and phy_hist:
        # Use last epoch values (or means)
        return float(emp_hist[-1]), float(phy_hist[-1])

    return None, None


def _non_dominated_indices(points):
    """
    Compute indices of Pareto-optimal points for a minimization problem
    in 2D. points is an (N,2) array where lower is better on both axes.
    """
    N = points.shape[0]
    is_nd = np.ones(N, dtype=bool)
    for i in range(N):
        if not is_nd[i]:
            continue
        # any j that dominates i?
        # dominates if: j <= i elementwise AND strictly better in at least one axis
        dominates = np.all(points <= points[i], axis=1) & np.any(points < points[i], axis=1)
        # if any point j dominates i, i is not Pareto
        if np.any(dominates):
            is_nd[i] = False
    return np.where(is_nd)[0]


def create_pareto_plot(successful_results, best_result=None, save_path="pareto_front.png"):
    """
    Creates a Pareto front plot using empirical and physics loss from successful runs.

    Parameters:
        successful_results (list): list of run dicts (e.g., your 'all_results').
        best_result (dict|None): optional run to highlight distinctly.
        save_path (str): output image path.
    """
    # Collect metrics
    emp_losses, phy_losses, weights, keep_idx = [], [], [], []
    for i, r in enumerate(successful_results):
        emp, phy = _extract_emp_phy(r)
        if emp is None or phy is None:
            continue
        emp_losses.append(emp)
        phy_losses.append(phy)
        # Physics weight (optional, used for annotation if present)
        w = None
        # grid_search style
        if 'hyperparameters' in r and isinstance(r['hyperparameters'], dict):
            w = r['hyperparameters'].get('physics_weight', None)
        # other scripts
        if w is None:
            w = r.get('physics_weight', None)
        weights.append(w)
        keep_idx.append(i)

    if not emp_losses:
        print("No runs had both empirical and physics loss available to plot.")
        return

    emp_losses = np.array(emp_losses, dtype=float)
    phy_losses = np.array(phy_losses, dtype=float)
    pts = np.column_stack([emp_losses, phy_losses])

    # Pareto front (min-min)
    nd_idx = _non_dominated_indices(pts)
    frontier = pts[nd_idx]

    # Sort frontier for nice line drawing (by empirical loss)
    order = np.argsort(frontier[:, 0])
    frontier_sorted = frontier[order]

    # Plot
    plt.figure(figsize=(8, 8))
    plt.scatter(emp_losses, phy_losses, s=60, alpha=0.7, label="All runs")
    # Pareto line + points
    plt.plot(frontier_sorted[:, 0], frontier_sorted[:, 1], linewidth=2, label="Pareto front")
    plt.scatter(frontier[:, 0], frontier[:, 1], s=80, edgecolor='k', facecolor='none', label="Non-dominated")

    # Annotate physics weights (optional)
    for (x, y), w in zip(pts, weights):
        if w is not None:
            plt.annotate(f"{w:.4g}", (x, y), fontsize=8, xytext=(4, 4), textcoords='offset points')

    # Highlight best_result if provided
    if best_result is not None:
        be, bp = _extract_emp_phy(best_result)
        if be is not None and bp is not None:
            plt.scatter([be], [bp], s=120, marker='*', label="Selected best", zorder=5)

    plt.xlabel("Empirical loss")
    plt.ylabel("Physics loss")
    plt.title("Pareto Front: Empirical vs Physics Loss (lower is better)")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(save_path, dpi=180)
    print(f"Saved {save_path}")
    plt.show()

def run_pareto_from_pickle(pickle_path, save_path="pareto_front.png"):
    data = load_and_explore_results(pickle_path)
    if not data:
        print("No data loaded.")
        return

    runs = data.get('all_results', [])
    if not runs:
        print("No results to plot (all_results is empty).")
        return

    # choose a best run using the same logic as plots_from_pickle
    best, metric = pick_best_run(runs)
    if not best:
        print("Could not determine a best run; plotting Pareto without a highlight.")
        create_pareto_plot(runs, best_result=None, save_path=save_path)
        return

    print(f"\nPareto plot will highlight best by '{metric}' (w={best.get('physics_weight')}).")
    create_pareto_plot(runs, best_result=best, save_path=save_path)


def main():
    # Specify the pickle file to load
    pickle_filename = 'PGGCN_results_100structures_20251105_205112.pkl'
    save_path = os.path.join(DIR, pickle_filename)

    # existing plots (parity, loss curve)
    plots_from_pickle(save_path)

    # NEW: Pareto front
    run_pareto_from_pickle(save_path, save_path=os.path.join(os.path.dirname(save_path), "pareto_front.png"))


if __name__ == "__main__":
    main()