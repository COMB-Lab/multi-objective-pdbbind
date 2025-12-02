#!/usr/bin/env python3
"""
Load and Analyze PyTorch PGGCN Results
Provides utilities for inspecting training results and generating plots.
"""

import pickle
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import sys


def load_results(filepath):
    """Load results from pickle file."""
    with open(filepath, 'rb') as f:
        results = pickle.load(f)
    return results


def print_summary(results):
    """Print summary statistics."""
    print('\n' + '='*70)
    print('RESULTS SUMMARY')
    print('='*70)
    
    if 'config' in results:
        config = results['config']
        print(f"\nConfiguration:")
        print(f"  - Dataset size: {config.get('dataset_size', 'N/A')}")
        print(f"  - Batch size: {config.get('batch_size', 'N/A')}")
        print(f"  - Epochs: {config.get('epochs', 'N/A')}")
        print(f"  - Learning rate: {config.get('learning_rate', 'N/A')}")
    
    if 'results' in results:
        print(f"\nPhysics Weight Sweep Results:")
        print(f"{'Weight':<15} {'Train RMSE':<15} {'Test RMSE':<15} {'Test MAE':<15}")
        print('-' * 60)
        
        for weight in sorted(results['results'].keys()):
            r = results['results'][weight]
            train_rmse = r.get('train_rmse', r['losses'][-1] if r['losses'] else 0)
            test_rmse = r.get('test_rmse', 'N/A')
            test_mae = r.get('test_mae', 'N/A')
            
            if isinstance(test_rmse, (int, float)):
                print(f"{weight:<15.2e} {train_rmse:<15.4f} {test_rmse:<15.4f} {test_mae:<15.4f}")
            else:
                print(f"{weight:<15.2e} {train_rmse:<15.4f} {test_rmse:<15} {test_mae:<15}")
    
    if 'best_weight' in results:
        print(f"\n✓ Best physics_weight: {results['best_weight']:.2e}")
    elif 'results' in results:
        best_weight = min(results['results'].keys(), 
                         key=lambda w: results['results'][w].get('test_rmse', float('inf')))
        print(f"\n✓ Best physics_weight: {best_weight:.2e}")
    
    print('='*70 + '\n')


def plot_convergence(results, output_file='convergence.png'):
    """Plot training convergence curves."""
    fig, axes = plt.subplots(1, len(results['results']), figsize=(15, 4))
    
    if len(results['results']) == 1:
        axes = [axes]
    
    for idx, (weight, data) in enumerate(sorted(results['results'].items())):
        ax = axes[idx] if len(results['results']) > 1 else axes[0]
        
        losses = data.get('losses', [])
        if losses:
            ax.plot(losses, linewidth=2, color='steelblue')
            ax.set_title(f'Physics Weight = {weight:.2e}')
            ax.set_xlabel('Epoch')
            ax.set_ylabel('Loss')
            ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_file, dpi=150, bbox_inches='tight')
    print(f"✓ Convergence plot saved to {output_file}")
    plt.close()


def plot_predictions(results, output_file='predictions.png'):
    """Plot predicted vs actual binding affinities."""
    y_test = results.get('y_test', None)
    if y_test is None:
        print("⚠️  No y_test data available for plotting")
        return
    
    fig, axes = plt.subplots(1, len(results['predictions']), figsize=(15, 4))
    
    if len(results['predictions']) == 1:
        axes = [axes]
    
    for idx, (weight, predictions) in enumerate(sorted(results['predictions'].items())):
        ax = axes[idx] if len(results['predictions']) > 1 else axes[0]
        
        # Scatter plot
        ax.scatter(y_test, predictions, alpha=0.6, s=100)
        
        # Perfect prediction line
        min_val = min(y_test.min(), predictions.min())
        max_val = max(y_test.max(), predictions.max())
        ax.plot([min_val, max_val], [min_val, max_val], 'r--', lw=2, label='Perfect prediction')
        
        ax.set_title(f'Physics Weight = {weight:.2e}')
        ax.set_xlabel('Actual Binding Affinity (kcal/mol)')
        ax.set_ylabel('Predicted Binding Affinity (kcal/mol)')
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_file, dpi=150, bbox_inches='tight')
    print(f"✓ Predictions plot saved to {output_file}")
    plt.close()


def compute_statistics(results):
    """Compute detailed statistics."""
    print('\nDetailed Statistics:')
    print('-' * 70)
    
    for weight, data in sorted(results['results'].items()):
        print(f"\nPhysics Weight = {weight:.2e}")
        
        losses = data.get('losses', [])
        if losses:
            print(f"  Loss statistics:")
            print(f"    - Initial: {losses[0]:.4f}")
            print(f"    - Final: {losses[-1]:.4f}")
            print(f"    - Min: {min(losses):.4f}")
            print(f"    - Max: {max(losses):.4f}")
            print(f"    - Mean: {np.mean(losses):.4f}")
            
            # Convergence rate
            if len(losses) > 1:
                convergence_rate = (losses[0] - losses[-1]) / losses[0] * 100
                print(f"    - Convergence: {convergence_rate:.1f}% reduction")
        
        if 'test_rmse' in data:
            print(f"  Test RMSE: {data['test_rmse']:.4f}")
        if 'test_mae' in data:
            print(f"  Test MAE: {data['test_mae']:.4f}")


def main():
    """Main function to load and display results."""
    if len(sys.argv) < 2:
        # Find most recent results file
        results_files = list(Path('.').glob('PGGCN_results_pytorch_*.pkl'))
        if not results_files:
            print("No results files found. Run a training script first:")
            print("  python3 quick_test_pytorch.py")
            print("  python3 full_training_pytorch.py")
            print("  python3 multi_objective_pdbbind_pytorch.py")
            sys.exit(1)
        
        results_file = max(results_files, key=lambda p: p.stat().st_mtime)
        print(f"Loading most recent results: {results_file.name}")
    else:
        results_file = sys.argv[1]
    
    # Load results
    results = load_results(results_file)
    
    # Print summary
    print_summary(results)
    
    # Compute statistics
    compute_statistics(results)
    
    # Generate plots
    try:
        plot_convergence(results, 'convergence.png')
    except Exception as e:
        print(f"⚠️  Could not generate convergence plot: {e}")
    
    try:
        plot_predictions(results, 'predictions.png')
    except Exception as e:
        print(f"⚠️  Could not generate predictions plot: {e}")
    
    print("\n✅ Analysis complete!")


if __name__ == '__main__':
    main()
