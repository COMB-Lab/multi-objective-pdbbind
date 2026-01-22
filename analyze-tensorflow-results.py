"""
TensorFlow Results Analysis Script
Extracts and displays all information from PGGCN results pickle file
"""

import pickle
import numpy as np
import json
from datetime import datetime
import os

def format_array_stats(arr, name="Array"):
    """Format statistics for numpy arrays"""
    if len(arr) == 0:
        return f"{name}: Empty array"
    
    arr = np.array(arr).flatten()
    stats = {
        'shape': arr.shape,
        'mean': float(np.mean(arr)),
        'std': float(np.std(arr)),
        'min': float(np.min(arr)),
        'max': float(np.max(arr)),
        'median': float(np.median(arr))
    }
    return stats


def analyze_training_history(history, run_num):
    """Analyze training history from a run"""
    print(f"\n  Training History:")
    print(f"  {'-'*60}")
    
    for key, values in history.items():
        if isinstance(values, (list, np.ndarray)):
            values = np.array(values)
            if len(values) > 0:
                print(f"    {key}:")
                print(f"      Length: {len(values)} epochs")
                print(f"      Initial: {values[0]:.4f}")
                print(f"      Final: {values[-1]:.4f}")
                print(f"      Best: {np.min(values):.4f} @ epoch {np.argmin(values)+1}")
                print(f"      Mean: {np.mean(values):.4f}")
                
                # Show first and last few values
                if len(values) > 10:
                    print(f"      First 5: {values[:5]}")
                    print(f"      Last 5: {values[-5:]}")


def analyze_predictions(y_true, y_pred, dataset_name="Dataset"):
    """Analyze prediction quality"""
    y_true = np.array(y_true).flatten()
    y_pred = np.array(y_pred).flatten()
    
    # Calculate metrics
    mae = np.mean(np.abs(y_true - y_pred))
    rmse = np.sqrt(np.mean((y_true - y_pred) ** 2))
    
    # R²
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
    r2 = 1 - (ss_res / ss_tot)
    
    # Error distribution
    errors = y_pred - y_true
    
    print(f"\n  {dataset_name} Predictions Analysis:")
    print(f"  {'-'*60}")
    print(f"    Number of samples: {len(y_true)}")
    print(f"    MAE:  {mae:.4f}")
    print(f"    RMSE: {rmse:.4f}")
    print(f"    R²:   {r2:.4f}")
    print(f"\n    True values:")
    print(f"      Mean: {np.mean(y_true):.4f}, Std: {np.std(y_true):.4f}")
    print(f"      Range: [{np.min(y_true):.4f}, {np.max(y_true):.4f}]")
    print(f"\n    Predicted values:")
    print(f"      Mean: {np.mean(y_pred):.4f}, Std: {np.std(y_pred):.4f}")
    print(f"      Range: [{np.min(y_pred):.4f}, {np.max(y_pred):.4f}]")
    print(f"\n    Error distribution:")
    print(f"      Mean error: {np.mean(errors):.4f}")
    print(f"      Std error:  {np.std(errors):.4f}")
    print(f"      Max overpredict:  {np.max(errors):.4f}")
    print(f"      Max underpredict: {np.min(errors):.4f}")
    
    # Sample predictions
    print(f"\n    Sample predictions (first 10):")
    print(f"      {'True':>8} | {'Pred':>8} | {'Error':>8}")
    print(f"      {'-'*28}")
    for i in range(min(10, len(y_true))):
        print(f"      {y_true[i]:>8.2f} | {y_pred[i]:>8.2f} | {errors[i]:>+8.2f}")
    
    return {'mae': mae, 'rmse': rmse, 'r2': r2}


def main():
    # Load the pickle file
    pkl_path = '/home/exouser/multi-objective-pdbbind/multi-objective-pdbbind/PGGCN_results_100structures_20250825_034719.pkl'
    
    print("="*80)
    print("TENSORFLOW PGGCN RESULTS - COMPREHENSIVE ANALYSIS")
    print("="*80)
    print(f"\nLoading: {pkl_path}")
    
    with open(pkl_path, 'rb') as f:
        results = pickle.load(f)
    
    print("✓ Loaded successfully\n")
    
    # ========================================================================
    # EXPERIMENT INFO
    # ========================================================================
    print("\n" + "="*80)
    print("EXPERIMENT INFORMATION")
    print("="*80)
    
    exp_info = results['experiment_info']
    print(f"\nTimestamp: {exp_info['timestamp']}")
    print(f"Dataset size: {exp_info['dataset_size']} structures")
    print(f"Total runtime: {exp_info['total_runtime_minutes']:.2f} minutes ({exp_info['total_runtime_minutes']/60:.2f} hours)")
    
    if 'config' in exp_info:
        print(f"\nConfiguration:")
        for key, value in exp_info['config'].items():
            print(f"  {key:25s}: {value}")
    
    # ========================================================================
    # SUMMARY METRICS
    # ========================================================================
    print("\n" + "="*80)
    print("SUMMARY METRICS (BEST ACROSS ALL RUNS)")
    print("="*80)
    
    summary = results['summary_metrics']
    print(f"\nBest Test MAE:        {summary['best_mae']:.4f} kcal/mol")
    print(f"Best Test Loss:       {summary['best_test_loss']:.4f}")
    print(f"Best Train Loss:      {summary['best_train_loss']:.4f}")
    print(f"Best Physics Weight:  {summary['best_physics_weight']:.2e}")
    
    # ========================================================================
    # ALL RUNS - OVERVIEW
    # ========================================================================
    print("\n" + "="*80)
    print("ALL RUNS OVERVIEW")
    print("="*80)
    
    print(f"\nTotal number of runs: {len(results['all_results'])}")
    print(f"\n{'Run':<5} {'Physics Wt':<12} {'Epochs':<8} {'Train Loss':<12} {'Test Loss':<12} {'MAE':<10}")
    print("-"*75)
    
    for i, run in enumerate(results['all_results'], 1):
        phys_wt = run.get('physics_weight', 'N/A')
        epochs = int(run.get('epochs_trained', 0))
        train_loss = run.get('final_train_loss', 0)
        test_loss = run.get('test_loss', 0)
        mae = run.get('mean_abs_diff', 0)
        
        # Mark the best run
        marker = " ← BEST" if abs(mae - summary['best_mae']) < 0.01 else ""
        
        print(f"{i:<5} {phys_wt:<12.2e} {epochs:<8} {train_loss:<12.2f} {test_loss:<12.2f} {mae:<10.2f}{marker}")
    
    # ========================================================================
    # DETAILED RUN ANALYSIS
    # ========================================================================
    print("\n" + "="*80)
    print("DETAILED RUN-BY-RUN ANALYSIS")
    print("="*80)
    
    for i, run in enumerate(results['all_results'], 1):
        print(f"\n{'='*80}")
        print(f"RUN {i}")
        print(f"{'='*80}")
        
        # Basic info
        print(f"\nBasic Information:")
        print(f"  Physics weight: {run.get('physics_weight', 'N/A'):.2e}")
        print(f"  Dataset size: {run.get('dataset_size', 'N/A')}")
        print(f"  Epochs trained: {run.get('epochs_trained', 'N/A')}")
        print(f"  Final train loss: {run.get('final_train_loss', 'N/A'):.4f}")
        print(f"  Test loss: {run.get('test_loss', 'N/A'):.4f}")
        print(f"  Mean absolute difference (MAE): {run.get('mean_abs_diff', 'N/A'):.4f}")
        
        # Training history
        if 'training_history' in run:
            analyze_training_history(run['training_history'], i)
        
        # Predictions analysis
        if 'y_true_train' in run and 'y_pred_train' in run:
            metrics_train = analyze_predictions(
                run['y_true_train'], 
                run['y_pred_train'], 
                "Training Set"
            )
        
        if 'y_true_test' in run and 'y_pred_test' in run:
            metrics_test = analyze_predictions(
                run['y_true_test'], 
                run['y_pred_test'], 
                "Test Set"
            )
        
        # Config
        if 'config' in run:
            print(f"\n  Configuration:")
            print(f"  {'-'*60}")
            for key, value in run['config'].items():
                print(f"    {key:25s}: {value}")
    
    # ========================================================================
    # PHYSICS WEIGHT COMPARISON
    # ========================================================================
    print("\n" + "="*80)
    print("PHYSICS WEIGHT COMPARISON")
    print("="*80)
    
    physics_weights = {}
    for run in results['all_results']:
        pw = run.get('physics_weight', None)
        if pw is not None:
            if pw not in physics_weights:
                physics_weights[pw] = []
            physics_weights[pw].append({
                'test_loss': run.get('test_loss', 0),
                'mae': run.get('mean_abs_diff', 0),
                'epochs': run.get('epochs_trained', 0)
            })
    
    print(f"\n{'Physics Weight':<15} {'Avg Test Loss':<15} {'Avg MAE':<12} {'Avg Epochs':<12} {'Count':<8}")
    print("-"*65)
    
    for pw in sorted(physics_weights.keys()):
        runs = physics_weights[pw]
        avg_test_loss = np.mean([r['test_loss'] for r in runs])
        avg_mae = np.mean([r['mae'] for r in runs])
        avg_epochs = np.mean([r['epochs'] for r in runs])
        count = len(runs)
        
        marker = " ← BEST" if abs(pw - summary['best_physics_weight']) < 1e-10 else ""
        
        print(f"{pw:<15.2e} {avg_test_loss:<15.2f} {avg_mae:<12.2f} {avg_epochs:<12.1f} {count:<8}{marker}")
    
    # ========================================================================
    # KEY TAKEAWAYS
    # ========================================================================
    print("\n" + "="*80)
    print("KEY TAKEAWAYS")
    print("="*80)
    
    best_run = None
    for run in results['all_results']:
        if abs(run.get('mean_abs_diff', 999) - summary['best_mae']) < 0.01:
            best_run = run
            break
    
    if best_run:
        print(f"\n✓ Best performance achieved with:")
        print(f"   - Physics weight: {best_run.get('physics_weight', 'N/A'):.2e}")
        print(f"   - Converged in: {best_run.get('epochs_trained', 'N/A')} epochs")
        print(f"   - Test MAE: {best_run.get('mean_abs_diff', 'N/A'):.4f} kcal/mol")
        print(f"   - Test Loss (RMSE): {best_run.get('test_loss', 'N/A'):.4f}")
    
    print(f"\n✓ Convergence analysis:")
    epochs_list = [run.get('epochs_trained', 0) for run in results['all_results']]
    print(f"   - Fastest convergence: {min(epochs_list)} epochs")
    print(f"   - Slowest convergence: {max(epochs_list)} epochs")
    print(f"   - Average convergence: {np.mean(epochs_list):.1f} epochs")
    
    print(f"\n✓ Performance range:")
    mae_list = [run.get('mean_abs_diff', 0) for run in results['all_results']]
    print(f"   - Best MAE: {min(mae_list):.4f} kcal/mol")
    print(f"   - Worst MAE: {max(mae_list):.4f} kcal/mol")
    print(f"   - Average MAE: {np.mean(mae_list):.4f} kcal/mol")
    print(f"   - Std Dev MAE: {np.std(mae_list):.4f} kcal/mol")
    
    print(f"\n✓ Configuration used:")
    if 'config' in exp_info:
        config = exp_info['config']
        print(f"   - Batch size: {config.get('batch_size', 'N/A')}")
        print(f"   - Max epochs: {config.get('epochs', 'N/A')}")
        print(f"   - Dataset: {config.get('dataset_size', 'N/A')} structures")
    
    # ========================================================================
    # SAVE SUMMARY TO FILE
    # ========================================================================
    print("\n" + "="*80)
    print("SAVING RESULTS")
    print("="*80)
    
    # Create summary dictionary
    summary_dict = {
        'experiment_info': exp_info,
        'summary_metrics': summary,
        'run_summaries': []
    }
    
    for i, run in enumerate(results['all_results'], 1):
        run_summary = {
            'run_number': i,
            'physics_weight': float(run.get('physics_weight', 0)),
            'epochs_trained': int(run.get('epochs_trained', 0)),
            'final_train_loss': float(run.get('final_train_loss', 0)),
            'test_loss': float(run.get('test_loss', 0)),
            'mae': float(run.get('mean_abs_diff', 0)),
        }
        
        # Add prediction metrics if available
        if 'y_true_test' in run and 'y_pred_test' in run:
            y_true = np.array(run['y_true_test']).flatten()
            y_pred = np.array(run['y_pred_test']).flatten()
            mae = np.mean(np.abs(y_true - y_pred))
            rmse = np.sqrt(np.mean((y_true - y_pred) ** 2))
            run_summary['test_mae_calculated'] = float(mae)
            run_summary['test_rmse_calculated'] = float(rmse)
        
        summary_dict['run_summaries'].append(run_summary)
    
    # Save to JSON (with numpy type conversion)
    def convert_numpy(obj):
        """Convert numpy types to Python types for JSON serialization"""
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, dict):
            return {k: convert_numpy(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [convert_numpy(item) for item in obj]
        return obj
    
    summary_dict = convert_numpy(summary_dict)
    
    output_json = '/home/exouser/multi-objective-pdbbind/multi-objective-pdbbind/tensorflow_results_summary.json'
    with open(output_json, 'w') as f:
        json.dump(summary_dict, f, indent=2)
    print(f"\n✓ Saved summary to: {output_json}")
    
    # Save detailed text report
    # We'll redirect stdout to capture everything
    print(f"\n✓ This detailed analysis is displayed above")
    print(f"\n✓ For PyTorch comparison, see: tensorflow_pytorch_comparison.md")
    
    print("\n" + "="*80)
    print("ANALYSIS COMPLETE")
    print("="*80)


if __name__ == "__main__":
    main()