"""
Overnight Grid Search - 0.01 Step (101 weights) - VERBOSE VERSION
Tracks and displays: Train/Val RMSE, Empirical Loss, Physics Loss
"""

import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import numpy as np
import sys
import os
import random
import time
import json
import pickle
from datetime import timedelta
from sklearn.model_selection import train_test_split

# Add paths
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PARENT_DIR = os.path.dirname(SCRIPT_DIR)

possible_paths = [
    SCRIPT_DIR,
    PARENT_DIR,
    os.path.join(PARENT_DIR, 'PGGCN'),
    '/home/exouser/multi-objective-pdbbind/multi-objective-pdbbind',
]

for path in possible_paths:
    if path not in sys.path:
        sys.path.insert(0, path)

try:
    from PGGCN.models.dcFeaturizer import atom_features as get_atom_features
    from PGGCN.models.layers_pytorch import PGGCNModel
except ImportError:
    try:
        from models.dcFeaturizer import atom_features as get_atom_features
        from models.layers_pytorch import PGGCNModel
    except ImportError as e:
        print(f"Error importing modules: {e}")
        sys.exit(1)

# Import PCGrad
try:
    from pcgrad_pytorch import PCGrad
    PCGRAD_AVAILABLE = True
    print("PCGrad optimizer available")
except ImportError:
    PCGRAD_AVAILABLE = False
    print("Warning: PCGrad not available, will use standard Adam")


def apply_maxnorm_constraint(model, max_norm=3.0):
    """Apply MaxNorm constraint."""
    with torch.no_grad():
        for param in model.parameters():
            if param.requires_grad and param.dim() >= 2:
                norm = param.norm(2, dim=0, keepdim=True)
                desired = torch.clamp(norm, max=max_norm)
                param.mul_(desired / (norm + 1e-7))


def compute_l2_loss(model, l2_weight=1e-4):
    """Compute L2 regularization for all parameters."""
    l2_loss = torch.tensor(0., device=next(model.parameters()).device)
    for param in model.parameters():
        if param.requires_grad:
            l2_loss += torch.sum(param ** 2)
    return (l2_weight / 2) * l2_loss


def featurize(molecule, info):
    """Featurize a molecule."""
    atom_features = []
    for atom in molecule.GetAtoms():
        base_feat = get_atom_features(atom)
        new_feature = base_feat.tolist()
        position = molecule.GetConformer().GetAtomPosition(atom.GetIdx())
        new_feature += [atom.GetMass(), atom.GetAtomicNum(), atom.GetFormalCharge()]
        new_feature += [position.x, position.y, position.z]
        neighbors = atom.GetNeighbors()[:2]
        for neighbor in neighbors:
            new_feature += [float(neighbor.GetIdx())]
        for i in range(2 - len(neighbors)):
            new_feature += [-1.0]
        full_feature = new_feature + info
        atom_features.append(full_feature)
    return np.array(atom_features)


def load_all_data(info_csv_path, hostguest_dir):
    """Load dataset."""
    print(f"Loading PDB files from: {hostguest_dir}")
    with open(hostguest_dir, 'rb') as f:
        pdb_dict = pickle.load(f)
    print(f"Loaded {len(pdb_dict)} PDB files")

    df_all = pd.read_csv(info_csv_path)
    print(f"Loaded {len(df_all)} CSV entries")

    feature_columns = [
        'pb_host_VDWAALS', 'pb_guest_VDWAALS', 'pb_complex_VDWAALS',
        'gb_host_1-4EEL', 'gb_guest_1-4EEL', 'gb_Complex_1-4EEL',
        'gb_host_EELEC', 'gb_guest_EELEC', 'gb_Complex_EELEC',
        'gb_host_EGB', 'gb_guest_EGB', 'gb_Complex_EGB',
        'gb_host_ESURF', 'gb_guest_ESURF', 'gb_Complex_ESURF'
    ]

    X, y = [], []
    
    for pdb_id in list(pdb_dict.keys()):
        if pdb_id not in df_all['Ids'].values:
            continue
        
        molecule = pdb_dict[pdb_id]
        row = df_all[df_all['Ids'] == pdb_id].iloc[0]
        info_array = row[feature_columns].tolist()
        target = row['Ex _G_(kcal/mol)']
        
        try:
            features = featurize(molecule, info_array)
            X.append(torch.FloatTensor(features))
            y.append(target)
        except:
            pass
    
    print(f"Successfully loaded {len(X)} complexes")
    return X, y


def compute_task_losses(predictions, targets, model_vars, physics_info, physics_weight):
    """Compute task losses."""
    targets = targets.view(-1, 1)
    empirical_loss = torch.sqrt(torch.mean((predictions - targets) ** 2))
    
    host_energy = physics_info[:, [0, 3, 6, 9, 12]].sum(dim=1, keepdim=True)
    guest_energy = physics_info[:, [1, 4, 7, 10, 13]].sum(dim=1, keepdim=True)
    complex_energy = physics_info[:, [2, 5, 8, 11, 14]].sum(dim=1, keepdim=True)
    
    dG_physics = complex_energy - (host_energy + guest_energy)
    physics_loss = torch.sqrt(torch.mean((predictions - dG_physics) ** 2))
    weighted_physics_loss = physics_weight * physics_loss
    
    return empirical_loss, weighted_physics_loss, physics_loss


def train_single_config(X_train, y_train, X_val, y_val, physics_weight, device, 
                       epochs=250, lr=0.005, use_pcgrad=True, verbose=True):
    """Train a single configuration with detailed tracking."""
    # Set seed for this config
    seed = 42
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    
    # Create model
    model = PGGCNModel(num_atom_features=36, r_out_channel=20, c_out_channel=128, dropout_rate=0.2)
    model.add_rule("sum", 0, 32)
    model.add_rule("multiply", 32, 33)
    model.add_rule("distance", 33, 36)
    model = model.to(device)
    
    # Optimizer
    base_optimizer = optim.Adam(model.parameters(), lr=lr)
    
    if use_pcgrad and PCGRAD_AVAILABLE:
        optimizer = PCGrad(base_optimizer)
    else:
        optimizer = base_optimizer
    
    # Move data
    X_train_dev = [x.to(device) for x in X_train]
    X_val_dev = [x.to(device) for x in X_val]
    y_train_tensor = torch.FloatTensor(y_train).unsqueeze(1).to(device)
    y_val_tensor = torch.FloatTensor(y_val).unsqueeze(1).to(device)
    
    # Track best metrics
    best_val_loss = float('inf')
    best_val_empirical = float('inf')
    best_val_physics = float('inf')
    best_val_rmse = float('inf')
    best_train_rmse = float('inf')
    
    # Track history
    train_rmse_history = []
    val_rmse_history = []
    train_empirical_history = []
    val_empirical_history = []
    train_physics_history = []
    val_physics_history = []
    
    for epoch in range(epochs):
        # Training
        model.train()
        
        predictions, model_var, physics_info = model(X_train_dev, training=True)
        train_empirical, train_weighted_physics, train_raw_physics = compute_task_losses(predictions, y_train_tensor, model_var, physics_info, physics_weight)
        optimizer.pc_backward([train_empirical, train_weighted_physics])

        optimizer.step()
        apply_maxnorm_constraint(model, max_norm=3.0)
        
        # Compute training RMSE
        train_rmse = torch.sqrt(nn.MSELoss()(predictions, y_train_tensor)).item()
        
        # Validation
        model.eval()
        with torch.no_grad():
            val_pred, val_var, val_phys = model(X_val_dev, training=False)
            val_empirical, val_weighted_physics, val_raw_physics = compute_task_losses(
                val_pred, y_val_tensor, val_var, val_phys, physics_weight
            )
            val_total = val_empirical + val_weighted_physics
            val_rmse = torch.sqrt(nn.MSELoss()(val_pred, y_val_tensor)).item()
        
        # Store history
        train_rmse_history.append(train_rmse)
        val_rmse_history.append(val_rmse)
        train_empirical_history.append(train_empirical.item())
        val_empirical_history.append(val_empirical.item())
        train_physics_history.append(train_raw_physics.item())
        val_physics_history.append(val_raw_physics.item())
        
        # Track best
        if val_total.item() < best_val_loss:
            best_val_loss = val_total.item()
            best_val_empirical = val_empirical.item()
            best_val_physics = val_raw_physics.item()
            best_val_rmse = val_rmse
            best_train_rmse = train_rmse
        
        # Verbose output
        if verbose and ((epoch + 1) % 50 == 0 or epoch == 0):
            print(f"      Epoch {epoch+1:3d}/{epochs} | "
                  f"Train RMSE: {train_rmse:.4f} | Val RMSE: {val_rmse:.4f} | "
                  f"Train Emp: {train_empirical.item():.4f} | Val Emp: {val_empirical.item():.4f} | "
                  f"Train Phys: {train_raw_physics.item():.4f} | Val Phys: {val_raw_physics.item():.4f}")
    
    # Final evaluation
    model.eval()
    with torch.no_grad():
        val_pred, _, _ = model(X_val_dev, training=False)
        final_val_rmse = torch.sqrt(nn.MSELoss()(val_pred, y_val_tensor)).item()
        val_mae = torch.mean(torch.abs(val_pred - y_val_tensor)).item()
        
        train_pred, _, _ = model(X_train_dev, training=False)
        final_train_rmse = torch.sqrt(nn.MSELoss()(train_pred, y_train_tensor)).item()
        train_mae = torch.mean(torch.abs(train_pred - y_train_tensor)).item()
    
    # Cleanup
    del model, optimizer, X_train_dev, X_val_dev
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    
    return {
        'physics_weight': physics_weight,
        
        # Final metrics
        'val_rmse': final_val_rmse,
        'val_mae': val_mae,
        'train_rmse': final_train_rmse,
        'train_mae': train_mae,
        
        # Best metrics during training
        'best_val_total_loss': best_val_loss,
        'best_val_empirical_loss': best_val_empirical,
        'best_val_physics_loss': best_val_physics,
        'best_val_rmse': best_val_rmse,
        'best_train_rmse': best_train_rmse,
        
        # Training history (for plotting)
        'train_rmse_history': train_rmse_history,
        'val_rmse_history': val_rmse_history,
        'train_empirical_history': train_empirical_history,
        'val_empirical_history': val_empirical_history,
        'train_physics_history': train_physics_history,
        'val_physics_history': val_physics_history,
    }


def main():
    print("="*80)
    print("OVERNIGHT GRID SEARCH - VERBOSE VERSION")
    print("Tracking: Train/Val RMSE, Empirical Loss, Physics Loss")
    print("="*80)
    
    # Configuration
    physics_weights = [round(i * 0.01, 6) for i in range(101)]  # 0.00 to 1.00
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    epochs = 250
    use_pcgrad = PCGRAD_AVAILABLE
    
    print(f"\nConfiguration:")
    print(f"  Device: {device}")
    print(f"  Optimizer: {'PCGrad + Adam' if use_pcgrad else 'Adam'}")
    print(f"  Weights: {len(physics_weights)} (0.00 to 1.00, step 0.01)")
    print(f"  Epochs per weight: {epochs}")
    
    # Time estimate
    time_per_weight_min = 30
    total_time_min = len(physics_weights) * time_per_weight_min
    total_hours = total_time_min / 60
    print(f"\n  Estimated time: {total_hours:.1f} hours ({total_hours/24:.1f} days)")
    print(f"  Start: {time.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"  Est. finish: {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(time.time() + total_time_min*60))}")
    
    # Paths
    info_csv_path = '/home/exouser/multi-objective-pdbbind/multi-objective-pdbbind/Datasets/Final_data_DDG.csv'
    hostguest_dir = '/home/exouser/multi-objective-pdbbind/multi-objective-pdbbind/Datasets/PDBs_RDKit_BFE.pkl'
    output_dir = '/home/exouser/multi-objective-pdbbind/multi-objective-pdbbind/pytorch-implementation/grid_search_results'
    
    print("\n" + "="*80)
    print("Loading Data...")
    print("="*80)
    X, y = load_all_data(info_csv_path, hostguest_dir)
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)
    print(f"Training: {len(X_train)}, Validation: {len(X_val)}")
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Run grid search
    results = []
    total_start = time.time()
    
    print("\n" + "="*80)
    print("Starting Grid Search")
    print("="*80)
    
    for i, weight in enumerate(physics_weights, 1):
        iter_start = time.time()
        
        print(f"\n[{i}/{len(physics_weights)}] Physics Weight = {weight:.6f}")
        print(f"  Started: {time.strftime('%Y-%m-%d %H:%M:%S')}")
        
        try:
            result = train_single_config(
                X_train, y_train, X_val, y_val,
                weight, device,
                epochs=epochs,
                use_pcgrad=use_pcgrad,
                verbose=True
            )
            
            results.append(result)
            
            iter_time = time.time() - iter_start
            print(f"\n  ✓ Completed in {timedelta(seconds=int(iter_time))}")
            print(f"  ┌─ Final Metrics ─────────────────────────")
            print(f"  │ Train RMSE: {result['train_rmse']:.4f}")
            print(f"  │ Val RMSE:   {result['val_rmse']:.4f}")
            print(f"  │ Train MAE:  {result['train_mae']:.4f}")
            print(f"  │ Val MAE:    {result['val_mae']:.4f}")
            print(f"  ├─ Best During Training ──────────────────")
            print(f"  │ Best Train RMSE: {result['best_train_rmse']:.4f}")
            print(f"  │ Best Val RMSE:   {result['best_val_rmse']:.4f}")
            print(f"  │ Best Empirical:  {result['best_val_empirical_loss']:.4f}")
            print(f"  │ Best Physics:    {result['best_val_physics_loss']:.4f}")
            print(f"  └─────────────────────────────────────────")
            
            # Progress estimate
            elapsed = time.time() - total_start
            avg_time = elapsed / i
            remaining = avg_time * (len(physics_weights) - i)
            pct_complete = (i / len(physics_weights)) * 100
            
            print(f"\n  Progress: {pct_complete:.1f}% ({i}/{len(physics_weights)})")
            print(f"  Elapsed: {timedelta(seconds=int(elapsed))}")
            print(f"  Estimated remaining: {timedelta(seconds=int(remaining))}")
            print(f"  Est. completion: {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(time.time() + remaining))}")
            
            # Save checkpoint every 10 iterations
            if i % 10 == 0 or i == len(physics_weights):
                checkpoint_path = os.path.join(output_dir, f'results_step0.01_checkpoint_{i}.json')
                with open(checkpoint_path, 'w') as f:
                    json.dump(results, f, indent=2)
                print(f"  ✓ Checkpoint saved: {checkpoint_path}")
            
        except Exception as e:
            print(f"  ✗ ERROR: {e}")
            import traceback
            traceback.print_exc()
            continue
    
    # Final save
    total_time = time.time() - total_start
    
    print("\n" + "="*80)
    print("GRID SEARCH COMPLETED")
    print("="*80)
    print(f"Total time: {timedelta(seconds=int(total_time))}")
    print(f"Successful runs: {len(results)}/{len(physics_weights)}")
    print(f"Finished: {time.strftime('%Y-%m-%d %H:%M:%S')}")
    
    # Save final results
    final_path = os.path.join(output_dir, 'results_step0.01_FINAL.json')
    with open(final_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\n✓ Final results saved: {final_path}")
    
    # Print top 10
    if results:
        sorted_results = sorted(results, key=lambda x: x['val_rmse'])
        
        print("\n" + "="*80)
        print("TOP 10 CONFIGURATIONS")
        print("="*80)
        print(f"{'Rank':<6} {'Weight':<10} {'Val RMSE':<10} {'Train RMSE':<11} {'Emp Loss':<10} {'Phys Loss':<10}")
        print("-"*67)
        
        for rank, result in enumerate(sorted_results[:10], 1):
            print(f"{rank:<6} {result['physics_weight']:<10.4f} "
                  f"{result['val_rmse']:<10.4f} {result['train_rmse']:<11.4f} "
                  f"{result['best_val_empirical_loss']:<10.4f} "
                  f"{result['best_val_physics_loss']:<10.4f}")
        
        best = sorted_results[0]
        print("\n" + "="*80)
        print("BEST CONFIGURATION")
        print("="*80)
        print(f"Physics Weight:     {best['physics_weight']:.6f}")
        print(f"Val RMSE:           {best['val_rmse']:.4f}")
        print(f"Train RMSE:         {best['train_rmse']:.4f}")
        print(f"Val MAE:            {best['val_mae']:.4f}")
        print(f"Train MAE:          {best['train_mae']:.4f}")
        print(f"Best Empirical:     {best['best_val_empirical_loss']:.4f}")
        print(f"Best Physics:       {best['best_val_physics_loss']:.4f}")
        print("="*80)
    
    print("\n✓ All done! You can now run plot_combined_results.py to visualize.")


if __name__ == "__main__":
    main()